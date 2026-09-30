"""Chunked, self-contained storage for a Raystrack run.

The manifest is the commit record. Chunk directories are immutable and become
visible to readers only after their descriptor is committed to the manifest.
"""

from __future__ import annotations

import copy
import json
import math
import os
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np

from .params import MatrixParams, SkyParams


FORMAT = "raystrack-store"
FORMAT_VERSION = 1
DEFAULT_CHUNK_BYTES = 4_194_304
MAX_RESULT_ROWS = 256
RESULT_KINDS = ("scene", "sky", "rest")
ResultRow = Tuple[str, Mapping[str, float]]
ResultRows = Union[Mapping[str, Mapping[str, float]], Iterable[ResultRow]]


def _store_path(path: Union[str, Path]) -> Path:
    result = Path(path)
    if not result.suffix:
        result = result.with_suffix(".raystrack")
    if result.suffix != ".raystrack":
        raise ValueError("Store path must end in .raystrack")
    return result


def _name(value: str, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{role} must be a non-empty string")
    return value


def _json_copy(value: object) -> object:
    """Copy JSON-compatible data, rejecting NaN and non-string object keys."""
    def check(item: object) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("Metadata object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _new_manifest(chunk_bytes: int) -> dict:
    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "complete": False,
        "chunk_bytes": chunk_bytes,
        "params": {"matrix": None, "sky": None},
        "metadata": {},
        "meshes": [],
        "results": {
            kind: {"senders": [], "receivers": [], "chunks": []}
            for kind in RESULT_KINDS
        },
    }


def _lock_writer(path: Path):
    """Hold a process-wide, nonblocking OS lock for this store's writer."""
    lock = (path / ".writer.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise RuntimeError(f"Another writer has {path} open") from exc
    return lock


class RunStore:
    """Read or append a versioned ``.raystrack`` directory.

    ``w`` creates a new directory, ``a`` resumes an unfinished directory, and
    ``r`` reads committed chunks. Pending rows are committed by ``flush()``,
    ``finalize()``, or a successful context-manager exit.
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "r",
        *,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self.path = _store_path(path).resolve()
        if mode not in ("r", "w", "a"):
            raise ValueError("mode must be 'r', 'w', or 'a'")
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 12:
            raise ValueError("chunk_bytes must be an integer of at least 12")
        self.mode = mode
        self._closed = False
        self._lock = None
        self._pending: Dict[str, List[ResultRow]] = {kind: [] for kind in RESULT_KINDS}
        self._pending_names = {kind: set() for kind in RESULT_KINDS}
        self._pending_bytes = {kind: 8 for kind in RESULT_KINDS}

        if mode == "w":
            self.path.mkdir(parents=True, exist_ok=False)
        elif not self.path.is_dir():
            raise FileNotFoundError(f"Store directory not found: {self.path}")

        try:
            if mode != "r":
                self._lock = _lock_writer(self.path)
            if mode == "w":
                self._manifest = _new_manifest(chunk_bytes)
                self._commit(copy.deepcopy(self._manifest))
            else:
                with (self.path / "manifest.json").open("r", encoding="utf-8") as fh:
                    self._manifest = json.load(fh)
                if self._manifest.get("format") != FORMAT:
                    raise ValueError("Not a Raystrack store")
                if self._manifest.get("version") != FORMAT_VERSION:
                    raise ValueError(
                        f"Unsupported Raystrack store version: {self._manifest.get('version')}"
                    )
                if mode == "a" and self._manifest["complete"]:
                    raise ValueError("Cannot append to a finalized store")
            self._mesh_entries = {entry["name"]: entry for entry in self._manifest["meshes"]}
            self._sender_indices = {
                kind: {name: index for index, name in enumerate(self._manifest["results"][kind]["senders"])}
                for kind in RESULT_KINDS
            }
        except BaseException:
            self._release_lock()
            raise

    @property
    def complete(self) -> bool:
        return bool(self._manifest["complete"])

    @property
    def metadata(self) -> dict:
        return copy.deepcopy(self._manifest["metadata"])

    @property
    def matrix_params(self) -> Optional[MatrixParams]:
        data = self._manifest["params"]["matrix"]
        return MatrixParams.from_dict(copy.deepcopy(data)) if data is not None else None

    @property
    def sky_params(self) -> Optional[SkyParams]:
        data = self._manifest["params"]["sky"]
        return SkyParams.from_dict(copy.deepcopy(data)) if data is not None else None

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("Store is closed")

    def _check_writable(self) -> None:
        self._check_open()
        if self.mode == "r":
            raise ValueError("Store is read-only")
        if self.complete:
            raise ValueError("Store is finalized")

    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _commit(self, manifest: dict) -> None:
        temporary = self.path / f".manifest-{os.getpid()}-{id(manifest)}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, self.path / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)
        self._manifest = manifest

    def _chunk_path(self, relative: str) -> Path:
        path = (self.path / relative).resolve()
        try:
            path.relative_to(self.path)
        except ValueError as exc:
            raise ValueError("Chunk path escapes the store") from exc
        return path

    @staticmethod
    def _next_directory(parent: Path, start: int = 0) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        index = start
        while True:
            directory = parent / f"{index:06d}"
            try:
                directory.mkdir()
                return directory
            except FileExistsError:
                index += 1

    def set_params(
        self,
        *,
        matrix: Optional[MatrixParams] = None,
        sky: Optional[SkyParams] = None,
    ) -> None:
        self._check_writable()
        if matrix is not None and not isinstance(matrix, MatrixParams):
            raise TypeError("matrix must be MatrixParams")
        if sky is not None and not isinstance(sky, SkyParams):
            raise TypeError("sky must be SkyParams")
        manifest = copy.deepcopy(self._manifest)
        if matrix is not None:
            manifest["params"]["matrix"] = matrix.as_dict()
        if sky is not None:
            manifest["params"]["sky"] = sky.as_dict()
        self._commit(manifest)

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        self._check_writable()
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        value = _json_copy(dict(metadata))
        manifest = copy.deepcopy(self._manifest)
        manifest["metadata"] = value
        self._commit(manifest)

    def add_mesh(self, name: str, vertices: np.ndarray, faces: np.ndarray) -> None:
        self._check_writable()
        name = _name(name, "Mesh name")
        if name in self._mesh_entries:
            raise ValueError(f"Duplicate mesh name: {name}")
        verts = np.asarray(vertices, dtype=np.float32)
        raw_faces = np.asarray(faces)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if not np.isfinite(verts).all():
            raise ValueError("vertices must be finite")
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if raw_faces.dtype.kind not in "iu":
            raise TypeError("face indices must be integers")
        if raw_faces.size and (raw_faces.min() < 0 or raw_faces.max() >= len(verts)):
            raise ValueError("face indices are out of bounds")
        if raw_faces.size and raw_faces.max() > np.iinfo(np.int32).max:
            raise ValueError("face indices exceed int32 range")
        face_array = np.asarray(raw_faces, dtype=np.int32)
        directory = self._next_directory(self.path / "geometry", len(self._manifest["meshes"]))
        descriptor = {"name": name, "vertices": {"rows": len(verts), "chunks": []},
                      "faces": {"rows": len(face_array), "chunks": []}}
        for key, array in (("vertices", verts), ("faces", face_array)):
            rows_per_chunk = max(1, self._manifest["chunk_bytes"] // array.dtype.itemsize // 3)
            for part, start in enumerate(range(0, len(array), rows_per_chunk)):
                filename = directory / f"{key}-{part:04d}.npy"
                with filename.open("wb") as fh:
                    np.save(fh, np.ascontiguousarray(array[start:start + rows_per_chunk]),
                            allow_pickle=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                descriptor[key]["chunks"].append({"path": filename.relative_to(self.path).as_posix(),
                                                  "rows": min(rows_per_chunk, len(array) - start)})
        manifest = copy.deepcopy(self._manifest)
        manifest["meshes"].append(descriptor)
        self._commit(manifest)
        self._mesh_entries[name] = descriptor

    def iter_mesh_chunks(self, name: str, array: str) -> Iterator[np.ndarray]:
        self._check_open()
        if array not in ("vertices", "faces"):
            raise ValueError("array must be 'vertices' or 'faces'")
        descriptor = self._mesh_entries.get(name)
        if descriptor is None:
            raise KeyError(name)
        for chunk in descriptor[array]["chunks"]:
            yield np.load(self._chunk_path(chunk["path"]), mmap_mode="r", allow_pickle=False)

    def load_meshes(self, names: Optional[Iterable[str]] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        self._check_open()
        available = self._mesh_entries.keys()
        wanted = available if names is None else set([names] if isinstance(names, str) else names)
        unknown = wanted - available
        if unknown:
            raise KeyError(next(iter(unknown)))
        result = []
        for entry in self._manifest["meshes"]:
            name = entry["name"]
            if name not in wanted:
                continue
            arrays = []
            for key, dtype in (("vertices", np.float32), ("faces", np.int32)):
                parts = list(self.iter_mesh_chunks(name, key))
                arrays.append(np.concatenate(parts) if parts else np.empty((0, 3), dtype=dtype))
            result.append((name, arrays[0], arrays[1]))
        return result

    def append_result_rows(self, kind: str, rows: ResultRows) -> None:
        self._check_writable()
        if kind not in RESULT_KINDS:
            raise ValueError(f"kind must be one of {RESULT_KINDS}")
        iterator = rows.items() if isinstance(rows, Mapping) else iter(rows)
        existing = self._sender_indices[kind]
        for sender, row in iterator:
            sender = _name(sender, "Sender name")
            if sender in existing or sender in self._pending_names[kind]:
                raise ValueError(f"Duplicate {kind} sender: {sender}")
            if not isinstance(row, Mapping):
                raise TypeError("Each result row must be a receiver mapping")
            clean: Dict[str, float] = {}
            for receiver, value in row.items():
                if not isinstance(receiver, str):
                    raise TypeError("Receiver names must be strings")
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TypeError("Result values must be numeric") from exc
                if not math.isfinite(number):
                    raise ValueError("Result values must be finite")
                clean[receiver] = number
            row_bytes = 8 + 12 * len(clean)
            if self._pending[kind] and (
                len(self._pending[kind]) >= MAX_RESULT_ROWS
                or self._pending_bytes[kind] + row_bytes > self._manifest["chunk_bytes"]
            ):
                self._flush_kind(kind)
            self._pending[kind].append((sender, clean))
            self._pending_bytes[kind] += row_bytes
            self._pending_names[kind].add(sender)
            if len(self._pending[kind]) >= MAX_RESULT_ROWS or self._pending_bytes[kind] >= self._manifest["chunk_bytes"]:
                self._flush_kind(kind)

    def _flush_kind(self, kind: str) -> None:
        rows = self._pending[kind]
        if not rows:
            return
        result = self._manifest["results"][kind]
        receiver_names = list(result["receivers"])
        receiver_ids = {name: index for index, name in enumerate(receiver_names)}
        offsets = [0]
        columns = []
        values = []
        for _, row in rows:
            for receiver, value in row.items():
                if receiver not in receiver_ids:
                    receiver_ids[receiver] = len(receiver_names)
                    receiver_names.append(receiver)
                columns.append(receiver_ids[receiver])
                values.append(value)
            offsets.append(len(values))
        if len(receiver_names) > np.iinfo(np.int32).max:
            raise ValueError("Too many receiver names for int32 columns")
        directory = self._next_directory(self.path / "results" / kind, len(result["chunks"]))
        for key, array in (
            ("offsets", np.asarray(offsets, dtype=np.int64)),
            ("columns", np.asarray(columns, dtype=np.int32)),
            ("values", np.asarray(values, dtype=np.float64)),
        ):
            with (directory / f"{key}.npy").open("wb") as fh:
                np.save(fh, array, allow_pickle=False)
                fh.flush()
                os.fsync(fh.fileno())
        manifest = copy.deepcopy(self._manifest)
        target = manifest["results"][kind]
        target["receivers"] = receiver_names
        target["chunks"].append({"path": directory.relative_to(self.path).as_posix(),
                                 "start": len(target["senders"]), "count": len(rows)})
        target["senders"].extend(sender for sender, _ in rows)
        self._commit(manifest)
        for index, (sender, _) in enumerate(rows, start=target["chunks"][-1]["start"]):
            self._sender_indices[kind][sender] = index
        self._pending[kind] = []
        self._pending_names[kind] = set()
        self._pending_bytes[kind] = 8

    def flush(self) -> None:
        self._check_writable()
        for kind in RESULT_KINDS:
            self._flush_kind(kind)

    def finalize(self) -> None:
        self.flush()
        manifest = copy.deepcopy(self._manifest)
        manifest["complete"] = True
        self._commit(manifest)

    def iter_result_rows(
        self, kind: str, senders: Optional[Iterable[str]] = None
    ) -> Iterator[Tuple[str, Dict[str, float]]]:
        self._check_open()
        if kind not in RESULT_KINDS:
            raise ValueError(f"kind must be one of {RESULT_KINDS}")
        result = self._manifest["results"][kind]
        if senders is None:
            selected_indices = None
        else:
            wanted = set([senders] if isinstance(senders, str) else senders)
            unknown = wanted - self._sender_indices[kind].keys()
            if unknown:
                raise KeyError(next(iter(unknown)))
            selected_indices = sorted(self._sender_indices[kind][name] for name in wanted)
        receiver_names = result["receivers"]
        for chunk in result["chunks"]:
            start = chunk["start"]
            stop = start + chunk["count"]
            if selected_indices is None:
                indices = range(start, stop)
            else:
                first = bisect_left(selected_indices, start)
                last = bisect_left(selected_indices, stop, first)
                indices = selected_indices[first:last]
            if not indices:
                continue
            directory = self._chunk_path(chunk["path"])
            offsets = np.load(directory / "offsets.npy", mmap_mode="r", allow_pickle=False)
            columns = np.load(directory / "columns.npy", mmap_mode="r", allow_pickle=False)
            values = np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False)
            for sender_index in indices:
                i = sender_index - start
                sender = result["senders"][sender_index]
                first, last = int(offsets[i]), int(offsets[i + 1])
                row = {receiver_names[int(columns[j])]: float(values[j])
                       for j in range(first, last)}
                yield sender, row

    def load_result(self, kind: str, senders: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
        return dict(self.iter_result_rows(kind, senders))

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.mode != "r" and not self.complete:
                self.flush()
        finally:
            self._release_lock()
            self._closed = True

    def __enter__(self) -> "RunStore":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._pending = {kind: [] for kind in RESULT_KINDS}
            self._pending_names = {kind: set() for kind in RESULT_KINDS}
            self._release_lock()
            self._closed = True


def open_store(
    path: Union[str, Path], mode: str = "r", *, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> RunStore:
    """Open a chunked Raystrack run as a context manager."""
    return RunStore(path, mode, chunk_bytes=chunk_bytes)


def save_run(
    path: Union[str, Path],
    *,
    meshes: Iterable[Tuple[str, np.ndarray, np.ndarray]],
    matrix_params: Optional[MatrixParams] = None,
    sky_params: Optional[SkyParams] = None,
    scene: Optional[ResultRows] = None,
    sky: Optional[ResultRows] = None,
    rest: Optional[ResultRows] = None,
    metadata: Optional[Mapping[str, object]] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> str:
    """Save a complete run; refuse to replace an existing directory."""
    with open_store(path, "w", chunk_bytes=chunk_bytes) as store:
        store.set_params(matrix=matrix_params, sky=sky_params)
        if metadata is not None:
            store.set_metadata(metadata)
        for name, vertices, faces in meshes:
            store.add_mesh(name, vertices, faces)
        for kind, rows in (("scene", scene), ("sky", sky), ("rest", rest)):
            if rows is not None:
                store.append_result_rows(kind, rows)
        store.finalize()
        return str(store.path)


__all__ = ["RunStore", "open_store", "save_run"]
