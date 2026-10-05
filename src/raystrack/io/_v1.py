"""Read-only adapters for v1 stores and JSON; never synthesize statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._arrays import contained_path, integer, read_array


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Legacy names must be non-empty strings")
    return value


def _names(value):
    if not isinstance(value, list):
        raise ValueError("Legacy name table must be a list")
    names = [_name(name) for name in value]
    if len(set(names)) != len(names):
        raise ValueError("Legacy name table contains duplicates")
    return names


def _legacy_geometry(root, description, dtype):
    if not isinstance(description, dict):
        raise ValueError("Invalid legacy geometry descriptor")
    rows = integer(description.get("rows"), "Legacy geometry rows")
    return read_array(root, {"dtype": np.dtype(dtype).str, "shape": [rows, 3],
                             "chunks": description.get("chunks")}, dtype, trailing_shape=(3,))


def _legacy_vector(filename, dtype):
    array = np.load(filename, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError("Legacy array chunks must be .npy arrays")
    if array.dtype != np.dtype(dtype) or array.ndim != 1:
        raise ValueError("Legacy result array has an invalid dtype or shape")
    return array


def _legacy_rows(root, description):
    if not isinstance(description, dict):
        raise ValueError("Invalid legacy result descriptor")
    senders = _names(description.get("senders"))
    receivers = _names(description.get("receivers"))
    chunks = description.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Legacy result chunks must be a list")
    rows = {}
    expected_start = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid legacy result chunk")
        start = integer(chunk.get("start"), "Legacy chunk start")
        count = integer(chunk.get("count"), "Legacy chunk count", minimum=1)
        if start != expected_start or start + count > len(senders):
            raise ValueError("Legacy result chunk sender range is invalid")
        relative = chunk.get("path")
        directory = contained_path(root, relative)
        offsets = _legacy_vector(contained_path(root, f"{relative}/offsets.npy"), np.int64)
        columns = _legacy_vector(contained_path(root, f"{relative}/columns.npy"), np.int32)
        values = _legacy_vector(contained_path(root, f"{relative}/values.npy"), np.float64)
        if not directory.is_dir() or len(offsets) != count + 1 or len(columns) != len(values):
            raise ValueError("Legacy sparse array lengths are inconsistent")
        if offsets[0] != 0 or offsets[-1] != len(values) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Legacy sparse offsets are invalid")
        if columns.size and (columns.min() < 0 or columns.max() >= len(receivers)):
            raise ValueError("Legacy sparse columns are out of bounds")
        if not np.isfinite(values).all():
            raise ValueError("Legacy result values must be finite")
        for index in range(count):
            first, last = int(offsets[index]), int(offsets[index + 1])
            row = {}
            for entry in range(first, last):
                receiver = receivers[int(columns[entry])]
                if receiver in row:
                    raise ValueError("Legacy result row contains duplicate receivers")
                row[receiver] = float(values[entry])
            rows[senders[start + index]] = row
        expected_start += count
    if expected_start != len(senders):
        raise ValueError("Legacy result chunks do not cover their sender table")
    return rows, receivers


def _channel(kind, name, surface_ids):
    from ..solver import Channel

    if kind == "sky":
        if name == "Sky":
            return Channel("sky")
        if name.startswith("Sky_Patch_"):
            try:
                patch = int(name[len("Sky_Patch_"):]) - 1
            except ValueError as exc:
                raise ValueError("Invalid legacy sky patch") from exc
            if patch < 0 or patch >= 145:
                raise ValueError("Invalid legacy sky patch")
            return Channel("sky", patch=patch)
        raise ValueError(f"Unknown legacy sky receiver: {name}")
    if kind == "rest":
        if name != "Rest":
            raise ValueError(f"Unknown legacy rest receiver: {name}")
        return Channel("rest")
    # A real surface called "panel_front" takes priority over suffix parsing.
    if name in surface_ids:
        return Channel("surface", surface_id=name)
    for suffix, side in (("_front", "front"), ("_back", "back")):
        if name.endswith(suffix):
            return Channel("surface", surface_id=name[:-len(suffix)], side=side)
    return Channel("surface", surface_id=name)


def _result(scene, results, receiver_tables, *, complete, source):
    from ..solver import Result, SparseValues

    surface_ids = {surface.surface_id for surface in scene.surfaces}
    senders = [surface.surface_id for surface in scene.surfaces]
    sender_indices = {sender: index for index, sender in enumerate(senders)}
    for rows in results.values():
        for sender in rows:
            if sender not in sender_indices:
                sender_indices[sender] = len(senders)
                senders.append(sender)
    channels = []
    channel_indices_by_value = {}
    receiver_channels = {}
    for kind in ("scene", "sky", "rest"):
        for receiver in receiver_tables[kind]:
            channel = _channel(kind, receiver, surface_ids)
            if channel not in channel_indices_by_value:
                channel_indices_by_value[channel] = len(channels)
                channels.append(channel)
            receiver_channels[kind, receiver] = channel_indices_by_value[channel]
    row_indices, channel_indices, estimates = [], [], []
    entries = {}
    for kind in ("scene", "sky", "rest"):
        for sender, row in results[kind].items():
            for receiver, estimate in row.items():
                key = (sender_indices[sender], receiver_channels[kind, receiver])
                if key in entries:
                    raise ValueError("Legacy receiver names collapse to the same structured channel")
                entries[key] = estimate
    for (row, column), estimate in sorted(entries.items()):
        row_indices.append(row)
        channel_indices.append(column)
        estimates.append(estimate)
    if not any(results.values()) and not channels:
        return None
    return Result(
        sender_ids=tuple(senders), channels=tuple(channels),
        data=SparseValues(np.asarray(row_indices, dtype=np.int64),
                          np.asarray(channel_indices, dtype=np.int64),
                          np.asarray(estimates, dtype=np.float64),
                          np.full(len(estimates), np.nan, dtype=np.float64)),
        coverage=np.full(len(senders), -1, dtype=np.int8),
        statistics={}, execution={}, scene_revision=scene.revision,
        rays_used=None, cumulative_rays=None, status="imported", converged=None,
        elapsed_ms=0.0,
        provenance=({"operation": "import_v1", "source": source,
                     "source_complete": bool(complete),
                     "unknown": ("coverage", "sampling_errors", "statistics", "ray_counts", "convergence"),
                     "sky_patch_mapping": "v1 one-based to v2 zero-based"},),
    )


def read_store(root, manifest):
    from ..model import Mesh, Scene
    from . import StoredRun

    if not isinstance(manifest.get("complete"), bool):
        raise ValueError("Legacy store complete flag must be a boolean")
    entries = manifest.get("meshes")
    if not isinstance(entries, list):
        raise ValueError("Legacy mesh table must be a list")
    meshes = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid legacy mesh descriptor")
        name = _name(entry.get("name"))
        if name in meshes:
            raise ValueError("Duplicate legacy mesh name")
        meshes[name] = Mesh(_legacy_geometry(root, entry.get("vertices"), np.float32),
                            _legacy_geometry(root, entry.get("faces"), np.int32))
    scene = Scene.from_meshes(meshes)
    descriptions = manifest.get("results")
    if not isinstance(descriptions, dict):
        raise ValueError("Legacy results must be an object")
    results, receiver_tables = {}, {}
    for kind in ("scene", "sky", "rest"):
        results[kind], receiver_tables[kind] = _legacy_rows(root, descriptions.get(kind))
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Legacy metadata must be an object")
    metadata = dict(metadata)
    metadata["v1_import"] = {"params": manifest.get("params", {}), "complete": bool(manifest.get("complete"))}
    return StoredRun(scene, _result(scene, results, receiver_tables,
                                   complete=manifest.get("complete", False), source="store"), metadata)


def _json_file(filename):
    from ._arrays import _invalid_constant
    with Path(filename).open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_invalid_constant)


def _json_result(filename):
    if filename is None:
        return {}, []
    rows = _json_file(filename)
    if not isinstance(rows, dict):
        raise ValueError("Legacy result JSON must contain an object of rows")
    result, receivers = {}, []
    for sender, row in rows.items():
        _name(sender)
        if not isinstance(row, dict):
            raise ValueError("Legacy result row must be an object")
        clean = {}
        for receiver, value in row.items():
            _name(receiver)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Legacy result values must be finite numbers")
            clean[receiver] = float(value)
            if receiver not in receivers:
                receivers.append(receiver)
        result[sender] = clean
    return result, receivers


def read_json(meshes_path, scene_path=None, sky_path=None, rest_path=None):
    from ..model import Mesh, Scene
    from . import StoredRun

    payload = _json_file(meshes_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("meshes"), list):
        raise ValueError("Legacy geometry JSON must contain a meshes list")
    meshes = {}
    for entry in payload["meshes"]:
        if not isinstance(entry, dict):
            raise ValueError("Legacy geometry entry must be an object")
        name = _name(entry.get("name"))
        if name in meshes:
            raise ValueError("Duplicate legacy mesh name")
        vertices, faces = entry.get("vertices"), entry.get("faces")
        # Empty arrays lose their second dimension in v1 JSON.
        if vertices == []:
            vertices = np.empty((0, 3), np.float32)
        if faces == []:
            faces = np.empty((0, 3), np.int32)
        meshes[name] = Mesh(vertices, faces)
    scene = Scene.from_meshes(meshes)
    results, tables = {}, {}
    for kind, filename in (("scene", scene_path), ("sky", sky_path), ("rest", rest_path)):
        results[kind], tables[kind] = _json_result(filename)
    result = _result(scene, results, tables, complete=False, source="json")
    return StoredRun(scene, result, {"v1_import": {"format": "json", "statistics_available": False}})
