"""Typed, chunked array storage shared by the v2 writer and v1 importer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np


DEFAULT_CHUNK_BYTES = 4_194_304
FORMAT = "raystrack-store"


def integer(value, name, *, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def store_path(path):
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".raystrack")
    if path.suffix != ".raystrack":
        raise ValueError("Store path must end in .raystrack")
    return path.resolve()


def contained_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid array path in store")
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Array path escapes the store")
    result = (root / path).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError("Array path escapes the store") from exc
    return result


def json_value(value):
    """Detach frozen mappings and reject ambiguous/non-finite JSON values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Metadata object keys must be strings")
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(child) for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Metadata and statistics must contain finite JSON numbers")
        return value
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def read_manifest(root):
    filename = root / "manifest.json"
    # A manifest contains descriptors, not numerical payloads.
    if filename.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Store manifest is too large")
    with filename.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, parse_constant=lambda value: _invalid_constant(value))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Not a Raystrack store")
    integer(manifest.get("version"), "Store version", minimum=1)
    return manifest


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def write_manifest(root, manifest):
    temporary = root / ".manifest.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def write_array(root, relative, array, chunk_bytes):
    array = np.asarray(array)
    if array.ndim < 1 or array.dtype.kind not in "biuf":
        raise TypeError("Stored arrays must be numeric and have at least one dimension")
    directory = contained_path(root, relative)
    directory.mkdir(parents=True, exist_ok=False)
    row_bytes = array.dtype.itemsize * max(1, int(np.prod(array.shape[1:])))
    rows_per_chunk = max(1, chunk_bytes // row_bytes)
    descriptor = {"dtype": array.dtype.str, "shape": list(array.shape), "chunks": []}
    for part, start in enumerate(range(0, len(array), rows_per_chunk)):
        filename = directory / f"{part:06d}.npy"
        section = np.ascontiguousarray(array[start:start + rows_per_chunk])
        with filename.open("xb") as stream:
            np.save(stream, section, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor["chunks"].append({"path": filename.relative_to(root).as_posix(), "rows": len(section)})
    return descriptor


def load_array_file(filename, dtype, shape):
    """Validate a memory-mapped payload before copying or concatenating it."""
    array = np.load(filename, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError("Array chunks must be .npy arrays")
    if array.dtype != np.dtype(dtype) or array.shape != tuple(shape):
        raise ValueError("Array chunk dtype or shape does not match its descriptor")
    return array


def read_array(root, descriptor, dtype, *, trailing_shape=None):
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid array descriptor")
    raw_shape = descriptor.get("shape")
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError("Array descriptor must specify its shape")
    shape = tuple(integer(size, "Array dimension") for size in raw_shape)
    if trailing_shape is not None and shape[1:] != tuple(trailing_shape):
        raise ValueError("Array descriptor has an invalid shape")
    try:
        if not isinstance(descriptor.get("dtype"), str):
            raise ValueError("Array dtype must be a string")
        stored_dtype = np.dtype(descriptor["dtype"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid array dtype") from exc
    if stored_dtype != np.dtype(dtype):
        raise ValueError("Array descriptor has an invalid dtype")
    chunks = descriptor.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Array descriptor must contain chunks")
    parts = []
    count = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid array chunk descriptor")
        rows = integer(chunk.get("rows"), "Chunk rows", minimum=1)
        count += rows
        if count > shape[0]:
            raise ValueError("Array chunk row count exceeds its descriptor")
        filename = contained_path(root, chunk.get("path"))
        parts.append(load_array_file(filename, dtype, (rows,) + shape[1:]))
    if count != shape[0]:
        raise ValueError("Array chunk row count does not match its descriptor")
    return np.concatenate(parts) if parts else np.empty(shape, dtype=dtype)
