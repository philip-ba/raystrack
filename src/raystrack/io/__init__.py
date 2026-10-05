"""Versioned Raystrack storage: write v2 and import isolated v1 formats."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ._arrays import (
    DEFAULT_CHUNK_BYTES, FORMAT, integer, json_value, read_array,
    read_manifest, store_path, write_array, write_manifest,
)


FORMAT_VERSION = 2


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class StoredRun:
    """Loaded scene, optional immutable result, and detached metadata."""

    scene: object
    result: object = None
    metadata: Mapping = None

    def __post_init__(self):
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(json_value(self.metadata or {})))


def save(path, scene, result=None, metadata=None, *, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Save a new v2 directory, publishing its manifest after all array chunks.

    Existing stores are never overwritten. A failed write has no committed
    manifest and cannot be mistaken for a complete run. Results must describe
    the current scene revision.
    """
    from ..model import Scene
    from ..solver import Result

    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene")
    if result is not None and not isinstance(result, Result):
        raise TypeError("result must be a Result or None")
    integer(chunk_bytes, "chunk_bytes", minimum=12)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    clean_metadata = json_value(metadata or {})
    with scene.lock:
        revision = scene.revision
        surfaces = tuple(scene.surfaces)
    if result is not None and result.scene_revision != revision:
        raise ValueError("Result belongs to a different scene revision")
    if result is not None:
        _validate_result_ids(result, {surface.surface_id for surface in surfaces})
    result_descriptor = None
    if result is not None:
        result_descriptor = {
            "sender_ids": list(result.sender_ids),
            "channels": [
                {"kind": channel.kind, "surface_id": channel.surface_id,
                 "side": channel.side, "patch": channel.patch}
                for channel in result.channels
            ],
            "statistics": json_value(result.statistics),
            "execution": json_value(result.execution),
            "scene_revision": result.scene_revision,
            "rays_used": result.rays_used,
            "cumulative_rays": result.cumulative_rays,
            "status": result.status,
            "converged": result.converged,
            "elapsed_ms": result.elapsed_ms,
            "provenance": json_value(result.provenance),
        }
        result_descriptor = json_value(result_descriptor)
    root = store_path(path)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": FORMAT, "version": FORMAT_VERSION, "complete": True,
        "chunk_bytes": chunk_bytes, "metadata": clean_metadata,
        "scene": {"revision": revision, "geometries": [], "instances": []},
        "result": result_descriptor,
    }
    geometry_ids = {}
    for surface in surfaces:
        mesh = surface.mesh
        identity = id(mesh)
        if identity not in geometry_ids:
            geometry_id = f"geometry-{len(geometry_ids):06d}"
            geometry_ids[identity] = geometry_id
            manifest["scene"]["geometries"].append({
                "id": geometry_id,
                "vertices": write_array(root, f"geometry/{geometry_id}/vertices", mesh.vertices, chunk_bytes),
                "faces": write_array(root, f"geometry/{geometry_id}/faces", mesh.faces, chunk_bytes),
            })
        manifest["scene"]["instances"].append({
            "surface_id": surface.surface_id, "geometry_id": geometry_ids[identity],
            "label": surface.label, "transform": np.asarray(surface.transform).tolist(),
        })
    if result is not None:
        for name, array in (
            ("row_indices", result.data.row_indices),
            ("channel_indices", result.data.channel_indices),
            ("estimates", result.data.estimates),
            ("errors", result.data.errors),
            ("coverage", result.coverage),
        ):
            result_descriptor[name] = write_array(root, f"result/{name}", array, chunk_bytes)
    if scene.revision != revision:
        raise RuntimeError("Scene changed during storage; no manifest was published")
    write_manifest(root, manifest)
    return str(root)


def _mapping(value, role):
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {role} descriptor")
    return value


def _entries(value, role):
    if not isinstance(value, list):
        raise ValueError(f"{role} must be a list")
    return value


def _name(value, role):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty string")
    return value


def _validate_result_ids(result, surface_ids):
    referenced = set(result.sender_ids) | {channel.surface_id for channel in result.channels if channel.kind == "surface"}
    missing = referenced - surface_ids
    if missing:
        raise ValueError(f"Result refers to surfaces absent from scene geometry: {sorted(missing)}")


def _read_v2(root, manifest):
    from ..model import Mesh, Scene, Surface
    from ..solver import Channel, Result, SparseValues

    if manifest.get("complete") is not True:
        raise ValueError("Version 2 store is not complete")
    description = _mapping(manifest.get("scene"), "scene")
    revision = integer(description.get("revision"), "Scene revision")
    geometries = {}
    for item in _entries(description.get("geometries"), "Geometries"):
        item = _mapping(item, "geometry")
        identifier = _name(item.get("id"), "Geometry ID")
        if identifier in geometries:
            raise ValueError("Duplicate geometry ID")
        vertices = read_array(root, item.get("vertices"), np.float32, trailing_shape=(3,))
        faces = read_array(root, item.get("faces"), np.int32, trailing_shape=(3,))
        geometries[identifier] = Mesh(vertices, faces)
    surfaces = []
    for item in _entries(description.get("instances"), "Instances"):
        item = _mapping(item, "instance")
        identifier = _name(item.get("surface_id"), "Surface ID")
        geometry_id = item.get("geometry_id")
        if not isinstance(geometry_id, str) or geometry_id not in geometries:
            raise ValueError("Instance refers to an unknown geometry ID")
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError("Instance label must be a string or None")
        transform = np.asarray(item.get("transform"), dtype=np.float64)
        surfaces.append(Surface(identifier, geometries[geometry_id], label, transform))
    scene = Scene(surfaces, revision=revision)
    result = None
    description = manifest.get("result")
    if description is not None:
        description = _mapping(description, "result")
        sender_ids = tuple(_name(value, "Sender ID") for value in _entries(description.get("sender_ids"), "Sender IDs"))
        channels = []
        for item in _entries(description.get("channels"), "Channels"):
            item = _mapping(item, "channel")
            channels.append(Channel(kind=item.get("kind"), surface_id=item.get("surface_id"),
                                    side=item.get("side"), patch=item.get("patch")))
        arrays = {name: read_array(root, description.get(name), dtype, trailing_shape=())
                  for name, dtype in (("row_indices", np.int64), ("channel_indices", np.int64),
                                      ("estimates", np.float64), ("errors", np.float64),
                                      ("coverage", np.int8))}
        result_revision = integer(description.get("scene_revision"), "Result scene revision")
        if result_revision != revision:
            raise ValueError("Stored result belongs to a different scene revision")
        for count in ("rays_used", "cumulative_rays"):
            if description.get(count) is not None:
                integer(description[count], count)
        converged = description.get("converged")
        if converged is not None and not isinstance(converged, bool):
            raise ValueError("Result converged must be a boolean or None")
        elapsed = description.get("elapsed_ms")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Result elapsed_ms must be finite and non-negative")
        result = Result(
            sender_ids=sender_ids, channels=tuple(channels),
            data=SparseValues(arrays["row_indices"], arrays["channel_indices"], arrays["estimates"], arrays["errors"]),
            coverage=arrays["coverage"],
            statistics=_mapping(description.get("statistics"), "statistics"),
            execution=_mapping(description.get("execution"), "execution"),
            scene_revision=result_revision, rays_used=description.get("rays_used"),
            cumulative_rays=description.get("cumulative_rays"),
            status=_name(description.get("status"), "Result status"), converged=converged,
            elapsed_ms=float(elapsed),
            provenance=tuple(_mapping(item, "provenance") for item in _entries(description.get("provenance"), "Provenance")),
        )
        _validate_result_ids(result, set(scene.surface_ids))
    metadata = _mapping(manifest.get("metadata", {}), "metadata")
    return StoredRun(scene, result, metadata)


def load(path):
    """Read a v2 store or import a v1 store without inventing missing accuracy."""
    root = store_path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Store directory not found: {root}")
    manifest = read_manifest(root)
    version = manifest["version"]
    if version == FORMAT_VERSION:
        return _read_v2(root, manifest)
    if version == 1:
        from ._v1 import read_store
        return read_store(root, manifest)
    raise ValueError(f"Unsupported Raystrack store version: {version}")


def import_v1_json(meshes_path, scene_path=None, sky_path=None, rest_path=None):
    """Import v1 mesh/result JSON; coverage and errors remain unknown."""
    from ._v1 import read_json
    return read_json(meshes_path, scene_path, sky_path, rest_path)


__all__ = ["StoredRun", "save", "load", "import_v1_json"]
