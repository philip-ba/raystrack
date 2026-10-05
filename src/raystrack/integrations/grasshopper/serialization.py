"""Detached JSON descriptions for the compiled Grasshopper bridge.

Surface IDs and receiver sides remain separate fields. Dense ``null`` entries
mean unavailable estimates or errors, and coverage distinguishes uncomputed
rows from sampled zero. Meshes with identical geometry share a Mesh object.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
import math
import numbers

import numpy as np

from ...model import Mesh, Scene, Surface
from ...solver import (Accuracy, Channel, Postprocessing, Query, Result,
                       Sampling, SolveOptions, SparseValues)


def json_value(value):
    """Convert immutable snapshots to JSON, marking nonfinite values unknown."""
    if isinstance(value, Mapping):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(child) for child in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _object(value, name):
    """Require a JSON object and retain its field path in validation errors."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object; received {type(value).__name__}")
    return value


def _sequence(value, name):
    """Require a JSON list rather than silently iterating text or mappings."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list; received {type(value).__name__}")
    return value


def _record(cls, value, name):
    """Build a validated dataclass with errors naming its nested input path."""
    value = _object(value, name)
    allowed = {field.name for field in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}; supported fields: {sorted(allowed)}")
    try:
        return cls(**value)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        prefix = "." if any(message.startswith(field + " ") for field in allowed) else ": "
        raise ValueError(f"{name}{prefix}{message}") from None


def _kind(value, expected, name):
    """Reject a connected description of the wrong Raystrack object kind."""
    actual = value.get("kind", expected)
    if actual != expected:
        raise ValueError(f"{name}.kind must be {expected!r}; received {actual!r}. "
                         f"Connect a Raystrack {expected.title()} description.")


def _rows(value, width, path, *, integers=False):
    """Validate fixed-width numeric JSON rows and report the first bad cell."""
    rows = _sequence(value, path)
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{path}[{index}] must contain exactly {width} numbers; received {row!r}")
        for column, number in enumerate(row):
            role = f"{path}[{index}][{column}]"
            if integers:
                if isinstance(number, bool) or not isinstance(number, numbers.Integral):
                    raise ValueError(f"{role} must be an integer vertex index; received {number!r}")
            elif isinstance(number, bool) or not isinstance(number, numbers.Real) or not math.isfinite(number):
                raise ValueError(f"{role} must be a finite real number; received {number!r}")
    return rows


def _names(value, path, *, optional=False):
    """Validate a list of unique stable IDs with paths for missing entries."""
    if value is None and optional:
        return None
    entries = _sequence(value, path)
    seen = {}
    for index, identifier in enumerate(entries):
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{path}[{index}] must be a nonempty stable surface ID; received {identifier!r}")
        if identifier in seen:
            raise ValueError(f"{path}[{index}] duplicates {path}[{seen[identifier]}] ({identifier!r}); select each ID once")
        seen[identifier] = index
    return entries


def query_from_json(value):
    """Build a Query using stable surface IDs, not display labels or indices."""
    value = dict(_object(value, "query"))
    _kind(value, "query", "query")
    value.pop("kind", None)
    for name in ("senders", "receivers"):
        if name in value:
            _names(value[name], f"query.{name}", optional=True)
    if "receiver_sides" in value:
        _sequence(value["receiver_sides"], "query.receiver_sides")
    return _record(Query, value, "query")


def options_from_json(value=None):
    """Build typed option groups while identifying invalid nested settings."""
    value = dict(_object({} if value is None else value, "options"))
    _kind(value, "options", "options")
    value.pop("kind", None)
    for name, cls in (("sampling", Sampling), ("accuracy", Accuracy),
                      ("postprocessing", Postprocessing)):
        if name in value:
            value[name] = _record(cls, value[name], f"options.{name}")
    return _record(SolveOptions, value, "options")


def mesh_key(mesh):
    """Hash immutable local geometry so repeated instances share one Mesh."""
    digest = hashlib.sha256()
    for array in (mesh.vertices, mesh.faces):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def scene_from_json(value, *, meshes=None, revision=None):
    """Build a Scene with shared geometry and contextual surface input errors.

    ``meshes`` optionally retains existing Mesh identities across launches.
    ``revision`` reconciles a saved result with the worker's cache revision.
    """
    value = _object(value, "scene")
    _kind(value, "scene", "scene")
    meshes = {} if meshes is None else meshes
    surfaces = []
    seen = {}
    for index, entry in enumerate(_sequence(value.get("surfaces"), "scene.surfaces")):
        path = f"scene.surfaces[{index}]"
        entry = _object(entry, path)
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{path}.id must be a nonempty stable surface ID; assign an ID in the Scene component")
        if identifier in seen:
            raise ValueError(f"{path}.id={identifier!r} duplicates scene.surfaces[{seen[identifier]}].id; "
                             "assign a unique stable ID to each instance")
        seen[identifier] = index
        path += f" (id={identifier!r})"
        geometry = _object(entry.get("mesh"), f"{path}.mesh")
        vertices = _rows(geometry.get("vertices"), 3, f"{path}.mesh.vertices")
        faces = _rows(geometry.get("faces"), 3, f"{path}.mesh.faces", integers=True)
        for row, face in enumerate(faces):
            for column, vertex in enumerate(face):
                if not 0 <= vertex < len(vertices):
                    raise ValueError(f"{path}.mesh.faces[{row}][{column}]={vertex} is outside the vertex range "
                                     f"0..{len(vertices)-1}; repair the triangle indices")
        if not vertices:
            vertices = np.empty((0, 3), np.float32)
        if not faces:
            faces = np.empty((0, 3), np.int32)
        try:
            mesh = Mesh(vertices, faces)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{path}.mesh: {exc}") from None
        mesh = meshes.setdefault(mesh_key(mesh), mesh)
        label = entry.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"{path}.label must be text or null; received {label!r}")
        transform = entry.get("transform")
        if transform is not None:
            _rows(transform, 4, f"{path}.transform")
            if len(transform) != 4:
                raise ValueError(f"{path}.transform must have four rows of four numbers")
        try:
            surfaces.append(Surface(identifier, mesh, label, transform))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.transform: {exc}") from None
    try:
        return Scene(surfaces, revision=value.get("revision", 0) if revision is None else revision)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scene.revision: {exc}") from None


def scene_to_json(scene):
    """Detach a coherent scene revision, IDs, local geometry and placements."""
    with scene.lock:
        return {"kind": "scene", "revision": scene.revision, "surfaces": [
            {"id": surface.surface_id, "label": surface.label,
             "mesh": {"vertices": surface.mesh.vertices.tolist(),
                      "faces": surface.mesh.faces.tolist()},
             "transform": surface.transform.tolist()}
            for surface in scene.surfaces]}


def scene_fingerprint(scene):
    """Numerical identity excludes labels and external revision bookkeeping."""
    digest = hashlib.sha256()
    for surface in scene.surfaces:
        digest.update(json.dumps(surface.surface_id, ensure_ascii=True).encode("ascii"))
        digest.update(mesh_key(surface.mesh).encode("ascii"))
        digest.update(surface.transform.tobytes())
    return digest.hexdigest()


def result_to_json(result, *, scene_hash=None, rays_used=None, elapsed_ms=None):
    """Detach a snapshot with dense nulls for unknown values and uncertainties.

    Optional counts describe the current GH launch rather than its last chunk.
    The scene hash protects Save from pairing a result with different geometry.
    """
    values = [[result.value(sender, channel) for channel in result.channels]
              for sender in result.sender_ids]
    errors = [[result.error(sender, channel) for channel in result.channels]
              for sender in result.sender_ids]
    payload = {"kind": "result", "sender_ids": list(result.sender_ids),
               "channels": [{"kind": c.kind, "surface_id": c.surface_id,
                             "side": c.side, "patch": c.patch} for c in result.channels],
               "values": values, "errors": errors, "coverage": result.coverage,
               "statistics": result.statistics, "execution": result.execution,
               "scene_revision": result.scene_revision,
               "rays_used": result.rays_used if rays_used is None else rays_used,
               "cumulative_rays": result.cumulative_rays, "status": result.status,
               "converged": result.converged,
               "elapsed_ms": result.elapsed_ms if elapsed_ms is None else elapsed_ms,
               "provenance": result.provenance}
    if scene_hash is not None:
        payload["scene_hash"] = scene_hash
    return json_value(payload)


def result_from_json(value):
    """Rebuild an immutable Result while preserving unknowns and sampled zero."""
    value = _object(value, "result")
    _kind(value, "result", "result")
    senders = tuple(_names(value.get("sender_ids"), "result.sender_ids"))
    channels = tuple(_record(Channel, item, f"result.channels[{index}]")
                     for index, item in enumerate(_sequence(value.get("channels"), "result.channels")))
    coverage = np.asarray(value.get("coverage"))
    if coverage.shape != (len(senders),) or not np.all(np.isin(coverage, (-1, 0, 1))):
        raise ValueError(f"result.coverage must contain one -1 (unknown), 0 (uncomputed) or 1 (sampled) "
                         f"per sender; expected {len(senders)} entries")
    values, errors = value.get("values"), value.get("errors")
    if (not isinstance(values, list) or not isinstance(errors, list) or
            len(values) != len(senders) or len(errors) != len(senders)):
        raise ValueError(f"result.values and result.errors must each have {len(senders)} rows matching result.sender_ids")
    rows, cols, estimates, uncertainties = [], [], [], []
    for row, (row_values, row_errors) in enumerate(zip(values, errors)):
        if (not isinstance(row_values, list) or not isinstance(row_errors, list) or
                len(row_values) != len(channels) or len(row_errors) != len(channels)):
            raise ValueError(f"result.values[{row}] and result.errors[{row}] must each have "
                             f"{len(channels)} columns matching result.channels (sender={senders[row]!r})")
        for col, (estimate, error) in enumerate(zip(row_values, row_errors)):
            if estimate is None:
                if error is not None:
                    raise ValueError(f"result.errors[{row}][{col}] must be null when its estimate is unknown")
                if coverage[row] == 1:
                    raise ValueError(f"result.values[{row}][{col}] cannot be null for a sampled row; "
                                     "use 0 for a sampled zero estimate")
                continue
            if coverage[row] == 0:
                raise ValueError(f"result.values[{row}][{col}] must be null for an uncomputed row (coverage=0)")
            for number, path in ((estimate, f"result.values[{row}][{col}]"),
                                 (error, f"result.errors[{row}][{col}]")):
                if number is not None and (isinstance(number, bool) or not isinstance(number, numbers.Real) or
                                           not math.isfinite(number) or number < 0):
                    raise ValueError(f"{path} must be a finite nonnegative number or null for unknown; received {number!r}")
            rows.append(row)
            cols.append(col)
            estimates.append(estimate)
            uncertainties.append(np.nan if error is None else error)
    for name in ("statistics", "execution"):
        if name in value:
            _object(value[name], f"result.{name}")
    if "provenance" in value:
        for index, entry in enumerate(_sequence(value["provenance"], "result.provenance")):
            _object(entry, f"result.provenance[{index}]")
    try:
        return Result(senders, channels,
                  SparseValues(np.asarray(rows, np.int64), np.asarray(cols, np.int64),
                               np.asarray(estimates, np.float64), np.asarray(uncertainties, np.float64)),
                  coverage, statistics=value.get("statistics", {}), execution=value.get("execution", {}),
                  scene_revision=value.get("scene_revision", 0), rays_used=value.get("rays_used"),
                  cumulative_rays=value.get("cumulative_rays"), status=value.get("status", "imported"),
                  converged=value.get("converged"), elapsed_ms=value.get("elapsed_ms", 0.),
                      provenance=tuple(value.get("provenance", ())))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"result: {exc}") from None

