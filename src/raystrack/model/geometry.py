"""Immutable local triangle geometry and rigid surface instances."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Use an immutable bytes owner, so even ``setflags`` cannot enable writes."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _vertices(vertices) -> np.ndarray:
    raw = np.asarray(vertices)
    if raw.dtype.kind == "c":
        raise ValueError("vertices must contain real coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        owned = np.array(raw, dtype=np.float32, order="C", copy=True)
    if owned.ndim != 2 or owned.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if not np.isfinite(owned).all():
        raise ValueError("vertices must fit finite float32 coordinates")
    return _immutable_array(owned)


def _faces(faces, vertex_count: int) -> np.ndarray:
    raw = np.asarray(faces)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("faces must have shape (n, 3)")
    if raw.dtype.kind not in "iu":
        raise ValueError("faces must contain integer vertex indices")
    if raw.size and (np.min(raw) < 0 or np.max(raw) >= vertex_count):
        raise ValueError("face vertex index is out of bounds")
    if raw.size and np.max(raw) > np.iinfo(np.int32).max:
        raise ValueError("face vertex indices must fit int32")
    return _immutable_array(np.array(raw, dtype=np.int32, order="C", copy=True))


def rigid_transform(transform=None) -> np.ndarray:
    """Validate a column-vector proper rigid transform and own it immutably."""
    transform = np.eye(4) if transform is None else transform
    if np.asarray(transform).dtype.kind == "c":
        raise ValueError("transform must contain real coordinates")
    matrix = np.array(transform, dtype=np.float64, order="C", copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-7)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("transform must be a proper rigid rotation and translation (no scale or shear)")
    with np.errstate(over="ignore", invalid="ignore"):
        fits = np.isfinite(matrix.astype(np.float32)).all()
    if not fits:
        raise ValueError("transform must fit finite float32 coordinates")
    return _immutable_array(matrix)


def _check_world_bounds(mesh: Mesh, transform: np.ndarray) -> None:
    if not len(mesh.vertices):
        return
    center, extent = mesh._bounds
    # Match the v1 instancing range check, including outward AABB rounding.
    limit = float(np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0)))
    for rotation in (transform[:3, :3], transform[:3, :3].astype(np.float32).astype(np.float64)):
        world_center = rotation @ center + transform[:3, 3]
        world_extent = np.abs(rotation) @ extent
        if np.any(np.abs(world_center) + world_extent > limit):
            raise ValueError("transformed instance bounds must fit finite float32 coordinates")


@dataclass(frozen=True, eq=False, init=False)
class Mesh:
    """Owned float32 vertices and int32 triangles, with identity equality.

    Empty and degenerate triangles are retained, preserving the v1 estimator's
    geometry rules. Input arrays are copied; exposed arrays cannot be made
    writable, including through their base arrays.
    """

    vertices: np.ndarray
    faces: np.ndarray
    _bounds: Tuple[np.ndarray, np.ndarray] = field(repr=False)

    def __init__(self, vertices, faces):
        owned_vertices = _vertices(vertices)
        owned_faces = _faces(faces, len(owned_vertices))
        object.__setattr__(self, "vertices", owned_vertices)
        object.__setattr__(self, "faces", owned_faces)
        if len(owned_vertices):
            lower = owned_vertices.min(axis=0).astype(np.float64)
            upper = owned_vertices.max(axis=0).astype(np.float64)
            center, extent = (lower + upper) * 0.5, (upper - lower) * 0.5
        else:
            center = extent = np.zeros(3, dtype=np.float64)
        object.__setattr__(self, "_bounds", (_immutable_array(center), _immutable_array(extent)))


@dataclass(frozen=True, eq=False)
class Surface:
    """A stable surface ID, display label, shared local mesh, and transform.

    ``world = local @ transform[:3, :3].T + transform[:3, 3]``. Two surfaces
    may share the same Mesh and label; surface IDs remain distinct.
    """

    surface_id: str
    mesh: Mesh
    label: Optional[str] = None
    transform: Optional[np.ndarray] = None

    def __post_init__(self):
        if not isinstance(self.surface_id, str) or not self.surface_id:
            raise ValueError("surface_id must be a nonempty string")
        if not isinstance(self.mesh, Mesh):
            raise TypeError("mesh must be a Mesh")
        label = self.surface_id if self.label is None else self.label
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        transform = rigid_transform(self.transform)
        _check_world_bounds(self.mesh, transform)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "transform", transform)

    def world_vertices(self) -> np.ndarray:
        """Materialize tracing coordinates without modifying local geometry."""
        if np.array_equal(self.transform, np.eye(4)):
            return self.mesh.vertices
        with np.errstate(over="ignore", invalid="ignore"):
            vertices = (self.mesh.vertices.astype(np.float64) @ self.transform[:3, :3].T
                        + self.transform[:3, 3])
        return _vertices(vertices)
