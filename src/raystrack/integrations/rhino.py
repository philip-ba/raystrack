"""Lazy Rhino/Grasshopper mesh conversion outside the numerical core.

Grasshopper Python usage::

    from raystrack import Solver, Query, Budget
    from raystrack.integrations.rhino import from_rhino_scene
    scene = from_rhino_scene({"wall": wall_mesh, "roof": roof_mesh})
    with Solver(scene) as solver:
        result = solver.solve(Query.matrix(), budget=Budget(rays=65536))

Importing this module never imports Rhino. Conversion copies vertices and
triangulates quads along A-C without changing the supplied Rhino meshes.
"""
from __future__ import annotations

import importlib

import numpy as np

from ..model import Mesh, Scene


def _rhino():
    try:
        return importlib.import_module("Rhino")
    except ImportError as error:
        raise ImportError("Rhino mesh conversion requires Rhino's Python runtime") from error


def from_rhino_mesh(mesh) -> Mesh:
    """Copy a Rhino.Geometry.Mesh as owned float32 triangles, preserving winding."""
    rhino = _rhino()
    if not isinstance(mesh, rhino.Geometry.Mesh):
        raise TypeError("mesh must be a Rhino.Geometry.Mesh")
    vertices = [(vertex.X, vertex.Y, vertex.Z) for vertex in mesh.Vertices]
    faces = []
    for face in mesh.Faces:
        faces.append((face.A, face.B, face.C))
        if not face.IsTriangle:
            faces.append((face.A, face.C, face.D))
    return Mesh(np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
                np.asarray(faces, dtype=np.int32).reshape((-1, 3)))


def from_rhino_scene(meshes, *, labels=None) -> Scene:
    """Convert a mapping of stable IDs to Rhino meshes into a Scene."""
    return Scene.from_meshes({surface_id: from_rhino_mesh(mesh)
                             for surface_id, mesh in meshes.items()}, labels=labels)
