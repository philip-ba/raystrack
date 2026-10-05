from __future__ import annotations

import dataclasses
import importlib
import os
import subprocess
import sys
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from raystrack.model import Mesh, Scene, Surface


def triangle():
    return Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])


def translated(x=0, y=0, z=0):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_mesh_owns_immutable_buffers_and_identity_equality():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    faces = np.asarray([[0, 1, 2]], np.int64)
    mesh = Mesh(vertices, faces)
    vertices[:] = 9
    faces[:] = 0
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])
    assert mesh.vertices.dtype == np.float32
    assert mesh.faces.dtype == np.int32
    assert mesh != Mesh(mesh.vertices, mesh.faces)
    assert len({mesh: "first", Mesh(mesh.vertices, mesh.faces): "second"}) == 2
    for array in (mesh.vertices, mesh.faces):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0] = 4
        with pytest.raises(ValueError):
            array.base.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mesh.vertices = vertices


@pytest.mark.parametrize("vertices,faces", [
    (np.zeros((3, 2)), [[0, 1, 2]]),
    ([[0, 0, float("nan")]], [[0, 0, 0]]),
    ([[0, 0, 1e40]], [[0, 0, 0]]),
    ([[0, 0, 1j]], [[0, 0, 0]]),
    ([[0, 0, 0]], [[0, 0, -1]]),
    ([[0, 0, 0]], [[0, 0, 1]]),
    ([[0, 0, 0]], [[0.0, 0.0, 0.0]]),
    ([[0, 0, 0]], [[False, False, False]]),
    ([[0, 0, 0]], [[0, 0]]),
])
def test_mesh_rejects_invalid_geometry(vertices, faces):
    with pytest.raises(ValueError):
        Mesh(vertices, faces)


def test_empty_and_degenerate_geometry_are_retained():
    empty = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    assert empty.vertices.shape == empty.faces.shape == (0, 3)
    degenerate = Mesh([[0, 0, 0]], [[0, 0, 0]])
    assert degenerate.faces.shape == (1, 3)
    assert Scene.from_meshes({"empty": empty, "degenerate": degenerate}).surface_ids == ("empty", "degenerate")


def test_stable_ids_independent_labels_and_snapshot_order():
    mesh = triangle()
    scene = Scene.from_meshes({"one": mesh, "two": mesh}, labels={"one": "shared", "two": "shared"})
    original = scene.surfaces
    scene.set_label("one", "changed")
    assert scene.surface_ids == ("one", "two")
    assert scene["one"].label == "changed"
    assert original[0].label == "shared"
    assert scene.revision == 0
    assert scene.remove("one") == 1
    assert scene.surface_ids == ("two",)
    assert scene.add(Surface("three", mesh, "shared")) == 2
    assert scene.surface_ids == ("two", "three")
    with pytest.raises(ValueError, match="unique"):
        scene.add(Surface("two", mesh))
    with pytest.raises(ValueError, match="unique"):
        Scene.from_meshes([("duplicate", mesh), ("duplicate", mesh)])
    with pytest.raises(KeyError):
        Scene.from_meshes({"one": mesh}, labels={"typo": "wrong"})


def test_instances_share_geometry_and_absolute_transform_updates():
    mesh = triangle()
    transform = translated(x=2)
    scene = Scene.from_instances({"prototype": mesh}, [
        ("a", "prototype", transform), ("b", "prototype", translated(z=3)),
    ])
    transform[0, 3] = 90
    assert scene["a"].mesh is scene["b"].mesh is mesh
    old_surface = scene["a"]
    scene.update_transform("a", translated(x=4))
    world = scene.world_meshes()
    np.testing.assert_array_equal(world[0][1], mesh.vertices + [4, 0, 0])
    np.testing.assert_array_equal(old_surface.world_vertices(), mesh.vertices + [2, 0, 0])
    assert world[0][0] == "a"
    assert world[0][2] is mesh.faces
    assert scene["b"].mesh is mesh
    with pytest.raises(ValueError):
        scene["a"].transform.setflags(write=True)
    with pytest.raises(ValueError):
        world[0][1].setflags(write=True)


def test_vertices_reset_transform_and_detach_only_changed_instance():
    mesh = triangle()
    scene = Scene.from_instances({"prototype": mesh}, [
        ("a", "prototype", translated(x=2)), ("b", "prototype", np.eye(4)),
    ])
    assert scene.update_vertices("a", mesh.vertices + [0, 0, 5]) == 1
    assert scene["a"].mesh is not mesh
    assert scene["b"].mesh is mesh
    np.testing.assert_array_equal(scene["a"].transform, np.eye(4))
    scene.update_transform("a", translated(x=3))
    np.testing.assert_array_equal(scene["a"].world_vertices(), mesh.vertices + [3, 0, 5])
    replacement = Mesh(mesh.vertices * 2, mesh.faces)
    assert scene.update_mesh("a", replacement) == 3
    assert scene["a"].mesh is replacement
    np.testing.assert_array_equal(scene["a"].transform, np.eye(4))


@pytest.mark.parametrize("kind", ["scale", "shear", "reflection", "projective", "nan", "overflow"])
def test_transform_validation_is_atomic_and_does_not_notify(kind):
    scene = Scene.from_meshes({"a": triangle()})
    old = scene["a"]
    notifications = []
    scene._subscribe(lambda: notifications.append(scene.revision))
    matrix = np.eye(4)
    if kind == "scale":
        matrix[0, 0] = 2
    elif kind == "shear":
        matrix[0, 1] = 0.1
    elif kind == "reflection":
        matrix[0, 0] = -1
    elif kind == "projective":
        matrix[3, 0] = 0.1
    elif kind == "nan":
        matrix[0, 3] = np.nan
    elif kind == "overflow":
        matrix[0, 3] = 1e40
    with pytest.raises(ValueError):
        scene.update_transform("a", matrix)
    assert scene["a"] is old
    assert scene.revision == 0
    assert notifications == []


def test_transformed_bounds_fail_before_geometry_changes():
    mesh = Mesh([[1e38, 0, 0]], [[0, 0, 0]])
    scene = Scene.from_meshes({"large": mesh})
    with pytest.raises(ValueError, match="bounds"):
        scene.update_transform("large", translated(x=3e38))
    assert scene["large"].mesh is mesh
    assert scene.revision == 0


def test_geometry_notifications_precede_lock_wait_and_can_unsubscribe():
    scene = Scene.from_meshes({"a": triangle()})
    notified, finished = Event(), Event()
    observed = []

    def callback():
        observed.append(scene.revision)
        notified.set()

    def mutate():
        scene.update_transform("a", translated(z=2))
        finished.set()

    scene._subscribe(callback)
    scene._subscribe(callback)
    with scene.lock:
        worker = Thread(target=mutate)
        worker.start()
        assert notified.wait(2)
        assert not finished.is_set()
        assert scene.revision == 0
    worker.join(2)
    assert finished.is_set()
    assert observed == [0]
    scene._unsubscribe(callback)
    scene.update_transform("a", np.eye(4))
    assert observed == [0]


def test_vertex_replacement_retains_topology_changed_while_waiting():
    original = triangle()
    replacement = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 3], [0, 3, 2]])
    scene = Scene.from_meshes({"a": original})
    notified = Event()
    scene._subscribe(notified.set)
    new_vertices = replacement.vertices + [0, 0, 5]
    with scene.lock:
        worker = Thread(target=scene.update_vertices, args=("a", new_vertices))
        worker.start()
        assert notified.wait(2)
        scene.update_mesh("a", replacement)
    worker.join(2)
    assert not worker.is_alive()
    assert scene.revision == 2
    np.testing.assert_array_equal(scene["a"].mesh.faces, replacement.faces)
    np.testing.assert_array_equal(scene["a"].mesh.vertices, new_vertices)


def test_revision_restore_validation():
    assert Scene([Surface("a", triangle())], revision=8).revision == 8
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            Scene(revision=invalid)


def test_application_imports_are_lazy_in_fresh_python():
    code = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'Rhino' or name.startswith('Rhino.'):
        raise AssertionError('Rhino imported eagerly')
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import raystrack.model
import raystrack.integrations
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True,
                   capture_output=True, text=True)


def test_rhino_conversion_triangulates_quads_without_mutating_source():
    class FakeMesh:
        def __init__(self):
            self.Vertices = [SimpleNamespace(X=x, Y=y, Z=0)
                             for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
            self.Faces = [SimpleNamespace(A=0, B=1, C=2, D=3, IsTriangle=False),
                          SimpleNamespace(A=0, B=2, C=3, D=3, IsTriangle=True)]

    fake_rhino = SimpleNamespace(Geometry=SimpleNamespace(Mesh=FakeMesh))
    adapter = importlib.import_module("raystrack.integrations.rhino")
    source = FakeMesh()
    with patch.dict(sys.modules, {"Rhino": fake_rhino}):
        scene = adapter.from_rhino_scene({"wall": source}, labels={"wall": "Wall"})
        mesh = scene["wall"].mesh
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3], [0, 2, 3]])
        assert scene["wall"].label == "Wall"
        assert len(source.Faces) == 2
        source.Vertices[0].X = 9
        assert mesh.vertices[0, 0] == 0
        with pytest.raises(TypeError):
            adapter.from_rhino_mesh(object())
        empty = FakeMesh()
        empty.Vertices, empty.Faces = [], []
        assert adapter.from_rhino_mesh(empty).vertices.shape == (0, 3)


def test_rhino_runtime_is_requested_only_at_conversion():
    adapter = importlib.import_module("raystrack.integrations.rhino")
    with patch.object(adapter.importlib, "import_module", side_effect=ImportError("missing Rhino")) as requested:
        with pytest.raises(ImportError, match="Rhino's Python runtime"):
            adapter.from_rhino_mesh(object())
        requested.assert_called_once_with("Rhino")
