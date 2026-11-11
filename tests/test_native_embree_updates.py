"""Real Embree buffer updates preserve ordered IDs and match fresh geometry."""

import os

import numpy as np
import pytest

from raystrack import CapabilityError, Mesh, Scene
from raystrack.backends.native import NativeGeometry, native_library
from raystrack.shortwave import Opaque, ShortwaveSolver


@pytest.fixture
def embree():
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("No native library configured")
    try:
        native_library("embree")
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise


def meshes(x=0, z=1, faces=None):
    vertices = np.array([[x - 1, -1, z], [x + 1, -1, z],
                         [x + 1, 1, z], [x - 1, 1, z]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]] if faces is None else faces, dtype=np.int32)
    return [("upper", vertices, triangles),
            ("lower", vertices - np.array([0, 0, 0.5], dtype=np.float32), triangles)]


def classify(geometry):
    origins = np.array([[0.2, -0.1, 2], [0.2, -0.1, 0], [3.2, -0.1, 2]], dtype=np.float32)
    directions = np.array([[0, 0, -1], [0, 0, 1], [0, 0, -1]], dtype=np.float32)
    hits = np.empty(3, dtype=np.int32)
    front = np.empty(3, dtype=np.uint8)
    blocked = np.empty(3, dtype=np.uint8)
    geometry.longwave(origins, directions, np.ones(2, dtype=np.uint8), -1, 0,
                      hits, front, blocked)
    return hits, front, blocked


def test_embree_refit_motion_deformation_and_fresh_parity(embree):
    geometry = NativeGeometry("embree", meshes())
    try:
        np.testing.assert_array_equal(classify(geometry)[0], [0, 1, -1])
        handle = geometry._handle.value
        deformed = meshes(x=3)
        deformed[0][1][0, 2] += 0.25
        for replacement in (meshes(x=3), deformed, meshes()):
            assert geometry.update(replacement)
            assert geometry._handle.value == handle
            fresh = NativeGeometry("embree", replacement)
            try:
                for actual, expected in zip(classify(geometry), classify(fresh)):
                    np.testing.assert_array_equal(actual, expected)
            finally:
                fresh.close()
        np.testing.assert_array_equal(classify(geometry)[1], [1, 0, 0])
    finally:
        geometry.close()


def test_embree_periodic_refresh_and_empty_surface_slots(embree):
    initial = meshes()
    empty = ("empty", np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32))
    geometry = NativeGeometry("embree", [empty, *initial])
    try:
        for frame in range(33):
            replacement = [empty, *meshes(x=0.001 * (frame + 1))]
            assert geometry.update(replacement) is (frame < 32)
        origins = np.array([[0.2, -0.1, 2]], dtype=np.float32)
        directions = np.array([[0, 0, -1]], dtype=np.float32)
        hits, front, blocked = np.empty(1, dtype=np.int32), np.empty(1, dtype=np.uint8), np.empty(1, dtype=np.uint8)
        geometry.longwave(origins, directions, np.ones(3, dtype=np.uint8), -1, 0, hits, front, blocked)
        np.testing.assert_array_equal(hits, [1])
        np.testing.assert_array_equal(front, [1])
        np.testing.assert_array_equal(blocked, [1])
    finally:
        geometry.close()


@pytest.mark.parametrize("triangles", [
    [[0, 1, 3], [1, 2, 3]],  # Equal counts, changed connectivity.
    [[2, 1, 0], [3, 2, 0]],  # Equal counts, reversed winding.
    [[0, 1, 2]],            # Changed triangle count.
])
def test_embree_topology_changes_rebuild_and_preserve_slots(embree, triangles):
    geometry = NativeGeometry("embree", meshes())
    try:
        replacement = meshes(faces=triangles)
        assert not geometry.update(replacement)
        fresh = NativeGeometry("embree", replacement)
        try:
            for actual, expected in zip(classify(geometry), classify(fresh)):
                np.testing.assert_array_equal(actual, expected)
        finally:
            fresh.close()
    finally:
        geometry.close()


def test_embree_invalid_update_does_not_change_committed_geometry(embree):
    geometry = NativeGeometry("embree", meshes())
    try:
        before = classify(geometry)
        replacement = meshes(x=3)
        replacement[1][1][0, 2] = np.float32(2e18)
        with pytest.raises((RuntimeError, CapabilityError), match="coordinate range"):
            geometry.update(replacement)
        for actual, expected in zip(classify(geometry), before):
            np.testing.assert_array_equal(actual, expected)
    finally:
        geometry.close()


def test_embree_shortwave_reuses_geometry_after_deformation_and_matches_fresh(embree):
    _, vertices, triangles = meshes()[0]
    scene = Scene.from_meshes({"sheet": Mesh(vertices, triangles)})
    args = ([[0.2, -0.1, 2], [0.2, -0.1, 0]], [[0, 0, -1], [0, 0, 1]], [0.2, 0, 1])
    options = dict(dni=700, dhi=100, samples=512, bounces=2, seed=19)
    materials = {"sheet": Opaque(0.3)}
    with ShortwaveSolver(scene, backend="embree", materials=materials) as solver:
        initial = solver.solve(*args, **options)
        snapshot = initial.irradiance.copy()
        native = solver._native
        deformed = vertices.copy()
        deformed[0, 2] += 0.25
        scene.update_vertices("sheet", deformed)
        updated = solver.solve(*args, **options)
        assert solver._native is native
        with ShortwaveSolver(scene, backend="embree", materials=materials) as fresh:
            expected = fresh.solve(*args, **options)
        np.testing.assert_allclose(updated.irradiance, expected.irradiance, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(initial.irradiance, snapshot)
        assert (initial.geometry_revision, updated.geometry_revision) == (0, 1)
