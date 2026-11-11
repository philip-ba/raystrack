"""Native shared BLAS instances preserve world hits and atomic pose boundaries."""
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
        library = native_library("embree")
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    if not hasattr(library, "rs_create_instances"):
        pytest.skip("Native library lacks shared instance extension")


def pose(x=0, z=0, angle=0):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, x], [0, 1, 0, 0], [-s, 0, c, z], [0, 0, 0, 1]])


def scene():
    panel = Mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                 [[0, 1, 2], [0, 2, 3]])
    return Scene.from_instances({"panel": panel}, [("a", "panel", pose(z=1)),
                                                    ("b", "panel", pose(x=3, z=2))])


def assert_fresh_parity(native, model):
    origins = np.array([[.3, -.2, 4], [3.3, -.2, 4], [.3, -.2, 0], [7, 0, 4]], np.float32)
    directions = np.array([[0, 0, -1], [0, 0, -1], [0, 0, 1], [0, 0, -1]], np.float32)
    flat = NativeGeometry("embree", model.world_meshes())
    try:
        for bound in (np.inf, 1.5, 3.5):
            actual, mask = native.trace(origins, directions, t_max=bound)
            expected, expected_mask = flat.trace(origins, directions, t_max=bound)
            np.testing.assert_array_equal(actual["surface"], expected["surface"])
            np.testing.assert_array_equal(actual["primitive"], expected["primitive"])
            np.testing.assert_array_equal(mask, expected_mask)
            hit = actual["surface"] >= 0
            for key in ("distance", "u", "v", "normal", "point"):
                np.testing.assert_allclose(actual[key][hit], expected[key][hit], atol=2e-6, rtol=2e-6)
    finally:
        flat.close()


def test_shared_prototype_transform_only_retains_blas_and_world_hit_metadata(embree):
    model = scene()
    native = NativeGeometry.from_scene("embree", model._snapshot())
    try:
        assert native.geometry_mode == "shared-prototype-instances"
        assert native.instance_statistics["prototype_builds"] == 1
        assert_fresh_parity(native, model)
        for angle in (.37, np.pi, -.7):
            model.update_transforms({"a": pose(z=1, angle=angle), "b": pose(x=3, z=2, angle=-angle)})
            assert native.update_scene(model._snapshot())
            assert_fresh_parity(native, model)
        stats = native.instance_statistics
        assert stats == {"prototype_builds": 1, "prototype_refits": 0,
                         "transform_commits": 6, "scene_commits": 4}
        assert native.update_scene(model._snapshot())
        assert native.instance_statistics == stats
    finally:
        native.close()


def test_instance_deformation_refit_topology_rebuild_and_empty_slots(embree):
    model = scene()
    # One shared replacement followed by one unique deformed mesh changes prototype count.
    native = NativeGeometry.from_scene("embree", model._snapshot())
    try:
        mesh = model["a"].mesh
        vertices = model["a"].world_vertices().copy()
        vertices[0, 2] += .2
        model.update_vertices("a", vertices)
        assert not native.update_scene(model._snapshot())
        assert_fresh_parity(native, model)
        vertices[0, 2] += .1
        model.update_vertices("a", vertices)
        assert native.update_scene(model._snapshot())
        assert native.instance_statistics["prototype_refits"] == 1
        assert_fresh_parity(native, model)
        model.update_mesh("a", Mesh(mesh.vertices, mesh.faces[:, ::-1]))
        assert not native.update_scene(model._snapshot())
        assert_fresh_parity(native, model)
        model.update_mesh("a", Mesh(np.empty((0, 3)), np.empty((0, 3), np.int32)))
        assert not native.update_scene(model._snapshot())
        assert_fresh_parity(native, model)
    finally:
        native.close()


def test_instance_invalid_update_rolls_back_every_pose(embree):
    model = scene()
    native = NativeGeometry.from_scene("embree", model._snapshot())
    try:
        previous = model._snapshot()
        model.update_transforms({"a": pose(x=7), "b": pose(x=2e18)})
        with pytest.raises(RuntimeError, match="coordinate range"):
            native.update_scene(model._snapshot())
        assert_fresh_parity(native, Scene(previous.surfaces))
        assert native.instance_statistics["scene_commits"] == 1
    finally:
        native.close()


def test_unique_prototype_periodic_deformation_refresh_preserves_hits(embree):
    model = scene()
    # Split one shared instance first; subsequent deformations retain slots.
    model.update_vertices("a", model["a"].world_vertices())
    native = NativeGeometry.from_scene("embree", model._snapshot())
    try:
        for index in range(33):
            vertices = model["a"].world_vertices().copy()
            vertices[0,2] += .001
            model.update_vertices("a",vertices)
            assert native.update_scene(model._snapshot()) is (index < 32)
        assert native.instance_statistics["prototype_refits"] == 0
        assert_fresh_parity(native,model)
    finally:
        native.close()


def test_shortwave_instance_motion_changes_shadow_and_keeps_snapshot(embree):
    model = scene()
    kwargs = dict(dni=700, dhi=80, samples=256, bounces=1, seed=17)
    args = ([[0.3, -.2, 0]], [[0, 0, 1]], [[0, 0, 1]])
    materials = {key: Opaque(.2) for key in model.surface_ids}
    with ShortwaveSolver(model, backend="embree", materials=materials) as solver:
        first = solver.solve(*args, **kwargs)
        snapshot = first.irradiance.copy()
        model.update_transforms({"a": pose(x=7, z=1), "b": pose(x=3, z=2)})
        updated = solver.solve(*args, **kwargs)
        with ShortwaveSolver(model, backend="embree", materials=materials) as fresh:
            expected = fresh.solve(*args, **kwargs)
        np.testing.assert_allclose(updated.irradiance, expected.irradiance, atol=1e-10)
        np.testing.assert_array_equal(first.irradiance, snapshot)
        assert updated.irradiance[0, 0, 0, 0] > first.irradiance[0, 0, 0, 0] + 600
        assert solver._native.instance_statistics["prototype_builds"] == 1


def test_diagnostic_trace_rejects_closed_geometry_and_invalid_rays(embree):
    native = NativeGeometry.from_scene("embree", scene()._snapshot())
    try:
        with pytest.raises(ValueError, match="nonzero"):
            native.trace([[0, 0, 0]], [[0, 0, 0]])
        with pytest.raises(ValueError, match="ray bounds"):
            native.trace([[0, 0, 0]], [[0, 0, 1]], t_min=2, t_max=1)
        with pytest.raises(RuntimeError, match="update_scene"):
            native.update(scene().world_meshes())
    finally:
        native.close()
    with pytest.raises(RuntimeError, match="closed"):
        native.trace([[0, 0, 0]], [[0, 0, 1]])
