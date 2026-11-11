"""Opt-in native view factors through the same public solver and run API."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from raystrack import Budget, CapabilityError, Channel, Query, Sampling, Scene, SolveOptions, Solver
from raystrack.backends.native import native_library
from raystrack.utils.prepared import PreparedSolver
from tests.self_view_cases import box_mesh, fixed_options as box_options
from tests.test_solver_v2 import assert_same_snapshot, fixed_options, pair_scene


@pytest.fixture(params=["native-builtin", "embree", "vulkan-rt"])
def native_device(request):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("No native library configured")
    try:
        native_library(request.param)
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    return request.param


def test_native_is_opt_in_and_missing_library_is_a_capability_error(monkeypatch):
    monkeypatch.delenv("RAYSTRACK_NATIVE_LIBRARY", raising=False)
    scene = pair_scene()
    query = Query.row("A")
    with Solver(scene, device="embree") as solver:
        run = solver.start(query)
        assert run.advance(Budget(rays=0)).execution == {}
        with pytest.raises(CapabilityError, match="RAYSTRACK_NATIVE_LIBRARY"):
            run.advance(Budget(rays=1))

    with Solver(scene) as solver:
        assert solver.solve(query, budget=Budget(rays=1)).execution["backend"] == "cpu"


@pytest.mark.parametrize("settings", [{"bvh": "off"}, {"acceleration": "instanced"}])
def test_native_does_not_claim_legacy_acceleration_settings(settings):
    with pytest.raises(CapabilityError, match="acceleration='flat' and bvh='auto'"):
        Solver(pair_scene(), device="embree", **settings)


def test_native_does_not_accept_area_pair_without_loading_a_library():
    with Solver(pair_scene(), device="embree") as solver:
        options = SolveOptions(sampling=Sampling(strategy="area_pair"))
        with pytest.raises(CapabilityError, match="CPU only"):
            solver.start(Query.pair("A", "B"), options)


def test_native_longwave_resume_matches_stationary_run_and_cpu(native_device):
    scene = pair_scene()
    options = fixed_options(rays=32, replicas=3, batch=128)
    query = Query.row("A", sky="tregenza145")
    with Solver(scene, device=native_device, auto_tune=False) as solver, patch.object(
        PreparedSolver, "get_scene", side_effect=AssertionError("native traversal built a legacy BVH"),
    ):
        assert solver.warmup(query, options, repeats=1, max_probe_rays=32).backend == native_device
        run = solver.start(query, options)
        partial = run.advance(Budget(rays=137))
        resumed = run.advance(Budget())
        uninterrupted = solver.solve(query, options, Budget())

    assert partial.cumulative_rays == 137
    assert partial.error("A", Channel("surface", "B", "front")) is None
    assert resumed.execution["backend"] == native_device
    assert_same_snapshot(resumed, uninterrupted)
    assert sum(resumed.row("A").values()) == pytest.approx(1)

    with Solver(scene, device="cpu", auto_tune=False) as solver:
        reference = solver.solve(query, options, Budget())
    assert resumed.cumulative_rays == reference.cumulative_rays
    # Both use the same rays. Permit three classifications to differ near
    # triangle boundaries across traversal implementations.
    np.testing.assert_allclose(
        resumed.dense(), reference.dense(), rtol=0, atol=3 / reference.cumulative_rays,
    )


def test_native_rebuilds_geometry_after_edit_and_preserves_old_result(native_device):
    scene = pair_scene(blocked=True)
    options = fixed_options(rays=32, batch=256)
    query = Query.row("A", receivers=("B",), sky="merged")
    channel = Channel("surface", "B", "front")
    with Solver(scene, device=native_device, auto_tune=False) as solver:
        run = solver.start(query, options)
        original = run.advance(Budget())
        transform = np.eye(4)
        transform[0, 3] = 100
        scene.update_transform("blocker", transform)
        assert run.status == "invalidated"
        moved = solver.solve(query, options, Budget())
        scene.update_vertices("blocker", scene["blocker"].mesh.vertices)
        restored = solver.solve(query, options, Budget())

    with Solver(scene, device=native_device, auto_tune=False) as solver:
        fresh = solver.solve(query, options, Budget())
    assert_same_snapshot(restored, fresh)
    assert original.value("A", channel) == restored.value("A", channel) == 0
    assert original.value("A", Channel("unrequested")) > 0.99
    assert moved.value("A", channel) > 0.1
    assert (original.scene_revision, moved.scene_revision, restored.scene_revision) == (0, 1, 2)


def test_native_closed_enclosure_self_views_the_interior_back_side(native_device):
    scene = Scene.from_meshes({"box": box_mesh(inward=False)})
    with Solver(scene, device=native_device, auto_tune=False) as solver:
        result = solver.solve(Query.row("box", sky="merged"), box_options(flip=True), Budget())
    assert result.value("box", Channel("surface", "box", "back")) == 1
    assert result.value("box", Channel("surface", "box", "front")) == 0
    assert result.value("box", Channel("sky")) == result.value("box", Channel("rest")) == 0
