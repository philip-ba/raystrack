"""Self-viewing must preserve first-hit visibility, side labels and conservation."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from raystrack import Accuracy, Budget, Channel, Mesh, Query, Sampling, Scene, SolveOptions, Solver, Surface
from tests.self_view_cases import backend_summary, box_mesh, fixed_options, joined_plates


def test_cpu_closed_box_interior_and_convex_exterior_across_traversals():
    assert len(backend_summary("cpu")) == 9


@pytest.mark.parametrize("acceleration,bvh", [("flat", "off"), ("flat", "builtin"), ("instanced", "builtin")])
def test_open_box_matches_aperture_area_ratio_and_self_occludes_sky(acceleration, bvh):
    # The unit opening sees all five unit interior faces. Reciprocity gives
    # F_walls,opening = 1/5 and F_walls,walls = 4/5, for one combined mesh.
    scene = Scene([Surface("box", box_mesh(open_top=True))])
    with Solver(scene, device="cpu", acceleration=acceleration, bvh=bvh, auto_tune=False) as solver:
        result = solver.solve(Query.row("box", sky="merged"), fixed_options(density=4, rays=64, replicas=8), Budget())
    assert result.value("box", Channel("surface", "box", "front")) == pytest.approx(.8, abs=.015)
    assert result.value("box", Channel("sky")) == pytest.approx(.2, abs=.015)
    assert result.value("box", Channel("surface", "box", "back")) == 0
    assert result.value("box", Channel("rest")) == 0
    assert sum(result.row("box").values()) == pytest.approx(1)


def test_unrequested_self_hit_still_blocks_external_receiver_and_sky():
    roof = Mesh([[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]], [[0, 2, 1], [0, 3, 2]])
    scene = Scene([Surface("box", box_mesh()), Surface("roof", roof)])
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        pair = solver.solve(Query.pair("box", "roof"), fixed_options(), Budget())
        sky = solver.solve(Query.sky(("box",), discrete=True), fixed_options(), Budget())
        excluded_side = solver.solve(Query.pair("box", "box", receiver_sides=("back",)), fixed_options(), Budget())
    for result in (pair, sky, excluded_side):
        assert result.value("box", Channel("unrequested")) == 1
        assert result.value("box", Channel("rest")) == 0
        assert sum(result.row("box").values()) == 1


def test_placed_self_pair_and_partial_resume_preserve_the_same_sample_prefix():
    transform = np.array([[0, -1, 0, 3], [1, 0, 0, -2], [0, 0, 1, 4], [0, 0, 0, 1]], float)
    scene = Scene([Surface("box", box_mesh(), transform=transform)])
    options = fixed_options(replicas=3, batch=13)
    with Solver(scene, device="cpu", acceleration="instanced", auto_tune=False) as solver:
        run = solver.start(Query.pair("box", "box"), options)
        partial = run.advance(Budget(rays=17))
        final = run.advance(Budget())
        uninterrupted = solver.solve(Query.pair("box", "box"), options, Budget())
    channel = Channel("surface", "box", "front")
    assert partial.value("box", channel) == final.value("box", channel) == 1
    assert partial.error("box", channel) is None
    assert final.error("box", channel) == 0
    np.testing.assert_array_equal(final.dense(), uninterrupted.dense())
    assert final.cumulative_rays == uninterrupted.cumulative_rays


@pytest.mark.parametrize("strategy", ["cosine", "area_pair"])
def test_same_mesh_pair_matches_parallel_plate_analytical_factor(strategy):
    scene = Scene([Surface("plates", joined_plates())])
    sampling = Sampling(strategy=strategy, density=4, rays_per_cell=128, pair_samples=32768, seed=19)
    options = SolveOptions(sampling, Accuracy(max_replicates=8, min_replicates=8, tolerance=0))
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.pair("plates", "plates"), options, Budget())
    assert result.value("plates", Channel("surface", "plates", "front")) == pytest.approx(.1998248957, abs=.004)
    assert result.value("plates", Channel("surface", "plates", "back")) == 0


def test_planar_mesh_has_no_artificial_self_hit():
    scene = Scene([Surface("plate", Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]]))])
    with Solver(scene, device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.pair("plate", "plate"), fixed_options(), Budget())
    assert result.value("plate", Channel("surface", "plate", "front")) == 0
    assert result.value("plate", Channel("surface", "plate", "back")) == 0
    assert result.value("plate", Channel("rest")) == 1


def test_cuda_sim_self_viewing_matches_cpu_with_both_ray_generation_paths():
    program = """
from raystrack import Solver
from tests.self_view_cases import backend_summary
assert len(backend_summary('cuda')) == 9
original = Solver.__init__
def host_ray_generation(self, *args, **kwargs):
    kwargs['gpu_raygen'] = False
    original(self, *args, **kwargs)
Solver.__init__ = host_ray_generation
assert len(backend_summary('cuda')) == 9
"""
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["NUMBA_NUM_THREADS"] = "4"
    result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec("taichi") is None, reason="optional Taichi dependency is absent")
def test_real_vulkan_self_viewing_matches_cpu():
    from raystrack.utils.taichi_trace import get_taichi_backend, TaichiUnavailableError
    try:
        get_taichi_backend(arch="vulkan")
    except TaichiUnavailableError as error:
        pytest.skip(str(error))
    assert len(backend_summary("vulkan")) == 9
