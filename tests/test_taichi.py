"""Small portable GPU parity checks; skipped when no supported GPU is present."""

import importlib.util
import gc
import os
import subprocess
import sys
import threading
import weakref

import numpy as np
import pytest

from raystrack.utils.prepared import prepare_emitters, prepare_scene
from raystrack.utils.taichi_trace import (
    TaichiBackend, TaichiUnavailableError, _escape_links, get_taichi_backend,
    probe_taichi,
)


def _square(name, z, *, half=1.0, x=0.0, downward=False):
    vertices = np.array([[x-half, -half, z], [x+half, -half, z],
                         [x+half, half, z], [x-half, half, z]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    if downward:
        faces = faces[:, [0, 2, 1]]
    return name, vertices, faces


def _public_scene():
    return [_square("lower", 0), _square("blocker", 1, half=.3, x=.4),
            _square("upper", 2, downward=True)]


def _assert_results_match(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        assert set(actual[name]) == set(expected[name])
        for key in actual[name]:
            np.testing.assert_allclose(actual[name][key], expected[name][key], atol=2e-7, rtol=2e-6)


def _triangle(x=0.0, z=1.0):
    vertices = np.array([[x - .8, -.8, z], [x + .8, -.8, z],
                         [x, .8, z]], np.float32)
    return str(x), vertices, np.array([[0, 1, 2]], np.int32)


@pytest.fixture(scope="module")
def gpu():
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("optional Taichi dependency is absent")
    try:
        return get_taichi_backend()
    except TaichiUnavailableError as exc:
        pytest.skip(str(exc))


def test_probe_contract():
    report = probe_taichi()
    assert isinstance(report["installed"], bool)
    assert isinstance(report["architectures"], list)
    assert set(report["architectures"]) <= {"vulkan", "metal", "cuda"}


def test_invalid_architecture():
    with pytest.raises(ValueError, match="arch"):
        TaichiBackend("cpu")


def test_existing_cpu_runtime_is_rejected():
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("optional Taichi dependency is absent")
    program = """
import taichi as ti
from raystrack.utils.taichi_trace import TaichiBackend, TaichiUnavailableError, probe_taichi
ti.init(arch=ti.cpu)
assert not probe_taichi()['available']
try:
    TaichiBackend()
except TaichiUnavailableError as exc:
    assert 'incompatible' in str(exc)
else:
    raise AssertionError('GPU request reused a CPU runtime')
assert ti.lang.impl.current_cfg().arch == ti.cpu
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_escape_links_visit_all_leaves():
    scene = prepare_scene([_triangle(x=3 * i) for i in range(24)], use_bvh=True)
    escape = _escape_links(scene)
    node = 0
    visited = []
    while node >= 0:
        visited.append(node)
        node = int(scene.left[node]) if scene.count[node] == 0 else int(escape[node])
    assert sorted(visited) == list(range(len(scene.count)))


@pytest.mark.parametrize("use_bvh", [False, True])
def test_supplied_rays_match_cpu(gpu, use_bvh):
    from raystrack.utils.cpu_trace import (
        trace_cpu_combined, trace_cpu_bvh_combined,
        reduce_first_hits, bin_tregenza_cpu,
    )
    meshes = [_triangle(x=3 * i) for i in range(24)]
    scene = prepare_scene(meshes, use_bvh=use_bvh)
    orig = np.zeros((50, 3), np.float32)
    orig[:24, 0] = 3 * np.arange(24)
    orig[24:48, 0] = 3 * np.arange(24)
    orig[24:48, 2] = 2
    orig[48:, 0] = 1000
    directions = np.zeros((50, 3), np.float32)
    directions[:, 2] = 1
    directions[24:48, 2] = -1
    active = np.ones(24, np.uint8)
    active[4] = 0
    hit = np.empty(50, np.int32)
    front = np.empty(50, np.uint8)
    any_hit = np.empty(50, np.uint8)
    if use_bvh:
        trace_cpu_bvh_combined(orig, directions, scene.v0, scene.e1, scene.e2,
                              scene.normals, scene.sid, active, scene.bb_min,
                              scene.bb_max, scene.left, scene.right, scene.start,
                              scene.count, -1, 6, hit, front, any_hit)
    else:
        trace_cpu_combined(orig, directions, scene.v0, scene.e1, scene.e2,
                           scene.normals, scene.sid, active, -1, 6, hit, front, any_hit)
    expected_f = np.zeros(24, np.int64)
    expected_b = np.zeros(24, np.int64)
    reduce_first_hits(hit, front, expected_f, expected_b)
    expected_sky = np.zeros(145, np.int64)
    bin_tregenza_cpu(directions, any_hit, expected_sky)
    actual = gpu.trace_rays(scene, orig, directions, surf_active=active,
                            min_sid=6, include_sky=True, discrete=True)
    for a, b in zip(actual, (expected_f, expected_b, expected_sky)):
        np.testing.assert_array_equal(a, b)
    assert gpu.arch in ("vulkan", "metal", "cuda")


def test_geometry_refresh_and_buffer_reuse(gpu):
    active = np.ones(1, np.uint8)
    origin = np.zeros((1, 3), np.float32)
    direction = np.array([[0, 0, 1]], np.float32)
    first = prepare_scene([_triangle()], use_bvh=True)
    assert gpu.trace_rays(first, origin, direction, surf_active=active)[1][0] == 1
    old_buffer = gpu._buffers["geom"]
    moved = prepare_scene([_triangle(x=100)], use_bvh=True)
    result = gpu.trace_rays(moved, origin, direction, surf_active=active, include_sky=True)
    assert result[1][0] == 0
    assert result[2][0] == 1
    assert gpu._buffers["geom"] is old_buffer


def test_gpu_raygen_and_chunk_offsets(gpu):
    emitter_mesh = _triangle(z=0)
    emitters = prepare_emitters([emitter_mesh], samples=4, rays=16, flip_faces=False)
    emitter = emitters[0]
    scene = prepare_scene([emitter_mesh, _triangle(z=1)], use_bvh=True)
    controls = dict(rays=16, cp_grid=np.array([.13, .37]),
                    cp_dims=np.array([.23, .11, .43, .17, .29]),
                    surf_active=np.ones(2, np.uint8), emit_sid=0,
                    include_sky=True, discrete=True)
    actual = gpu.trace_batch(scene, emitter, **controls)
    total = emitter.n_cells * 16
    gpu_rays = gpu._buffers["raydata"].to_numpy()[:total]
    expected = gpu.trace_batch(scene, emitter, gpu_raygen=False, **controls)
    host_rays = gpu._buffers["raydata"].to_numpy()[:total]
    np.testing.assert_allclose(gpu_rays, host_rays, rtol=2e-5, atol=2e-6)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    part1 = gpu.trace_batch(scene, emitter, ray_count=13, **controls)
    part2 = gpu.trace_batch(scene, emitter, ray_offset=13, ray_count=total - 13, **controls)
    for complete, a, b in zip(actual, part1, part2):
        np.testing.assert_array_equal(complete, a + b)


def test_bounded_counters_split_work(gpu):
    original_limit = gpu.max_batch_rays
    try:
        gpu.max_batch_rays = 7
        scene = prepare_scene([_triangle()], use_bvh=False)
        f, b, s = gpu.trace_rays(scene, np.zeros((29, 3), np.float32),
                                  np.tile([0, 0, 1], (29, 1)),
                                  surf_active=np.ones(1, np.uint8))
        assert f[0] == 0 and b[0] == 29 and s[0] == 0
        assert b.dtype == np.int64
    finally:
        gpu.max_batch_rays = original_limit


def test_grouped_iterations_match_individual_readbacks(gpu):
    emitter_mesh = _triangle(z=0)
    emitter = prepare_emitters([emitter_mesh], samples=4, rays=16, flip_faces=False)[0]
    scene = prepare_scene([emitter_mesh, _triangle(z=1)], use_bvh=True)
    grids = np.array([[.13, .37], [.3, .71], [.61, .09]], np.float32)
    dims = np.array([[.23, .11, .43, .17, .29], [.12, .79, .61, .34, .54],
                     [.9, .6, .3, .1, .4]], np.float32)
    controls = dict(rays=16, surf_active=np.ones(2, np.uint8), emit_sid=0,
                    include_sky=True, discrete=True, ray_count=31, ray_offset=7)
    expected = [gpu.trace_batch(scene, emitter, cp_grid=g, cp_dims=d, **controls)
                for g, d in zip(grids, dims)]
    actual = gpu.trace_iterations(scene, emitter, cp_grids=grids, cp_dims=dims, **controls)
    for i, array in enumerate(actual):
        np.testing.assert_array_equal(array, np.stack([row[i] for row in expected]))


def test_public_matrix_budget_and_stationary_repeat(gpu):
    from dataclasses import replace
    from tests.v2_cases import MatrixCase, matrix_case
    from raystrack.utils.prepared import PreparedSolver
    meshes = _public_scene()
    prepared = PreparedSolver(meshes)
    params = MatrixCase(samples=4, rays=16, max_iters=4, min_iters=2,
                          seed=11, bvh="builtin", device=gpu.arch, reciprocity=False,
                          emitter_names=["lower"], max_total_rays=37, ray_batch_size=7)
    used = []
    actual = matrix_case(meshes, params, prepared=prepared, progress=used.append)
    expected = matrix_case(meshes, replace(params, device="cpu"))
    _assert_results_match(actual, expected)
    assert sum(used) == 37 and max(used) <= 7
    _assert_results_match(matrix_case(meshes, params, prepared=prepared), actual)


def test_public_combined_scene_discrete_sky_parity(gpu):
    from dataclasses import replace
    from tests.v2_cases import MatrixCase, SkyCase, outside_case
    meshes = _public_scene()
    matrix = MatrixCase(samples=4, rays=16, max_iters=4, min_iters=4, seed=11,
                          bvh="builtin", device="taichi", reciprocity=False, tol=0)
    sky = SkyCase(samples=4, rays=16, max_iters=4, min_iters=4, seed=11,
                    bvh="builtin", device="taichi", discrete=True, tol=0)
    actual = outside_case(meshes, matrix_params=matrix, sky_params=sky)
    expected = outside_case(meshes,
                 matrix_params=replace(matrix, device="cpu"),
                 sky_params=replace(sky, device="cpu"))
    for a, b in zip(actual, expected):
        _assert_results_match(a, b)


def test_public_dynamic_refit_matches_fresh_solve(gpu):
    from tests.v2_cases import MatrixCase, matrix_case
    from raystrack.utils.prepared import PreparedSolver
    prepared = PreparedSolver(_public_scene())
    params = MatrixCase(samples=4, rays=16, max_iters=3, min_iters=3,
                          seed=11, bvh="builtin", device="taichi", reciprocity=False,
                          emitter_names=["lower"], max_total_rays=150, ray_batch_size=19)
    before = matrix_case(prepared.meshes, params, prepared=prepared)
    transform = np.eye(4)
    transform[0, 3] = 2
    prepared.update_transform("blocker", transform)
    updated = matrix_case(prepared.meshes, params, prepared=prepared)
    fresh = matrix_case(prepared.meshes, params)
    _assert_results_match(updated, fresh)
    assert before != updated


def test_public_deferred_checkpoints_match_cpu(gpu):
    from dataclasses import replace
    from tests.v2_cases import MatrixCase, SkyCase, matrix_case, sky_case
    meshes = _public_scene()
    matrix = MatrixCase(samples=4, rays=16, max_iters=6, min_iters=2,
                          seed=11, bvh="builtin", device="taichi", reciprocity=False,
                          tol=0, convergence_interval=3)
    actual = matrix_case(meshes, matrix)
    expected = matrix_case(meshes, replace(matrix, device="cpu"))
    _assert_results_match(actual, expected)
    sky = SkyCase(samples=4, rays=16, max_iters=6, min_iters=2,
                    seed=11, bvh="builtin", device="taichi", discrete=False,
                    tol=0, convergence_interval=3)
    actual_sky = sky_case(meshes, sky)
    expected_sky = sky_case(meshes, replace(sky, device="cpu"))
    _assert_results_match(actual_sky, expected_sky)


def test_empty_scene_sky_and_zero_rays(gpu):
    scene = prepare_scene([], use_bvh=True)
    orig = np.zeros((3, 3), np.float32)
    directions = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0]], np.float32)
    front, back, sky = gpu.trace_rays(scene, orig, directions,
                                     surf_active=np.ones(1, np.uint8),
                                     include_matrix=False, include_sky=True, discrete=True)
    assert front[0] == back[0] == 0
    assert sky.sum() == 1 and sky[144] == 1
    empty = gpu.trace_rays(scene, orig[:0], directions[:0],
                           surf_active=np.ones(1, np.uint8), include_sky=True)
    assert all(value.sum() == 0 for value in empty)


def test_sample_index_overflow_rejected_before_allocation(gpu):
    emitter_mesh = _triangle(z=0)
    emitter = prepare_emitters([emitter_mesh], samples=2, rays=4, flip_faces=False)[0]
    scene = prepare_scene([emitter_mesh], use_bvh=False)
    with pytest.raises(ValueError, match="int32"):
        gpu.trace_batch(scene, emitter, rays=2**31, cp_grid=[0, 0], cp_dims=[0]*5,
                        surf_active=np.ones(1, np.uint8), emit_sid=0)


def test_portable_run_single_worker(gpu):
    from raystrack import Solver, Query, SolveOptions, Sampling, Accuracy, Budget
    from tests.v2_cases import scene_for
    current = scene_for(_public_scene())
    opts = SolveOptions(Sampling(density=2, rays_per_cell=8),
                        Accuracy(max_replicates=3, min_replicates=3, tolerance=0), batch_size=7)
    with Solver(current, device=gpu.arch, auto_tune=False, bvh="builtin") as solver:
        first_run = solver.start(Query.row("lower"), opts)
        first = first_run.submit(Budget(rays=37)).result(timeout=30)
        assert first.rays_used == 37 and first.status != "cancelled"
        transform = np.eye(4); transform[0,3] = 1
        current.update_transform("blocker", transform)
        assert first_run.status == "invalidated"
        updated = solver.start(Query.row("lower"), opts).submit(Budget(rays=37)).result(timeout=30)
        assert updated.rays_used == 37 and updated.scene_revision == first.scene_revision+1


def test_dynamic_device_cache_does_not_retain_old_snapshots(gpu):
    from raystrack.utils.prepared import PreparedSolver
    prepared = PreparedSolver(_public_scene())
    emitter = prepared.get_emitter(0, samples=2, rays=8, flip_faces=False)
    controls = dict(rays=8, cp_grid=[.11, .19], cp_dims=[.3, .1, .6, .2, .8],
                    surf_active=np.ones(3, np.uint8), emit_sid=0,
                    include_sky=True)
    scene = prepared.get_scene(use_bvh=True)
    old_scene = weakref.ref(scene)
    gpu.trace_batch(scene, emitter, **controls)
    keys = set(gpu._buffers)
    buffers = {name: gpu._buffers[name] for name in ("geom", "bounds", "nodes", "escape")}
    del scene
    for frame in range(6):
        transform = np.eye(4)
        transform[0, 3] = frame * .2
        prepared.update_transform("blocker", transform)
        gpu.trace_batch(prepared.get_scene(use_bvh=True), emitter, **controls)
        assert set(gpu._buffers) == keys
        assert all(gpu._buffers[name] is buffer for name, buffer in buffers.items())
    gc.collect()
    assert old_scene() is None


def test_portable_runtime_can_initialize_on_run_worker(gpu):
    program = """
import numpy as np
from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Budget, Channel
v = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], np.float32)
f = np.array([[0,1,2],[0,2,3]], np.int32)
current = Scene.from_meshes({'lower': Mesh(v,f), 'upper': Mesh(v+[0,0,1],f[:,::-1])})
opts = SolveOptions(Sampling(density=2,rays_per_cell=8),
                    Accuracy(max_replicates=3,min_replicates=3,tolerance=0),batch_size=7)
with Solver(current, device='vulkan', auto_tune=False) as solver:
    run = solver.start(Query.row('lower'),opts)
    result = run.submit(Budget(rays=19)).result(timeout=30)
    assert result.rays_used == 19
    assert result.value('lower',Channel('surface','upper','front')) > 0
"""
    result = subprocess.run([sys.executable,"-c",program],capture_output=True,
                            text=True,timeout=40,env=os.environ.copy())
    assert result.returncode == 0, result.stdout+result.stderr


def test_singleton_sky_matches_cpu(gpu):
    from dataclasses import replace
    from tests.v2_cases import SkyCase, sky_case
    meshes = [_square("upward", 0)]
    params = SkyCase(samples=2, rays=8, max_iters=3, min_iters=3,
                       device="taichi", seed=11)
    actual = sky_case(meshes, params)
    expected = sky_case(meshes, replace(params, device="cpu"))
    _assert_results_match(actual, expected)
    assert actual["upward"]["Sky"] == 1.0


def test_portable_run_queues_additive_advances(gpu, monkeypatch):
    from raystrack import Solver, Query, SolveOptions, Sampling, Accuracy, Budget
    from tests.v2_cases import scene_for
    started,release=threading.Event(),threading.Event()
    generate=gpu._generate
    def gated_generate(*args):
        if not started.is_set():
            started.set()
            assert release.wait(10)
        return generate(*args)
    monkeypatch.setattr(gpu,"_generate",gated_generate)
    opts=SolveOptions(Sampling(density=2,rays_per_cell=8),
                      Accuracy(max_replicates=3,min_replicates=3,tolerance=0),batch_size=7)
    with Solver(scene_for(_public_scene()),device=gpu.arch,auto_tune=False) as solver:
        run=solver.start(Query.row("lower"),opts)
        first=run.submit(Budget(rays=19))
        try:
            assert started.wait(10)
            queued=run.submit(Budget(rays=23))
            latest=run.submit(Budget(rays=37))
            assert not queued.done() and not latest.done()
        finally:
            release.set()
        assert first.result(timeout=30).cumulative_rays == 19
        assert queued.result(timeout=30).cumulative_rays == 42
        result=latest.result(timeout=30)
        assert result.rays_used == 37 and result.cumulative_rays == 79
        assert run.result is result
