"""Fair sampling, complete-replicate errors, and exact seeded continuation."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from raystrack import MatrixParams, SkyParams, PreparedSolver
from raystrack.execution import SolveAccumulator, _CpuWorkspace, solve
from raystrack.tuning import ExecutionPlan


def scene(count=3):
    vertices = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
    faces = np.array([[0,1,2], [0,2,3]], np.int32)
    return [(f"row{i}", vertices + [0,0,float(i)], faces) for i in range(count)]


def sky_params(**overrides):
    values = dict(samples=2, rays=8, max_iters=100, min_iters=3, tol=0,
                  device="cpu", auto_tune=False, seed=17, ray_batch_size=65536)
    values.update(overrides)
    return SkyParams(**values)


def test_small_budget_is_distributed_substantially_across_every_emitter():
    prepared = PreparedSolver(scene())
    acc = SolveAccumulator()
    _, results = solve(prepared.meshes, prepared=prepared,
                       sky_params=sky_params(max_total_rays=100), accumulator=acc)
    rows = acc.stats()["emitters"]
    assert set(results) == set(rows) == {"row0", "row1", "row2"}
    assert acc.cumulative_rays == 100
    shares = [row["rays"] for row in rows.values()]
    assert min(shares) >= 20
    assert max(shares) - min(shares) <= 16
    assert all(row["replicates"] == 0 for row in rows.values())


def test_minimal_global_budget_serves_each_selected_row():
    prepared = PreparedSolver(scene())
    acc = SolveAccumulator()
    solve(prepared.meshes, prepared=prepared,
          sky_params=sky_params(max_total_rays=3), accumulator=acc)
    assert [row["rays"] for row in acc.stats()["emitters"].values()] == [1, 1, 1]


def test_resume_matches_uninterrupted_prefix_without_duplicate_rays():
    prepared = PreparedSolver(scene())
    params = sky_params(max_total_rays=31, ray_batch_size=17)
    resumed, full = SolveAccumulator(), SolveAccumulator()
    trace = _CpuWorkspace.trace
    seen = set()

    def record(workspace, scene, emitter, n_surf, **options):
        shift = options["cp_grid"].tobytes()
        start, size = options["ray_offset"], options["ray_count"]
        samples = {(options["emit_sid"], shift, ray) for ray in range(start, start+size)}
        assert seen.isdisjoint(samples), "a previously traced ray was traced again"
        seen.update(samples)
        return trace(workspace, scene, emitter, n_surf, **options)

    with patch.object(_CpuWorkspace, "trace", record):
        solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=resumed)
        actual = solve(prepared.meshes, prepared=prepared,
                       sky_params=replace(params, max_total_rays=33), accumulator=resumed)
    expected = solve(prepared.meshes, prepared=prepared,
                     sky_params=replace(params, max_total_rays=64), accumulator=full)
    assert actual == expected
    assert len(seen) == resumed.cumulative_rays == full.cumulative_rays == 64
    assert resumed.stats()["emitters"] == full.stats()["emitters"]


def test_incomplete_replicates_do_not_claim_zero_variance_convergence():
    prepared = PreparedSolver(scene(1))
    params = sky_params(max_total_rays=31)
    acc = SolveAccumulator()
    n_once = prepared.get_emitter(0, samples=params.samples, rays=params.rays,
                                 flip_faces=False).n_cells * params.rays
    solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    info = acc.stats()["emitters"]["row0"]
    assert info["replicates"] == 0 and info["ray_offset"] == 31
    assert info["sky"]["stderr"] is None and not info["sky"]["converged"]
    solve(prepared.meshes, prepared=prepared,
          sky_params=replace(params, max_total_rays=n_once-31), accumulator=acc)
    assert acc.stats()["emitters"]["row0"]["sky"]["replicates"] == 1
    solve(prepared.meshes, prepared=prepared,
          sky_params=replace(params, max_total_rays=n_once), accumulator=acc)
    assert not acc.stats()["emitters"]["row0"]["sky"]["converged"]
    solve(prepared.meshes, prepared=prepared,
          sky_params=replace(params, max_total_rays=n_once), accumulator=acc)
    assert acc.status == "converged"
    assert acc.cumulative_rays == 3 * n_once


def test_resume_reopens_a_completed_iteration_cap_without_restarting():
    prepared = PreparedSolver(scene(1))
    params = sky_params(max_iters=1, min_iters=1)
    acc = SolveAccumulator()
    solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    previous = acc.cumulative_rays
    assert acc.status == "max_iters"
    progress = []
    solve(prepared.meshes, prepared=prepared,
          sky_params=replace(params, max_iters=3, min_iters=2), accumulator=acc,
          progress=progress.append)
    assert sum(progress) == previous
    assert acc.cumulative_rays == 2 * previous and acc.status == "converged"


def test_sampling_and_scene_changes_reject_existing_accumulation():
    prepared = PreparedSolver(scene())
    params = sky_params(max_total_rays=17)
    acc = SolveAccumulator()
    solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    for changed in (replace(params, seed=18), replace(params, samples=3),
                    replace(params, emitter_names=["row1"])):
        with pytest.raises(ValueError, match="fresh"):
            solve(prepared.meshes, prepared=prepared, sky_params=changed, accumulator=acc)
    transform = np.eye(4)
    transform[0,3] = .4
    prepared.update_transform("row1", transform)
    with pytest.raises(ValueError, match="fresh"):
        solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    assert acc.cumulative_rays == 17


def test_accuracy_allocation_prioritizes_noisy_rows_after_fair_exploration():
    def model(workspace, scene, emitter, n_surf, **options):
        # A deterministic Bernoulli prefix with known row-specific variability
        # across randomized replicates, independent of ray chunk boundaries.
        row, jitter = options["emit_sid"], float(options["cp_grid"][0])
        probability = ((.3 + .15*jitter) if row == 0 else
                       (.1 + .8*jitter) if row == 1 else (.4 + .1*jitter))
        offset, count = options["ray_offset"], options["ray_count"]
        hits = int((offset+count)*probability) - int(offset*probability)
        return np.zeros(n_surf, np.int64), np.zeros(n_surf, np.int64), np.array([hits], np.int64)

    prepared = PreparedSolver(scene())
    fair, adaptive = SolveAccumulator(), SolveAccumulator()
    params = sky_params(max_total_rays=4096, ray_batch_size=32)
    with patch.object(_CpuWorkspace, "trace", model):
        solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=fair)
        solve(prepared.meshes, prepared=prepared,
              sky_params=replace(params, sampling_mode="adaptive"), accumulator=adaptive)
    fair_rows, adapted = fair.stats()["emitters"], adaptive.stats()["emitters"]
    assert adaptive.cumulative_rays == fair.cumulative_rays == 4096
    assert all(row["sky"]["replicates"] >= params.min_iters for row in adapted.values())
    assert adapted["row1"]["rays"] > 1.3 * fair_rows["row1"]["rays"]
    assert adapted["row1"]["rays"] > max(adapted["row0"]["rays"], adapted["row2"]["rays"])
    assert min(row["rays"] for row in adapted.values()) > 3 * 128


def test_resume_keeps_numerical_backend_when_auto_calibration_changes():
    prepared = PreparedSolver(scene(1))
    acc = SolveAccumulator()
    params = sky_params(device="auto", max_total_rays=17)
    with patch("raystrack.tuning.plan_execution", return_value=ExecutionPlan("cpu", 17)):
        solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    with patch("raystrack.tuning.plan_execution", return_value=ExecutionPlan("taichi", 64)):
        solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    assert acc.cumulative_rays == 34
    assert acc.stats()["execution"]["backend"] == "cpu"
    assert prepared._last_execution_plan["backend"] == "cpu"


def test_child_branches_report_one_cumulative_total():
    prepared = PreparedSolver(scene())
    acc = SolveAccumulator()
    first, second = acc.child("matrix"), acc.child("sky")
    solve(prepared.meshes, prepared=prepared,
          matrix_params=MatrixParams(samples=2, rays=8, device="cpu", auto_tune=False,
                                     reciprocity=False, max_total_rays=19), accumulator=first)
    solve(prepared.meshes, prepared=prepared, sky_params=sky_params(max_total_rays=23), accumulator=second)
    assert acc.cumulative_rays == 42
    assert set(acc.stats()["children"]) == {"matrix", "sky"}
    assert acc.child("matrix") is first


def test_zero_additional_budget_returns_existing_result_without_tracing():
    prepared = PreparedSolver(scene(1))
    params = sky_params(max_total_rays=13)
    acc = SolveAccumulator()
    expected = solve(prepared.meshes, prepared=prepared, sky_params=params, accumulator=acc)
    with patch.object(_CpuWorkspace, "trace", side_effect=AssertionError("tracing with zero cap")):
        actual = solve(prepared.meshes, prepared=prepared,
                       sky_params=replace(params, max_total_rays=0), accumulator=acc)
    assert actual == expected and acc.cumulative_rays == 13
