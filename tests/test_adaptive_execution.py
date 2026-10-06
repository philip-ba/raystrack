"""Fair sampling, complete-replicate errors, and exact seeded continuation."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Budget
from raystrack.backends.workspaces import _CpuWorkspace
from raystrack.backends.calibration import ExecutionPlan


def scene(count=3):
    vertices = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], np.float32)
    faces = np.array([[0,1,2],[0,2,3]], np.int32)
    return Scene.from_meshes({f"row{i}": Mesh(vertices+[0,0,float(i)], faces) for i in range(count)})


def options(**changes):
    sampling = Sampling(density=2, rays_per_cell=8, seed=17, mode=changes.pop("mode", "fair"))
    accuracy = Accuracy(max_replicates=changes.pop("max_replicates",100),
                        min_replicates=changes.pop("min_replicates",3), tolerance=changes.pop("tolerance",0))
    return SolveOptions(sampling, accuracy, batch_size=changes.pop("batch_size",65536))


def rows(result):
    return result.statistics["emitters"]


def test_small_budget_is_distributed_substantially_across_every_emitter():
    with Solver(scene(), device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.sky(), options(), Budget(rays=100))
    assert set(rows(result)) == set(result.sender_ids) == {"row0","row1","row2"}
    assert result.cumulative_rays == 100
    shares = [row["rays"] for row in rows(result).values()]
    assert min(shares) >= 20 and max(shares)-min(shares) <= 16
    assert all(row["replicates"] == 0 for row in rows(result).values())


def test_minimal_global_budget_serves_each_selected_row():
    with Solver(scene(), device="cpu", auto_tune=False) as solver:
        result = solver.solve(Query.sky(), options(), Budget(rays=3))
    assert [row["rays"] for row in rows(result).values()] == [1,1,1]


def test_resume_matches_uninterrupted_prefix_without_duplicate_rays():
    trace, seen = _CpuWorkspace.trace, set()
    def record(workspace, scene, emitter, n_surf, **opts):
        shift = opts["cp_grid"].tobytes()
        start,size = opts["ray_offset"],opts["ray_count"]
        row = int(round(float(emitter.plane_origin[2])))
        samples = {(row,shift,ray) for ray in range(start,start+size)}
        assert seen.isdisjoint(samples), "a previously traced ray was traced again"
        seen.update(samples)
        return trace(workspace,scene,emitter,n_surf,**opts)
    with Solver(scene(),device="cpu",auto_tune=False) as solver:
        run = solver.start(Query.sky(), options(batch_size=17))
        with patch.object(_CpuWorkspace,"trace",record):
            run.advance(Budget(rays=31))
            actual = run.advance(Budget(rays=33))
        expected = solver.solve(Query.sky(),options(batch_size=17),Budget(rays=64))
    np.testing.assert_array_equal(actual.dense(),expected.dense())
    assert len(seen) == actual.cumulative_rays == expected.cumulative_rays == 64
    assert rows(actual) == rows(expected)


def test_incomplete_replicates_do_not_claim_zero_variance_convergence():
    with Solver(scene(1),device="cpu",auto_tune=False) as solver:
        run = solver.start(Query.sky(),options())
        first = run.advance(Budget(rays=31))
        info = rows(first)["row0"]
        assert info["replicates"] == 0 and info["ray_offset"] == 31
        assert info["sky"]["stderr"] is None and not info["sky"]["converged"]
        one = run.advance(Budget(rays=97))
        assert rows(one)["row0"]["sky"]["replicates"] == 1
        two = run.advance(Budget(rays=128))
        assert not rows(two)["row0"]["sky"]["converged"]
        three = run.advance(Budget(rays=128))
    assert three.status == "converged" and three.cumulative_rays == 384


def test_resume_reopens_a_completed_iteration_cap_without_restarting():
    with Solver(scene(1),device="cpu",auto_tune=False) as solver:
        opts = options(max_replicates=1,min_replicates=1)
        run = solver.start(Query.sky(),opts)
        first = run.advance()
        assert first.status == "max_iters"
        progress = []
        result = run.advance(options=replace(opts,accuracy=Accuracy(max_replicates=3,min_replicates=2,tolerance=0)),progress=progress.append)
    assert sum(progress) == first.cumulative_rays
    assert result.cumulative_rays == 2*first.cumulative_rays and result.status == "converged"


def test_sampling_and_scene_changes_reject_existing_accumulation():
    current,opts = scene(),options()
    with Solver(current,device="cpu",auto_tune=False) as solver:
        run = solver.start(Query.sky(),opts)
        run.advance(Budget(rays=17))
        for changed in (replace(opts,sampling=replace(opts.sampling,seed=18)),
                        replace(opts,sampling=replace(opts.sampling,density=3))):
            with pytest.raises(ValueError,match="Sampling changed"):
                run.advance(Budget(rays=17),options=changed)
        transform=np.eye(4); transform[0,3]=.4
        current.update_transform("row1",transform)
        with pytest.raises(RuntimeError,match="Scene geometry changed"):
            run.advance(Budget(rays=17))
        assert run.cumulative_rays == 17


def test_accuracy_allocation_prioritizes_noisy_rows_after_fair_exploration():
    def model(workspace,scene,emitter,n_surf,**opts):
        # emit_sid controls whole-surface exclusion; identify these rows by height.
        row = int(round(float(emitter.plane_origin[2])))
        jitter = float(opts["cp_grid"][0])
        probability = (.3+.15*jitter) if row == 0 else (.1+.8*jitter) if row == 1 else (.4+.1*jitter)
        offset,count = opts["ray_offset"],opts["ray_count"]
        hits = int((offset+count)*probability)-int(offset*probability)
        return np.zeros(n_surf,np.int64),np.zeros(n_surf,np.int64),np.array([hits],np.int64)
    with Solver(scene(),device="cpu",auto_tune=False) as solver, patch.object(_CpuWorkspace,"trace",model):
        fair = solver.solve(Query.sky(),options(batch_size=32),Budget(rays=4096))
        adaptive = solver.solve(Query.sky(),options(mode="adaptive",batch_size=32),Budget(rays=4096))
    adapted=rows(adaptive)
    assert adaptive.cumulative_rays == fair.cumulative_rays == 4096
    assert all(row["sky"]["replicates"] >= 3 for row in adapted.values())
    assert adapted["row1"]["rays"] > 1.3*rows(fair)["row1"]["rays"]
    assert adapted["row1"]["rays"] > max(adapted["row0"]["rays"],adapted["row2"]["rays"])
    assert min(row["rays"] for row in adapted.values()) > 3*128


def test_resume_keeps_numerical_backend_when_auto_calibration_changes():
    with Solver(scene(1),device="auto") as solver:
        run=solver.start(Query.sky(),options())
        with patch("raystrack.engine.scheduler.plan_execution",return_value=ExecutionPlan("cpu",17)):
            run.advance(Budget(rays=17))
        with patch("raystrack.engine.scheduler.plan_execution",return_value=ExecutionPlan("taichi",64)):
            result=run.advance(Budget(rays=17))
    assert result.cumulative_rays == 34 and result.execution["backend"] == "cpu"


def test_independent_queries_keep_independent_cumulative_totals():
    with Solver(scene(),device="cpu",auto_tune=False) as solver:
        matrix = solver.start(Query.matrix(),options())
        sky = solver.start(Query.sky(),replace(options(),sampling=replace(options().sampling,seed=18)))
        first=matrix.advance(Budget(rays=19)); second=sky.advance(Budget(rays=23))
        assert first.cumulative_rays+second.cumulative_rays == 42
        matrix.advance(Budget(rays=7))
        assert matrix.cumulative_rays == 26 and sky.cumulative_rays == 23
        assert second.cumulative_rays == 23


def test_zero_additional_budget_returns_existing_result_without_tracing():
    with Solver(scene(1),device="cpu",auto_tune=False) as solver:
        run=solver.start(Query.sky(),options())
        expected=run.advance(Budget(rays=13))
        with patch.object(_CpuWorkspace,"trace",side_effect=AssertionError("tracing with zero cap")):
            actual=run.advance(Budget(rays=0))
    np.testing.assert_array_equal(actual.dense(),expected.dense())
    assert actual.cumulative_rays == 13 and actual.rays_used == 0
