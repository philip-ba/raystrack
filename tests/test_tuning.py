"""Execution plans must reflect timings and never spend capped solve rays."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from raystrack import MatrixParams, PreparedSolver, SkyParams, warmup
from raystrack import SolveAccumulator, view_factor_matrix
from raystrack.tuning import ExecutionPlan, _calibrate, plan_execution


def solver():
    vertices = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
    faces = np.array([[0,1,2], [0,2,3]], np.int32)
    return PreparedSolver([("lower", vertices, faces),
                           ("upper", vertices + [0,0,1], faces[:,::-1].copy())])


def timed_probe(clock, rates, overhead=None):
    overhead = overhead or {}
    def factory(prepared, params, backend, *args, **kwargs):
        def trace(count):
            clock[0] += overhead.get(backend, 0) + count * rates[backend]
        return trace
    return factory


def test_calibration_uses_measured_throughput_and_records_diagnostic_rays():
    prepared = solver()
    params = MatrixParams(device="auto", samples=64, rays=256, tune_budget_ms=10000)
    clock = [0.0]
    with patch("raystrack.tuning._candidates", return_value=["cpu", "taichi"]), \
         patch("raystrack.tuning.time.perf_counter", side_effect=lambda: clock[0]), \
         patch("raystrack.tuning._probe", side_effect=timed_probe(clock, {"cpu": 2e-7, "taichi": 2e-8})):
        plan = warmup(prepared, params, repeats=2)
    assert plan.backend == "taichi"
    assert plan.calibrated and plan.warmup_rays > 0
    assert plan.ray_batch_size == 16384
    assert set(row["backend"] for row in plan.measurements) == {"cpu", "taichi"}
    with patch("raystrack.tuning._calibrate", side_effect=AssertionError("must reuse cached plan")):
        reused = plan_execution(prepared, replace(params, max_total_rays=3), allow_calibration=False)
    assert reused.backend == "taichi"
    assert reused.warmup_rays == 0


def test_small_batches_can_favor_cpu_over_gpu():
    prepared = solver()
    params = MatrixParams(device="auto", samples=1, rays=8, tune_budget_ms=10000)
    clock = [0.0]
    with patch("raystrack.tuning._candidates", return_value=["cpu", "taichi"]), \
         patch("raystrack.tuning.time.perf_counter", side_effect=lambda: clock[0]), \
         patch("raystrack.tuning._probe", side_effect=timed_probe(clock,
             {"cpu": 2e-7, "taichi": 2e-8}, {"taichi": 0.001})):
        plan = warmup(prepared, params)
    assert plan.backend == "cpu"


def test_capped_preview_does_not_initialize_gpu_or_trace_probes():
    params = MatrixParams(device="auto", max_total_rays=8, max_time_ms=50)
    with patch("raystrack.tuning._calibrate", side_effect=AssertionError("no probe tracing")), \
         patch("raystrack.tuning.resolve_backend", side_effect=AssertionError("small scene stays on CPU")):
        plan = plan_execution(solver(), params, allow_calibration=False)
    assert plan.backend == "cpu" and not plan.calibrated and plan.warmup_rays == 0


def test_explicit_device_is_preserved_and_batch_cap_respected():
    prepared = solver()
    params = MatrixParams(device="vulkan", ray_batch_size=17)
    with patch("raystrack.tuning.resolve_backend", return_value="vulkan"):
        plan = plan_execution(prepared, params)
    assert plan.backend == "vulkan" and plan.ray_batch_size == 17


def test_gpu_failure_is_reported_for_auto_and_raised_for_explicit():
    prepared = solver()
    params = MatrixParams(device="auto", tune_budget_ms=10000)
    clock = [0.0]
    normal = timed_probe(clock, {"cpu": 1e-7})
    def probe(*args, **kwargs):
        if args[2] == "vulkan":
            raise RuntimeError("probe device failure")
        return normal(*args, **kwargs)
    with patch("raystrack.tuning._candidates", return_value=["cpu", "vulkan"]), \
         patch("raystrack.tuning.time.perf_counter", side_effect=lambda: clock[0]), \
         patch("raystrack.tuning._probe", side_effect=probe):
        plan = warmup(prepared, params)
    assert plan.backend == "cpu"
    assert "probe device failure" in plan.failures["vulkan"]
    with patch("raystrack.tuning._candidates", return_value=["vulkan"]), \
         patch("raystrack.tuning._probe", side_effect=probe), pytest.raises(RuntimeError, match="device failure"):
        warmup(prepared, replace(params, device="vulkan"))


def test_cached_plan_survives_rigid_motion_but_separates_outputs():
    prepared = solver()
    params = MatrixParams(device="cpu")
    with patch("raystrack.tuning._calibrate", return_value=ExecutionPlan("cpu", 32, True)):
        warmup(prepared, params)
    transform = np.eye(4)
    transform[0,3] = 2
    prepared.update_transform("upper", transform)
    assert plan_execution(prepared, params).calibrated
    assert not plan_execution(prepared, params, include_sky=True).calibrated


def test_invalid_tuning_controls_and_incompatible_outputs():
    for budget in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tune_budget_ms"):
            plan_execution(solver(), MatrixParams(tune_budget_ms=budget))
    with pytest.raises(ValueError, match="auto_tune"):
        plan_execution(solver(), MatrixParams(auto_tune=1))
    with pytest.raises(ValueError, match="compatible"):
        warmup(solver(), MatrixParams(samples=3), SkyParams(samples=4))
    with pytest.raises(ValueError, match="repeats"):
        warmup(solver(), MatrixParams(device="cpu"), repeats=0)


def test_real_cpu_warmup_preserves_geometry_and_can_be_serialized():
    prepared = solver()
    params = MatrixParams(samples=1, rays=2, device="cpu", tune_budget_ms=25)
    plan = warmup(prepared, params, repeats=1, max_probe_rays=32)
    assert plan.calibrated and plan.backend == "cpu"
    assert plan.warmup_rays > 0
    assert prepared.version == 0
    assert plan.as_dict()["measurements"]
    assert plan_execution(prepared, params).warmup_rays == 0


def test_public_auto_cap_counts_only_solve_rays_without_calibration():
    prepared = solver()
    params = MatrixParams(samples=1, rays=4, device="auto", min_iters=3,
                          max_iters=3, max_total_rays=37, reciprocity=False)
    accumulator = SolveAccumulator()
    counts = []
    with patch("raystrack.tuning._calibrate", side_effect=AssertionError("capped solve cannot probe")):
        result = view_factor_matrix(prepared.meshes, params, prepared=prepared,
                                    accumulator=accumulator, progress=counts.append)
    assert set(result) == {"lower", "upper"}
    assert sum(counts) == accumulator.cumulative_rays == 37
    assert prepared._last_execution_plan["backend"] == "cpu"
    assert prepared._last_execution_plan["warmup_rays"] == 0


def test_warmup_does_not_change_seeded_estimate_or_retained_ray_count():
    prepared = solver()
    params = MatrixParams(samples=1, rays=4, device="auto", min_iters=3,
                          max_iters=3, max_total_rays=100, reciprocity=False)
    with patch("raystrack.tuning._candidates", return_value=["cpu"]):
        plan = warmup(prepared, params, repeats=1, max_probe_rays=32)
    assert plan.warmup_rays > 0
    accumulated = SolveAccumulator()
    result = view_factor_matrix(prepared.meshes, params, prepared=prepared,
                                accumulator=accumulated)
    reference = solver()
    expected = view_factor_matrix(reference.meshes, replace(params, device="cpu"), prepared=reference)
    assert result == expected
    assert accumulated.cumulative_rays == 100
    assert prepared._last_execution_plan["calibrated"]


@pytest.mark.parametrize("device", ["auto", "AUTO", None])
def test_unrestricted_public_auto_uses_calibration_plan(device):
    prepared = solver()
    params = MatrixParams(samples=1, rays=2, device=device, min_iters=2,
                          max_iters=2, tol=0, reciprocity=False)
    with patch("raystrack.tuning._calibrate", return_value=ExecutionPlan("cpu", 7, True)) as calibrate:
        result = view_factor_matrix(prepared.meshes, params, prepared=prepared)
    calibrate.assert_called_once()
    assert set(result) == {"lower", "upper"}
    assert prepared._last_execution_plan["backend"] == "cpu"
    assert prepared._last_execution_plan["ray_batch_size"] == 7
