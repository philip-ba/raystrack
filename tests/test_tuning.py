"""Measured plans remain diagnostic work outside bounded solve budgets."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Budget
from raystrack.backends.calibration import ExecutionPlan, plan_execution
from raystrack.engine.controls import controls


def solver(**settings):
    vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],np.float32)
    faces=np.array([[0,1,2],[0,2,3]],np.int32)
    current=Scene.from_meshes({"lower":Mesh(vertices,faces),"upper":Mesh(vertices+[0,0,1],faces[:,::-1])})
    return Solver(current,**settings)


def options(density=16,rays=128,batch_size=65536):
    return SolveOptions(Sampling(density=density,rays_per_cell=rays),
                        Accuracy(max_replicates=3,min_replicates=3,tolerance=0),batch_size=batch_size)


def plan_for(solver,opts,query=None):
    query=Query.matrix() if query is None else query
    cfg=controls(solver,query,opts,Budget())
    return plan_execution(solver._sync_scene(),cfg,include_matrix=True,
                          include_sky=query.sky_mode is not None,discrete=query.sky_mode == "tregenza145")


def timed_probe(clock,rates,overhead=None):
    overhead=overhead or {}
    def factory(prepared,params,backend,*args,**kwargs):
        def trace(count):
            clock[0]+=overhead.get(backend,0)+count*rates[backend]
        return trace
    return factory


def test_calibration_uses_measured_throughput_and_records_diagnostic_rays():
    clock=[0.0]
    with solver(device="auto",tune_budget_ms=10000) as current:
        opts=options(64,256)
        with patch("raystrack.backends.calibration._candidates",return_value=["cpu","taichi"]), \
             patch("raystrack.backends.calibration.time.perf_counter",side_effect=lambda:clock[0]), \
             patch("raystrack.backends.calibration._probe",side_effect=timed_probe(clock,{"cpu":2e-7,"taichi":2e-8})):
            plan=current.warmup(Query.matrix(),opts,repeats=2)
        assert plan.backend == "taichi" and plan.calibrated and plan.warmup_rays > 0
        assert plan.ray_batch_size == 16384
        assert {row["backend"] for row in plan.measurements} == {"cpu","taichi"}
        with patch("raystrack.backends.calibration._calibrate",side_effect=AssertionError("must reuse cached plan")):
            reused=plan_for(current,opts)
        assert reused.backend == "taichi" and reused.warmup_rays == 0


def test_small_batches_can_favor_cpu_over_gpu():
    clock=[0.0]
    with solver(device="auto",tune_budget_ms=10000) as current:
        with patch("raystrack.backends.calibration._candidates",return_value=["cpu","taichi"]), \
             patch("raystrack.backends.calibration.time.perf_counter",side_effect=lambda:clock[0]), \
             patch("raystrack.backends.calibration._probe",side_effect=timed_probe(clock,{"cpu":2e-7,"taichi":2e-8},{"taichi":.001})):
            plan=current.warmup(Query.matrix(),options(1,8))
    assert plan.backend == "cpu"


def test_capped_preview_does_not_initialize_gpu_or_trace_probes():
    with solver(device="auto") as current:
        with patch("raystrack.backends.calibration._calibrate",side_effect=AssertionError("no probe tracing")), \
             patch("raystrack.backends.calibration.resolve_backend",side_effect=AssertionError("small scene stays on CPU")):
            plan=plan_for(current,options())
    assert plan.backend == "cpu" and not plan.calibrated and plan.warmup_rays == 0


def test_explicit_device_is_preserved_and_batch_cap_respected():
    with solver(device="vulkan") as current:
        with patch("raystrack.backends.calibration.resolve_backend",return_value="vulkan"):
            plan=plan_for(current,options(batch_size=17))
    assert plan.backend == "vulkan" and plan.ray_batch_size == 17


def test_gpu_failure_is_reported_for_auto_and_raised_for_explicit():
    clock=[0.0]; normal=timed_probe(clock,{"cpu":1e-7})
    def probe(*args,**kwargs):
        if args[2] == "vulkan":
            raise RuntimeError("probe device failure")
        return normal(*args,**kwargs)
    with solver(device="auto",tune_budget_ms=10000) as current:
        with patch("raystrack.backends.calibration._candidates",return_value=["cpu","vulkan"]), \
             patch("raystrack.backends.calibration.time.perf_counter",side_effect=lambda:clock[0]), \
             patch("raystrack.backends.calibration._probe",side_effect=probe):
            plan=current.warmup(Query.matrix(),options())
    assert plan.backend == "cpu" and "probe device failure" in plan.failures["vulkan"]
    with solver(device="vulkan") as current:
        with patch("raystrack.backends.calibration._candidates",return_value=["vulkan"]), \
             patch("raystrack.backends.calibration._probe",side_effect=probe), pytest.raises(RuntimeError,match="device failure"):
            current.warmup(Query.matrix(),options())


def test_cached_plan_survives_rigid_motion_but_separates_outputs():
    with solver(device="cpu") as current:
        opts=options()
        with patch("raystrack.backends.calibration._calibrate",return_value=ExecutionPlan("cpu",32,True)):
            current.warmup(Query.matrix(),opts)
        transform=np.eye(4); transform[0,3]=2
        current.scene.update_transform("upper",transform)
        assert plan_for(current,opts).calibrated
        assert not plan_for(current,opts,Query.matrix(sky="merged")).calibrated


def test_invalid_tuning_controls_and_shared_output_options():
    for budget in (-1,float("nan"),float("inf")):
        with pytest.raises(ValueError,match="tune_budget_ms"):
            solver(tune_budget_ms=budget)
    with pytest.raises(ValueError,match="auto_tune"):
        solver(auto_tune=1)
    with pytest.raises(TypeError,match="sampling"):
        SolveOptions(sampling={"density":3})
    with solver(device="cpu") as current, pytest.raises(ValueError,match="repeats"):
        current.warmup(Query.matrix(sky="merged"),options(),repeats=0)


def test_real_cpu_warmup_preserves_geometry_and_can_be_serialized():
    with solver(device="cpu",tune_budget_ms=25) as current:
        opts=options(1,2)
        plan=current.warmup(Query.matrix(),opts,repeats=1,max_probe_rays=32)
        assert plan.calibrated and plan.backend == "cpu" and plan.warmup_rays > 0
        assert current.scene.revision == 0 and plan.as_dict()["measurements"]
        assert plan_for(current,opts).warmup_rays == 0


def test_public_auto_cap_counts_only_solve_rays_without_calibration():
    counts=[]
    with solver(device="auto") as current:
        with patch("raystrack.backends.calibration._calibrate",side_effect=AssertionError("capped solve cannot probe")):
            result=current.solve(Query.matrix(),options(1,4),Budget(rays=37),progress=counts.append)
    assert set(result.sender_ids) == {"lower","upper"}
    assert sum(counts) == result.cumulative_rays == 37
    assert result.execution["backend"] == "cpu" and result.execution["warmup_rays"] == 0


def test_warmup_does_not_change_seeded_estimate_or_retained_ray_count():
    with solver(device="auto") as current:
        opts=options(1,4)
        with patch("raystrack.backends.calibration._candidates",return_value=["cpu"]):
            plan=current.warmup(Query.matrix(),opts,repeats=1,max_probe_rays=32)
        result=current.solve(Query.matrix(),opts,Budget(rays=100))
        assert plan.warmup_rays > 0 and result.execution["calibrated"]
    with solver(device="cpu") as reference:
        expected=reference.solve(Query.matrix(),opts,Budget(rays=100))
    np.testing.assert_array_equal(result.dense(),expected.dense())
    assert result.cumulative_rays == 100


@pytest.mark.parametrize("device",["auto","AUTO",None])
def test_unrestricted_public_auto_uses_calibration_plan(device):
    with solver(device=device) as current:
        opts=replace(options(1,2),accuracy=Accuracy(max_replicates=2,min_replicates=2,tolerance=0))
        with patch("raystrack.backends.calibration._calibrate",return_value=ExecutionPlan("cpu",7,True)) as calibrate:
            result=current.solve(Query.matrix(),opts)
    calibrate.assert_called_once()
    assert set(result.sender_ids) == {"lower","upper"}
    assert result.execution["backend"] == "cpu" and result.execution["ray_batch_size"] == 7
