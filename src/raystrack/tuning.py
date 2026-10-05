"""Measure representative solver work and reuse CPU/GPU execution plans.

Calibration rays are diagnostic work, not view-factor samples. Automatic
calibration is restricted to unrestricted solves; previews and hard ray caps
use cached plans or a provisional choice. Call ``warmup`` explicitly before
interactive work to compile kernels and compare synchronized batch timings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from .devices import resolve_backend


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    ray_batch_size: int
    calibrated: bool = False
    reason: str = "provisional"
    measurements: tuple = ()
    failures: dict = field(default_factory=dict)
    warmup_rays: int = 0
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "ray_batch_size": self.ray_batch_size,
                "calibrated": self.calibrated, "reason": self.reason,
                "measurements": [dict(row) for row in self.measurements],
                "failures": dict(self.failures), "warmup_rays": self.warmup_rays,
                "elapsed_ms": self.elapsed_ms}


def _validate(params, repeats=3, max_probe_rays=16384, target_batch_ms=10.0):
    if not isinstance(params.auto_tune, bool):
        raise ValueError("auto_tune must be a bool")
    if not math.isfinite(params.tune_budget_ms) or params.tune_budget_ms < 0:
        raise ValueError("tune_budget_ms must be finite and nonnegative")
    for name, value in (("repeats", repeats), ("max_probe_rays", max_probe_rays)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(target_batch_ms) or target_batch_ms <= 0:
        raise ValueError("target_batch_ms must be finite and positive")


def _representative(prepared, params):
    emitters = prepared.get_emitters(samples=params.samples, rays=params.rays,
                                    flip_faces=bool(getattr(params, "flip_faces", False)))
    names = [m[0] for m in prepared.meshes]
    selection = list(names if params.emitter_names is None else params.emitter_names)
    unknown = set(selection) - set(names)
    if unknown:
        raise ValueError(f"Unknown emitter names: {sorted(unknown)}")
    indices = [names.index(name) for name in selection]
    indices = [i for i in indices if emitters[i].total_area > 0]
    if not indices:
        return None, None, 0
    index = max(indices, key=lambda i: emitters[i].n_cells)
    emitter = emitters[index]
    return index, emitter, emitter.n_cells * params.rays


def _key(prepared, params, include_matrix, include_sky, discrete, n_once):
    # Power-of-two workload buckets survive rigid motion but change with
    # substantial tessellation or sampling changes. Cache belongs to a solver.
    def bucket(value):
        return max(1, int(value)).bit_length()
    return (str(params.device or "auto").lower(), getattr(prepared, "acceleration", "flat"),
            bucket(prepared.total_faces), bucket(n_once), params.bvh,
            bool(params.gpu_raygen), bool(params.cuda_async), bool(include_matrix),
            bool(include_sky), bool(discrete))


def _candidates(mode):
    if mode != "auto":
        return [resolve_backend(mode)]
    candidates = ["cpu"]
    from numba import cuda, config
    # Simulator is useful for correctness but provides no GPU timings.
    if cuda.is_available() and not config.ENABLE_CUDASIM:
        candidates.append("cuda")
    try:
        from .utils.taichi_trace import get_taichi_backend
        get_taichi_backend()
        candidates.append("taichi")
    except (RuntimeError, ImportError):
        pass
    return candidates


def _probe(prepared, params, backend, index, emitter, scene, *, include_matrix,
           include_sky, discrete):
    from .execution import _CpuWorkspace, _CudaWorkspace
    from .main import _build_emitter_surface_mask
    centers, extents = prepared.get_mesh_bounds()
    active = _build_emitter_surface_mask(index, emitter, centers, extents)
    cp_grid = np.asarray([0.321, 0.713], np.float32)
    cp_dims = np.asarray([0.127, 0.391, 0.547, 0.811, 0.239], np.float32)
    options = dict(samples=params.samples, rays=params.rays, surf_active=active,
                   emit_sid=index, min_sid=0, include_matrix=include_matrix,
                   include_sky=include_sky, discrete=discrete,
                   gpu_raygen=params.gpu_raygen,
                   flip_faces=bool(getattr(params, "flip_faces", False)))
    cache = getattr(prepared, "_execution_workspace_cache", None)
    if cache is None:
        cache = prepared._execution_workspace_cache = {}
    if backend == "cpu":
        workspace = cache.setdefault("cpu", _CpuWorkspace())
        return lambda count: workspace.trace(scene, emitter, len(prepared.meshes),
            cp_grid=cp_grid, cp_dims=cp_dims, ray_count=count, ray_offset=0, **options)
    if backend == "cuda":
        from numba import cuda
        key = ("cuda", *prepared._cuda_key(cuda), params.cuda_async)
        if key not in cache:
            cache[key] = _CudaWorkspace(params.cuda_async)
        workspace = cache[key]
        return lambda count: workspace.trace_many(prepared, index, scene, emitter,
            len(prepared.meshes), [(cp_grid, cp_dims, count, 0)], **options)
    from .utils.taichi_trace import get_taichi_backend
    workspace = get_taichi_backend(arch="auto" if backend == "taichi" else backend)
    portable_options = {k: v for k, v in options.items() if k not in ("samples", "flip_faces")}
    return lambda count: workspace.trace_batch(scene, emitter, cp_grid=cp_grid,
        cp_dims=cp_dims, ray_count=count, ray_offset=0, **portable_options)


def _calibrate(prepared, params, *, include_matrix, include_sky, discrete,
               repeats=3, max_probe_rays=16384, target_batch_ms=10.0):
    _validate(params, repeats, max_probe_rays, target_batch_ms)
    started = time.perf_counter()
    index, emitter, n_once = _representative(prepared, params)
    mode = str(params.device or "auto").lower()
    if emitter is None:
        return ExecutionPlan("cpu" if mode == "auto" else resolve_backend(mode),
                             params.ray_batch_size, reason="no nonempty selected emitters")
    from .main import _select_bvh
    use_bvh = _select_bvh(params.bvh, prepared.total_faces)
    scene = (prepared.get_instanced_scene() if getattr(prepared, "acceleration", None) == "instanced"
             else prepared.get_scene(use_bvh=use_bvh))
    maximum = max(1, min(n_once, max_probe_rays, params.ray_batch_size))
    sizes = sorted({min(maximum, size) for size in (1024, 4096, 16384, maximum)}, reverse=True)
    measurements, failures, warmup_rays = [], {}, 0
    for backend in _candidates(mode):
        try:
            trace = _probe(prepared, params, backend, index, emitter, scene,
                           include_matrix=include_matrix, include_sky=include_sky, discrete=discrete)
            # Compile/upload before sampling warmed synchronized wall time.
            trace(maximum)
            warmup_rays += maximum
            backend_rows = []
            for size in sizes:
                timings = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    trace(size)
                    timings.append((time.perf_counter() - t0) * 1000)
                    warmup_rays += size
                ms = float(np.median(timings))
                row = {"backend": backend, "ray_count": size, "median_ms": ms,
                       "ms_per_ray": ms / size}
                backend_rows.append(row)
                # A soft budget always permits one warm measurement per
                # candidate, even when compilation exhausted the deadline.
                if (time.perf_counter() - started) * 1000 >= params.tune_budget_ms:
                    break
            measurements.extend(backend_rows)
        except Exception as exc:
            failures[backend] = f"{type(exc).__name__}: {exc}"
            if mode != "auto":
                raise
    if not measurements:
        raise RuntimeError(f"No backend completed calibration: {failures}")
    # Choose by actual throughput; use a measured batch fitting latency when
    # possible. Avoid extrapolating beyond measured sizes for predictable chunks.
    feasible = [row for row in measurements if row["median_ms"] <= target_batch_ms]
    pool = feasible or measurements
    best = min(pool, key=lambda row: row["ms_per_ray"] if feasible else row["median_ms"])
    backend = best["backend"]
    backend_rows = [row for row in pool if row["backend"] == backend]
    if feasible:
        fastest = min(row["ms_per_ray"] for row in backend_rows)
        # Timings fluctuate; prefer larger chunks when throughput is within
        # five percent, rather than picking tiny batches due to measurement noise.
        close = [row for row in backend_rows if row["ms_per_ray"] <= fastest * 1.05]
        best = max(close, key=lambda row: row["ray_count"])
    else:
        best = min(backend_rows, key=lambda row: row["median_ms"])
    return ExecutionPlan(backend, best["ray_count"], True,
                         "measured synchronized ray generation, tracing and reduction",
                         tuple(measurements), failures, warmup_rays,
                         (time.perf_counter() - started) * 1000)


def warmup(prepared, matrix_params=None, sky_params=None, *, repeats=3,
           max_probe_rays=16384, target_batch_ms=10.0) -> ExecutionPlan:
    """Compile representative kernels and measure a reusable execution plan.

    Explicit devices stay explicit. Calibration rays are reported separately
    and never added to estimates or resumable accumulators. The time budget
    is soft because a compilation or an already running probe must finish.
    """
    from .params import MatrixParams, SkyParams
    from .utils.prepared import PreparedSolver
    if not isinstance(prepared, PreparedSolver):
        raise TypeError("prepared must be a PreparedSolver")
    if matrix_params is None and sky_params is None:
        matrix_params = MatrixParams()
    for params, expected in ((matrix_params, MatrixParams), (sky_params, SkyParams)):
        if params is not None and not isinstance(params, expected):
            raise TypeError(f"expected {expected.__name__}")
    if matrix_params is not None and sky_params is not None:
        from .main import outside_workflow_shareable
        if not outside_workflow_shareable(matrix_params, sky_params):
            raise ValueError("warmup requires compatible matrix and sky parameters; warm each pass separately")
    params = matrix_params if matrix_params is not None else sky_params
    from .execution import validate_controls
    validate_controls(params)
    with prepared.solve_lock:
        plan = _calibrate(prepared, params, include_matrix=matrix_params is not None,
                          include_sky=sky_params is not None,
                          discrete=bool(sky_params and sky_params.discrete), repeats=repeats,
                          max_probe_rays=max_probe_rays, target_batch_ms=target_batch_ms)
        _, _, n_once = _representative(prepared, params)
        cache = getattr(prepared, "_tuning_cache", None)
        if cache is None:
            cache = prepared._tuning_cache = {}
        key = _key(prepared, params, matrix_params is not None, sky_params is not None,
                   bool(sky_params and sky_params.discrete), n_once)
        cache[key] = plan
        # Bound plans across changing topology/density in long live sessions.
        while len(cache) > 16:
            del cache[next(iter(cache))]
        return plan


def plan_execution(prepared, params, *, include_matrix=True, include_sky=False,
                   discrete=False, allow_calibration=False) -> ExecutionPlan:
    """Internal selection; bounded solves never trace calibration rays."""
    _validate(params)
    mode = str(params.device or "auto").lower()
    if not params.auto_tune:
        return ExecutionPlan(resolve_backend(mode), params.ray_batch_size, reason="tuning disabled")
    _, _, n_once = _representative(prepared, params)
    key = _key(prepared, params, include_matrix, include_sky, discrete, n_once)
    cache = getattr(prepared, "_tuning_cache", {})
    if key in cache:
        plan = cache[key]
        return ExecutionPlan(plan.backend, min(params.ray_batch_size, plan.ray_batch_size),
                             plan.calibrated, "cached calibration", plan.measurements,
                             plan.failures, 0, 0)
    if mode == "auto" and allow_calibration and params.tune_budget_ms > 0:
        plan = _calibrate(prepared, params, include_matrix=include_matrix,
                          include_sky=include_sky, discrete=discrete)
        if not hasattr(prepared, "_tuning_cache"):
            prepared._tuning_cache = {}
        prepared._tuning_cache[key] = plan
        while len(prepared._tuning_cache) > 16:
            del prepared._tuning_cache[next(iter(prepared._tuning_cache))]
        return plan
    backend = "cpu" if mode == "auto" and n_once <= 16384 else resolve_backend(mode)
    return ExecutionPlan(backend, params.ray_batch_size,
                         reason="explicit device" if mode != "auto" else "provisional; call warmup to measure")


__all__ = ["ExecutionPlan", "warmup"]
