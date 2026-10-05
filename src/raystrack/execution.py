"""Bounded, shared CPU/CUDA/portable solves for dynamic scenes.

The original CUDA matrix scheduler remains available for unrestricted matrix
solves. This executor keeps preview budgets global, uses round-robin emitter
scheduling, and shares scene/sky ray intersections.
"""
from __future__ import annotations

import math
import time
import weakref
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .devices import resolve_backend
from .utils.ray_builder import build_rays
from .utils.cpu_trace import (trace_cpu_combined, trace_cpu_bvh_combined,
                             reduce_first_hits, bin_tregenza_cpu,
                             count_upward_misses_cpu)


def validate_controls(params) -> None:
    for key in ("samples", "rays", "max_iters", "min_iters", "ray_batch_size"):
        value = getattr(params, key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if params.min_total_rays < 0:
        raise ValueError("min_total_rays must be nonnegative")
    if params.max_total_rays is not None:
        value = params.max_total_rays
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("max_total_rays must be a nonnegative integer or None")
    if params.max_time_ms is not None and (not math.isfinite(params.max_time_ms) or params.max_time_ms < 0):
        raise ValueError("max_time_ms must be finite and nonnegative or None")
    if params.tol_mode not in ("stderr", "delta"):
        raise ValueError("tol_mode must be stderr or delta")
    if not math.isfinite(params.tol) or params.tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if params.emitter_names is not None:
        if isinstance(params.emitter_names, str) or len(set(params.emitter_names)) != len(params.emitter_names):
            raise ValueError("emitter_names must be a sequence of unique names")
    if getattr(params, "sampling_mode", "fair") not in ("fair", "adaptive"):
        raise ValueError("sampling_mode must be fair or adaptive")


def common_required(params, cancel=None, progress=None, *, shared=False, accumulator=None) -> bool:
    validate_controls(params)
    if (accumulator is not None or getattr(params, "sampling_mode", "fair") == "adaptive"
            or cancel is not None or progress is not None or params.emitter_names is not None
            or params.max_total_rays is not None or params.max_time_ms is not None):
        return True
    backend = resolve_backend(params.device)
    return backend not in ("cpu", "cuda") or (shared and backend == "cuda")


def _build(emitter, rays, orig, dirs, cp_grid, cp_dims, offset=0):
    build_rays(emitter.u_grid, emitter.v_grid, emitter.halton_tri,
               emitter.halton_u, emitter.halton_v, emitter.halton_r1,
               emitter.halton_r2, emitter.cdf, emitter.tri_a, emitter.tri_e1,
               emitter.tri_e2, emitter.tri_u, emitter.tri_v, emitter.tri_n,
               emitter.tri_origin_eps, rays, orig, dirs, cp_grid, cp_dims, offset)


class _CpuWorkspace:
    def __init__(self):
        self.capacity = 0

    def trace(self, scene, emitter, n_surf, *, rays, cp_grid, cp_dims,
              surf_active, emit_sid, discrete, include_sky, ray_count,
              ray_offset=0, **unused):
        if ray_count > self.capacity:
            self.capacity = ray_count
            self.orig = np.empty((ray_count, 3), np.float32)
            self.dirs = np.empty_like(self.orig)
            self.hit = np.empty(ray_count, np.int32)
            self.front = np.empty(ray_count, np.uint8)
            self.mask = np.empty(ray_count, np.uint8)
        orig, dirs = self.orig[:ray_count], self.dirs[:ray_count]
        hit, front, mask = self.hit[:ray_count], self.front[:ray_count], self.mask[:ray_count]
        _build(emitter, rays, orig, dirs, cp_grid, cp_dims, ray_offset)
        if getattr(scene, "instanced", False):
            from .utils.cpu_trace import trace_cpu_instanced_combined
            trace_cpu_instanced_combined(orig, dirs, scene, surf_active, emit_sid, 0, hit, front, mask)
        elif scene.use_bvh:
            args = (orig, dirs, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid, surf_active)
            trace_cpu_bvh_combined(*args, scene.bb_min, scene.bb_max, scene.left,
                                   scene.right, scene.start, scene.count,
                                   emit_sid, 0, hit, front, mask)
        else:
            args = (orig, dirs, scene.v0, scene.e1, scene.e2, scene.normals, scene.sid, surf_active)
            trace_cpu_combined(*args, emit_sid, 0, hit, front, mask)
        hf, hb = np.zeros(n_surf, np.int64), np.zeros(n_surf, np.int64)
        reduce_first_hits(hit, front, hf, hb)
        sky = np.zeros(145 if discrete else 1, np.int64)
        if include_sky:
            if discrete:
                bin_tregenza_cpu(dirs, mask, sky)
            else:
                sky[0] = count_upward_misses_cpu(dirs, mask)
        return hf, hb, sky


class _CudaWorkspace:
    """Persistent buffers; multiple replicates read back in one compact copy."""
    def __init__(self, async_):
        from numba import cuda
        self.stream = cuda.stream() if async_ else 0
        self.capacity = self.summary_capacity = self.columns = 0

    def trace_many(self, prepared, emitter_index, scene, emitter, n_surf, specs, **options):
        from numba import cuda
        from .utils.cuda_trace import (kernel_build_rays, kernel_trace_combined,
            kernel_trace_bvh_combined, kernel_reduce_hits, kernel_bin_tregenza,
            kernel_count_upward_misses, kernel_zero_i64)
        from .main import _compute_cuda_launch
        stream = self.stream
        capacity = max(spec[2] for spec in specs)
        columns = 2 * n_surf + (145 if options["discrete"] else 1)
        if capacity > self.capacity:
            self.capacity = capacity
            self.orig = cuda.device_array((capacity, 3), np.float32)
            self.dirs = cuda.device_array((capacity, 3), np.float32)
            self.hit = cuda.device_array(capacity, np.int32)
            self.front = cuda.device_array(capacity, np.uint8)
            self.mask = cuda.device_array(capacity, np.uint8)
            self.host_orig = np.empty((capacity, 3), np.float32)
            self.host_dirs = np.empty_like(self.host_orig)
        if columns != self.columns or len(specs) > self.summary_capacity:
            self.columns, self.summary_capacity = columns, len(specs)
            self.summary = cuda.device_array((len(specs), columns), np.int64)
            self.active = cuda.device_array(n_surf, np.uint8)
        self.active.copy_to_device(options["surf_active"], stream=stream)
        ds = (prepared.get_device_instanced_scene() if getattr(scene, "instanced", False)
              else prepared.get_device_scene(use_bvh=scene.use_bvh))
        de = prepared.get_device_emitter(emitter_index, samples=options["samples"],
                    rays=options["rays"], flip_faces=options["flip_faces"])
        # Geometry uploads may use the default stream. Finish them before the
        # workspace stream traverses updated buffers.
        if getattr(self, "_scene_version", None) != prepared.version:
            cuda.synchronize()
            self._scene_version = prepared.version
        for slot, (cp_grid, cp_dims, ray_count, ray_offset) in enumerate(specs):
            orig, dirs = self.orig[:ray_count], self.dirs[:ray_count]
            hit, front, mask = self.hit[:ray_count], self.front[:ray_count], self.mask[:ray_count]
            blocks, threads = _compute_cuda_launch(ray_count, None)
            if options["gpu_raygen"]:
                kernel_build_rays[blocks, threads, stream](de.u_grid, de.v_grid,
                    de.halton_tri, de.halton_u, de.halton_v, de.halton_r1, de.halton_r2,
                    de.cdf, de.tri_a, de.tri_e1, de.tri_e2, de.tri_u, de.tri_v,
                    de.tri_n, de.tri_origin_eps, options["rays"], orig, dirs,
                    cp_grid, cp_dims, ray_offset)
            else:
                ho, hd = self.host_orig[:ray_count], self.host_dirs[:ray_count]
                _build(emitter, options["rays"], ho, hd, cp_grid, cp_dims, ray_offset)
                # The host buffers are shared: synchronous copies prevent
                # overwriting them while an earlier upload is pending.
                orig.copy_to_device(ho)
                dirs.copy_to_device(hd)
            row = self.summary[slot]
            zblocks, zthreads = _compute_cuda_launch(columns, None)
            kernel_zero_i64[zblocks, zthreads, stream](row)
            if getattr(scene, "instanced", False):
                from .utils.cuda_trace import kernel_trace_instanced_combined
                kernel_trace_instanced_combined[blocks, threads, stream](
                    orig, dirs, *ds, self.active, emitter_index, 0, hit, front, mask)
            elif scene.use_bvh:
                args = (orig, dirs, ds.v0, ds.e1, ds.e2, ds.normals, ds.sid, self.active)
                kernel_trace_bvh_combined[blocks, threads, stream](*args,
                    ds.bb_min, ds.bb_max, ds.left, ds.right, ds.start, ds.count,
                    emitter_index, 0, hit, front, mask)
            else:
                args = (orig, dirs, ds.v0, ds.e1, ds.e2, ds.normals, ds.sid, self.active)
                kernel_trace_combined[blocks, threads, stream](*args,
                    emitter_index, 0, hit, front, mask)
            kernel_reduce_hits[blocks, threads, stream](hit, front, row[:n_surf], row[n_surf:2*n_surf], n_surf)
            if options["include_sky"]:
                if options["discrete"]:
                    kernel_bin_tregenza[blocks, threads, stream](dirs, mask, row[2*n_surf:])
                else:
                    kernel_count_upward_misses[blocks, threads, stream](dirs, mask, row[2*n_surf:])
        host = self.summary[:len(specs)].copy_to_host(stream=stream)
        if stream:
            stream.synchronize()
        else:
            cuda.synchronize()
        return [(r[:n_surf], r[n_surf:2*n_surf], r[2*n_surf:]) for r in host]


@dataclass
class _Estimate:
    params: object
    width: int
    selected: object = None
    empty: bool = False

    def __post_init__(self):
        self.hits = np.zeros(self.width, np.int64)
        self.mean = np.zeros(self.width, np.float64)
        self.m2 = np.zeros(self.width, np.float64)
        self.previous = None
        self.delta_change = None
        self.pending_hits = np.zeros(self.width, np.int64)
        self.pending_rays = 0
        self.completed_hits = np.zeros(self.width, np.int64)
        self.completed_rays = 0
        self.rays = self.iterations = 0
        self.converged = bool(self.empty)

    @property
    def done(self):
        return self.empty or self.converged or self.iterations >= self.params.max_iters

    def stderr(self):
        if self.empty:
            return 0.0
        if self.iterations < 2:
            return None
        se = np.sqrt(np.maximum(self.m2 / (self.iterations - 1), 0) / self.iterations)
        values = se if self.selected is None else se[self.selected]
        return float(np.max(values)) if len(values) else 0.0

    def _checkpoint(self):
        from .main import _convergence_checkpoint
        p = self.params
        return _convergence_checkpoint(self.iterations, min_iters=p.min_iters,
            interval=p.convergence_interval, max_iters=p.max_iters,
            needs_variance=p.tol_mode == "stderr", total_rays=self.completed_rays,
            min_total_rays=p.min_total_rays)

    def refresh(self, params):
        """Re-evaluate a tighter tolerance or larger cap from stored statistics."""
        self.params = params
        self.converged = bool(self.empty)
        if self.empty or not self._checkpoint():
            return
        if params.tol_mode == "stderr":
            se = self.stderr()
            self.converged = se is not None and se <= params.tol
        elif self.delta_change is not None:
            values = self.delta_change if self.selected is None else self.delta_change[self.selected]
            self.converged = bool(np.all(values < params.tol))

    def add_chunk(self, counts, rays, *, complete):
        """Accumulate all rays; only complete randomized replicates enter M2."""
        self.hits += counts
        self.rays += rays
        self.pending_hits += counts
        self.pending_rays += rays
        if not complete:
            return
        self.iterations += 1
        self.completed_hits += self.pending_hits
        self.completed_rays += self.pending_rays
        sample = self.pending_hits.astype(np.float64) / self.pending_rays
        delta = sample - self.mean
        self.mean += delta / self.iterations
        self.m2 += delta * (sample - self.mean)
        self.pending_hits.fill(0)
        self.pending_rays = 0
        if self._checkpoint() and self.params.tol_mode == "delta":
            current = self.completed_hits / self.completed_rays
            if self.previous is not None:
                self.delta_change = np.abs(current - self.previous)
            self.previous = current.copy()
        self.refresh(self.params)

    def values(self):
        return self.hits / max(1, self.rays)

    def stats(self):
        status = ("converged" if self.converged else "max_iters" if self.done
                  else "sampling" if self.rays else "not_started")
        return dict(rays=int(self.rays), replicates=int(self.iterations),
                    stderr=self.stderr(), converged=bool(self.converged), status=status)


@dataclass
class _EmitterState:
    active: np.ndarray
    receivers: np.ndarray
    matrix: object
    sky: object
    n_once: int
    iteration: int = 0
    offset: int = 0
    rays: int = 0

    @property
    def done(self):
        return all(estimate is None or estimate.done for estimate in (self.matrix, self.sky))

    def exploring(self):
        for estimate in (self.matrix, self.sky):
            if estimate is None or estimate.done:
                continue
            # A variance estimate requires independent complete replicates.
            floor = max(2, int(estimate.params.min_iters))
            if estimate.iterations < floor or estimate.completed_rays < estimate.params.min_total_rays:
                return True
        return False

    def priority(self):
        scores = []
        for estimate in (self.matrix, self.sky):
            if estimate is None or estimate.done:
                continue
            error = estimate.stderr()
            if error is None:
                return float("inf")
            tolerance = float(estimate.params.tol)
            scores.append(error / tolerance if tolerance > 0 else error)
        return max(scores, default=0.0)


class SolveAccumulator:
    """Raw same-scene estimates that can resume without tracing old rays.

    A solve budget counts *additional* rays; results and ``cumulative_rays``
    include previous calls. Sampling, selection and geometry must match.
    Tolerances, iteration caps, deadlines and chunk sizes may change.

    Complete randomized replicates contribute to Welford variance. A partial
    replicate retains its exact seed and ray offset for the next call. Shared
    outputs retain both kinds of counts while either needs more rays, allowing
    stricter refinement to use intersections already computed for the other.
    """

    def __init__(self):
        self._solver = None
        self._version = None
        self._signature = None
        self._states = {}
        self._names = []
        self._order = []
        self._cursor = 0
        self._steps = 0
        self._scheduled_idx = None
        self._chunk_remaining = 0
        self._chunk_is_fair = True
        self._fair_quantum = 1
        self._fair_pending = None
        self._rays = 0
        self._children = {}
        self.plan = None

    def child(self, name):
        """Independent matrix/sky branches for incompatible sampling tables."""
        if self._signature is not None:
            raise ValueError("A bound accumulator cannot become a branch container")
        if name not in ("matrix", "sky"):
            raise ValueError("Accumulator child must be matrix or sky")
        return self._children.setdefault(name, SolveAccumulator())

    @property
    def cumulative_rays(self):
        return int(self._rays + sum(child.cumulative_rays for child in self._children.values()))

    @property
    def status(self):
        if self._children:
            statuses = [child.status for child in self._children.values()]
        else:
            statuses = [estimate.stats()["status"] for state in self._states.values()
                        for estimate in (state.matrix, state.sky) if estimate is not None]
        if not statuses:
            return "not_started"
        if all(status == "converged" for status in statuses):
            return "converged"
        if all(status in ("converged", "max_iters") for status in statuses):
            return "max_iters"
        return "sampling" if self.cumulative_rays else "not_started"

    def stats(self):
        result = dict(status=self.status, cumulative_rays=self.cumulative_rays)
        if self.plan is not None:
            result["execution"] = self.plan
        if self._children:
            result["children"] = {name: child.stats() for name, child in self._children.items()}
        else:
            result["emitters"] = {self._names[idx]: dict(
                rays=int(state.rays), replicates=int(state.iteration), ray_offset=int(state.offset),
                matrix=None if state.matrix is None else state.matrix.stats(),
                sky=None if state.sky is None else state.sky.stats())
                for idx, state in self._states.items()}
        return result

    @staticmethod
    def _sampling_signature(matrix_params, sky_params):
        p = matrix_params if matrix_params is not None else sky_params
        selection = None if p.emitter_names is None else tuple(p.emitter_names)
        return (p.samples, p.rays, p.seed, p.bvh, p.device, p.cuda_async,
                p.gpu_raygen, selection, getattr(p, "sampling_mode", "fair"),
                matrix_params is not None, sky_params is not None,
                bool(matrix_params and matrix_params.flip_faces),
                bool(sky_params and sky_params.discrete),
                bool(matrix_params and matrix_params.reciprocity),
                None if matrix_params is None else matrix_params.reciprocity_mode)

    def compatible(self, prepared, matrix_params=None, sky_params=None):
        if self._signature is None:
            return not self._children
        return (self._solver() is prepared and self._version == prepared.version
                and self._signature == self._sampling_signature(matrix_params, sky_params))

    def _bind(self, solver, matrix_params, sky_params, names, order, states):
        signature = self._sampling_signature(matrix_params, sky_params)
        if self._children:
            raise ValueError("Use the appropriate accumulator child for this solve")
        if self._signature is None:
            self._solver, self._version = weakref.ref(solver), solver.version
            self._signature = signature
            self._names, self._order, self._states = list(names), list(order), states
        elif not self.compatible(solver, matrix_params, sky_params):
            raise ValueError("Accumulator scene or sampling changed; start a fresh preview/solve")
        for state in self._states.values():
            if state.matrix is not None:
                state.matrix.refresh(matrix_params)
            if state.sky is not None:
                state.sky.refresh(sky_params)


def solve(meshes, *, matrix_params=None, sky_params=None, prepared=None,
          cancel: Callable[[], bool] | None = None,
          progress: Callable[[int], None] | None = None, accumulator=None):
    """Solve fresh or resume a compatible accumulator with an additional budget."""
    from .main import _ensure_prepared
    if accumulator is not None and not isinstance(accumulator, SolveAccumulator):
        raise TypeError("accumulator must be a SolveAccumulator or None")
    solver = _ensure_prepared(meshes, prepared)
    with solver.solve_lock:
        return _solve(meshes, matrix_params=matrix_params, sky_params=sky_params,
                      prepared=solver, cancel=cancel, progress=progress,
                      accumulator=accumulator)


def _solve(meshes, *, matrix_params=None, sky_params=None, prepared=None,
           cancel=None, progress=None, accumulator=None):
    from .main import (_select_bvh, _build_emitter_surface_mask,
                       _average_bidirectional_front_flux, outside_workflow_shareable)
    from .utils.helpers import enforce_reciprocity_and_rowsum
    configs = [p for p in (matrix_params, sky_params) if p is not None]
    if not configs:
        raise ValueError("A matrix or sky parameter set is required")
    for p in configs:
        validate_controls(p)
    if len(configs) == 2 and not outside_workflow_shareable(matrix_params, sky_params):
        raise ValueError("Shared accumulation requires matching matrix/sky sampling")
    p, solver = configs[0], prepared
    started = time.perf_counter()
    names = [mesh[0] for mesh in meshes]
    selected = list(names if p.emitter_names is None else p.emitter_names)
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError(f"Unknown emitter names: {sorted(unknown)}")
    if matrix_params is not None and matrix_params.reciprocity_mode not in ("shortcut", "bidirectional"):
        raise ValueError("reciprocity_mode must be shortcut or bidirectional")
    if matrix_params is not None and matrix_params.reciprocity_mode == "bidirectional" and not matrix_params.reciprocity:
        raise ValueError("bidirectional reciprocity requires reciprocity=True")
    if matrix_params is not None and matrix_params.enforce_reciprocity_rowsum and set(selected) != set(names):
        raise ValueError("Row-sum enforcement requires all emitter rows")
    limits = [c.max_total_rays for c in configs if c.max_total_rays is not None]
    ray_limit = min(limits) if limits else None
    deadlines = [c.max_time_ms for c in configs if c.max_time_ms is not None]
    time_limit = min(deadlines) if deadlines else None
    version, used = solver.version, 0
    persistent = accumulator is not None
    acc = accumulator if persistent else SolveAccumulator()
    if acc._signature is not None and not acc.compatible(solver, matrix_params, sky_params):
        raise ValueError("Accumulator scene or sampling changed; start a fresh preview/solve")

    def stopped():
        return ((cancel is not None and cancel()) or solver.version != version
                or (ray_limit is not None and used >= ray_limit)
                or (time_limit is not None and (time.perf_counter() - started)*1000 >= time_limit))

    if not selected:
        return {}, {}
    # A paused accumulator can still return its previous estimate with zero new
    # rays. Fresh zero-budget/cancelled requests need no device initialization.
    if stopped() and acc._signature is None:
        return {}, {}
    use_bvh = _select_bvh(p.bvh, solver.total_faces)
    scene = (solver.get_instanced_scene() if getattr(solver, "acceleration", "flat") == "instanced"
             else solver.get_scene(use_bvh=use_bvh))
    emitters = solver.get_emitters(samples=p.samples, rays=p.rays,
                    flip_faces=bool(matrix_params and matrix_params.flip_faces))
    centers, extents = solver.get_mesh_bounds()
    n_surf = len(meshes)
    unrestricted = cancel is None and progress is None and ray_limit is None and time_limit is None
    shortcut = bool(not persistent and unrestricted and matrix_params and matrix_params.reciprocity
                    and matrix_params.reciprocity_mode == "shortcut" and p.emitter_names is None)
    discrete = bool(sky_params and sky_params.discrete)
    states = {}
    order = [names.index(name) for name in selected]
    if acc._signature is None:
        for idx in order:
            em = emitters[idx]
            active = _build_emitter_surface_mask(idx, em, centers, extents)
            recv = np.flatnonzero(active)
            if shortcut:
                recv = recv[recv > idx]
            matrix = (_Estimate(matrix_params, 2*n_surf,
                                np.concatenate((recv, recv+n_surf)),
                                empty=not len(recv) or em.total_area <= 0)
                      if matrix_params else None)
            sky = (_Estimate(sky_params, 145 if discrete else 1, empty=em.total_area <= 0)
                   if sky_params else None)
            states[idx] = _EmitterState(active, recv, matrix, sky, em.n_cells * p.rays)
    acc._bind(solver, matrix_params, sky_params, names, order, states)
    states = acc._states

    from .tuning import plan_execution
    previous_plan = acc.plan
    plan = plan_execution(solver, p, include_matrix=matrix_params is not None,
                          include_sky=sky_params is not None, discrete=discrete,
                          allow_calibration=unrestricted and not stopped())
    if (persistent and acc.cumulative_rays and previous_plan is not None
            and plan.backend != previous_plan["backend"]):
        # Calibration can choose a new device after an earlier preview. Keep
        # one numerical backend within an accumulator's seeded sample stream.
        plan = replace(plan, backend=previous_plan["backend"],
                       ray_batch_size=min(p.ray_batch_size, previous_plan["ray_batch_size"]),
                       reason="resumed accumulator retains its original backend")
    acc.plan = solver._last_execution_plan = plan.as_dict()
    backend = plan.backend
    batch_size = min(int(p.ray_batch_size), int(plan.ray_batch_size))
    workspace_cache = getattr(solver, "_execution_workspace_cache", None)
    if workspace_cache is None:
        workspace_cache = solver._execution_workspace_cache = {}
    if backend == "cpu":
        if "cpu" not in workspace_cache:
            workspace_cache["cpu"] = _CpuWorkspace()
        workspace = workspace_cache["cpu"]
    elif backend == "cuda":
        from numba import cuda
        key = ("cuda", *solver._cuda_key(cuda), p.cuda_async)
        if key not in workspace_cache:
            workspace_cache[key] = _CudaWorkspace(p.cuda_async)
        workspace = workspace_cache[key]
    else:
        from .utils.taichi_trace import get_taichi_backend
        workspace = get_taichi_backend(arch="auto" if backend == "taichi" else backend)

    def pending_order():
        count = len(order)
        return [order[(acc._cursor + j) % count] for j in range(count)
                if not states[order[(acc._cursor + j) % count]].done]

    def schedule(pending):
        if acc._scheduled_idx in pending and acc._chunk_remaining > 0:
            return acc._scheduled_idx
        acc._scheduled_idx, acc._chunk_remaining = None, 0
        mode = getattr(p, "sampling_mode", "fair")
        exploring = any(states[idx].exploring() for idx in pending)
        if mode == "adaptive" and not exploring and acc._steps % 4 != 3:
            # Error/tolerance ratio prioritizes the rows furthest from their
            # requested accuracy; every fourth chunk retains a fair service.
            idx = max(pending, key=lambda key: states[key].priority())
            quantum, acc._chunk_is_fair = batch_size, False
        else:
            if acc._fair_pending is None:
                acc._fair_pending = set(pending)
            acc._fair_pending.intersection_update(pending)
            if not acc._fair_pending:
                acc._fair_quantum = min(batch_size, 2 * acc._fair_quantum)
                acc._fair_pending = set(pending)
            idx = next(key for key in pending if key in acc._fair_pending)
            quantum = batch_size if len(pending) == 1 else min(batch_size, acc._fair_quantum)
            acc._chunk_is_fair = True
        acc._scheduled_idx = idx
        acc._chunk_remaining = min(quantum, states[idx].n_once - states[idx].offset)
        return idx

    def finish_chunk(idx):
        if acc._chunk_remaining > 0:
            return
        acc._cursor = (order.index(idx) + 1) % len(order)
        acc._steps += 1
        if acc._chunk_is_fair and acc._fair_pending is not None:
            acc._fair_pending.discard(idx)
        acc._scheduled_idx = None

    def accept(idx, counts, ray_count):
        nonlocal used
        state = states[idx]
        state.offset += ray_count
        state.rays += ray_count
        complete = state.offset == state.n_once
        if state.matrix is not None and (persistent or not state.matrix.done):
            state.matrix.add_chunk(np.concatenate((counts[0], counts[1])), ray_count, complete=complete)
        if state.sky is not None and (persistent or not state.sky.done):
            state.sky.add_chunk(counts[2], ray_count, complete=complete)
        if complete:
            state.iteration += 1
            state.offset = 0
        used += ray_count
        acc._rays += ray_count
        if progress is not None:
            progress(ray_count)

    while not stopped():
        pending = pending_order()
        if not pending:
            break
        idx = schedule(pending)
        state, em = states[idx], emitters[idx]
        include_matrix = state.matrix is not None and (persistent or not state.matrix.done)
        include_sky = state.sky is not None and (persistent or not state.sky.done)
        opts = dict(samples=p.samples, rays=p.rays, surf_active=state.active,
                    emit_sid=idx, min_sid=0, include_matrix=include_matrix,
                    include_sky=include_sky, discrete=discrete, gpu_raygen=p.gpu_raygen,
                    flip_faces=bool(matrix_params and matrix_params.flip_faces))
        group = 1
        # Preserve deferred GPU reductions for unrestricted fair solves. Their
        # per-row seeds and counts are independent of cross-row ordering.
        if (backend != "cpu" and unrestricted and state.offset == 0
                and getattr(p, "sampling_mode", "fair") == "fair"
                and state.n_once <= 1048576 and p.gpu_raygen):
            from .main import _convergence_checkpoint
            for count in range(1, 65):
                group, checkpoint = count, False
                for estimate in (state.matrix, state.sky):
                    if estimate is None or estimate.done:
                        continue
                    cfg, done = estimate.params, estimate.iterations + count
                    checkpoint |= done >= cfg.max_iters or _convergence_checkpoint(done,
                        min_iters=cfg.min_iters, interval=cfg.convergence_interval,
                        max_iters=cfg.max_iters, needs_variance=cfg.tol_mode == "stderr",
                        total_rays=estimate.completed_rays + count*state.n_once,
                        min_total_rays=cfg.min_total_rays)
                if checkpoint:
                    break
        if group > 1:
            specs = []
            for j in range(group):
                rng = np.random.default_rng(p.seed + idx + state.iteration + j)
                specs.append((rng.random(2, dtype=np.float32), rng.random(5, dtype=np.float32), state.n_once, 0))
            if backend == "cuda":
                results = workspace.trace_many(solver, idx, scene, em, n_surf, specs, **opts)
            else:
                portable_opts = {key: value for key, value in opts.items() if key not in ("samples", "flip_faces")}
                front, back, bins = workspace.trace_iterations(scene, em,
                    cp_grids=np.asarray([spec[0] for spec in specs], np.float32),
                    cp_dims=np.asarray([spec[1] for spec in specs], np.float32),
                    ray_count=state.n_once, ray_offset=0, **portable_opts)
                results = list(zip(front, back, bins))
            for result in results:
                accept(idx, result, state.n_once)
            acc._chunk_remaining = 0
            finish_chunk(idx)
            continue
        size = min(batch_size, acc._chunk_remaining, state.n_once - state.offset)
        if ray_limit is not None:
            size = min(size, ray_limit - used)
        if size <= 0:
            break
        rng = np.random.default_rng(p.seed + idx + state.iteration)
        cp_grid, cp_dims = rng.random(2, dtype=np.float32), rng.random(5, dtype=np.float32)
        if backend == "cpu":
            result = workspace.trace(scene, em, n_surf, cp_grid=cp_grid,
                                     cp_dims=cp_dims, ray_count=size, ray_offset=state.offset, **opts)
        elif backend == "cuda":
            result = workspace.trace_many(solver, idx, scene, em, n_surf,
                       [(cp_grid, cp_dims, size, state.offset)], **opts)[0]
        else:
            portable_opts = {key: value for key, value in opts.items() if key not in ("samples", "flip_faces")}
            result = workspace.trace_batch(scene, em, cp_grid=cp_grid, cp_dims=cp_dims,
                        ray_count=size, ray_offset=state.offset, **portable_opts)
        acc._chunk_remaining -= size
        accept(idx, result, size)
        finish_chunk(idx)

    scene_result, sky_result = {}, {}
    for idx, state in states.items():
        name, recv, matrix, sky = names[idx], state.receivers, state.matrix, state.sky
        if matrix is not None and (matrix.rays or matrix.empty):
            values = matrix.values()
            scene_result[name] = {names[j] + suffix: float(values[j+offset])
                for offset, suffix in ((0, "_front"), (n_surf, "_back"))
                for j in recv if values[j+offset] > 0}
        if sky is not None and sky.rays:
            values = sky.values()
            sky_result[name] = ({f"Sky_Patch_{i+1}": float(v) for i, v in enumerate(values)}
                               if discrete else {"Sky": float(values[0])})
    if shortcut:
        areas = [em.total_area for em in emitters]
        for idx, state in states.items():
            for j in state.receivers:
                front = scene_result.get(names[idx], {}).get(names[j] + "_front", 0.0)
                if front > 0 and areas[j] > 0:
                    scene_result.setdefault(names[j], {})[names[idx] + "_front"] = front * areas[idx] / areas[j]
    elif matrix_params and matrix_params.reciprocity and matrix_params.reciprocity_mode == "bidirectional":
        areas = [em.total_area for em in emitters]
        sampled = [mesh for i, mesh in enumerate(meshes)
                   if i in states and states[i].matrix.rays > 0]
        if len(sampled) > 1:
            indices = [names.index(mesh[0]) for mesh in sampled]
            _average_bidirectional_front_flux(scene_result, sampled, [areas[i] for i in indices])
    if (matrix_params and matrix_params.enforce_reciprocity_rowsum
            and all(state.matrix.done for state in states.values())):
        enforce_reciprocity_and_rowsum(scene_result, meshes, [em.total_area for em in emitters])
    return scene_result, sky_result


__all__ = ["SolveAccumulator", "solve"]
