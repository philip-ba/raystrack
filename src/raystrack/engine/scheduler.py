"""One fair/adaptive scheduler for scene, sky and area-pair queries.

Ray offsets, independently shifted replicates and convergence checkpoints are
preserved from the validated v1 executor. Every public solve uses this pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import time
import numpy as np
from .sampling import _select_bvh, _build_emitter_surface_mask
from ..backends.workspaces import workspace_for
from ..backends.calibration import plan_execution

@dataclass
class _Estimate:
    params: object
    width: int
    selected: object = None
    empty: bool = False

    def __post_init__(self):
        self.hits = np.zeros(self.width, np.float64 if self.params.strategy == "area_pair" else np.int64)
        self.mean = np.zeros(self.width, np.float64)
        self.m2 = np.zeros(self.width, np.float64)
        self.previous = None
        self.delta_change = None
        self.pending_hits = np.zeros(self.width, np.float64 if self.params.strategy == "area_pair" else np.int64)
        self.pending_rays = 0
        self.completed_hits = np.zeros(self.width, np.float64 if self.params.strategy == "area_pair" else np.int64)
        self.completed_rays = 0
        self.rays = self.iterations = 0
        self.converged = bool(self.empty)

    @property
    def done(self):
        return self.empty or self.converged or self.iterations >= self.params.max_iters

    def standard_errors(self):
        if self.iterations < 2:
            return None
        return np.sqrt(np.maximum(self.m2 / (self.iterations - 1), 0) / self.iterations)

    def stderr(self):
        if self.empty:
            return 0.0
        if self.iterations < 2:
            return None
        se = self.standard_errors()
        values = se if self.selected is None else se[self.selected]
        return float(np.max(values)) if len(values) else 0.0

    def _checkpoint(self):
        from .sampling import _convergence_checkpoint
        p = self.params
        return _convergence_checkpoint(self.iterations, min_iters=p.min_iters,
            interval=p.convergence_interval, max_iters=p.max_iters,
            needs_variance=p.tol_mode == "stderr", total_rays=self.completed_rays,
            min_total_rays=p.min_total_rays)

    def refresh(self, params):
        """Re-evaluate a tighter tolerance or larger cap from stored statistics."""
        if (params.tol_mode != self.params.tol_mode
                or params.convergence_interval != self.params.convergence_interval):
            self.previous = None
            self.delta_change = None
        self.params = params
        self.converged = bool(self.empty)
        if self.empty or self.pending_rays or not self._checkpoint():
            return
        if params.tol_mode == "stderr":
            se = self.stderr()
            self.converged = se is not None and se <= params.tol
        elif self.delta_change is not None:
            values = self.delta_change if self.selected is None else self.delta_change[self.selected]
            self.converged = bool(np.all(values < params.tol))

    def add_chunk(self, counts, rays, *, complete, weights=None, sides=None, columns=None):
        """Accumulate all rays; only complete randomized replicates enter M2."""
        self.rays += rays
        if weights is None:
            self.hits += counts
            self.pending_hits += counts
            self.pending_rays += rays
        else:
            # Weighted sums depend on reduction order. Keep one replicate's
            # ordered samples and reduce its prefix canonically on every call.
            if not hasattr(self, "_weight_buffer"):
                self._weight_buffer = np.empty(self.params.pair_samples, np.float64)
                self._side_buffer = np.empty(self.params.pair_samples, np.float64)
            sl = slice(self.pending_rays, self.pending_rays + rays)
            self._weight_buffer[sl], self._side_buffer[sl] = weights, sides
            self.pending_rays += rays
            w, side = self._weight_buffer[:self.pending_rays], self._side_buffer[:self.pending_rays]
            self.pending_hits.fill(0)
            self.pending_hits[columns[0]] = np.sum(w[side > 0])
            self.pending_hits[columns[1]] = np.sum(w[side < 0])
            self.hits[:] = self.completed_hits + self.pending_hits
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
                    completed_rays=int(self.completed_rays), pending_rays=int(self.pending_rays),
                    stderr=self.stderr(), stderr_basis="completed_replicates",
                    converged=bool(self.converged), status=status)


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


class Accumulator:
    """A raw seeded sample stream owned by exactly one Run."""
    def __init__(self):
        self._signature = None
        self._version = None
        self._states = {}
        self._names = []
        self._order = []
        self._cursor = self._steps = 0
        self._scheduled_idx = None
        self._chunk_remaining = 0
        self._chunk_is_fair = True
        self._fair_quantum = 1
        self._fair_pending = None
        self._rays = 0
        self.plan = None

    @property
    def cumulative_rays(self):
        return int(self._rays)

    @property
    def status(self):
        estimates = [est for state in self._states.values() for est in (state.matrix, state.sky) if est is not None and not est.empty]
        if not self._states:
            return "not_started"
        if all(est.converged for est in estimates):
            return "converged"
        if all(est.done for est in estimates):
            return "max_iters"
        return "sampling" if self._rays else "not_started"

    def stats(self):
        return {"status": self.status, "cumulative_rays": self.cumulative_rays,
                "emitters": {self._names[idx]: dict(rays=int(state.rays), replicates=int(state.iteration), ray_offset=int(state.offset),
                    matrix=state.matrix.stats(), sky=None if state.sky is None else state.sky.stats())
                    for idx, state in self._states.items()}}

    def bind(self, prepared, cfg, names, order, states):
        if self._signature is None:
            self._signature = cfg.signature
            self._version = prepared.version
            self._names, self._order, self._states = list(names), list(order), states
        elif self._signature != cfg.signature or self._version != prepared.version:
            raise ValueError("Scene or sampling changed; start a new run")
        for state in self._states.values():
            state.matrix.refresh(cfg)
            if state.sky is not None:
                state.sky.refresh(cfg)

def advance(prepared, cfg, acc, *, cancel=None, progress=None, allow_calibration=False):
    p, solver = cfg, prepared
    started = time.perf_counter()
    names = [m[0] for m in prepared.meshes]
    order = [names.index(name) for name in p.emitter_names]
    ray_limit, time_limit = p.max_total_rays, p.max_time_ms
    version, used = prepared.version, 0
    def stopped():
        return ((cancel is not None and cancel()) or solver.version != version
                or (ray_limit is not None and used >= ray_limit)
                or (time_limit is not None and (time.perf_counter() - started)*1000 >= time_limit))
    if not order or (stopped() and acc._signature is None):
        return
    n_surf = len(names)
    if p.strategy == "area_pair":
        # Direct area sampling has its own uniform points and CPU segment
        # visibility. It needs neither cosine tables nor a separate BLAS/TLAS.
        scene, emitters = None, [None] * n_surf
    else:
        use_bvh = _select_bvh(p.bvh, solver.total_faces)
        scene = (solver.get_instanced_scene() if solver.acceleration == "instanced"
                 else solver.get_scene(use_bvh=use_bvh))
        emitters = solver.get_emitters(samples=p.samples, rays=p.rays, flip_faces=p.flip_faces)
        centers, extents = solver.get_mesh_bounds()
    discrete = p.discrete
    states = {}
    if acc._signature is None:
        for idx in order:
            em = emitters[idx]
            active = (np.ones(n_surf, np.uint8) if em is None
                      else _build_emitter_surface_mask(idx, em, centers, extents))
            recv = np.asarray([names.index(name) for name in p.receiver_names if name != names[idx]], np.int32)
            selected = np.asarray([j+offset for offset, side in ((0,"front"),(n_surf,"back"))
                                   if p.include_matrix and side in p.receiver_sides for j in recv], np.int32)
            matrix = _Estimate(p, 2*n_surf+2, selected,
                               empty=not p.include_matrix or em is not None and em.total_area <= 0)
            sky = _Estimate(p, 145 if discrete else 1, empty=em.total_area <= 0) if p.include_sky else None
            n_once = p.pair_samples if p.strategy == "area_pair" else em.n_cells * p.rays
            states[idx] = _EmitterState(active, recv, matrix, sky, n_once)
    acc.bind(solver, p, names, order, states)
    states = acc._states
    unrestricted = allow_calibration and ray_limit is None and time_limit is None and progress is None
    previous_plan = acc.plan
    if p.strategy == "area_pair":
        from ..backends.calibration import ExecutionPlan
        plan = ExecutionPlan("cpu", p.ray_batch_size, reason="area_pair is a CPU estimator")
    else:
        plan = plan_execution(solver, p, include_matrix=True, include_sky=p.include_sky,
                              discrete=discrete, allow_calibration=unrestricted and not stopped())
    if (acc.cumulative_rays and previous_plan is not None and plan.backend != previous_plan["backend"]):
        plan = replace(plan, backend=previous_plan["backend"],
                       ray_batch_size=min(p.ray_batch_size, previous_plan["ray_batch_size"]),
                       reason="resumed run retains its original backend")
    acc.plan = solver._last_execution_plan = plan.as_dict()
    backend = plan.backend
    batch_size = min(p.ray_batch_size, plan.ray_batch_size)
    if p.strategy == "area_pair":
        from ..backends.area_pair import AreaPairWorkspace
        cache = getattr(solver, "_execution_workspace_cache", None)
        if cache is None:
            cache = solver._execution_workspace_cache = {}
        workspace = cache.setdefault("area_pair", AreaPairWorkspace())
    else:
        workspace = workspace_for(solver, backend, p)

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
        if state.matrix is not None:
            raw = np.concatenate((counts[0], counts[1]))
            requested = np.sum(raw[state.matrix.selected])
            extra = [np.sum(raw) - requested, ray_count - np.sum(raw) - np.sum(counts[2])]
            if p.strategy == "area_pair":
                extra = [0, 0]
            if p.strategy == "area_pair":
                target = int(state.receivers[0])
                state.matrix.add_chunk(np.concatenate((raw, extra)), ray_count, complete=complete,
                    weights=workspace.last_weights, sides=workspace.last_sides, columns=(target, target+n_surf))
            else:
                state.matrix.add_chunk(np.concatenate((raw, extra)), ray_count, complete=complete)
        if state.sky is not None:
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
        include_matrix = True
        include_sky = state.sky is not None
        opts = dict(samples=p.samples, rays=p.rays, surf_active=state.active,
                    emit_sid=idx, min_sid=0, include_matrix=include_matrix,
                    include_sky=include_sky, discrete=discrete, gpu_raygen=p.gpu_raygen,
                    flip_faces=p.flip_faces)
        group = 1
        # Preserve deferred GPU reductions for unrestricted fair solves. Their
        # per-row seeds and counts are independent of cross-row ordering.
        if (backend != "cpu" and unrestricted and state.offset == 0
                and getattr(p, "sampling_mode", "fair") == "fair"
                and state.n_once <= 1048576 and p.gpu_raygen):
            from .sampling import _convergence_checkpoint
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
            results = workspace.trace_many(solver, idx, scene, em, n_surf, specs, **opts)
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
        if p.strategy == "area_pair":
            result = workspace.trace(prepared=solver, source_index=idx, target_index=state.receivers[0], cfg=p,
                                     iteration=state.iteration, ray_count=size, ray_offset=state.offset)
        else:
            result = workspace.trace_many(solver, idx, scene, em, n_surf,
                       [(cp_grid, cp_dims, size, state.offset)], **opts)[0]
        acc._chunk_remaining -= size
        accept(idx, result, size)
        finish_chunk(idx)
