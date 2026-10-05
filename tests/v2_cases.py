"""Test-only workload adapters used to preserve numerical regression cases.

These records translate historical test settings into the v2 public API. They
do not call v1 calculation entry points and are not shipped with the package.
Lifecycle, validation and typed-result tests exercise v2 directly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

import numpy as np

from raystrack import (Mesh, Scene, Solver, Query, SolveOptions, Sampling,
                       Accuracy, Postprocessing, Budget, Channel)
from raystrack.utils.prepared import PreparedSolver


def _log(message):
    pass


@dataclass
class MatrixCase:
    samples: int = 16
    rays: int = 128
    seed: int = 1
    bvh: str = "auto"
    device: str = "auto"
    cuda_async: bool = True
    gpu_raygen: bool = True
    max_iters: int = 100
    tol: float = 1e-4
    tol_mode: str = "stderr"
    min_iters: int = 5
    convergence_interval: int = 1
    reciprocity: bool = True
    enforce_reciprocity_rowsum: bool = False
    flip_faces: bool = False
    min_total_rays: int = 0
    reciprocity_mode: str = "shortcut"
    emitter_names: list | None = None
    max_total_rays: int | None = None
    max_time_ms: float | None = None
    ray_batch_size: int = 65536
    sampling_mode: str = "fair"
    auto_tune: bool = True
    tune_budget_ms: float = 250

    def as_dict(self):
        return asdict(self)


@dataclass
class SkyCase:
    samples: int = 16
    rays: int = 128
    seed: int = 1
    bvh: str = "auto"
    device: str = "auto"
    cuda_async: bool = True
    gpu_raygen: bool = True
    max_iters: int = 100
    tol: float = 1e-4
    tol_mode: str = "stderr"
    min_iters: int = 5
    convergence_interval: int = 1
    discrete: bool = False
    min_total_rays: int = 0
    emitter_names: list | None = None
    max_total_rays: int | None = None
    max_time_ms: float | None = None
    ray_batch_size: int = 65536
    sampling_mode: str = "fair"
    auto_tune: bool = True
    tune_budget_ms: float = 250

    def as_dict(self):
        return asdict(self)


def options_for(case, *, matrix=True):
    post = "none"
    if matrix and getattr(case, "reciprocity", False):
        post = "bidirectional"
    if matrix and getattr(case, "enforce_reciprocity_rowsum", False):
        post = "rowsum"
    return SolveOptions(
        sampling=Sampling(density=case.samples, rays_per_cell=case.rays,
                          seed=case.seed, mode=case.sampling_mode,
                          flip_faces=bool(getattr(case, "flip_faces", False))),
        accuracy=Accuracy(max_replicates=case.max_iters, tolerance=case.tol,
                          mode=case.tol_mode, min_replicates=case.min_iters,
                          check_interval=case.convergence_interval, min_rays=case.min_total_rays),
        postprocessing=Postprocessing(reciprocity=post), batch_size=case.ray_batch_size)


def scene_for(meshes):
    names = [name for name, _, _ in meshes]
    if len(set(names)) != len(names):
        raise ValueError("surface IDs must be unique")
    return Scene.from_meshes({name: Mesh(vertices, faces) for name, vertices, faces in meshes})


def context_for(meshes, case, prepared=None):
    if prepared is None:
        current = scene_for(meshes)
        cache = {}
    else:
        prepared.validate_meshes(meshes)
        current = getattr(prepared, "_case_scene", None)
        if current is None:
            current = prepared._case_scene = scene_for(meshes)
            prepared._case_version = prepared.version
            prepared._case_world = [(name, v.copy(), f.copy()) for name, v, f in meshes]
            prepared._case_solvers = {}
        elif prepared._case_version != prepared.version:
            for (name, v, f), (_, old_v, old_f) in zip(meshes, prepared._case_world):
                if not np.array_equal(v, old_v) or not np.array_equal(f, old_f):
                    current.update_mesh(name, Mesh(v, f))
            prepared._case_version = prepared.version
            prepared._case_world = [(name, v.copy(), f.copy()) for name, v, f in meshes]
        cache = prepared._case_solvers
    settings = (case.device or "auto", case.bvh, case.auto_tune,
                case.cuda_async, case.gpu_raygen, case.tune_budget_ms)
    if settings not in cache:
        cache[settings] = Solver(current, device=settings[0], bvh=case.bvh,
             acceleration=getattr(prepared, "acceleration", "flat"), auto_tune=case.auto_tune,
             cuda_async=case.cuda_async, gpu_raygen=case.gpu_raygen,
             tune_budget_ms=case.tune_budget_ms)
    return current, cache[settings]


def split_rows(result):
    matrix, sky, rest = {}, {}, {}
    for index, sender in enumerate(result.sender_ids):
        if result.coverage[index] != 1:
            continue
        matrix[sender], sky[sender], rest[sender] = {}, {}, {}
        for channel, value in result.row(sender).items():
            if channel.kind == "surface" and value > 0:
                matrix[sender][f"{channel.surface_id}_{channel.side}"] = float(value)
            elif channel.kind == "sky":
                key = "Sky" if channel.patch is None else f"Sky_Patch_{channel.patch + 1}"
                sky[sender][key] = float(value)
            elif channel.kind == "rest":
                rest[sender]["Rest"] = float(value)
    return matrix, sky, rest


class CaseAccumulator:
    def __init__(self):
        self.run = None
        self.query = None
        self.solver = None

    @property
    def cumulative_rays(self):
        return 0 if self.run is None or self.run.result is None else self.run.result.cumulative_rays

    @property
    def status(self):
        return "empty" if self.run is None else self.run.status

    def stats(self):
        return {} if self.run is None or self.run.result is None else self.run.result.statistics


def execute_case(meshes, case, query, *, prepared=None, progress=None, cancel=None, accumulator=None):
    _, solver = context_for(meshes, case, prepared)
    options = options_for(case, matrix=isinstance(case, MatrixCase))
    if accumulator is None and cancel is None:
        budget = (None if case.max_total_rays is None and case.max_time_ms is None
                  else Budget(rays=case.max_total_rays, time_ms=case.max_time_ms))
        result = solver.solve(query, options=options, budget=budget, progress=progress)
        if prepared is not None:
            prepared._case_result = result
            prepared._last_execution_plan = dict(result.execution)
        return result
    if accumulator is None:
        run = solver.start(query, options=options)
    elif accumulator.run is None:
        run = accumulator.run = solver.start(query, options=options)
        accumulator.query, accumulator.solver = query, solver
    else:
        if accumulator.query != query or accumulator.solver is not solver:
            raise ValueError("Query changed; start a new run")
        run = accumulator.run
    if cancel is not None and cancel():
        run.cancel()
    def on_progress(count):
        if progress is not None:
            progress(count)
        if cancel is not None and cancel():
            run.cancel()
    result = run.advance(Budget(rays=case.max_total_rays, time_ms=case.max_time_ms),
                         options=options, progress=on_progress)
    if prepared is not None:
        prepared._case_result = result
        prepared._last_execution_plan = dict(result.execution)
    return result


def matrix_case(meshes, params, *, prepared=None, progress=None, cancel=None, accumulator=None):
    result = execute_case(meshes, params, Query.matrix(senders=params.emitter_names),
                          prepared=prepared, progress=progress, cancel=cancel, accumulator=accumulator)
    return split_rows(result)[0]


def sky_case(meshes, params, *, prepared=None, progress=None, cancel=None, accumulator=None):
    result = execute_case(meshes, params, Query.sky(senders=params.emitter_names, discrete=params.discrete),
                          prepared=prepared, progress=progress, cancel=cancel, accumulator=accumulator)
    return split_rows(result)[1]


def pair_case(meshes, sender, receiver, *, samples=8192, seed=1, sequence="shifted_halton", prepared=None):
    case = MatrixCase(device="cpu", reciprocity=False)
    _, solver = context_for(meshes, case, prepared)
    options = SolveOptions(sampling=Sampling(strategy="area_pair", pair_samples=samples,
                                             seed=seed, sequence=sequence),
                           accuracy=Accuracy(min_replicates=1, max_replicates=1))
    result = solver.solve(Query.pair(sender, receiver), options=options)
    return split_rows(result)[0].get(sender, {})


def outside_case(meshes, *, matrix_params, sky_params, prepared=None, cancel=None,
                 progress=None, accumulator=None, **unused):
    query = Query.matrix(senders=matrix_params.emitter_names,
                         sky="tregenza145" if sky_params.discrete else "merged")
    result = execute_case(meshes, matrix_params, query, prepared=prepared,
                          cancel=cancel, progress=progress, accumulator=accumulator)
    return split_rows(result)


class CaseResult:
    def __init__(self, result, revision):
        self.raw = result
        self.scene_version = revision
        self.scene, self.sky, self.rest = split_rows(result)
        self.statistics = result.statistics
        self.rays_used, self.cumulative_rays = result.rays_used, result.cumulative_rays
        self.status, self.converged = result.status, result.converged
        self.cancelled = result.status in ("cancelled", "invalidated")
        self.completed = not self.cancelled

    def as_dict(self):
        def thaw(value):
            if hasattr(value, "items"):
                return {key: thaw(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [thaw(item) for item in value]
            return value
        return {"scene": deepcopy(self.scene), "sky": deepcopy(self.sky), "rest": deepcopy(self.rest),
                "statistics": thaw(self.statistics)}


class CaseSession:
    """Scenario convenience over a v2 Solver/Run, used only by regression tests."""
    def __init__(self, prepared, matrix_params=None, sky_params=None):
        self.prepared = prepared
        self.mp = deepcopy(matrix_params)
        self.sp = deepcopy(sky_params)
        self.case = self.mp or self.sp or MatrixCase(device="cpu", reciprocity=False)
        self.scene, self.solver = context_for(prepared.meshes, self.case, prepared)
        self.accumulator = None
        self.latest = None
        self._runs = []

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        for run in self._runs:
            run.close()

    def _settings(self, overrides):
        replacement = (overrides.get("matrix_params") or
                       (overrides.get("sky_params") if self.mp is None else None))
        self.mp = overrides.pop("matrix_params", self.mp)
        self.sp = overrides.pop("sky_params", self.sp)
        case = replacement or self.case
        valid = set(case.__dataclass_fields__)
        self.case = replace(case, **{key: value for key, value in overrides.items() if key in valid})
        return self.case

    def _query(self):
        if self.mp is None and self.sp is not None:
            return Query.sky(senders=self.case.emitter_names, discrete=self.sp.discrete)
        return Query.matrix(senders=self.case.emitter_names,
                            sky=None if self.sp is None else "tregenza145" if self.sp.discrete else "merged")

    def preview(self, **overrides):
        case = self._settings(overrides)
        self.accumulator = CaseAccumulator()
        raw = execute_case(self.prepared.meshes, case, self._query(), prepared=self.prepared,
                            accumulator=self.accumulator, progress=overrides.get("progress"),
                            cancel=overrides.get("cancel"))
        self._runs.append(self.accumulator.run)
        self.latest = CaseResult(raw, self.scene.revision)
        return self.latest

    solve = preview

    def refine(self, level=1, **overrides):
        if self.accumulator is None:
            raise RuntimeError("start a new run before refining")
        settings = self.mp, self.sp, self.case
        previous = self.case
        case = self._settings(overrides)
        if "max_total_rays" not in overrides:
            case = replace(case, max_total_rays=self.accumulator.cumulative_rays * (2**level - 1))
        if "max_iters" not in overrides:
            case = replace(case, max_iters=max(case.max_iters, previous.max_iters * 2**level))
        self.case = case
        try:
            raw = execute_case(self.prepared.meshes, case, self._query(), prepared=self.prepared,
                                accumulator=self.accumulator, progress=overrides.get("progress"),
                                cancel=overrides.get("cancel"))
        except Exception:
            self.mp, self.sp, self.case = settings
            raise
        self.latest = CaseResult(raw, self.scene.revision)
        return self.latest

    def update_transform(self, *args):
        self.prepared.update_transform(*args)
        context_for(self.prepared.meshes, self.case, self.prepared)
        self.latest = None

    def warmup(self, **kwargs):
        return self.solver.warmup(self._query(), options=options_for(self.case), **kwargs)
