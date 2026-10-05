"""Bounded, cancellable solves for interactive and moving scenes.

Use a session's update methods when a worker is tracing the scene. They cancel
the old frame before waiting for the prepared solver's lock. A time budget is
soft: preparation and an already running ray chunk finish before cancellation
or a deadline can be checked.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
import threading
import time
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Tuple

from .api import view_factor_outside_workflow
from .main import view_factor_matrix
from .params import MatrixParams, SkyParams
from .utils.prepared import PreparedSolver


_UNSET = object()
Factors = Mapping[str, Mapping[str, float]]


def _freeze_factors(values: Factors) -> Factors:
    return MappingProxyType({
        name: MappingProxyType({key: float(value) for key, value in row.items()})
        for name, row in values.items()
    })


def _freeze_metadata(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _plain_metadata(value):
    if isinstance(value, Mapping):
        return {key: _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_metadata(item) for item in value]
    return value


def _new_accumulator():
    from .execution import SolveAccumulator
    return SolveAccumulator()


def _sampling_signature(mp, sp):
    """Settings that cannot change while retaining sampled estimates."""
    controls = {
        "max_iters", "min_iters", "tol", "tol_mode", "min_total_rays",
        "convergence_interval", "max_total_rays", "max_time_ms",
        "ray_batch_size", "auto_tune", "tune_budget_ms",
    }
    return tuple(None if params is None else {
        key: value for key, value in params.as_dict().items() if key not in controls
    } for params in (mp, sp))


@dataclass(frozen=True)
class PreviewResult:
    """An immutable result from one scene version.

    ``completed`` means the solve returned a usable result for the requested
    scene. A budget-limited preview is completed; this flag does not certify
    Monte Carlo convergence. Superseded or cancelled solves return empty
    mappings, so results from different scene positions cannot be combined by
    accident. ``rays_used`` counts new rays traced during this call, including
    cancelled work. ``cumulative_rays`` counts retained rays used by the result;
    it is zero for cancelled work. ``statistics`` reports per-emitter completed
    replicates, partial offsets and standard errors when enough full replicates
    exist. ``status`` describes the estimator state; only ``converged`` confirms
    the configured statistical convergence checks were satisfied.
    """

    scene_version: int
    scene: Factors
    sky: Factors
    rest: Factors
    completed: bool
    cancelled: bool
    rays_used: int
    elapsed_ms: float
    cumulative_rays: int = 0
    status: str = "completed"
    converged: bool = False
    statistics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field in ("scene", "sky", "rest"):
            object.__setattr__(self, field, _freeze_factors(getattr(self, field)))
        object.__setattr__(self, "statistics", _freeze_metadata(self.statistics))

    def as_dict(self) -> Dict[str, object]:
        """Return independent ordinary dictionaries, suitable for JSON export."""
        return {
            "scene_version": self.scene_version,
            "scene": {name: dict(row) for name, row in self.scene.items()},
            "sky": {name: dict(row) for name, row in self.sky.items()},
            "rest": {name: dict(row) for name, row in self.rest.items()},
            "completed": self.completed,
            "cancelled": self.cancelled,
            "rays_used": self.rays_used,
            "elapsed_ms": self.elapsed_ms,
            "cumulative_rays": self.cumulative_rays,
            "status": self.status,
            "converged": self.converged,
            "statistics": _plain_metadata(self.statistics),
        }


def _copy_params(params):
    names = params.emitter_names
    if isinstance(names, str):
        raise TypeError("emitter_names must be a sequence of names, not a string")
    return replace(params, emitter_names=None if names is None else list(names))


class PreviewSession:
    """Reuse prepared geometry for previews and stationary refinement.

    ``preview()`` defaults to at most 65,536 rays and a soft 50 ms deadline.
    ``solve()`` uses the configured solve parameters without those defaults.
    Both accept emitter selection and ray/time-budget overrides. The seed is
    preserved across frames to reduce sampling-induced visual noise.

    ``submit()`` schedules a preview on a single worker. A newer request cancels
    queued futures and asks the running request to stop at its next ray chunk.
    A queued future's ``result()`` can therefore raise ``CancelledError``; a
    running superseded request returns a cancelled ``PreviewResult``.

    ``refine()`` resumes the latest successful result, tracing only additional
    rays. Completed and partial replicates retain their original seed and ray
    offset. Refinement requires unchanged geometry, sampling configuration and
    emitter selection; it cannot blend estimates across scene versions. Fresh
    ``preview()`` and ``solve()`` calls restart accumulation. Use this session's
    ``update_*`` methods for responsive cancellation during motion. Direct
    prepared-solver updates are serialized by its lock, but wait for the active
    frame to end.
    """

    def __init__(
        self,
        prepared: PreparedSolver,
        matrix_params: Optional[MatrixParams] = None,
        sky_params: Optional[SkyParams] = None,
    ):
        if not isinstance(prepared, PreparedSolver):
            raise TypeError("prepared must be a PreparedSolver instance")
        if matrix_params is not None and not isinstance(matrix_params, MatrixParams):
            raise TypeError("matrix_params must be a MatrixParams instance")
        if sky_params is not None and not isinstance(sky_params, SkyParams):
            raise TypeError("sky_params must be a SkyParams instance or None")
        self.prepared = prepared
        self.matrix_params = _copy_params(matrix_params or MatrixParams())
        self.sky_params = None if sky_params is None else _copy_params(sky_params)
        self._state_lock = threading.RLock()
        self._generation = 0
        self._closed = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures = set()
        self._latest: Optional[PreviewResult] = None
        self._refinement: Optional[Tuple[int, MatrixParams, Optional[SkyParams]]] = None
        self._accumulator = None

    @property
    def latest(self) -> Optional[PreviewResult]:
        """Latest completed result, or None if the scene has since changed."""
        with self._state_lock:
            result = self._latest
            return result if result is not None and result.scene_version == self.prepared.version else None

    def _invalidate_locked(self) -> None:
        self._generation += 1
        for future in tuple(self._futures):
            future.cancel()

    def _new_request(self, *, fresh=True) -> Tuple[int, int]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("PreviewSession is closed")
            self._invalidate_locked()
            if fresh:
                self._accumulator = None
                self._refinement = None
            return self._generation, self.prepared.version

    def _parameters(
        self,
        *,
        matrix_params=None,
        sky_params=_UNSET,
        emitter_names=_UNSET,
        max_total_rays=_UNSET,
        max_time_ms=_UNSET,
        ray_batch_size=_UNSET,
        preview=False,
    ) -> Tuple[MatrixParams, Optional[SkyParams]]:
        mp = self.matrix_params if matrix_params is None else matrix_params
        sp = self.sky_params if sky_params is _UNSET else sky_params
        if not isinstance(mp, MatrixParams):
            raise TypeError("matrix_params must be a MatrixParams instance")
        if sp is not None and not isinstance(sp, SkyParams):
            raise TypeError("sky_params must be a SkyParams instance or None")
        overrides = {}
        if emitter_names is not _UNSET:
            if isinstance(emitter_names, str):
                raise TypeError("emitter_names must be a sequence of names, not a string")
            overrides["emitter_names"] = None if emitter_names is None else list(emitter_names)
        if max_total_rays is not _UNSET:
            overrides["max_total_rays"] = max_total_rays
        if max_time_ms is not _UNSET:
            overrides["max_time_ms"] = max_time_ms
        if ray_batch_size is not _UNSET:
            overrides["ray_batch_size"] = ray_batch_size
        result = []
        for params in (mp, sp):
            if params is None:
                result.append(None)
                continue
            values = dict(overrides)
            if preview:
                if max_total_rays is _UNSET:
                    values["max_total_rays"] = min(params.max_total_rays, 65536) if params.max_total_rays is not None else 65536
                if max_time_ms is _UNSET:
                    values["max_time_ms"] = min(params.max_time_ms, 50.0) if params.max_time_ms is not None else 50.0
                if ray_batch_size is _UNSET:
                    values["ray_batch_size"] = min(params.ray_batch_size, 8192)
            result.append(replace(_copy_params(params), **values))
        return result[0], result[1]

    def _run(
        self,
        request: Tuple[int, int],
        mp: MatrixParams,
        sp: Optional[SkyParams],
        cancel: Optional[Callable[[], bool]],
        progress: Optional[Callable[[int], None]],
        accumulator=None,
        previous_rays: int = 0,
    ) -> PreviewResult:
        generation, version = request
        started = time.perf_counter()
        rays_used = 0
        cancellation_seen = False

        def should_cancel() -> bool:
            nonlocal cancellation_seen
            with self._state_lock:
                stale = self._closed or self._generation != generation or self.prepared.version != version
            if stale or (cancel is not None and bool(cancel())):
                cancellation_seen = True
            return cancellation_seen

        def report(rays: int) -> None:
            nonlocal rays_used
            rays_used += int(rays)
            if progress is not None:
                progress(int(rays))

        scene, sky, rest = {}, {}, {}
        with self.prepared.solve_lock:
            if not should_cancel():
                if accumulator is None:
                    accumulator = _new_accumulator()
                if sp is None:
                    scene = view_factor_matrix(
                        self.prepared.meshes, mp, prepared=self.prepared,
                        cancel=should_cancel, progress=report, accumulator=accumulator,
                    )
                else:
                    scene, sky, rest = view_factor_outside_workflow(
                        self.prepared.meshes, matrix_params=mp, sky_params=sp,
                        prepared=self.prepared, cancel=should_cancel, progress=report,
                        accumulator=accumulator,
                    )
            was_cancelled = should_cancel()
            with self._state_lock:
                # Publication and generation checks share a lock with submit/update.
                was_cancelled = was_cancelled or self._generation != generation or self._closed
                statistics = {} if was_cancelled or accumulator is None else accumulator.stats()
                status = "cancelled" if was_cancelled else statistics.get("status", "completed")
                result = PreviewResult(
                    scene_version=version,
                    scene={} if was_cancelled else scene,
                    sky={} if was_cancelled else sky,
                    rest={} if was_cancelled else rest,
                    completed=not was_cancelled,
                    cancelled=was_cancelled,
                    rays_used=rays_used,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    cumulative_rays=0 if was_cancelled else previous_rays + rays_used,
                    status=status,
                    converged=status == "converged",
                    statistics=statistics,
                )
                if not was_cancelled:
                    self._latest = result
                    self._refinement = (version, _copy_params(mp), None if sp is None else _copy_params(sp))
                    self._accumulator = accumulator
                return result

    def solve(self, *, cancel=None, progress=None, **kwargs) -> PreviewResult:
        """Solve the current scene; optional overrides use dataclass replacement."""
        mp, sp = self._parameters(**kwargs)
        return self._run(self._new_request(), mp, sp, cancel, progress)

    def preview(self, *, cancel=None, progress=None, **kwargs) -> PreviewResult:
        """Solve with interactive defaults or explicit ray/time limits."""
        mp, sp = self._parameters(preview=True, **kwargs)
        return self._run(self._new_request(), mp, sp, cancel, progress)

    def submit(self, *, cancel=None, progress=None, **kwargs) -> Future:
        """Submit a preview, superseding queued and running older requests."""
        mp, sp = self._parameters(preview=True, **kwargs)
        with self._state_lock:
            request = self._new_request()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raystrack-preview")
            future = self._executor.submit(self._run, request, mp, sp, cancel, progress)
            self._futures.add(future)
            future.add_done_callback(self._finished)
            return future

    def _finished(self, future: Future) -> None:
        with self._state_lock:
            self._futures.discard(future)

    def refine(self, *, level: int = 1, factor: float = 2.0, cancel=None, progress=None, **kwargs) -> PreviewResult:
        """Continue the latest stationary frame with additional rays.

        By default, ``factor ** level`` scales the cumulative ray target and
        iteration cap. Only missing rays are traced. An explicit
        ``max_total_rays`` is an additional per-call budget, matching ``solve``
        parameter semantics. Statistical convergence can still stop early.
        The time deadline is cleared unless ``max_time_ms`` is supplied.

        Tolerances, iteration caps and ray chunk sizes can change. Changing
        sampling, output settings or selected emitters raises ``ValueError``;
        run a fresh preview to restart with those settings.
        """
        if isinstance(level, bool) or not isinstance(level, int) or level < 1:
            raise ValueError("level must be a positive integer")
        if not math.isfinite(factor) or factor <= 1.0:
            raise ValueError("factor must be finite and greater than one")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("PreviewSession is closed")
            if self._refinement is None:
                raise RuntimeError("run a preview or solve before refining")
            version, mp, sp = self._refinement
            if self.prepared.version != version:
                raise RuntimeError("scene changed; run a new preview before refining")
            if self._accumulator is None:
                raise RuntimeError("no resumable state; wait for active work or run a new preview")
            try:
                multiplier = factor ** level
            except OverflowError:
                raise ValueError("refinement multiplier is too large") from None
            if not math.isfinite(multiplier):
                raise ValueError("refinement multiplier is too large")
            previous_rays = self._latest.cumulative_rays
            budgets = [base.max_total_rays for base in (mp, sp)
                       if base is not None and base.max_total_rays is not None]
            base_target = previous_rays or (min(budgets) if budgets else mp.ray_batch_size)
            additional = max(1, int(math.ceil(base_target * multiplier)) - previous_rays)
            params = []
            for base in (mp, sp):
                params.append(None if base is None else replace(
                    base,
                    max_iters=max(base.max_iters + 1, int(math.ceil(base.max_iters * multiplier))),
                    max_total_rays=additional,
                    max_time_ms=None,
                ))
            settings = {"matrix_params": params[0], "sky_params": params[1]}
            settings.update(kwargs)
            revised_mp, revised_sp = self._parameters(**settings)
            if _sampling_signature(mp, sp) != _sampling_signature(revised_mp, revised_sp):
                raise ValueError("sampling, output settings or emitter selection changed; run a fresh preview")
            from .execution import validate_controls
            for params in (revised_mp, revised_sp):
                if params is not None:
                    validate_controls(params)
            mp, sp = revised_mp, revised_sp
            accumulator = self._accumulator
            self._accumulator = None
            request = self._new_request(fresh=False)
            # An update can apply between preparation and lock acquisition;
            # keeping the original version makes that request cancel safely.
            request = (request[0], version)
        return self._run(request, mp, sp, cancel, progress, accumulator, previous_rays)

    def warmup(self, *, matrix_params=None, sky_params=_UNSET,
               repeats: int = 3, max_probe_rays: int = 16384,
               target_batch_ms: float = 10.0):
        """Calibrate execution under the solver lock without changing estimates.

        Probe rays are independent diagnostics and are excluded from solve
        result ray counts. The returned execution plan reports measured choices.
        """
        mp, sp = self._parameters(matrix_params=matrix_params, sky_params=sky_params)
        from .tuning import warmup
        with self.prepared.solve_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("PreviewSession is closed")
            return warmup(self.prepared, mp, sp, repeats=repeats,
                          max_probe_rays=max_probe_rays, target_batch_ms=target_batch_ms)

    def cancel(self) -> None:
        """Ask active work to stop at the next chunk and cancel queued requests."""
        with self._state_lock:
            self._invalidate_locked()
            self._accumulator = None

    def _update(self, method: str, *args, **kwargs):
        with self._state_lock:
            if self._closed:
                raise RuntimeError("PreviewSession is closed")
            self._invalidate_locked()
            self._accumulator = None
        with self.prepared.solve_lock:
            value = getattr(self.prepared, method)(*args, **kwargs)
            with self._state_lock:
                self._latest = None
            return value

    def update_transform(self, *args, **kwargs):
        """Cancel stale work and delegate to PreparedSolver.update_transform."""
        return self._update("update_transform", *args, **kwargs)

    def update_vertices(self, *args, **kwargs):
        """Cancel stale work and delegate to PreparedSolver.update_vertices."""
        return self._update("update_vertices", *args, **kwargs)

    def update_mesh(self, *args, **kwargs):
        """Cancel stale work and delegate to PreparedSolver.update_mesh."""
        return self._update("update_mesh", *args, **kwargs)

    def rebuild_bvh(self) -> int:
        """Cancel stale work and repartition the BVH after substantial motion."""
        return self._update("rebuild_bvh")

    def close(self, *, wait: bool = True) -> None:
        """Cancel pending work and release the optional preview worker."""
        with self._state_lock:
            self._closed = True
            self._invalidate_locked()
            self._accumulator = None
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "PreviewSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["PreviewSession", "PreviewResult"]
