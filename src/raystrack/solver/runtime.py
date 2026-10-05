"""Reusable solver resources and cancellable, resumable runs."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from threading import RLock, Event, get_ident
import math
import time
import weakref
import numpy as np
from ..model import Scene
from ..utils.prepared import PreparedSolver
from ..backends import CapabilityError
from ..backends.calibration import warmup
from ..engine.controls import controls
from ..engine.scheduler import Accumulator, advance
from ..engine.sampling import _select_bvh
from .options import SolveOptions, Budget
from .query import Query
from .snapshot import snapshot


class Solver:
    """Own reusable acceleration and backend resources for one mutable Scene.

    Use as a context manager to close resources. All query shapes delegate to
    the same resumable Run machinery; selection does not remove occluders.
    """

    def __init__(self, scene, *, device="auto", acceleration="flat", bvh="auto",
                 auto_tune=True, cuda_async=True, gpu_raygen=True, tune_budget_ms=250.):
        """Bind a Scene and validate backend, acceleration and tuning settings."""
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        if acceleration not in ("flat", "instanced"):
            raise ValueError("acceleration must be flat or instanced")
        device = (device or "auto").lower()
        if device not in ("auto", "cpu", "gpu", "cuda", "taichi", "vulkan", "metal"):
            raise ValueError("Unknown device")
        _select_bvh(bvh, 0)
        for name, value in (("auto_tune", auto_tune), ("cuda_async", cuda_async), ("gpu_raygen", gpu_raygen)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        if not math.isfinite(tune_budget_ms) or tune_budget_ms < 0:
            raise ValueError("tune_budget_ms must be finite and nonnegative")
        self._scene, self._device, self._acceleration, self._bvh = scene, device, acceleration, bvh
        self._auto_tune, self._cuda_async, self._gpu_raygen = auto_tune, cuda_async, gpu_raygen
        self._tune_budget_ms = tune_budget_ms
        self._lock = RLock()
        self._submit_lock = RLock()
        self._runs_lock = RLock()
        self._worker_ident = None
        self._prepared, self._seen, self._revision = None, (), -1
        self._runs = weakref.WeakSet()
        self._executor = None
        self._closed = False
        scene._subscribe(self._geometry_changed)

    def __repr__(self):
        """Describe bound geometry and execution settings without its buffers."""
        return (f"Solver(surfaces={len(self.scene)}, revision={self.scene.revision}, "
                f"device={self.device!r}, acceleration={self.acceleration!r}, "
                f"bvh={self.bvh!r}, closed={self._closed})")

    @property
    def scene(self):
        """Return the scene whose revisions invalidate incompatible runs."""
        return self._scene

    @property
    def device(self):
        """Return the requested backend policy, such as CPU, CUDA or auto."""
        return self._device

    @property
    def acceleration(self):
        """Return the flat or instanced acceleration strategy."""
        return self._acceleration

    @property
    def bvh(self):
        """Return the bounding-volume acceleration selection policy."""
        return self._bvh

    @property
    def auto_tune(self):
        """Return whether unrestricted solves may calibrate backend selection."""
        return self._auto_tune

    @property
    def cuda_async(self):
        """Return whether CUDA tracing may use asynchronous execution buffers."""
        return self._cuda_async

    @property
    def gpu_raygen(self):
        """Return whether supported GPU backends generate rays on the device."""
        return self._gpu_raygen

    @property
    def tune_budget_ms(self):
        """Return the soft calibration time allowance in milliseconds."""
        return self._tune_budget_ms

    def _geometry_changed(self):
        with self._runs_lock:
            runs = tuple(self._runs)
        for run in runs:
            run._invalidated.set()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Solver is closed")

    def _sync_scene(self):
        surfaces = self.scene.surfaces
        ids = tuple(s.surface_id for s in surfaces)
        if self._prepared is None or ids != tuple(s.surface_id for s in self._seen):
            local = [(s.surface_id, s.mesh.vertices, s.mesh.faces) for s in surfaces]
            if self.acceleration == "instanced":
                self._prepared = PreparedSolver(local, acceleration="instanced", transforms=[s.transform for s in surfaces])
            else:
                self._prepared = PreparedSolver(local)
                for s in surfaces:
                    if not np.array_equal(s.transform, np.eye(4)):
                        self._prepared.update_transform(s.surface_id, s.transform)
        elif self._revision != self.scene.revision:
            for old, current in zip(self._seen, surfaces):
                if old.mesh is not current.mesh:
                    self._prepared.update_mesh(current.surface_id, current.mesh.vertices, current.mesh.faces)
                    if not np.array_equal(current.transform, np.eye(4)):
                        self._prepared.update_transform(current.surface_id, current.transform)
                elif not np.array_equal(old.transform, current.transform):
                    self._prepared.update_transform(current.surface_id, current.transform)
        self._seen, self._revision = surfaces, self.scene.revision
        return self._prepared

    def _validate(self, query, options):
        if not isinstance(query, Query) or not isinstance(options, SolveOptions):
            raise TypeError("Expected Query and SolveOptions")
        senders, receivers = query.resolve(self.scene.surface_ids)
        if options.sampling.strategy == "area_pair":
            if self.device not in ("auto", "cpu"):
                raise CapabilityError("area_pair supports CPU only")
            if not query.scene or query.sky_mode is not None or len(senders) != 1 or len(receivers) != 1 or senders[0] == receivers[0]:
                raise CapabilityError("area_pair requires one distinct sender/receiver pair and no sky")
            if options.postprocessing.reciprocity != "none":
                raise CapabilityError("area_pair does not support reciprocity postprocessing")
        mode = options.postprocessing.reciprocity
        if mode != "none":
            if mode == "rowsum" and query.sky_mode is not None:
                raise ValueError("Row-sum reciprocity requires a closed scene query without sky")
            if not query.scene or set(senders) != set(self.scene.surface_ids) or set(receivers) != set(senders):
                raise ValueError("Reciprocity requires all sender and receiver rows")
            if "front" not in query.receiver_sides or (mode == "rowsum" and set(query.receiver_sides) != {"front", "back"}):
                raise ValueError("Reciprocity requires complete receiver sides")

    def start(self, query, options=None):
        """Create a validated Run without tracing or initializing a GPU."""
        self._check_open()
        options = SolveOptions() if options is None else options
        with self.scene.lock, self._runs_lock:
            self._check_open()
            self._validate(query, options)
            run = Run(self, query, options)
            self._runs.add(run)
            return run

    def solve(self, query, options=None, budget=None, *, progress=None):
        """Start a Run and advance it to completion or the supplied budget.

        The optional callback receives additional rays after each traced chunk.
        A bounded solve returns a partial Result that may remain unconverged.
        """
        run = self.start(query, options)
        return run._advance(Budget() if budget is None else budget, progress=progress,
                            allow_calibration=budget is None and progress is None)

    def warmup(self, query=None, options=None, **kwargs):
        """Explicitly prepare and calibrate cosine tracing for this scene."""
        self._check_open()
        query = Query.matrix() if query is None else query
        options = SolveOptions() if options is None else options
        with self._lock, self.scene.lock:
            self._validate(query, options)
            if options.sampling.strategy == "area_pair":
                raise CapabilityError("Calibration supports cosine tracing; area_pair is CPU only")
            cfg = controls(self, query, options, Budget())
            return warmup(self._sync_scene(), cfg, include_matrix=True,
                          include_sky=cfg.include_sky, discrete=cfg.discrete, **kwargs)

    def close(self):
        """Cancel registered runs and close owned execution resources once."""
        # Set events before waiting for an executing chunk/solver lock.
        with self._runs_lock:
            for run in tuple(self._runs):
                run.cancel()
        with self._lock, self._submit_lock:
            if self._closed:
                return
            with self._runs_lock:
                self._closed = True
                for run in tuple(self._runs):
                    run.cancel()
            self.scene._unsubscribe(self._geometry_changed)
            executor, self._executor = self._executor, None
            self._prepared = None
        if executor is not None:
            executor.shutdown(wait=get_ident() != self._worker_ident, cancel_futures=True)

    def _execute(self, run, budget, options, progress):
        self._worker_ident = get_ident()
        return run.advance(budget, options=options, progress=progress)

    def __enter__(self):
        """Enter an open solver context with deterministic resource cleanup."""
        self._check_open()
        return self

    def __exit__(self, *args):
        """Close the solver when its context ends, including on an exception."""
        self.close()


class Run:
    """Own a cancellable, resumable seeded sample prefix for one query.

    Each advance budget means additional work. Geometry revisions invalidate
    the run; changing sampling requires a new run, while accuracy can refine it.
    """

    def __init__(self, solver, query, options):
        """Capture the validated query and its current scene revision."""
        self._solver, self._query, self._options = solver, query, options
        self.scene_revision = solver.scene.revision
        self._surface_ids = solver.scene.surface_ids
        self._sender_ids, self._receiver_ids = query.resolve(self._surface_ids)
        self._accumulator = Accumulator()
        self._cancelled, self._invalidated = Event(), Event()
        self._lock = RLock()
        self._closed = False
        self._result = None

    def __repr__(self):
        """Describe query and cumulative progress without accumulation arrays."""
        return (f"Run(query={self.query!r}, scene_revision={self.scene_revision}, "
                f"cumulative_rays={self.cumulative_rays}, status={self.status!r}, "
                f"closed={self._closed})")

    @property
    def result(self):
        """Return the latest immutable snapshot, or ``None`` before advancing."""
        return self._result

    @property
    def solver(self):
        """Return the reusable Solver that owns this run's execution resources."""
        return self._solver

    @property
    def query(self):
        """Return the immutable output selection captured at run creation."""
        return self._query

    @property
    def options(self):
        """Return the current sampling, accuracy and postprocessing settings."""
        return self._options

    @property
    def cumulative_rays(self):
        """Return all traced rays retained in this run's exact sample prefix."""
        return self._accumulator.cumulative_rays

    @property
    def status(self):
        """Return sampling, completion, cancellation or invalidation state."""
        return "invalidated" if self._invalidated.is_set() else "cancelled" if self._cancelled.is_set() else self._accumulator.status

    def advance(self, budget=None, *, options=None, progress=None):
        """Trace additional work and return a detached immutable Result.

        ``options`` may adjust accuracy, batching or explicit postprocessing;
        sampling changes require a new Run. The callback receives chunk rays.
        """
        return self._advance(Budget() if budget is None else budget, options=options, progress=progress)

    def _advance(self, budget, *, options=None, progress=None, allow_calibration=False):
        if not isinstance(budget, Budget):
            raise TypeError("budget must be a Budget")
        started = time.perf_counter()
        with self._lock, self.solver._lock, self.solver.scene.lock:
            self.solver._check_open()
            if self._closed:
                raise RuntimeError("Run is closed")
            if self._invalidated.is_set() or self.scene_revision != self.solver.scene.revision:
                self._invalidated.set()
                raise RuntimeError("Scene geometry changed; start a new run")
            if options is not None:
                self.solver._validate(self.query, options)
                if options.sampling != self.options.sampling:
                    raise ValueError("Sampling changed; start a new run")
                self._options = options
            before = self.cumulative_rays
            cfg = controls(self.solver, self.query, self.options, budget)
            for state in self._accumulator._states.values():
                state.matrix.refresh(cfg)
                if state.sky is not None:
                    state.sky.refresh(cfg)
            if not self._cancelled.is_set() and budget.rays != 0 and budget.time_ms != 0:
                prepared = self.solver._sync_scene()
                if budget.time_ms is not None:
                    cfg.max_time_ms = max(0., budget.time_ms - (time.perf_counter() - started) * 1000)
                with prepared.solve_lock:
                    advance(prepared, cfg, self._accumulator,
                            cancel=lambda: self._cancelled.is_set() or self._invalidated.is_set(),
                            progress=progress, allow_calibration=allow_calibration)
            self._result = snapshot(self, self.cumulative_rays - before, (time.perf_counter() - started)*1000)
            return self._result

    def submit(self, budget=None, *, options=None, progress=None):
        """Queue an advance on the solver's single worker and return a Future."""
        with self.solver._submit_lock:
            self.solver._check_open()
            if self._closed:
                raise RuntimeError("Run is closed")
            if self.solver._executor is None:
                self.solver._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raystrack")
            return self.solver._executor.submit(self.solver._execute, self, budget, options, progress)

    def cancel(self):
        """Request cooperative cancellation while retaining the sample prefix."""
        self._cancelled.set()

    def close(self):
        """Cancel this run and reject subsequent asynchronous submissions."""
        self.cancel()
        self._closed = True

    def __enter__(self):
        """Enter an open run context with cancellation when the context ends."""
        if self._closed:
            raise RuntimeError("Run is closed")
        return self

    def __exit__(self, *args):
        """Close this run when its context ends."""
        self.close()
