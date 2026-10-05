"""Persistent JSON-lines worker for the compiled Grasshopper plug-in.

Launch with ``python -u -m raystrack.integrations.grasshopper.worker``. Requests
are ``{protocol: 2, id: ..., operation: ..., arguments: {...}}``. Replies carry
the same ID and either ``ok: true, result: ...`` or ``ok: false, error: ...``.

``solve`` queues a job immediately; ``status`` and ``cancel`` never wait for a
kernel, GPU initialization or JIT compilation. One persistent executor owns all
solvers and GPU work. Each component key retains its scene/acceleration cache;
an identical new launch resumes only a paused Run, with an additional ray budget.
Repeated launch IDs are idempotent and changed arguments are rejected.

The CLI duplicates the original stdout descriptor for protocol replies, then
redirects both Python and native stdout to stderr. Taichi and driver banners
cannot corrupt JSON. EOF cancels every job, drains the worker, closes its solvers
and exits; the parent owns this process and may terminate it during application
shutdown if native initialization does not return promptly.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
import importlib.metadata
import json
import math
import os
import platform
import sys
from threading import Event, RLock
import time
import traceback

import numpy as np

from ... import Budget, Postprocessing, Solver, available_devices, load, save
from .serialization import (json_value, mesh_key, options_from_json,
                            query_from_json, result_from_json, result_to_json,
                            scene_fingerprint, scene_from_json, scene_to_json)

PROTOCOL = 2
TERMINAL = frozenset(("succeeded", "paused", "cancelled", "failed"))


def _name(value, role):
    """Require an explicit nonempty protocol identifier or filesystem path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a nonempty string; received {value!r}")
    return value


def _budget(value):
    """Validate the launch's additional ray allowance; zero means unlimited."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ray_budget must be a nonnegative integer of additional rays "
                         f"(0 is unlimited); received {value!r}")
    return value


def _signature(arguments):
    """Canonicalize a launch payload for idempotency and changed-input checks."""
    return json.dumps(arguments, sort_keys=True, allow_nan=False, separators=(",", ":"))


@dataclass
class _Job:
    """Retain one launch's immutable inputs, cancellation event and snapshot."""
    key: str
    launch: str
    arguments: dict
    signature: str
    cancelled: Event = field(default_factory=Event)
    run: object = None
    payload: dict = field(default_factory=dict)


@dataclass
class _Cache:
    """Retain one component's geometry, solver resources and resumable run."""
    scene: object
    solver: object
    meshes: dict
    scene_hash: str
    settings: tuple
    resume_signature: object = None
    run: object = None
    status: str = "not_started"


class WorkerService:
    """Thread-safe bridge state; all numerical operations use one worker."""

    def __init__(self):
        """Create responsive protocol state and one persistent numerical thread."""
        self._lock = RLock()
        self._jobs = {}
        self._caches = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raystrack-gh")
        self._closed = False

    def request(self, operation, arguments=None):
        """Dispatch a request, returning a detached value or queued IO Future.

        Solve, status and cancel return without waiting for numerical execution.
        Runtime probes and storage stay on the thread that owns GPU resources.
        """
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, dict):
            raise ValueError(f"arguments must be a JSON object; received {type(arguments).__name__}")
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker is closed")
        if operation == "solve":
            return self._solve(arguments)
        if operation in ("status", "cancel"):
            return self._status(arguments, cancel=operation == "cancel")
        if operation == "release":
            key = _name(arguments.get("key"), "key")
            with self._lock:
                job = self._jobs.pop(key, None)
                if job is not None:
                    self._cancel(job)
                self._executor.submit(self._release, key)
            return {"released": True, "key": key}
        if operation in ("runtime", "save", "load"):
            return self._executor.submit(self._operation, operation, deepcopy(arguments))
        raise ValueError(f"Unknown operation: {operation!r}; supported operations are "
                         "solve, status, cancel, release, runtime, save and load")

    def _solve(self, arguments):
        """Queue a new launch or return an identical launch's existing state."""
        key = _name(arguments.get("key"), "key")
        launch = _name(arguments.get("launch"), "launch")
        _budget(arguments.get("ray_budget", 0))
        device = arguments.get("device", "auto")
        if not isinstance(device, str) or device.lower() not in ("auto", "cpu", "gpu", "cuda", "taichi", "vulkan", "metal"):
            raise ValueError(f"device must be auto, cpu, gpu, cuda, taichi, vulkan or metal; received {device!r}")
        acceleration = arguments.get("acceleration", "instanced")
        if acceleration not in ("flat", "instanced"):
            raise ValueError(f"acceleration must be 'flat' or 'instanced'; received {acceleration!r}")
        signature = _signature(arguments)
        with self._lock:
            previous = self._jobs.get(key)
            if previous is not None and previous.launch == launch:
                if previous.signature != signature:
                    raise ValueError("Arguments changed for an existing launch; use a new launch ID")
                return deepcopy(previous.payload)
            if previous is not None and previous.payload["status"] not in TERMINAL:
                self._cancel(previous)
            job = _Job(key, launch, deepcopy(arguments), signature)
            job.payload = {"key": key, "launch": launch, "status": "preparing",
                           "message": "Preparing scene and solver", "progress": 0.0,
                           "cumulative_rays": 0, "result": None, "error": None}
            self._jobs[key] = job
            self._executor.submit(self._execute, job)
            return deepcopy(job.payload)

    @staticmethod
    def _cancel(job):
        """Retain cancellation even when JIT has not yet produced a Run."""
        job.cancelled.set()
        if job.run is not None:
            job.run.cancel()

    def _status(self, arguments, *, cancel=False):
        """Read the current launch state or request cooperative cancellation."""
        key = _name(arguments.get("key"), "key")
        launch = _name(arguments.get("launch"), "launch")
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                raise KeyError(f"No job for component {key}")
            if job.launch != launch:
                raise ValueError("Stale launch ID; this component has a newer job")
            if cancel and job.payload["status"] not in TERMINAL:
                self._cancel(job)
                job.payload = dict(job.payload, message="Cancellation requested")
            elif cancel and job.payload["status"] == "paused":
                self._cancel(job)
                snapshot = deepcopy(job.payload["result"])
                if snapshot is not None:
                    snapshot.update(status="cancelled", converged=False)
                job.payload = dict(job.payload, status="cancelled", message="Cancelled",
                                   result=snapshot)
                self._executor.submit(self._mark_cancelled, key, job.run)
            return deepcopy(job.payload)

    def _publish(self, job, **changes):
        """Atomically replace a job snapshot without exposing mutable arrays."""
        with self._lock:
            job.payload = dict(job.payload, **changes)

    def _prepare(self, job):
        """Synchronize cached geometry and resume only an identical paused run."""
        arguments = job.arguments
        settings = (arguments.get("device", "auto"), arguments.get("acceleration", "instanced"))
        previous = self._caches.get(job.key)
        meshes = {} if previous is None else previous.meshes
        incoming = scene_from_json(arguments.get("scene"), meshes=meshes)
        fingerprint = scene_fingerprint(incoming)
        query = query_from_json(arguments.get("query"))
        options = options_from_json(arguments.get("options"))
        try:
            query.resolve(incoming.surface_ids)
        except (TypeError, ValueError) as exc:
            available = ", ".join(repr(identifier) for identifier in incoming.surface_ids[:12]) or "(scene is empty)"
            if len(incoming) > 12:
                available += f", ... ({len(incoming)} surfaces total)"
            raise ValueError(f"query.senders/query.receivers: {exc}. Available Scene IDs: {available}. "
                             "Select stable surface IDs from the Scene component, not display labels.") from None
        if (previous is None or previous.settings != settings or
                previous.scene.surface_ids != incoming.surface_ids):
            solver = Solver(incoming, device=settings[0], acceleration=settings[1], auto_tune=False)
            if previous is not None:
                previous.solver.close()
            cache = _Cache(incoming, solver, meshes, fingerprint, settings)
            self._caches[job.key] = cache
        else:
            cache = previous
            for current in incoming.surfaces:
                old = cache.scene[current.surface_id]
                if old.mesh is not current.mesh:
                    cache.scene.update_mesh(current.surface_id, current.mesh)
                    old = cache.scene[current.surface_id]
                if not np.array_equal(old.transform, current.transform):
                    cache.scene.update_transform(current.surface_id, current.transform)
                if old.label != current.label:
                    cache.scene.set_label(current.surface_id, current.label)
            cache.scene_hash = fingerprint
        cache.meshes = {mesh_key(s.mesh): s.mesh for s in cache.scene.surfaces}
        resume_signature = (fingerprint, query, options)
        if cache.status == "paused" and cache.resume_signature == resume_signature:
            run = cache.run
        else:
            # Starting validates final postprocessing before any preview. A
            # malformed request cannot close an otherwise resumable run.
            run = cache.solver.start(query, options)
            if cache.run is not None:
                cache.run.close()
        cache.run, cache.resume_signature = run, resume_signature
        cache.status = "running"
        with self._lock:
            job.run = run
            if job.cancelled.is_set():
                run.cancel()
        return cache, run, options

    @staticmethod
    def _maximum_rays(scene, query, options):
        """Estimate the work cap used for progress, without initializing a GPU."""
        senders, _ = query.resolve(scene.surface_ids)
        if options.sampling.strategy == "area_pair":
            return len(senders) * options.sampling.pair_samples * options.accuracy.max_replicates
        total = 0
        for sender in senders:
            mesh = scene[sender].mesh
            points = mesh.vertices[mesh.faces].astype(np.float64)
            area = float(np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                                  points[:, 2] - points[:, 0]), axis=1).sum() / 2)
            grid = max(4, math.ceil(math.sqrt(area * options.sampling.density)))
            total += grid * grid * options.sampling.rays_per_cell * options.accuracy.max_replicates
        return total

    def _execute(self, job):
        """Advance bounded chunks and publish throttled previews plus final state."""
        cache = None
        started = time.perf_counter()
        try:
            if job.cancelled.is_set():
                self._publish(job, status="cancelled", message="Cancelled before preparation")
                return
            cache, run, options = self._prepare(job)
            baseline = run.cumulative_rays
            ray_budget = job.arguments.get("ray_budget", 0)
            allowance = ray_budget or max(1, self._maximum_rays(cache.scene, run.query, options) - baseline)
            last_publish = -math.inf
            preview_options = replace(options, postprocessing=Postprocessing())
            self._publish(job, status="running", message="Sampling", cumulative_rays=baseline)
            while True:
                if job.cancelled.is_set():
                    run.cancel()
                remaining = ray_budget - (run.cumulative_rays - baseline) if ray_budget else None
                chunk = min(65536, options.batch_size)
                if remaining is not None:
                    chunk = min(chunk, max(0, remaining))
                result = run.advance(Budget(rays=chunk, time_ms=100), options=preview_options)
                used = run.cumulative_rays - baseline
                elapsed = (time.perf_counter() - started) * 1000
                complete = result.status in ("converged", "max_iters")
                cancelled = job.cancelled.is_set() or result.status == "cancelled"
                paused = bool(ray_budget and used >= ray_budget and not complete and not cancelled)
                if complete and not cancelled and options.postprocessing.reciprocity != "none":
                    result = run.advance(Budget(rays=0), options=options)
                terminal = complete or cancelled or paused
                now = time.perf_counter()
                if terminal or now - last_publish >= 0.25:
                    snapshot = result_to_json(result, scene_hash=cache.scene_hash,
                                              rays_used=used, elapsed_ms=elapsed)
                    status = "cancelled" if cancelled else "succeeded" if complete else "paused" if paused else "running"
                    message = ("Cancelled" if cancelled else "Converged" if result.converged else
                               "Maximum replicates reached" if complete else
                               "Ray budget reached; ready to resume" if paused else "Sampling")
                    self._publish(job, status=status, message=message,
                                  progress=100.0 if complete or paused else min(99.9, 100 * used / allowance),
                                  cumulative_rays=run.cumulative_rays, result=snapshot)
                    last_publish = now
                if terminal:
                    cache.status = job.payload["status"]
                    return
                # Initial preparation can use the first soft deadline without
                # tracing. The next bounded chunk continues on the same thread.
        except Exception as exc:
            cache = self._caches.get(job.key) if cache is None else cache
            if cache is not None:
                cache.status = "failed"
            traceback.print_exc(file=sys.stderr)
            self._publish(job, status="failed", message=str(exc),
                          error={"type": type(exc).__name__, "message": str(exc)})

    def _operation(self, operation, arguments):
        """Probe runtime or read/write versioned storage on the owning thread."""
        if operation == "runtime":
            try:
                version = importlib.metadata.version("raystrack")
            except importlib.metadata.PackageNotFoundError:
                version = "development"
            return {"protocol": PROTOCOL, "python": sys.executable,
                    "python_version": platform.python_version(), "version": version,
                    "devices": json_value(available_devices())}
        path = _name(arguments.get("path"), "path")
        if operation == "load":
            stored = load(path)
            return {"path": os.path.abspath(path), "scene": scene_to_json(stored.scene),
                    "result": None if stored.result is None else result_to_json(
                        stored.result, scene_hash=scene_fingerprint(stored.scene)),
                    "metadata": json_value(stored.metadata)}
        payload = arguments.get("result")
        result = None if payload is None else result_from_json(payload)
        # A component's original Scene description has no knowledge of the
        # worker's cache revision. Preserve the result's revision after checking
        # its geometry fingerprint, rather than inventing a new numerical run.
        scene = scene_from_json(arguments.get("scene"), revision=None if result is None else result.scene_revision)
        if payload is not None and payload.get("scene_hash") is not None:
            if payload["scene_hash"] != scene_fingerprint(scene):
                raise ValueError("Result belongs to different scene geometry")
        return {"path": save(path, scene, result, arguments.get("metadata"))}

    def _release(self, key):
        """Discard a component's cached solver after its active work is stopped."""
        cache = self._caches.pop(key, None)
        if cache is not None:
            cache.solver.close()

    def _mark_cancelled(self, key, run):
        """Prevent a cancelled paused run from being resumed by a new launch."""
        cache = self._caches.get(key)
        if cache is not None and cache.run is run:
            cache.status = "cancelled"

    def close(self):
        """Cancel launches, drain their thread and release all cached resources."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in self._jobs.values():
                self._cancel(job)
            self._executor.submit(self._close_caches)
        self._executor.shutdown(wait=True)

    def _close_caches(self):
        """Close every solver from the same thread that performed its tracing."""
        for key in tuple(self._caches):
            self._release(key)

    def __enter__(self):
        """Enter the worker lifecycle context."""
        return self

    def __exit__(self, *args):
        """Drain and close worker resources when the lifecycle context ends."""
        self.close()


def _protocol_stream():
    """Keep protocol stdout while redirecting fd-level native output too."""
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    return protocol


def main():
    """Serve UTF-8 JSON-lines requests until EOF, then cancel and close jobs."""
    output = _protocol_stream()
    write_lock = RLock()
    service = WorkerService()

    def reply(identifier, result=None, error=None):
        """Write one response envelope without interleaving concurrent replies."""
        envelope = {"protocol": PROTOCOL, "id": identifier, "ok": error is None}
        envelope["result" if error is None else "error"] = json_value(result if error is None else error)
        with write_lock:
            try:
                output.write(json.dumps(envelope, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
                output.flush()
            except (BrokenPipeError, OSError):
                # The parent has gone away; EOF/close will cancel resources.
                pass

    def finished(future, identifier):
        """Publish a queued operation's value or actionable exception details."""
        try:
            reply(identifier, future.result())
        except Exception as exc:
            reply(identifier, error={"type": type(exc).__name__, "message": str(exc)})

    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            identifier = None
            try:
                request = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {token}")))
                if not isinstance(request, dict):
                    raise ValueError("Request must be an object")
                identifier = request.get("id")
                if request.get("protocol") != PROTOCOL:
                    raise ValueError(f"Expected protocol {PROTOCOL}")
                result = service.request(request.get("operation"), request.get("arguments", {}))
                if isinstance(result, Future):
                    result.add_done_callback(lambda future, identifier=identifier: finished(future, identifier))
                else:
                    reply(identifier, result)
            except Exception as exc:
                reply(identifier, error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        service.close()
        output.close()


if __name__ == "__main__":
    main()

