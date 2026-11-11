"""Measure native commit/update/trace separately; all output is machine readable.

Run sequentially on the baseline and candidate DLL, with identical arguments.
The optional ABI-2 update extension falls back to rebuilding older libraries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from raystrack.backends.native import NativeGeometry


def peak_memory_bytes():
    """Process high-water mark, including fresh and updated geometry together."""
    if os.name == "nt":
        import ctypes as ct

        class Counters(ct.Structure):
            _fields_ = [("cb", ct.c_uint32), ("faults", ct.c_uint32),
                        ("peak_working_set", ct.c_size_t), ("working_set", ct.c_size_t),
                        ("peak_paged", ct.c_size_t), ("paged", ct.c_size_t),
                        ("peak_nonpaged", ct.c_size_t), ("nonpaged", ct.c_size_t),
                        ("pagefile", ct.c_size_t), ("peak_pagefile", ct.c_size_t)]
        counters = Counters()
        counters.cb = ct.sizeof(counters)
        current = ct.windll.kernel32.GetCurrentProcess
        current.restype = ct.c_void_p
        query = ct.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ct.c_void_p, ct.POINTER(Counters), ct.c_uint32]
        return int(counters.peak_working_set) if query(current(), ct.byref(counters), counters.cb) else None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def mesh(grid):
    xy = np.indices((grid + 1, grid + 1)).reshape(2, -1).T / grid
    vertices = np.ascontiguousarray(np.column_stack((xy, np.ones(len(xy)))), dtype=np.float32)
    a = (np.arange(grid)[:, None] * (grid + 1) + np.arange(grid)).ravel()
    faces = np.ascontiguousarray(np.stack((np.column_stack((a, a + 1, a + grid + 2)),
                                          np.column_stack((a, a + grid + 2, a + grid + 1))),
                                         axis=1).reshape(-1, 3), dtype=np.int32)
    return vertices, faces


def timed(operation):
    start = time.perf_counter()
    value = operation()
    return value, 1000 * (time.perf_counter() - start)


def run(backend, grid, rays, repeats):
    vertices, faces = mesh(grid)
    meshes = [("canopy", vertices, faces)]
    origins = np.ascontiguousarray(np.column_stack((
        np.random.default_rng(11).uniform(0.001, 0.999, (rays, 2)), np.zeros(rays))), dtype=np.float32)
    directions = np.tile(np.array([0, 0, 1], dtype=np.float32), (rays, 1))
    active = np.ones(1, np.uint8)
    hits, front, blocked = np.empty(rays, np.int32), np.empty(rays, np.uint8), np.empty(rays, np.uint8)
    geometry, create_ms = timed(lambda: NativeGeometry(backend, meshes))

    def trace():
        geometry.longwave(origins, directions, active, -1, 0, hits, front, blocked)

    try:
        _, first_trace_ms = timed(trace)
        stationary = [timed(trace)[1] for _ in range(repeats)]
        records = []
        for mode in ("transform", "deform", "topology"):
            for frame in range(repeats):
                def prepare():
                    changed = vertices.copy()
                    changed[:, 0] += (frame + 1) * 0.01
                    if mode == "deform":
                        changed[:, 2] += 0.1 * np.sin(changed[:, 0] * 20)
                    # Alternate winding so every topology iteration rebuilds.
                    indices = np.ascontiguousarray(faces[:, ::-1]) if mode == "topology" and frame % 2 == 0 else faces
                    return [("canopy", changed, indices)]

                entries, preparation_ms = timed(prepare)
                retained, update_ms = timed(lambda: geometry.update(entries))
                _, trace_ms = timed(trace)
                snapshot = (hits.copy(), front.copy(), blocked.copy())
                fresh, fresh_build_ms = timed(lambda: NativeGeometry(backend, entries))
                try:
                    _, fresh_trace_ms = timed(lambda: fresh.longwave(
                        origins, directions, active, -1, 0, hits, front, blocked))
                    for before, after in zip(snapshot, (hits, front, blocked)):
                        np.testing.assert_array_equal(before, after)
                finally:
                    fresh.close()
                records.append(dict(mode=mode, frame=frame, retained=retained,
                                    preparation_ms=preparation_ms, update_ms=update_ms, trace_ms=trace_ms,
                                    fresh_build_ms=fresh_build_ms, fresh_trace_ms=fresh_trace_ms,
                                    fresh_parity=True))
        return dict(backend=backend, triangles=len(faces), rays=rays, create_ms=create_ms,
                    vertices=len(vertices), geometry_input_bytes=vertices.nbytes + faces.nbytes,
                    first_trace_ms=first_trace_ms, stationary_ms=stationary,
                    stationary_median_ms=statistics.median(stationary), updates=records)
    finally:
        geometry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["embree", "vulkan-rt"])
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--rays", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.grid, args.rays, args.repeats) < 1:
        parser.error("grid, rays and repeats must be positive")
    library = Path(os.environ["RAYSTRACK_NATIVE_LIBRARY"])
    report = dict(platform=platform.platform(), processor=platform.processor(),
                  python=platform.python_version(), cpu_logical_count=os.cpu_count(),
                  library=str(library.resolve()), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  seed=11, thread_policy="native defaults; sequential backend execution",
                  results=[run(b, args.grid, args.rays, args.repeats) for b in args.backends])
    report["process_peak_working_set_bytes"] = peak_memory_bytes()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
