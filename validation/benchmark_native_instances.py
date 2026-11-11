"""Sequential equal-input flat/shared Embree trials, with separate geometry/solve costs.

Use one fresh process per --mode/--library combination. Old ABI4 DLLs support
flat mode; shared mode requires the additive instance extension. No GPU timings
are inferred from host wall clocks. Run only after other workloads are idle.
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

from raystrack import Mesh, Scene
from raystrack.backends.native import Lighting, NativeGeometry, OpticalMaterial
from validation.benchmark_native_updates import mesh, peak_memory_bytes


def timed(operation):
    begin = time.perf_counter()
    value = operation()
    return value, (time.perf_counter() - begin)*1000


def fixture(count, grid, static_grid):
    vertices, faces = mesh(grid)
    panel = Mesh(vertices, faces)
    static_vertices, static_faces = mesh(static_grid)
    static_vertices[:, :2] *= 20
    static_vertices[:, 0] -= 30
    static = Mesh(static_vertices, static_faces)
    instances = [("static", "static", np.eye(4))]
    for i in range(count):
        transform = np.eye(4)
        transform[:2, 3] = (i % 32)*1.5, (i//32)*1.5
        instances.append((f"person-{i}", "panel", transform))
    return Scene.from_instances({"static": static, "panel": panel}, instances)


def run(mode, count, grid, static_grid, rays, repeats, samples):
    scene = fixture(count, grid, static_grid)
    origins = np.zeros((rays, 3), np.float32)
    origins[:, :2] = np.random.default_rng(17).uniform([0, 0], [48, max(1, count/32)*1.5], (rays, 2))
    directions = np.tile(np.array([0, 0, 1], np.float32), (rays, 1))
    hits, front, blocked = np.empty(rays, np.int32), np.empty(rays, np.uint8), np.empty(rays, np.uint8)
    active = np.ones(len(scene), np.uint8)
    sensors = np.ascontiguousarray(np.column_stack((origins[:32], directions[:32])))
    sun = np.array([[0, 0, 1]], np.float32)
    dni, sky = np.full((1, 3), 700.), np.full((1, 3), 80/np.pi)
    light = Lighting(sun.ctypes.data, dni.ctypes.data, sky.ctypes.data, None, 0, 1, 0, 0)
    materials = (OpticalMaterial * len(scene))() # black opaque: traversal/forcing contract identical
    # Component count is exported by the public result contract.
    from raystrack.shortwave import ShortwaveResult
    out = np.empty((1, len(sensors), len(ShortwaveResult.components), 3), np.float64)

    def prepare():
        return scene._snapshot() if mode == "shared" else scene.world_meshes()

    def create(value):
        return NativeGeometry.from_scene("embree", value) if mode == "shared" else NativeGeometry("embree", value)

    def update(value):
        return native.update_scene(value) if mode == "shared" else native.update(value)

    value, preparation_ms = timed(prepare)
    native, create_ms = timed(lambda: create(value))
    if mode == "shared" and native.geometry_mode != "shared-prototype-instances":
        native.close()
        raise RuntimeError("Shared mode requires rebuilt native instance extension")

    def trace():
        native.longwave(origins, directions, active, -1, 0, hits, front, blocked)

    def solve():
        native.shortwave(sensors, light, 1, materials, samples, 17, 0, 1e-5, out)

    try:
        _, first_trace_ms = timed(trace)
        trace_times = [timed(trace)[1] for _ in range(repeats)]
        _, cold_solve_ms = timed(solve)
        warm_solve_ms = [timed(solve)[1] for _ in range(repeats)]
        records = []
        for frame in range(repeats):
            poses = {}
            for surface in scene.surfaces[1:]:
                transform = surface.transform.copy()
                transform[0, 3] += .03
                poses[surface.surface_id] = transform
            _, edit_ms = timed(lambda: scene.update_transforms(poses))
            value, pack_ms = timed(prepare)
            retained, commit_ms = timed(lambda: update(value))
            _, moved_solve_ms = timed(solve)
            snapshot = out.copy()
            _, moved_trace_ms = timed(trace)
            classifications = (hits.copy(), front.copy(), blocked.copy())
            fresh, rebuild_ms = timed(lambda: create(value))
            try:
                fresh.longwave(origins, directions, active, -1, 0, hits, front, blocked)
                for previous, actual in zip(classifications, (hits, front, blocked)):
                    np.testing.assert_array_equal(previous, actual)
                fresh.shortwave(sensors, light, 1, materials, samples, 17, 0, 1e-5, out)
                np.testing.assert_array_equal(snapshot, out)
            finally:
                fresh.close()
            records.append(dict(frame=frame, scene_edit_ms=edit_ms, geometry_pack_ms=pack_ms,
                                native_commit_ms=commit_ms, moved_solve_ms=moved_solve_ms,
                                moved_trace_ms=moved_trace_ms, fresh_rebuild_ms=rebuild_ms,
                                retained=retained, fresh_hit_and_solve_parity=True))
        return dict(mode=mode, instances=len(scene), prototypes=2, world_triangles=sum(len(s.mesh.faces) for s in scene.surfaces),
                    unique_triangles=2*grid*grid + 2*static_grid*static_grid,
                    sensor_count=len(sensors), samples=samples, seed=17, rays=rays,
                    preparation_ms=preparation_ms, create_ms=create_ms, first_trace_ms=first_trace_ms,
                    stationary_trace_ms=trace_times, stationary_trace_median_ms=statistics.median(trace_times),
                    cold_solve_ms=cold_solve_ms, warm_solve_ms=warm_solve_ms,
                    output_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                    instance_statistics=native.instance_statistics, updates=records,
                    peak_host_working_set_bytes=peak_memory_bytes())
    finally:
        native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("flat", "shared"), required=True)
    parser.add_argument("--instances", type=int, default=256)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--static-grid", type=int, default=128)
    parser.add_argument("--rays", type=int, default=65536)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.instances, args.grid, args.static_grid, args.rays, args.samples, args.repeats) < 1:
        parser.error("counts must be positive")
    library = Path(os.environ["RAYSTRACK_NATIVE_LIBRARY"])
    report = dict(platform=platform.platform(), cpu_logical_count=os.cpu_count(),
                  library=str(library.resolve()), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  timing="host perf_counter; sequential; native automatic thread policy",
                  result=run(args.mode, args.instances, args.grid, args.static_grid, args.rays, args.repeats, args.samples))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
