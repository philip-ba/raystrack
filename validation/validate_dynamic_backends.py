#!/usr/bin/env python3
"""Check dynamic/portable execution against existing closed-form references.

    python validation/validate_dynamic_backends.py --devices cpu vulkan --json report.json

Selected sender rows exercise the new execution path. Square receivers move
through three gaps using one prepared BVH; other cases check shape/orientation.
Fixed, disjoint seed ranges measure sampling variation without early stopping.
This is an accuracy smoke check, not a confidence-interval coverage study.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from common_validation import disk_xy, rectangle_xy, rectangle_yz
from validate_01_parallel_equal_square import analytical_equal_square
from validate_02_parallel_equal_rectangle import analytical_equal_rectangles
from validate_03_equal_coaxial_discs import analytical_equal_discs
from validate_05_perpendicular_square_rectangle import analytical_square_to_adjacent_rectangle


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def cases(prepared_type):
    square = prepared_type([
        rectangle_xy("emitter", 1, 1, 0),
        rectangle_xy("receiver", 1, 1, 1, normal=-1),
    ])
    for gap in (0.5, 1.0, 2.0):
        transform = np.eye(4)
        transform[2, 3] = gap - 1.0
        square.update_transform("receiver", transform)
        yield f"moving_square_gap_{gap:g}", square, analytical_equal_square(1, gap)
    yield "parallel_rectangle_2x1", prepared_type([
        rectangle_xy("emitter", 2, 1, 0),
        rectangle_xy("receiver", 2, 1, 1, normal=-1),
    ]), analytical_equal_rectangles(2, 1, 1)
    yield "coaxial_discs_256_segments", prepared_type([
        disk_xy("emitter", 1, 0, segments=256),
        disk_xy("receiver", 1, 1, segments=256, normal=-1),
    ]), analytical_equal_discs(1, 1)
    yield "perpendicular_square_rectangle", prepared_type([
        rectangle_xy("emitter", 1, 1, 0, center=(0.5, 0)),
        rectangle_yz("receiver", 1, 1, 0),
    ]), analytical_square_to_adjacent_rectangle(1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu"],
                        choices=("cpu", "cuda", "vulkan", "metal", "taichi"))
    parser.add_argument("--accelerations", nargs="+", default=["flat"], choices=("flat", "instanced"))
    parser.add_argument("--sampling-mode", choices=("fair", "adaptive"), default="fair")
    parser.add_argument("--samples", type=positive, default=256, help="sampling density per unit area")
    parser.add_argument("--rays", type=positive, default=256)
    parser.add_argument("--iterations", type=positive, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 1009, 10007])
    parser.add_argument("--atol", type=float, default=1e-3, help="absolute view-factor error per seed")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    ordered = sorted(args.seeds)
    if any(b - a < args.iterations for a, b in zip(ordered, ordered[1:])):
        parser.error("seed ranges must be disjoint (spacing >= iterations)")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Channel

    report = {
        "settings": {"samples": args.samples, "rays": args.rays,
                     "iterations": args.iterations, "seeds": args.seeds,
                     "sampling_mode": args.sampling_mode,
                     "absolute_tolerance_per_seed": args.atol},
        "notes": ["V2 Query.row selects the emitter; moving square cases reuse one Solver and scene acceleration.",
                  "Every seed runs the full iteration count without convergence early stopping.",
                  "Disc meshes approximate curved boundaries with 256 segments.",
                  "Elapsed times include preparation/cache loading and compilation; not a speed comparison.",
                  "This does not validate uncertainty coverage or all supported hardware."],
        "measurements": [],
    }
    for device in dict.fromkeys(args.devices):
        for acceleration in dict.fromkeys(args.accelerations):
            cache = {}
            factory = lambda meshes: Scene.from_meshes({sid: Mesh(v, f) for sid, v, f in meshes})
            try:
                for name, scene, reference in cases(factory):
                    if id(scene) not in cache:
                        cache[id(scene)] = Solver(scene, device=device, acceleration=acceleration,
                                                  bvh="builtin", auto_tune=False)
                    solver = cache[id(scene)]
                    measurements = []
                    for seed in args.seeds:
                        options = SolveOptions(Sampling(density=args.samples, rays_per_cell=args.rays,
                                                        seed=seed, mode=args.sampling_mode),
                            Accuracy(max_replicates=args.iterations, min_replicates=args.iterations,
                                     tolerance=0), batch_size=65536)
                        started = time.perf_counter()
                        result = solver.solve(Query.row("emitter", receivers=["receiver"]), options)
                        value = result.value("emitter", Channel("surface", "receiver", "front"))
                        error = abs(value-reference)
                        measurements.append({"seed": seed, "view_factor": value,
                            "absolute_error": error, "passed": bool(error <= args.atol),
                            "rays": result.rays_used, "status": result.status,
                            "elapsed_ms": (time.perf_counter()-started)*1000})
                    backend = result.execution["backend"]
                    row = {"case": name, "device": device, "resolved_backend": backend,
                        "acceleration": acceleration, "scene_revision": scene.revision, "analytical": reference,
                        "mean_view_factor": float(np.mean([m["view_factor"] for m in measurements])),
                        "max_absolute_error": max(m["absolute_error"] for m in measurements),
                        "passed": all(m["passed"] for m in measurements), "seeds": measurements}
                    report["measurements"].append(row)
                    print(f"{device:7s} {acceleration:9s} {name:33s} analytical={reference:.8f} "
                          f"mean={row['mean_view_factor']:.8f} "
                          f"max_error={row['max_absolute_error']:.3g} pass={row['passed']}", flush=True)
            finally:
                for solver in cache.values():
                    solver.close()

    report["all_passed"] = all(row["passed"] for row in report["measurements"])
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.json}")
    if not report["all_passed"]:
        raise SystemExit("A closed-form comparison exceeded the per-seed absolute tolerance.")


if __name__ == "__main__":
    main()
