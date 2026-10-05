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
from unittest.mock import patch

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
    if square.acceleration == "instanced":
        square.get_instanced_scene()
    else:
        square.get_scene(use_bvh=True)
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
    from raystrack import MatrixParams, PreparedSolver, view_factor_matrix
    from raystrack.devices import resolve_backend

    report = {
        "settings": {"samples": args.samples, "rays": args.rays,
                     "iterations": args.iterations, "seeds": args.seeds,
                     "sampling_mode": args.sampling_mode,
                     "absolute_tolerance_per_seed": args.atol},
        "notes": ["Selected emitter forces dynamic execution; square cases reuse/refit one BVH.",
                  "Every seed runs the full iteration count without convergence early stopping.",
                  "Disc meshes approximate curved boundaries with 256 segments.",
                  "Elapsed times include preparation/cache loading and compilation; not a speed comparison.",
                  "This does not validate uncertainty coverage or all supported hardware."],
        "measurements": [],
    }
    with patch("raystrack.main._log"):
        for device in dict.fromkeys(args.devices):
            backend = resolve_backend(device)
            for acceleration in dict.fromkeys(args.accelerations):
                factory = lambda meshes: PreparedSolver(meshes, acceleration=acceleration)
                for name, prepared, reference in cases(factory):
                    measurements = []
                    for seed in args.seeds:
                        params = MatrixParams(samples=args.samples, rays=args.rays,
                            min_iters=args.iterations, max_iters=args.iterations,
                            seed=seed, tol=0, device=device, bvh="builtin",
                            reciprocity=False, emitter_names=["emitter"],
                            sampling_mode=args.sampling_mode,
                            ray_batch_size=65536)
                        started = time.perf_counter()
                        result = view_factor_matrix(prepared.meshes, params=params, prepared=prepared)
                        value = result["emitter"].get("receiver_front", 0.0)
                        n_once = prepared.get_emitter(0, samples=args.samples,
                            rays=args.rays, flip_faces=False).n_cells * args.rays
                        error = abs(value - reference)
                        measurements.append({"seed": seed, "view_factor": value,
                            "absolute_error": error, "passed": bool(error <= args.atol),
                            "rays": n_once * args.iterations,
                            "elapsed_ms": (time.perf_counter() - started) * 1000})
                    row = {"case": name, "device": device, "resolved_backend": backend,
                        "acceleration": acceleration, "scene_version": prepared.version, "analytical": reference,
                        "mean_view_factor": float(np.mean([m["view_factor"] for m in measurements])),
                        "max_absolute_error": max(m["absolute_error"] for m in measurements),
                        "passed": all(m["passed"] for m in measurements), "seeds": measurements}
                    report["measurements"].append(row)
                    print(f"{device:7s} {acceleration:9s} {name:33s} analytical={reference:.8f} "
                          f"mean={row['mean_view_factor']:.8f} "
                          f"max_error={row['max_absolute_error']:.3g} pass={row['passed']}", flush=True)

    report["all_passed"] = all(row["passed"] for row in report["measurements"])
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.json}")
    if not report["all_passed"]:
        raise SystemExit("A closed-form comparison exceeded the per-seed absolute tolerance.")


if __name__ == "__main__":
    main()
