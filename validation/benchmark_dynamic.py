#!/usr/bin/env python3
"""Measure dynamic BVH refits and verify updated-vs-fresh seeded solves.

Run without installing the source checkout:
    python validation/benchmark_dynamic.py --device cpu --frames 3
    python validation/benchmark_dynamic.py --device vulkan --json results.json

Small smoke run:
    python validation/benchmark_dynamic.py --frames 2 --subdivisions 2 --budget 512

Timing reports separate scene preparation, first tracing call, warmed tracing,
refit/cache refresh, and a fresh BVH preparation at each blocker position.
Comparing the same seed and ray budget checks update correctness; it does not
measure Monte Carlo error against an analytic ground truth or promise a speedup.
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


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def plane(name: str, z: float, radius: float, subdivisions: int, downward=False):
    axis = np.linspace(-radius, radius, subdivisions + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z))).astype(np.float32)
    faces = []
    width = subdivisions + 1
    for row in range(subdivisions):
        for column in range(subdivisions):
            a = row * width + column
            faces.extend(((a, a + 1, a + width + 1), (a, a + width + 1, a + width)))
    faces = np.asarray(faces, dtype=np.int32)
    if downward:
        faces = faces[:, ::-1].copy()
    return name, vertices, faces


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, (time.perf_counter() - started) * 1000.0


def flattened(result):
    return {(kind, emitter, receiver): float(value)
            for kind in ("scene", "sky", "rest")
            for emitter, row in getattr(result, kind).items()
            for receiver, value in row.items()}


def difference(updated, fresh):
    a, b = flattened(updated), flattened(fresh)
    return max((abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in set(a) | set(b)), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "vulkan", "metal", "taichi"), default="cpu")
    parser.add_argument("--acceleration", choices=("flat", "instanced"), default="flat")
    parser.add_argument("--warmup", action="store_true", help="measure a plan before the first solve")
    parser.add_argument("--frames", type=positive, default=5)
    parser.add_argument("--subdivisions", type=positive, default=8, help="grid subdivisions per plane")
    parser.add_argument("--budget", type=positive, default=4096, help="maximum traced rays per solve")
    parser.add_argument("--samples", type=positive, default=4, help="emission sampling density")
    parser.add_argument("--rays", type=positive, default=32, help="rays per sampling cell")
    parser.add_argument("--batch-size", type=positive, default=1024)
    parser.add_argument("--atol", type=float, default=1e-6, help="updated/fresh absolute VF tolerance")
    parser.add_argument("--json", type=Path, help="save a machine-readable timing and correctness report")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from raystrack import MatrixParams, PreparedSolver, SkyParams
    from raystrack.devices import resolve_backend
    from raystrack.preview import PreviewSession

    meshes = [plane("emitter", 0.0, 1.0, args.subdivisions),
              plane("receiver", 2.0, 1.0, args.subdivisions, downward=True),
              plane("blocker", 1.0, 1.2, args.subdivisions)]
    # Minimum g is four cells per side; the emitter area is four square units.
    cells = max(4, int(math.ceil(math.sqrt(4.0 * args.samples)))) ** 2
    per_iteration = cells * args.rays
    iterations = max(2, int(math.ceil(args.budget / per_iteration)))
    common = dict(samples=args.samples, rays=args.rays, seed=13, device=args.device,
                  bvh="builtin", min_iters=iterations, max_iters=iterations,
                  tol=0.0, ray_batch_size=args.batch_size, emitter_names=["emitter"],
                  max_total_rays=args.budget)
    mp, sp = MatrixParams(**common, reciprocity=False), SkyParams(**common)

    if args.device == "auto":
        backend, device_init_ms = "auto", 0.0
    else:
        backend, device_init_ms = timed(lambda: resolve_backend(args.device))

    def prepare(meshes):
        item = PreparedSolver(meshes, acceleration=args.acceleration)
        refresh(item)
        return item

    def refresh(item):
        if item.acceleration == "instanced":
            item.get_instanced_scene()
        else:
            item.get_scene(use_bvh=True)
        item.get_emitters(samples=mp.samples, rays=mp.rays, flip_faces=False)
        item.get_mesh_bounds()

    prepared, cold_prepare_ms = timed(lambda: prepare(meshes))
    report = {
        "requested_device": args.device,
        "resolved_backend": backend,
        "acceleration": args.acceleration,
        "triangles": prepared.total_faces,
        "selected_emitters": ["emitter"],
        "frames": args.frames,
        "max_total_rays_per_solve": args.budget,
        "absolute_tolerance": args.atol,
        "device_initialization_ms": device_init_ms,
        "cold_prepare_ms": cold_prepare_ms,
        "notes": ["Time budget disabled for equal-budget comparisons.",
                  "Cold timings include first-call compilation/cache loading and device setup where applicable.",
                  "The first frame's refit update can include first-call compilation/cache loading; later frames reuse it.",
                  "Tracing solve timings include ray generation, reduction and result processing.",
                  "Updated/fresh agreement checks cache correctness, not statistical accuracy."],
        "measurements": [],
    }

    # Keep benchmark output in the current terminal, without log-console windows.
    with patch("raystrack.main._log"):
        with PreviewSession(prepared, mp, sp) as session:
            if args.warmup:
                report["warmup"] = session.warmup().as_dict()
                report["notes"].append("First solve timing follows explicit calibration when --warmup is used.")
            cold, report["cold_solve_ms"] = timed(session.solve)
            warm, report["warm_solve_ms"] = timed(session.solve)
            report["execution_plan"] = getattr(prepared, "_last_execution_plan", {})
            backend = report["execution_plan"].get("backend", backend)
            report["resolved_backend"] = backend
            report["cold_rays"] = cold.rays_used
            report["warm_rays"] = warm.rays_used
            report["warm_repeat_max_abs_difference"] = difference(cold, warm)
            positions = np.linspace(0.0, 3.0, args.frames) if args.frames > 1 else [0.0]
            for frame, x in enumerate(positions):
                transform = np.eye(4)
                transform[0, 3] = x
                _, update_ms = timed(lambda: session.update_transform("blocker", transform))
                _, refresh_ms = timed(lambda: refresh(prepared))
                updated, updated_solve_ms = timed(session.solve)
                fresh, fresh_prepare_ms = timed(lambda: prepare(prepared.meshes))
                with PreviewSession(fresh, mp, sp) as fresh_session:
                    reference, fresh_solve_ms = timed(fresh_session.solve)
                error = difference(updated, reference)
                matched = (updated.completed and reference.completed
                           and updated.rays_used == reference.rays_used
                           and error <= args.atol)
                report["measurements"].append({
                    "frame": frame,
                    "scene_version": updated.scene_version,
                    "blocker_x": float(x),
                    "refit_update_ms": update_ms,
                    "updated_cache_refresh_ms": refresh_ms,
                    "updated_prepare_total_ms": update_ms + refresh_ms,
                    "fresh_bvh_prepare_ms": fresh_prepare_ms,
                    "updated_solve_ms": updated_solve_ms,
                    "fresh_solve_ms": fresh_solve_ms,
                    "updated_frame_total_ms": update_ms + refresh_ms + updated_solve_ms,
                    "fresh_frame_total_ms": fresh_prepare_ms + fresh_solve_ms,
                    "rays_used": updated.rays_used,
                    "fresh_rays_used": reference.rays_used,
                    "max_abs_difference": error,
                    "accuracy_pass": bool(matched),
                    "receiver_vf": updated.scene.get("emitter", {}).get("receiver_front", 0.0),
                    "sky_vf": updated.sky.get("emitter", {}).get("Sky", 0.0),
                })
            refined, refinement_ms = timed(session.refine)
            report["stationary_refinement"] = {
                "scene_version": refined.scene_version,
                "rays_used": refined.rays_used,
                "cumulative_rays": refined.cumulative_rays,
                "elapsed_ms": refinement_ms,
                "receiver_vf": refined.scene.get("emitter", {}).get("receiver_front", 0.0),
            }

    rows = report["measurements"]
    report["all_accuracy_checks_passed"] = all(row["accuracy_pass"] for row in rows)
    report["mean_updated_prepare_ms"] = float(np.mean([row["updated_prepare_total_ms"] for row in rows]))
    report["mean_fresh_prepare_ms"] = float(np.mean([row["fresh_bvh_prepare_ms"] for row in rows]))
    print(f"Backend={backend}; acceleration={args.acceleration}; triangles={prepared.total_faces}; budget={args.budget} rays/solve")
    print(f"Cold prepare={cold_prepare_ms:.3f} ms; first solve={report['cold_solve_ms']:.3f} ms; "
          f"warm solve={report['warm_solve_ms']:.3f} ms")
    print("frame  x     refit+refresh  fresh_prepare  updated_solve  fresh_solve  max_abs_diff  pass")
    for row in rows:
        print(f"{row['frame']:5d}  {row['blocker_x']:4.1f}  {row['updated_prepare_total_ms']:13.3f}  "
              f"{row['fresh_bvh_prepare_ms']:13.3f}  {row['updated_solve_ms']:13.3f}  "
              f"{row['fresh_solve_ms']:11.3f}  {row['max_abs_difference']:12.3g}  {row['accuracy_pass']}")
    print(f"Stationary refinement: {refined.rays_used} additional rays, {refined.cumulative_rays} cumulative rays, {refinement_ms:.3f} ms, "
          f"version {refined.scene_version}")
    print("Timings depend on scene, workload, device and cache state.")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.json}")
    if not report["all_accuracy_checks_passed"]:
        raise SystemExit("Updated/fresh solve comparison exceeded tolerance or ray budgets differed.")


if __name__ == "__main__":
    main()
