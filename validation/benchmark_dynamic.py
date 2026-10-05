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


def difference(a, b):
    return float(np.max(np.abs(a.dense()-b.dense()), initial=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu","auto","cuda","vulkan","metal","taichi"), default="cpu")
    parser.add_argument("--acceleration", choices=("flat","instanced"), default="flat")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--frames", type=positive, default=5)
    parser.add_argument("--subdivisions", type=positive, default=8)
    parser.add_argument("--budget", type=positive, default=4096)
    parser.add_argument("--samples", type=positive, default=4)
    parser.add_argument("--rays", type=positive, default=32)
    parser.add_argument("--batch-size", type=positive, default=1024)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--json", type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from dataclasses import replace
    from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Budget, Channel
    triples=[plane("emitter",0,1,args.subdivisions),
             plane("receiver",2,1,args.subdivisions,downward=True),
             plane("blocker",1,1.2,args.subdivisions)]
    scene=Scene.from_meshes({sid:Mesh(v,f) for sid,v,f in triples})
    cells=max(4,int(math.ceil(math.sqrt(4*args.samples))))**2
    iterations=max(2,int(math.ceil(args.budget/(cells*args.rays))))
    options=SolveOptions(Sampling(density=args.samples,rays_per_cell=args.rays,seed=13),
                        Accuracy(max_replicates=iterations,min_replicates=iterations,tolerance=0),
                        batch_size=args.batch_size)
    query=Query.row("emitter",sky="merged")
    def make(current):
        return Solver(current,device=args.device,acceleration=args.acceleration,bvh="builtin",auto_tune=False)
    def refresh(solver):
        # Internal instrumentation separates lazy preparation from public solve.
        # The benchmark performs the same preparation for reused and fresh solvers.
        prepared=solver._sync_scene()
        if args.acceleration == "instanced":
            prepared.get_instanced_scene()
        else:
            prepared.get_scene(use_bvh=True)
        prepared.get_emitters(samples=args.samples,rays=args.rays,flip_faces=False)
        prepared.get_mesh_bounds()
        return prepared
    report={"api":"v2","requested_device":args.device,"acceleration":args.acceleration,
            "triangles":sum(len(f) for _,_,f in triples),"frames":args.frames,
            "max_total_rays_per_solve":args.budget,"absolute_tolerance":args.atol,
            "notes":["Equal additional ray budgets; no time deadline or automatic probes.",
                     "Preparation instrumented with internal cache refresh so lazy work is measured separately.",
                     "Transform commit, solver synchronization/emitter preparation, and tracing are separate timings.",
                     "Cold timings include compilation/cache loading; only warmed timings compare steady workloads.",
                     "Updated/fresh agreement checks numerical cache correctness, not statistical accuracy."],
            "measurements":[]}
    with make(scene) as solver:
        prepared,report["cold_prepare_ms"]=timed(lambda:refresh(solver))
        if args.warmup:
            report["warmup"]=solver.warmup(query,options).as_dict()
        cold,report["cold_solve_ms"]=timed(lambda:solver.solve(query,options,Budget(rays=args.budget)))
        warm,report["warm_solve_ms"]=timed(lambda:solver.solve(query,options,Budget(rays=args.budget)))
        report["execution_plan"]=dict(warm.execution)
        report["resolved_backend"]=warm.execution["backend"]
        report["cold_rays"],report["warm_rays"]=cold.rays_used,warm.rays_used
        report["warm_repeat_max_abs_difference"]=difference(cold,warm)
        positions=np.linspace(0,3,args.frames) if args.frames > 1 else [0]
        for frame,x in enumerate(positions):
            transform=np.eye(4); transform[0,3]=x
            _,update_ms=timed(lambda:scene.update_transform("blocker",transform))
            _,refresh_ms=timed(lambda:refresh(solver))
            run=solver.start(query,options)
            updated,solve_ms=timed(lambda:run.advance(Budget(rays=args.budget)))
            with make(Scene(scene.surfaces)) as fresh:
                _,fresh_prepare_ms=timed(lambda:refresh(fresh))
                reference,fresh_solve_ms=timed(lambda:fresh.solve(query,options,Budget(rays=args.budget)))
            error=difference(updated,reference)
            report["measurements"].append({"frame":frame,"scene_revision":scene.revision,
                "blocker_x":float(x),"transform_commit_ms":update_ms,
                "updated_cache_refresh_ms":refresh_ms,"updated_prepare_total_ms":update_ms+refresh_ms,
                "fresh_bvh_prepare_ms":fresh_prepare_ms,"updated_solve_ms":solve_ms,
                "fresh_solve_ms":fresh_solve_ms,"updated_frame_total_ms":update_ms+refresh_ms+solve_ms,
                "fresh_frame_total_ms":fresh_prepare_ms+fresh_solve_ms,"rays_used":updated.rays_used,
                "fresh_rays_used":reference.rays_used,"max_abs_difference":error,
                "accuracy_pass":bool(error <= args.atol and updated.rays_used == reference.rays_used),
                "receiver_vf":updated.value("emitter",Channel("surface","receiver","front")),
                "sky_vf":updated.value("emitter",Channel("sky"))})
        refined,refine_ms=timed(lambda:run.advance(Budget(rays=args.budget),
             options=replace(options,accuracy=replace(options.accuracy,max_replicates=2*iterations))))
        report["stationary_refinement"]={"rays_used":refined.rays_used,
            "cumulative_rays":refined.cumulative_rays,"elapsed_ms":refine_ms,
            "scene_revision":refined.scene_revision}
    rows=report["measurements"]
    report["all_accuracy_checks_passed"]=all(row["accuracy_pass"] for row in rows)
    report["mean_updated_prepare_ms"]=float(np.mean([row["updated_prepare_total_ms"] for row in rows]))
    report["mean_fresh_prepare_ms"]=float(np.mean([row["fresh_bvh_prepare_ms"] for row in rows]))
    print(json.dumps(report,indent=2,default=dict))
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2,default=dict)+"\n",encoding="utf-8")
    if not report["all_accuracy_checks_passed"]:
        raise SystemExit("Updated/fresh mismatch")


if __name__ == "__main__":
    main()
