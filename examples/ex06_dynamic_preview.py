#!/usr/bin/env python3
"""Move a blocker, preview one stationary emitter, then refine the last frame.

Run from the repository root:
    python examples/ex06_dynamic_preview.py --device cpu --frames 5
    python examples/ex06_dynamic_preview.py --device vulkan --warmup

The portable backend requires the optional ``raystrack[portable-gpu]`` extra.
Time limits are soft and cannot interrupt preparation or a running ray chunk.
This small command-line demo uses a fixed ray budget so cold kernel compilation
does not leave the first frame empty. An interactive application can additionally
pass ``Budget(time_ms=50)`` and use ``Run.submit(...)`` for async work.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def square(name: str, z: float, radius: float, downward: bool = False):
    vertices = np.asarray([
        [-radius, -radius, z], [radius, -radius, z],
        [radius, radius, z], [-radius, radius, z],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    if downward:
        faces = faces[:, ::-1].copy()
    return name, vertices, faces


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "vulkan", "metal", "taichi"), default="cpu")
    parser.add_argument("--frames", type=positive, default=5)
    parser.add_argument("--budget", type=positive, default=4096, help="maximum rays per preview")
    parser.add_argument("--warmup", action="store_true", help="compile kernels and measure an execution plan before frames")
    parser.add_argument("--acceleration", choices=("flat", "instanced"), default="flat")
    parser.add_argument("--sampling-mode", choices=("fair", "adaptive"), default="fair")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Budget, Channel
    scene=Scene.from_meshes({sid:Mesh(v,f) for sid,v,f in [
        square("emitter",0,1),square("receiver",2,1,downward=True),square("blocker",1,1.2)]})
    options=SolveOptions(Sampling(density=4,rays_per_cell=64,seed=7,mode=args.sampling_mode),
                        Accuracy(max_replicates=max(8,args.budget//1024+2),min_replicates=2,tolerance=0),
                        batch_size=1024)
    query=Query.row("emitter",sky="merged")
    with Solver(scene,device=args.device,acceleration=args.acceleration,bvh="builtin") as solver:
        if args.warmup:
            print("Warmup:",solver.warmup(query,options).as_dict())
        for x in np.linspace(0,3,args.frames):
            transform=np.eye(4); transform[0,3]=x
            scene.update_transform("blocker",transform)
            run=solver.start(query,options)
            result=run.advance(Budget(rays=args.budget))
            print("revision",result.scene_revision,"blocker x",float(x),
                  "receiver",result.value("emitter",Channel("surface","receiver","front")),
                  "sky",result.value("emitter",Channel("sky")),"rays",result.rays_used,
                  "backend",result.execution["backend"])
        refined=run.advance(Budget(rays=args.budget))
        print("Refined:",refined.rays_used,"additional;",refined.cumulative_rays,"cumulative")
        # For async applications use run.submit(Budget(...)); repeated submissions
        # add work. Cancel an obsolete run before starting a replacement frame.


if __name__ == "__main__":
    main()
