#!/usr/bin/env python3
"""Move a blocker, preview one stationary emitter, then refine the last frame.

Run from the repository root:
    python examples/ex06_dynamic_preview.py --device cpu --frames 5
    python examples/ex06_dynamic_preview.py --device vulkan --warmup

The portable backend requires the optional ``raystrack[portable-gpu]`` extra.
Time limits are soft and cannot interrupt preparation or a running ray chunk.
This small command-line demo uses a fixed ray budget so cold kernel compilation
does not leave the first frame empty. An interactive application can additionally
pass ``max_time_ms=50`` and use ``session.submit(...)`` to supersede old frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

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
    from raystrack import MatrixParams, PreparedSolver, SkyParams
    from raystrack.preview import PreviewSession

    meshes = [square("emitter", 0.0, 1.0),
              square("receiver", 2.0, 1.0, downward=True),
              square("blocker", 1.0, 1.2)]
    common = dict(samples=4, rays=64, seed=7, device=args.device,
                  bvh="builtin", min_iters=2, max_iters=max(8, args.budget // 1024 + 1),
                  tol=0.0, ray_batch_size=1024, sampling_mode=args.sampling_mode)
    matrix_params = MatrixParams(**common, reciprocity=False)
    sky_params = SkyParams(**common)
    prepared = PreparedSolver(meshes, acceleration=args.acceleration)
    print(f"Requested device: {args.device}; acceleration: {args.acceleration}; selected emitter: emitter")
    print("The blocker moves away; stationary emitter-to-receiver visibility changes.")

    # Avoid the solver's optional separate log-console window in this CLI demo.
    with patch("raystrack.main._log"):
        with PreviewSession(prepared, matrix_params, sky_params) as session:
            if args.warmup:
                plan = session.warmup(repeats=1, max_probe_rays=min(args.budget, 4096))
                print(f"Warmup: backend={plan.backend}, batch={plan.ray_batch_size}, "
                      f"diagnostic_rays={plan.warmup_rays}, elapsed_ms={plan.elapsed_ms:.3f}")
            print("frame  version  blocker_x  receiver_vf  blocker_vf  sky_vf   rays  elapsed_ms")
            positions = np.linspace(0.0, 3.0, args.frames) if args.frames > 1 else [0.0]
            for frame, x in enumerate(positions):
                transform = np.eye(4)
                transform[0, 3] = x
                # Transforms are absolute relative to the blocker's local vertices.
                session.update_transform("blocker", transform)
                result = session.preview(emitter_names=["emitter"],
                                         max_total_rays=args.budget, max_time_ms=None)
                scene = result.scene.get("emitter", {})
                receiver = scene.get("receiver_front", 0.0)
                blocker = scene.get("blocker_back", 0.0)
                sky = result.sky.get("emitter", {}).get("Sky", 0.0)
                print(f"{frame:5d}  {result.scene_version:7d}  {x:9.3f}  {receiver:11.5f}  "
                      f"{blocker:10.5f}  {sky:6.4f}  {result.rays_used:5d}  {result.elapsed_ms:10.3f}")

            # The geometry is stationary now. Refinement resumes this version's
            # partial and completed replicates, tracing only additional rays.
            refined = session.refine(factor=2.0)
            receiver = refined.scene.get("emitter", {}).get("receiver_front", 0.0)
            print(f"Refined version {refined.scene_version}: receiver_vf={receiver:.6f}, "
                  f"additional_rays={refined.rays_used}, cumulative_rays={refined.cumulative_rays}, "
                  f"elapsed_ms={refined.elapsed_ms:.3f}, status={refined.status}")
            print("First-frame elapsed time can include kernel compilation and uploads.")


if __name__ == "__main__":
    main()
