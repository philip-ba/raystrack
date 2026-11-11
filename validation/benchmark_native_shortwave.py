"""Time cold, steady and edited shortwave solves on a fixed canopy scene."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import platform
import statistics
import time

import numpy as np

from raystrack.model import Mesh, Scene
from raystrack.shortwave import ShortwaveSolver, ThinDiffuse


def canopy(grid):
    tiles = np.indices((grid, grid)).reshape(2, -1).T - grid / 2
    corners = np.array([[0, 0], [0.75, 0], [0.75, 0.75], [0, 0.75]])
    xy = (tiles[:, None, :] + corners).reshape(-1, 2)
    vertices = np.column_stack((xy, np.ones(len(xy))))
    faces = (np.arange(len(tiles))[:, None, None] * 4 + [[0, 1, 2], [0, 2, 3]]).reshape(-1, 3)
    return Scene.from_meshes({"canopy": Mesh(vertices, faces)})


def run(backend, grid, samples, repeats):
    scene = canopy(grid)
    rng = np.random.default_rng(11)
    points = np.column_stack((rng.uniform(-grid / 4, grid / 4, (64, 2)), np.repeat([0.0, 2.0], 32)))
    normals = np.column_stack((np.zeros((64, 2)), np.repeat([1.0, -1.0], 32)))
    angle = np.linspace(-np.pi / 3, np.pi / 3, 24)
    sun = np.column_stack((np.sin(angle), np.zeros(24), np.cos(angle)))
    dni = 700 * sun[:, 2]

    with ShortwaveSolver(scene, materials={"canopy": ThinDiffuse(0.35, 0.1)}, backend=backend) as solver:
        def timed_solve():
            start = time.perf_counter()
            result = solver.solve(points, normals, sun, dni=dni, dhi=100, samples=samples, seed=11)
            return result.irradiance, 1000 * (time.perf_counter() - start)

        initial, cold_ms = timed_solve()
        steady_ms = [timed_solve()[1] for _ in range(repeats)]
        move = np.eye(4)
        move[0, 3] = 0.375
        scene.update_transform("canopy", move)
        edited, edit_ms = timed_solve()
    return initial, edited, {
        "cold_ms": cold_ms,
        "steady_ms": steady_ms,
        "steady_median_ms": statistics.median(steady_ms),
        "edited_ms": edit_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=128, help="Canopy tiles per side")
    parser.add_argument("--samples", type=int, default=1024, help="Cosine samples per sensor")
    parser.add_argument("--repeats", type=int, default=3, help="Number of steady solves")
    parser.add_argument("--embree-dll-dir", help="Embree runtime directory on Windows")
    args = parser.parse_args()
    if min(args.grid, args.samples, args.repeats) < 1:
        parser.error("grid, samples, and repeats must be positive")
    if args.embree_dll_dir and os.name != "nt":
        parser.error("--embree-dll-dir applies to Windows only")

    dll_scope = os.add_dll_directory(args.embree_dll_dir) if args.embree_dll_dir else nullcontext()
    with dll_scope:
        embree, embree_edited, cpu = run("embree", args.grid, args.samples, args.repeats)
        vulkan, vulkan_edited, gpu = run("vulkan-rt", args.grid, args.samples, args.repeats)
    print(json.dumps({
        "platform": platform.platform(),
        "triangles": 2 * args.grid ** 2,
        "sensors": 64,
        "timesteps": 24,
        "samples_per_sensor": args.samples,
        "seed": 11,
        "embree": cpu,
        "vulkan_rt": gpu,
        "initial_max_abs_error_w_m2": float(np.max(np.abs(embree - vulkan))),
        "edited_max_abs_error_w_m2": float(np.max(np.abs(embree_edited - vulkan_edited))),
    }, indent=2))


if __name__ == "__main__":
    main()
