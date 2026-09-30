"""Compare ordinary cosine rays with targeted sampling for a rare receiver.

Run with ``PYTHONPATH=src python validation/benchmark_sampling.py``.
The reference uses the point-receiver limit for a 1 cm square at 1 m height.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np

from raystrack import MatrixParams, PreparedSolver, view_factor_matrix, view_factor_targeted


def square(name: str, z: float, half_width: float, downward: bool = False):
    h = half_width
    vertices = np.array([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]],
                        dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    if downward:
        faces = faces[:, ::-1].copy()
    return name, vertices, faces


def main():
    meshes = [square("emitter", 0.0, 1.0),
              square("tiny", 1.0, 0.005, downward=True)]
    prepared = PreparedSolver(meshes)
    grid = (np.arange(500) + 0.5) / 250.0 - 1.0
    x, y = np.meshgrid(grid, grid)
    reference = 0.01**2 / np.pi * np.mean(1.0 / (1.0 + x*x + y*y)**2)
    rays = 512
    seeds = range(20)

    # Warm Numba kernels so compilation does not enter the timings.
    with patch("raystrack.main._log"):
        view_factor_matrix(meshes, MatrixParams(samples=4, rays=32, seed=0,
                            device="cpu", bvh="off", min_iters=1,
                            max_iters=1), prepared=prepared)
    view_factor_targeted(meshes, "emitter", "tiny", samples=rays,
                         seed=0, prepared=prepared)

    for method in ("cosine", "targeted_random", "targeted_halton"):
        estimates = []
        started = time.perf_counter()
        for seed in seeds:
            if method == "cosine":
                with patch("raystrack.main._log"):
                    result = view_factor_matrix(
                        meshes,
                        MatrixParams(samples=4, rays=32, seed=seed,
                                     device="cpu", bvh="off", min_iters=1,
                                     max_iters=1),
                        prepared=prepared,
                    )
                value = result["emitter"].get("tiny_front", 0.0)
            else:
                result = view_factor_targeted(
                    meshes, "emitter", "tiny", samples=rays,
                    seed=seed, prepared=prepared,
                    sequence="shifted_halton" if method == "targeted_halton" else "random")
                value = result.get("tiny_front", 0.0)
            estimates.append(value)
        elapsed = time.perf_counter() - started
        error = np.mean(np.abs(np.asarray(estimates) - reference))
        print(f"{method:15s}  rays/seed={rays}  mean_abs_error={error:.6g}  "
              f"zero_estimates={sum(value == 0 for value in estimates)}/20  "
              f"time_for_20={elapsed:.3f}s")
    print(f"reference={reference:.8g}")

    # Same total ray budget: one 1,024-ray direction versus two 512-ray
    # directions, averaged by exchanged area.
    facing = [square("lower", 0.0, 1.0),
              square("upper", 1.0, 1.0, downward=True)]
    facing_prepared = PreparedSolver(facing)
    delta = (np.arange(1000) + 0.5) * 4.0 / 1000 - 2.0
    dx, dy = np.meshgrid(delta, delta)
    facing_reference = (np.sum((2.0 - np.abs(dx)) * (2.0 - np.abs(dy))
                               / (1.0 + dx*dx + dy*dy)**2)
                        * (4.0 / 1000)**2 / (4.0 * np.pi))
    for mode, rays_per_cell in (("shortcut", 64), ("bidirectional", 32)):
        estimates = []
        for seed in range(50):
            with patch("raystrack.main._log"):
                result = view_factor_matrix(
                    facing,
                    MatrixParams(samples=4, rays=rays_per_cell, seed=seed,
                                 bvh="off", device="cpu", min_iters=1,
                                 max_iters=1, reciprocity_mode=mode),
                    prepared=facing_prepared,
                )
            estimates.append(result["lower"].get("upper_front", 0.0))
        error = np.sqrt(np.mean((np.asarray(estimates) - facing_reference)**2))
        print(f"{mode:15s}  rays/pair=1024  pair_rmse={error:.6g}  "
              f"reference={facing_reference:.8g}")


if __name__ == "__main__":
    main()
