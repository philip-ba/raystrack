#!/usr/bin/env python3
"""
ex05_prepared_seed_compare

Uses the street canyon geometry from ex00 and runs the matrix solver five times
with one shared PreparedSolver while varying only the random seed.

Purpose:
- demonstrate what PreparedSolver reuses across repeated solves
- show that changing only `seed` still works with the same prepared state
- print a compact comparison of representative VF entries between seeds

Notes:
- The prepared state is tied to the mesh set. Reuse it only for the same
  geometry.
- Changing `seed` does not invalidate the prepared caches, so this is the
  best-case reuse scenario.
- This example keeps the sample budget fixed (`min_iters == max_iters`) so the
  seed-to-seed comparison uses the same number of iterations each time.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple


def ensure_repo_on_path() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def flatten_matrix(vf: Dict[str, Dict[str, float]]) -> Dict[Tuple[str, str], float]:
    flat: Dict[Tuple[str, str], float] = {}
    for sender, row in vf.items():
        for receiver, value in row.items():
            flat[(sender, receiver)] = float(value)
    return flat


def mean_abs_diff(
    current: Dict[str, Dict[str, float]],
    reference: Dict[str, Dict[str, float]],
) -> float:
    flat_cur = flatten_matrix(current)
    flat_ref = flatten_matrix(reference)
    keys = set(flat_cur) | set(flat_ref)
    if not keys:
        return 0.0
    return sum(abs(flat_cur.get(key, 0.0) - flat_ref.get(key, 0.0)) for key in keys) / float(len(keys))


def main() -> None:
    ensure_repo_on_path()
    from numba.core.errors import NumbaPerformanceWarning
    import raystrack.main as raystrack_main
    from raystrack import PreparedSolver, view_factor_matrix
    from raystrack.io import load_meshes_json
    from raystrack.params import MatrixParams

    warnings.simplefilter("ignore", NumbaPerformanceWarning)

    # Keep this example output focused on the final seed comparison table.
    raystrack_main._log = lambda msg: None

    here = Path(__file__).resolve().parent
    geom = here / "street_canyon.json"
    if not geom.exists():
        raise FileNotFoundError(
            f"Geometry not found at {geom}. Run ex00_street_canyon_geometry.py first."
        )

    meshes = load_meshes_json(str(geom))
    prepared = PreparedSolver(meshes)
    seeds = [1, 2, 3, 4, 5]

    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    runtimes: Dict[int, float] = {}

    print("Running 5 solves with one shared PreparedSolver")
    print(f"Meshes: {len(meshes)}")
    print(f"Seeds: {seeds}")
    print()

    for seed in seeds:
        params = MatrixParams(
            samples=12,
            rays=96,
            seed=seed,
            bvh="builtin",
            device="auto",
            cuda_async=True,
            gpu_raygen=True,
            max_iters=10,
            min_iters=10,
            convergence_interval=10,
            tol=0.0,
            tol_mode="stderr",
            reciprocity=False,
            enforce_reciprocity_rowsum=False,
            flip_faces=False,
        )

        t0 = time.perf_counter()
        results[seed] = view_factor_matrix(meshes, params=params, prepared=prepared)
        runtimes[seed] = time.perf_counter() - t0

    baseline_seed = seeds[0]
    baseline = results[baseline_seed]

    print("Comparison against baseline seed")
    print(
        "Seed   Runtime[s]   Road->East0_f   Road->West0_f   East0->Road_f   Road row sum   Mean |delta| vs seed 1"
    )
    print("-" * 110)

    for seed in seeds:
        vf = results[seed]
        road_row = vf.get("road", {})
        east0_row = vf.get("east_side_0", {})
        road_to_east0 = float(road_row.get("east_side_0_front", 0.0))
        road_to_west0 = float(road_row.get("west_side_0_front", 0.0))
        east0_to_road = float(east0_row.get("road_front", 0.0))
        road_sum = float(sum(float(v) for v in road_row.values()))
        mad = mean_abs_diff(vf, baseline)

        print(
            f"{seed:>4d}   "
            f"{runtimes[seed]:>10.3f}   "
            f"{road_to_east0:>13.6f}   "
            f"{road_to_west0:>13.6f}   "
            f"{east0_to_road:>13.6f}   "
            f"{road_sum:>12.6f}   "
            f"{mad:>22.6e}"
        )


if __name__ == "__main__":
    main()
    from raystrack.utils.helpers import hold_console_open

    hold_console_open()
