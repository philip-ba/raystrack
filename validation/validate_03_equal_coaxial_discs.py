#!/usr/bin/env python3
from __future__ import annotations

import math

from common_validation import (
    disk_xy,
    row_front_to,
    run_raystrack,
    write_case_result,
)


def analytical_equal_discs(R: float, H: float) -> float:
    # Radiation View factors.pdf, "Equal discs":
    # F = 1 + (1 - sqrt(1 + 4 r^2)) / (2 r^2), r = R/H.
    r = R / H
    return 1.0 + (1.0 - math.sqrt(1.0 + 4.0 * r * r)) / (2.0 * r * r)


def main() -> None:
    R = 1.0
    H = 1.0
    segments = 256
    samples = 16
    rays = 512
    min_iters = 40
    max_iters = 500
    tolerance = 1.0e-4

    meshes = [
        disk_xy("disc_1", R, 0.0, segments=segments, normal=+1),
        disk_xy("disc_2", R, H, segments=segments, normal=-1),
    ]
    analytical = analytical_equal_discs(R, H)
    run = run_raystrack(
        meshes,
        samples=samples,
        rays=rays,
        min_iters=min_iters,
        max_iters=max_iters,
    )
    raystrack = row_front_to(run.vf["disc_1"], "disc_2")

    path = write_case_result(
        "03_equal_coaxial_discs",
        description="Two identical coaxial discs with R/H=1, receiver approximated by triangles.",
        formula="F = 1 + (1 - sqrt(1+4r^2))/(2r^2)",
        analytical=analytical,
        raystrack=raystrack,
        tolerance=tolerance,
        settings={
            "R": R,
            "H": H,
            "segments": segments,
            "samples": samples,
            "rays": rays,
            "min_iters": min_iters,
            "max_iters": max_iters,
        },
        run=run,
    )
    print(path)


if __name__ == "__main__":
    main()
