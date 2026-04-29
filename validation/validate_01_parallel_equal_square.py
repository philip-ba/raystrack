#!/usr/bin/env python3
from __future__ import annotations

import math

from common_validation import (
    rectangle_xy,
    row_front_to,
    run_raystrack,
    write_case_result,
)


def analytical_equal_square(W: float, H: float) -> float:
    # Radiation View factors.pdf, "Equal square plates":
    # F = 1/(pi w^2) * [ln(x^4/(1+2w^2)) + 4 w y],
    # x = sqrt(1+w^2), y = x atan(w/x) - atan(w), w = W/H.
    w = W / H
    x = math.sqrt(1.0 + w * w)
    y = x * math.atan(w / x) - math.atan(w)
    return (math.log(x**4 / (1.0 + 2.0 * w * w)) + 4.0 * w * y) / (math.pi * w * w)


def main() -> None:
    W = 1.0
    H = 1.0
    samples = 32
    rays = 1024
    min_iters = 40
    max_iters = 500
    tolerance = 1.0e-4

    meshes = [
        rectangle_xy("plate_1", W, W, 0.0, normal=+1),
        rectangle_xy("plate_2", W, W, H, normal=-1),
    ]
    analytical = analytical_equal_square(W, H)
    run = run_raystrack(
        meshes,
        samples=samples,
        rays=rays,
        min_iters=min_iters,
        max_iters=max_iters,
    )
    raystrack = row_front_to(run.vf["plate_1"], "plate_2")

    path = write_case_result(
        "01_parallel_equal_square",
        description="Two identical parallel square plates with W/H=1.",
        formula="F = [ln(x^4/(1+2w^2)) + 4 w (x atan(w/x)-atan(w))] / (pi w^2)",
        analytical=analytical,
        raystrack=raystrack,
        tolerance=tolerance,
        settings={
            "W": W,
            "H": H,
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
