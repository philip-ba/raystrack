#!/usr/bin/env python3
from __future__ import annotations

import math

from common_validation import (
    rectangle_xy,
    row_front_to,
    run_raystrack,
    write_case_result,
)


def analytical_equal_rectangles(W1: float, W2: float, H: float) -> float:
    # Radiation View factors.pdf, "Equal rectangular plates".
    x = W1 / H
    y = W2 / H
    x1 = math.sqrt(1.0 + x * x)
    y1 = math.sqrt(1.0 + y * y)
    term_log = math.log((x1 * x1 * y1 * y1) / (x1 * x1 + y1 * y1 - 1.0))
    term_x = 2.0 * x * (y1 * math.atan(x / y1) - math.atan(x))
    term_y = 2.0 * y * (x1 * math.atan(y / x1) - math.atan(y))
    return (term_log + term_x + term_y) / (math.pi * x * y)


def main() -> None:
    W1 = 2.0
    W2 = 1.0
    H = 1.0
    samples = 16
    rays = 512
    min_iters = 40
    max_iters = 500
    tolerance = 1.0e-4

    meshes = [
        rectangle_xy("plate_1", W1, W2, 0.0, normal=+1),
        rectangle_xy("plate_2", W1, W2, H, normal=-1),
    ]
    analytical = analytical_equal_rectangles(W1, W2, H)
    run = run_raystrack(
        meshes,
        samples=samples,
        rays=rays,
        min_iters=min_iters,
        max_iters=max_iters,
    )
    raystrack = row_front_to(run.vf["plate_1"], "plate_2")

    path = write_case_result(
        "02_parallel_equal_rectangle",
        description="Two identical parallel rectangular plates with W1/H=2 and W2/H=1.",
        formula="PDF equal-rectangles closed form with x=W1/H, y=W2/H.",
        analytical=analytical,
        raystrack=raystrack,
        tolerance=tolerance,
        settings={
            "W1": W1,
            "W2": W2,
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
