#!/usr/bin/env python3
from __future__ import annotations

import math

from common_validation import (
    rectangle_xy,
    rectangle_yz,
    row_front_to,
    run_raystrack,
    write_case_result,
)


def analytical_square_to_adjacent_rectangle(H: float, W: float) -> float:
    # Radiation View factors.pdf, "Square plate to rectangular plate".
    # F = 1/4 + 1/pi [h atan(1/h) - h1 atan(1/h1) - 1/4 ln(h2)]
    # h = H/W, h1 = sqrt(1+h^2), h2 = h1^4 / (h^2(2+h^2)).
    h = H / W
    h1 = math.sqrt(1.0 + h * h)
    h2 = h1**4 / (h * h * (2.0 + h * h))
    return 0.25 + (h * math.atan(1.0 / h) - h1 * math.atan(1.0 / h1) - 0.25 * math.log(h2)) / math.pi


def main() -> None:
    W = 1.0
    H = 1.0
    samples = 32
    rays = 512
    min_iters = 40
    max_iters = 500
    tolerance = 1.0e-4

    # A horizontal square emits upward to an adjacent vertical rectangle.
    meshes = [
        rectangle_xy("square", W, W, 0.0, normal=+1, center=(W / 2.0, 0.0)),
        rectangle_yz("adjacent_rectangle", W, H, 0.0, normal=+1, y_center=0.0, z_min=0.0),
    ]
    analytical = analytical_square_to_adjacent_rectangle(H, W)
    run = run_raystrack(
        meshes,
        samples=samples,
        rays=rays,
        min_iters=min_iters,
        max_iters=max_iters,
    )
    raystrack = row_front_to(run.vf["square"], "adjacent_rectangle")

    path = write_case_result(
        "05_perpendicular_square_rectangle",
        description="Square plate to adjacent perpendicular rectangle with H/W=1.",
        formula="F = 1/4 + [h atan(1/h) - h1 atan(1/h1) - 1/4 ln(h2)]/pi",
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
