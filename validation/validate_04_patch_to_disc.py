#!/usr/bin/env python3
from __future__ import annotations

from common_validation import (
    disk_xy,
    rectangle_xy,
    row_front_to,
    run_raystrack,
    write_case_result,
)


def analytical_patch_to_disc(R: float, H: float) -> float:
    # Radiation View factors.pdf, "Patch to disc":
    # F = 1 / (1 + h^2), h = H/R.
    h = H / R
    return 1.0 / (1.0 + h * h)


def main() -> None:
    R = 1.0
    H = 1.0
    patch_side = 0.04
    segments = 256
    samples = 8
    rays = 1024
    min_iters = 40
    max_iters = 500
    tolerance = 1.0e-4

    meshes = [
        rectangle_xy("patch", patch_side, patch_side, 0.0, normal=+1),
        disk_xy("disc", R, H, segments=segments, normal=-1),
    ]
    analytical = analytical_patch_to_disc(R, H)
    run = run_raystrack(
        meshes,
        samples=samples,
        rays=rays,
        min_iters=min_iters,
        max_iters=max_iters,
    )
    raystrack = row_front_to(run.vf["patch"], "disc")

    path = write_case_result(
        "04_patch_to_disc",
        description="Small square patch approximating a differential patch to a parallel concentric disc.",
        formula="F = 1/(1+h^2), h=H/R",
        analytical=analytical,
        raystrack=raystrack,
        tolerance=tolerance,
        settings={
            "R": R,
            "H": H,
            "patch_side": patch_side,
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
