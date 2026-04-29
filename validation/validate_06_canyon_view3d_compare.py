#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Dict

from common_validation import (
    RESULTS_ROOT,
    VIEW3D_REFERENCE_ROOT,
    build_canyon_meshes,
    max_abs_pair_diff,
    raystrack_base_matrix,
    run_raystrack,
    write_json,
)


REFERENCE_BASE_JSON = VIEW3D_REFERENCE_ROOT / "canyon_view3d_base.json"


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    meshes = build_canyon_meshes()
    names = [name for name, _, _ in meshes]

    samples = 8
    rays = 512
    min_iters = 40
    max_iters = 500
    tolerance = 1.0e-4

    run = run_raystrack(
        meshes,
        samples=samples,
        rays=rays,
        min_iters=min_iters,
        max_iters=max_iters,
        seed=31,
    )
    ray_base = raystrack_base_matrix(run.vf)
    write_json(RESULTS_ROOT / "06_canyon_view3d_raystrack_raw.json", run.vf)
    write_json(RESULTS_ROOT / "06_canyon_view3d_raystrack_base.json", ray_base)

    lines = [
        "case: 06_canyon_view3d",
        "description: Street canyon comparison between Raystrack and saved View3D reference.",
        "",
        "raystrack:",
        f"  samples: {samples}",
        f"  rays: {rays}",
        f"  min_iters: {min_iters}",
        f"  max_iters: {max_iters}",
        f"  raw_json: {RESULTS_ROOT / '06_canyon_view3d_raystrack_raw.json'}",
        f"  base_json: {RESULTS_ROOT / '06_canyon_view3d_raystrack_base.json'}",
        "",
        "convergence:",
        f"  tol_mode: {run.tol_mode}",
        f"  tol: {run.tol:.10f}",
        f"  min_iters: {run.min_iters}",
        f"  max_iters: {run.max_iters}",
        f"  converged_before_max: {run.converged_before_max}",
        "  iterations:",
    ]
    for name, iters in run.iterations.items():
        lines.append(f"    {name}: {iters}")
    lines.append("")

    if not REFERENCE_BASE_JSON.exists():
        lines.extend(
            [
                "view3d:",
                "  status: skipped",
                f"  reason: saved reference missing: {REFERENCE_BASE_JSON}",
                "",
                "comparison:",
                "  status: skipped because saved View3D reference was not found.",
                "  run: python validation\\generate_canyon_view3d_reference.py",
            ]
        )
        result_path = RESULTS_ROOT / "06_canyon_view3d.txt"
        result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(result_path)
        return

    view3d_base: Dict[str, Dict[str, float]] = json.loads(
        REFERENCE_BASE_JSON.read_text(encoding="utf-8")
    )
    max_diff, pair, ray_value, view3d_value = max_abs_pair_diff(
        ray_base,
        view3d_base,
        names=names,
    )
    passed = max_diff <= tolerance

    lines.extend(
        [
            "view3d:",
            "  status: loaded_reference",
            f"  base_json: {REFERENCE_BASE_JSON}",
            "",
            "comparison:",
            f"  max_abs_diff: {max_diff:.10f}",
            f"  max_pair: {pair[0]} -> {pair[1]}",
            f"  raystrack_at_max_pair: {ray_value:.10f}",
            f"  view3d_at_max_pair:    {view3d_value:.10f}",
            f"  tolerance: {tolerance:.10f}",
            f"  passed: {passed}",
        ]
    )
    result_path = RESULTS_ROOT / "06_canyon_view3d.txt"
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
