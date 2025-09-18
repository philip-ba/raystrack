#!/usr/bin/env python3
"""
ex03_workflow

Demonstrates the high-level view_factor_workflow:
- Computes the regular scene view-factor matrix
- Computes Radiance-style Tregenza sky view factors (merged by default)
- Reconciles per-emitter rows so that sum(scene) == 1 - sky_total,
  redistributing large discrepancies to boundary ground surfaces when needed.

Outputs:
- Saves the reconciled scene VF to vf_scene_workflow.json
- Saves the sky VF to sky_vf_workflow.json
"""
import sys
from pathlib import Path


def ensure_repo_on_path():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main():
    ensure_repo_on_path()
    from raystrack.io import load_meshes_json, save_vf_matrix_json
    from raystrack import view_factor_outside_workflow

    here = Path(__file__).resolve().parent
    geom = here / "street_canyon.json"
    if not geom.exists():
        raise FileNotFoundError(
            f"Geometry not found at {geom}. Run ex00_street_canyon_geometry.py first.")

    meshes = load_meshes_json(str(geom))

    # Reasonable defaults for a quick-yet-stable run
    matrix_params = dict(
        samples=16,
        rays=128,
        seed=7,
        bvh="auto",
        device="auto",
        cuda_async=True,
        gpu_raygen=True,
        max_iters=25,
        tol=1e-4,
        tol_mode="stderr",  # matrix algorithm supports 'delta' and 'stderr'
        min_iters=5,
        enforce_reciprocity_rowsum=False,
    )
    sky_params = dict(
        samples=16,
        rays=128,
        seed=7,
        bvh="auto",
        device="auto",
        cuda_async=True,
        gpu_raygen=True,
        max_iters=25,
        tol=1e-4,          # stderr target for merged sky
        tol_mode="stderr", # use true stderr convergence for sky
        min_iters=5,
    )

    # Use merged sky for speed (set discrete=True for 145 Tregenza bins)
    vf_scene, sky_vf = view_factor_outside_workflow(
        meshes,
        matrix_params=matrix_params,
        sky_params=sky_params,
        discrete=False,
        threshold_multiplier=5.0,
    )

    # Print a short reconciliation summary
    print("Emitter                            sum(scene)        Sky        1-sum(scene)-Sky")
    print("-" * 86)
    for name, row in vf_scene.items():
        scene_sum = float(sum(float(v) for v in row.values()))
        sky_total = float(sky_vf.get(name, {}).get("Sky", 0.0))
        resid = 1.0 - scene_sum - sky_total
        print(f"{name:32s}  {scene_sum:>10.6f}   {sky_total:>10.6f}        {resid:+.6e}")

    # Save results next to the example
    out_scene = here / "vf_scene_workflow.json"
    out_sky = here / "sky_vf_workflow.json"
    save_vf_matrix_json(vf_scene, str(out_scene))
    save_vf_matrix_json(sky_vf, str(out_sky))
    print(f"Saved reconciled scene VF to: {out_scene}")
    print(f"Saved sky VF to: {out_sky}")


if __name__ == "__main__":
    main()

