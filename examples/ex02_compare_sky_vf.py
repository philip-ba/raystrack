#!/usr/bin/env python3
"""
ex02_compare_sky_vf

Loads the street canyon geometry, computes:
1) Regular view-factor matrix between scene meshes and derives a "sky" VF
   as 1 - sum(scene hits) per emitter.
2) Directional (Radiance-style) Tregenza sky VF via view_factor_to_tregenza_sky
   (merged "Sky").

Prints a side-by-side comparison for each emitter.

Inputs and tunables:
- Requires `street_canyon.json` from ex00; the script augments it with an
  artificial "infinite" ground plane built from the scene bounds (`margin`
  controls the extra extent).
- Shared sampling `settings` define `samples`, `rays`, `bvh`, `device`,
  and optional CUDA optimizations (`cuda_async`, `gpu_raygen`).
- Scene VF call: `max_iters`, `tol`, `tol_mode`, and reciprocity flags keep the
  solver on a single progressive pass suited for comparison.
- Sky conversion: `view_factor_to_tregenza_sky` uses the same sampling and
  returns a merged "Sky" result.
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
    from raystrack.io import load_meshes_json
    from raystrack import view_factor_matrix, view_factor_to_tregenza_sky
    from raystrack.params import MatrixParams, SkyParams

    here = Path(__file__).resolve().parent
    geom = here / "street_canyon.json"
    if not geom.exists():
        raise FileNotFoundError(
            f"Geometry not found at {geom}. Run ex00_street_canyon_geometry.py first.")

    meshes = load_meshes_json(str(geom))

    # Add a very large ground mesh named "infite_ground" to catch downward rays
    # that miss the local road. We size it from the scene bounds with a large margin
    # and place it slightly below the lowest z to avoid coplanar intersections.
    import numpy as np
    allV = np.concatenate([V for (_, V, _) in meshes], axis=0)
    min_xyz = allV.min(axis=0)
    max_xyz = allV.max(axis=0)
    extent = np.maximum(max_xyz - min_xyz, 1.0)
    # Use a very large ground extent to catch shallow downward rays
    margin = float(100)
    x0 = float(min_xyz[0] - margin)
    x1 = float(max_xyz[0] + margin)
    y0 = float(min_xyz[1] - margin)
    y1 = float(max_xyz[1] + margin)
    z_plane = float(min_xyz[2]) - 1e-3
    Vg = np.asarray(
        [
            [x0, y0, z_plane],
            [x1, y0, z_plane],
            [x1, y1, z_plane],
            [x0, y1, z_plane],
        ],
        dtype=np.float32,
    )
    # +Z normal (front-facing up): two triangles (0,1,2) and (0,2,3)
    Fg = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    meshes_with_ground = list(meshes)
    meshes_with_ground.append(("infite_ground", Vg, Fg))

    # Shared settings for a stable comparison
    settings = dict(
        samples=16,
        rays=128,
        bvh="auto",
        device="auto",
        cuda_async=True,
        gpu_raygen=True,
    )

    matrix_params = MatrixParams(
        **settings,
        max_iters=50,  # single pass (directional sky is single pass too)
        tol=1e-4,
        reciprocity=False,
        tol_mode="stderr",
        min_iters=1,
        enforce_reciprocity_rowsum=False,
    )

    sky_params = SkyParams(
        **settings,
        max_iters=50,
        tol=1e-4,
        seed=20,
        discrete=False,
    )

    print("Computing regular VF matrix (all-to-all including ground)...")
    VF_scene = view_factor_matrix(meshes_with_ground, params=matrix_params)

    # 1 - sum(row) gives the fraction of rays that missed all scene geometry
    # (includes sky and any downward misses beyond geometry extents)
    derived_sky = {}
    for emitter, row in VF_scene.items():
        total_scene = sum(float(v) for v in row.values())
        sky_vf = max(0.0, 1.0 - total_scene)
        derived_sky[emitter] = sky_vf

    print("Computing directional Tregenza sky VF (no sky geometry, merged)...")
    # Exclude the infinite ground mesh from sky VF (not an emitter, not considered)
    meshes_no_ground = [m for m in meshes_with_ground if m[0] != "infite_ground"]
    VF_sky = view_factor_to_tregenza_sky(meshes_no_ground, params=sky_params)

    # Compare per emitter
    names = [name for name, _, _ in meshes_with_ground if name != "infite_ground"]
    print("\nEmitter                            VF(1-sum(scene))    VF(dir-sky)     Diff")
    print("-" * 78)
    for name in names:
        v1 = derived_sky.get(name, 0.0)
        v2 = VF_sky.get(name, {}).get("Sky", 0.0)
        print(f"{name:32s}  {v1:>10.6f}          {v2:>10.6f}   {v2 - v1:+.6f}")


if __name__ == "__main__":
    main()
    from raystrack.utils.helpers import hold_console_open
    hold_console_open()
