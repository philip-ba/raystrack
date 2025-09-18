#!/usr/bin/env python3
"""
ex01_compute_vf

Loads the street canyon geometry from ex00 and computes a view-factor matrix.
Results are saved to vf_matrix.json in this folder.
"""
import os
import sys
from pathlib import Path
import json


def ensure_repo_on_path():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main():
    ensure_repo_on_path()
    from raystrack.io import load_meshes_json, save_vf_matrix_json
    from raystrack import view_factor_matrix

    here = Path(__file__).resolve().parent
    geom = here / "street_canyon.json"
    if not geom.exists():
        raise FileNotFoundError(
            f"Geometry not found at {geom}. Run ex00_street_canyon_geometry.py first.")

    meshes = load_meshes_json(str(geom))

    # Modest settings 
    params = dict(
        samples=16,     # grid per side
        rays=128,       # rays per cell
        seed=1,
        bvh="auto",
        device="auto",
        max_iters=200,
        tol=1e-4,
        tol_mode="stderr",
        min_iters=10,
        reciprocity=True,
        enforce_reciprocity_rowsum=True,
        cuda_async=True,
    )

    VF = view_factor_matrix(meshes, **params)
    out = here / "vf_matrix.json"
    save_path = save_vf_matrix_json(VF, str(out))
    print(f"Saved view-factor matrix to: {save_path}")


if __name__ == "__main__":
    main()

