#!/usr/bin/env python3
"""
ex04_inside_enclosure

Constructs a simple 6-face box (unit cube) with outward-facing normals,
then computes the inside view-factor matrix by enabling `flip_faces=True`.

Results are printed and saved to `inside_vf_matrix.json` in this folder.

Inputs and parameters to know:
- Geometry is created on the fly by `make_box_unit_cube`; edit the vertex
  positions there if you want different dimensions or orientations.
- `flip_faces=True` in the solver call reverses emission so rays travel inward.
- Sampling and convergence controls mirror earlier examples: adjust `samples`,
  `rays`, `seed`, `max_iters`, `tol`, `tol_mode`, and `min_iters` to trade cost
  versus accuracy. Reciprocity enforcement is left disabled to preserve the raw
  view factors of the closed cavity.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


def ensure_repo_on_path():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def make_box_unit_cube():
    """Return 6 meshes (name, V, F) forming a closed unit cube [0,1]^3.

    Faces are quads triangulated into two triangles, with OUTWARD normals.
    Enabling `flip_faces=True` when computing view factors will then emit
    rays inward, suitable for inside-enclosure view factors.
    """
    def oriented_quad(name, p0, p1, p2, p3, desired_normal):
        V = np.array([p0, p1, p2, p3], dtype=np.float32)
        # Base triangulation
        F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        # Ensure the first triangle's normal points approximately to desired_normal
        a, b, c = V[0], V[1], V[2]
        n = np.cross(b - a, c - a)
        if np.dot(n, desired_normal) < 0.0:
            F = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
        return (name, V, F)

    # Cube corners
    O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    Y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Define faces with outward orientation (match desired outward normals)
    bottom = oriented_quad("Bottom", O, X, X + Y, Y, desired_normal=np.array([0.0, 0.0, -1.0], np.float32))
    top    = oriented_quad("Top",    O + Z, Y + Z, X + Y + Z, X + Z, desired_normal=np.array([0.0, 0.0, +1.0], np.float32))
    front  = oriented_quad("Front",  O, O + Z, X + Z, X, desired_normal=np.array([0.0, -1.0, 0.0], np.float32))
    back   = oriented_quad("Back",   Y, X + Y, X + Y + Z, Y + Z, desired_normal=np.array([0.0, +1.0, 0.0], np.float32))
    left   = oriented_quad("Left",   O, Y, Y + Z, O + Z, desired_normal=np.array([-1.0, 0.0, 0.0], np.float32))
    right  = oriented_quad("Right",  X, X + Z, X + Y + Z, X + Y, desired_normal=np.array([+1.0, 0.0, 0.0], np.float32))

    return [bottom, top, front, back, left, right]


def main():
    ensure_repo_on_path()
    from raystrack import view_factor_matrix
    from raystrack.io import save_vf_matrix_json
    from raystrack.params import MatrixParams
    from raystrack.utils.geometry import flip_meshes

    meshes = make_box_unit_cube()
    meshes = flip_meshes(meshes)

    # Faces are flipped so emitters shoot rays inward (inside the enclosure)
    params = MatrixParams(
        samples=16,                 # modest density for a quick example
        rays=128,
        seed=42,
        bvh="auto",
        device="auto",
        reciprocity=False,         
        enforce_reciprocity_rowsum=False,  
        max_iters=1000,
        tol=1e-3,
        tol_mode="stderr",
        min_iters=10,
        cuda_async=True,
    )

    VF = view_factor_matrix(meshes, params=params)

    # Print a compact summary per face
    for name in VF:
        row = VF[name]
        row_sum = float(sum(row.values()))
        print(f"{name}: receivers={len(row):2d}, sum={row_sum:.6f}")

    # Save to JSON
    here = Path(__file__).resolve().parent
    out_path = here / "inside_vf_matrix.json"
    save_path = save_vf_matrix_json(VF, str(out_path))
    print(f"Saved inside view-factor matrix to: {save_path}")


if __name__ == "__main__":
    main()
    from raystrack.utils.helpers import hold_console_open
    hold_console_open()
