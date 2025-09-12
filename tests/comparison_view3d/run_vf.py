import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from raystrack.main import view_factor_matrix
from raystrack.io import save_vf_matrix_json


def load_meshes_flexible(path: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Load meshes from a JSON file with either {vertices,faces} or {V,F} keys.

    Returns list of (name, V(float32[N,3]), F(int32[M,3])).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "meshes" not in data or not isinstance(data["meshes"], list):
        raise ValueError("Expected an object with a 'meshes' list")
    out = []
    for i, m in enumerate(data["meshes"]):
        if not isinstance(m, dict):
            raise ValueError(f"Entry {i} is not an object")
        name = m.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Entry {i}: missing or invalid 'name'")
        V = m.get("vertices") if "vertices" in m else m.get("V")
        F = m.get("faces") if "faces" in m else m.get("F")
        if V is None or F is None:
            raise ValueError(f"Entry {i} ('{name}') missing vertices/faces (or V/F)")
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"Entry {i} ('{name}'): V must have shape (N,3)")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"Entry {i} ('{name}'): F must have shape (M,3)")
        out.append((name, V, F))
    return out


def main():
    here = Path(__file__).resolve().parent
    # Default geometry file name (can be adjusted)
    geom = here / "philip_test2_geometry.json"
    if not geom.exists():
        # Fallback: first *.json in folder
        candidates = sorted(here.glob("*.json"))
        if not candidates:
            raise FileNotFoundError("No geometry JSON found in the folder")
        geom = candidates[0]

    meshes = load_meshes_flexible(geom)

    # Parameters for fastest CUDA path with reciprocity and stderr stopping
    params = dict(
        samples=16,
        rays=256,
        use_bvh=True,
        max_iters=1000,
        tol=1e-4,
        tol_mode="stderr",
        min_iters=5,
        reciprocity=True,
        cuda_async=True,
        gpu_threads=256,
        seed=1000,
    )

    # Try on-GPU ray generation if available flag is supported
    try:
        # Newer builds support 'gpu_raygen'
        res = view_factor_matrix(meshes, gpu_raygen=True, **params)
    except TypeError:
        # Fallback if the current version doesn't expose gpu_raygen
        res = view_factor_matrix(meshes, **params)

    out_path = here / "vf_matrix.json"
    save_vf_matrix_json(res, str(out_path))
    print(f"Saved view-factor matrix to: {out_path}")


if __name__ == "__main__":
    main()

