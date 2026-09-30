"""Compare the chunked store with JSON on a deterministic, sparse workload."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from raystrack import open_store, save_meshes_json, save_run, save_vf_matrix_json


def main() -> None:
    rng = np.random.default_rng(17)
    meshes = []
    faces = np.column_stack((np.arange(1998), np.arange(1, 1999), np.arange(2, 2000)))
    for i in range(60):
        vertices = rng.random((2000, 3), dtype=np.float32)
        meshes.append((f"surface_{i:03d}", vertices, faces.astype(np.int32)))
    scene = {
        f"surface_{i:03d}": {f"surface_{(i + j) % 60:03d}_front": j / 100 for j in range(1, 9)}
        for i in range(60)
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        saved = Path(save_run(root / "run.raystrack", meshes=meshes, scene=scene))
        geometry_json = Path(save_meshes_json(meshes, str(root / "geometry.json")))
        result_json = Path(save_vf_matrix_json(scene, str(root / "scene.json")))
        store_bytes = sum(path.stat().st_size for path in saved.rglob("*") if path.is_file())
        json_bytes = geometry_json.stat().st_size + result_json.stat().st_size
        started = time.perf_counter()
        with open_store(saved) as store:
            for _ in range(100):
                store.load_result("scene", ["surface_059"])
        selected_ms = (time.perf_counter() - started) * 10
        print(json.dumps({"store_bytes": store_bytes, "json_bytes": json_bytes,
                          "store_vs_json": round(store_bytes / json_bytes, 3),
                          "selected_row_ms": round(selected_ms, 3)}, indent=2))
        assert store_bytes < json_bytes


if __name__ == "__main__":
    main()
