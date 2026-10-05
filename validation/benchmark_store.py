"""Measure v2 full-load cost and storage size on a deterministic sparse fixture.

This is a synthetic storage fixture; accuracy and solve ray counts are unknown.
V2 loads immutable owned geometry and validates the full snapshot. It currently
has no selective on-disk row reader; Result.row selects rows after loading.
"""
from pathlib import Path
import json
import tempfile
import time
import numpy as np
from raystrack import Mesh, Scene, Result, SparseValues, Channel
from raystrack.io import save, load


def main():
    rng=np.random.default_rng(17)
    faces=np.column_stack((np.arange(1998),np.arange(1,1999),np.arange(2,2000))).astype(np.int32)
    scene=Scene.from_meshes({f"surface_{i:03d}":Mesh(rng.random((2000,3),dtype=np.float32),faces) for i in range(60)})
    channels=tuple(Channel("surface",sid,"front") for sid in scene.surface_ids)
    rows=np.repeat(np.arange(60),8)
    cols=np.array([(i+j)%60 for i in range(60) for j in range(1,9)])
    values=np.tile(np.arange(1,9)/100,60)
    result=Result(scene.surface_ids,channels,SparseValues(rows,cols,values,np.full(480,np.nan)),
                  np.ones(60,np.int8),rays_used=None,cumulative_rays=None,status="synthetic",converged=None)
    with tempfile.TemporaryDirectory(prefix="raystrack-store-benchmark-") as directory:
        started=time.perf_counter()
        path=Path(save(Path(directory)/"run.raystrack",scene,result))
        write_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter(); stored=load(path)
        load_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter()
        for _ in range(1000):
            stored.result.row("surface_059")
        row_us=(time.perf_counter()-started)*1000
        print(json.dumps({"triangles":119880,"sparse_entries":480,
            "bytes":sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
            "write_ms":write_ms,"full_load_ms":load_ms,"in_memory_row_us":row_us},indent=2))
        np.testing.assert_array_equal(stored.result.dense(),result.dense())


if __name__ == "__main__":
    main()
