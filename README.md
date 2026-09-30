# Raystrack

<p align="left">
  <img src="raystrack_icon.svg" alt="Raystrack icon" width="160">
</p>

Lightweight Monte-Carlo view-factor solver for polygonal meshes.

Raystrack computes radiative view factors F(i->j) between triangulated surfaces
using quasi-Monte-Carlo ray tracing. It runs on CPU, can leverage Numba/CUDA on
NVIDIA GPUs when available, and optionally accelerates ray intersection with a
BVH. The repository also ships a pure-Python API you can use outside Rhino.

## Features
- Efficient Monte-Carlo view factors: front/back hits, optional reciprocity
- CPU and optional CUDA GPU backends (Numba)
- Optional BVH acceleration structures
- Python API plus Rhino 8 / Grasshopper components

## Installation

### Python package (PyPI)
Use Raystrack as a normal Python package outside Rhino or Grasshopper.

Install the latest published release from PyPI:
```
pip install raystrack
```

From a local clone of this repository:
```
pip install .
```

Or from an absolute path:
```
pip install /path/to/raystrack
```

Requirements: Python 3.9+, `numpy`, `numba`. CUDA acceleration is enabled
automatically when `numba.cuda` detects a compatible GPU.

### Rhino 8 / Grasshopper
Raystrack is also available through the Rhino 8 Package Manager for use in
Rhino and Grasshopper.

In Rhino 8, run the `PackageManager` command, search for `Raystrack`, install
the package, and restart Rhino if prompted.

## Examples

All examples live in `examples/`. Start by running `ex00_street_canyon_geometry`
to generate `street_canyon.json`; subsequent scripts expect that file. Each
Python example now documents its own inputs and parameters inline, so open the
scripts directly for usage guidance and tunable options.

## Quick start (Python)
```python
import numpy as np
from raystrack import view_factor_matrix, MatrixParams

# Each mesh is a tuple: (name: str, V: (N,3) float32, F: (M,3) int32)
V_a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
F_a = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)  # two triangles

V_b = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
F_b = F_a.copy()

meshes = [
    ("A", V_a, F_a),
    ("B", V_b, F_b),
]

params = MatrixParams(
    samples=256,   # sampling density per unit area (QMC grid)
    rays=256,      # rays per cell
    bvh="builtin", # optional BVH acceleration (auto|off|builtin)
    reciprocity=True,
)

res = view_factor_matrix(meshes, params=params)

print(res["A"])  # e.g. {"B_front": 0.5, "B_back": 0.0, ...}
```

To persist a matrix without directional suffixes in the saved JSON:
```python
from raystrack import save_vf_matrix_json

save_vf_matrix_json(res, "vf_matrix.json", strip_dir=True)
```
This collapses receiver keys like `"B_front"` and `"B_back"` into `"B"` and
sums both values per sender row.

## Saving a complete run

The `.raystrack` directory format keeps meshes, parameters, and results together.
It stores numeric data in NumPy chunks, so you can load one mesh or a few sender
rows without reading the entire run. The JSON functions above remain available.

```text
case.raystrack/
  manifest.json                 # format version, names, parameters, metadata
  geometry/000000/vertices-0000.npy
  geometry/000000/faces-0000.npy
  results/scene/000000/offsets.npy
  results/scene/000000/columns.npy
  results/scene/000000/values.npy
  results/sky/...
  results/rest/...
```

Each result chunk stores up to 256 sender rows with indexed receiver names.
Explicit zero values and directional names are preserved. Chunks are uncompressed
for fast selective reads; the default target size is 4 MiB.

```python
from raystrack import open_store, save_run

save_run("case.raystrack", meshes=meshes, matrix_params=params, scene=res)

with open_store("case.raystrack") as store:
    one_mesh = store.load_meshes(names=["A"])
    one_row = store.load_result("scene", senders=["A"])
    saved_params = store.matrix_params
```

For incremental writes, open a new store with `mode="w"`, call `add_mesh` for
each mesh and `append_result_rows(kind, rows)` as rows become available, then
call `finalize()`. The result kind is `"scene"`, `"sky"`, or `"rest"`.
`flush()` commits buffered rows earlier;
closing a store without finalizing leaves it open for `mode="a"`. Readers see
only committed chunks. `iter_mesh_chunks(name, "vertices" | "faces")` and
`iter_result_rows(kind, senders=...)` provide streaming reads. New stores refuse
to overwrite existing directories.

## Parameter presets
Raystrack uses two parameter containers to keep configuration consistent:
- `MatrixParams`: controls the scene-to-scene view-factor solve (sampling, BVH,
  device selection, convergence tolerances, and reciprocity enforcement).
- `SkyParams`: controls the sky view-factor solve (sampling, device selection,
  and convergence tolerances, plus `discrete=True` for 145 sky patches or
  `False` for a single merged `"Sky"` output).

Typical usage:
```python
from raystrack import MatrixParams, SkyParams, view_factor_matrix, view_factor_to_tregenza_sky

matrix_params = MatrixParams(samples=32, rays=256, reciprocity=True, flip_faces=False)
sky_params = SkyParams(samples=32, rays=256)

vf_scene = view_factor_matrix(meshes, params=matrix_params)
vf_sky = view_factor_to_tregenza_sky(meshes, params=sky_params)
```

## Tracing performance

For a mesh with area `A`, one iteration emits approximately
`max(4, ceil(sqrt(A * samples))) ** 2 * rays` rays. This work repeats until the
convergence tolerance is met or `max_iters` is reached. A very small `tol` can
therefore dominate runtime even on a scene with few triangles.

The [outside workflow example](examples/ex03_workflow.py) uses a moderate
preview budget. Its matrix and sky sampling settings match, so the workflow
shares traced rays. Keep those settings aligned when changing the example;
otherwise the two results are computed in separate passes. Use `bvh="auto"`
for mixed scene sizes, and benchmark `device="cpu"` against `device="gpu"` on
your hardware. For repeated solves of the same geometry, pass one shared
`PreparedSolver` to reuse prepared geometry and ray tables. The first CPU run
may also spend time compiling Numba kernels.

For receivers that are rarely hit, a zero replicate variance can make the
adaptive solver stop before it has sampled enough rays. Set
`min_total_rays` per emitter when those receivers matter; `max_iters` remains
the hard cap. For example, `MatrixParams(min_total_rays=262_144)` prevents an
earlier convergence stop until that many rays have been traced. The same option
is available on `SkyParams`. This is a sampling floor, not a confidence bound.

For a selected rare receiver, direct area-pair sampling uses each sample to
connect a point on the sender to a point on that receiver. It weights the
sample by the cosine and distance geometry term and checks the connection for
occluders:

```python
from raystrack import view_factor_targeted

row = view_factor_targeted(meshes, "sender", "small_receiver",
                           samples=8192, seed=7)
# {'small_receiver_front': ..., 'small_receiver_back': ...} when visible
```

This CPU method estimates one pair, so it is most useful for a few important
small receivers. It does not fill the rest of the matrix or the sky result.
Its visibility check scans scene triangles for each connection, so large
meshes can favor the built-in BVH cosine tracer.
It defaults to independently shifted Halton points; use `sequence="random"`
to compare ordinary pseudorandom sampling. In a reproducible 512-sample
small-receiver benchmark, the cosine tracer returned zero for all 20 seeds,
while targeted shifted Halton sampling returned a nonzero estimate for all
20 and reduced mean absolute error from `1.39e-5` to `5.50e-8`. Run
`python validation/benchmark_sampling.py` to reproduce the comparison.

When both surfaces face each other, `MatrixParams(reciprocity_mode="bidirectional")`
traces both directions and averages the two front-side exchanged-area estimates.
The default `"shortcut"` traces one direction and applies area reciprocity.
Back-side hits remain direct estimates in bidirectional mode. At the same
1,024-ray pair budget in the benchmark, bidirectional sampling reduced pair
RMSE from about `0.00935` to `0.00874`; its benefit depends on geometry and
sample budget. The shared scene and sky workflow uses one ray set for both
results in either mode. Its CUDA path also reuses ray and result buffers across
emitters.

## Author
Philip Balizki <philip@metis.earth>

## Citation / Attribution
If you use Raystrack in a project, publication, report, tool, or derived work,
please cite it as:

Balizki, P. (2026). Raystrack (Version 1.0.2) [Computer software]. GitHub. https://github.com/philip-ba/raystrack

For informal attribution:

Raystrack by Philip Balizki

## License
MIT - see `LICENSE`.
