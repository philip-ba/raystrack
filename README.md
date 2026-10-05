# Raystrack

<p align="left">
  <img src="raystrack_icon.svg" alt="Raystrack icon" width="160">
</p>

Lightweight Monte-Carlo view-factor solver for polygonal meshes.

Raystrack computes radiative view factors F(i->j) between triangulated surfaces
using quasi-Monte-Carlo ray tracing. It runs on CPU, supports Numba/CUDA on
NVIDIA GPUs, and offers optional Taichi GPU tracing through Vulkan or Metal.
Prepared scenes support geometry updates and BVH refitting for moving scenes.
The repository also ships a Python API you can use outside Rhino.

## Features
- Efficient Monte-Carlo view factors: front/back hits, optional reciprocity
- CPU, CUDA, and optional portable Vulkan/Metal GPU backends
- Optional BVH acceleration structures
- Persistent geometry updates, transforms, deformation, and topology changes
- Shared object instances with local BVHs and a refittable scene hierarchy
- Fair/adaptive previews, global budgets, cancellation, and resumable refinement
- Measured CPU/GPU selection and batch tuning
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

Requirements: Python 3.9–3.13, `numpy`, `numba`. CUDA acceleration is enabled
automatically when `numba.cuda` detects a compatible GPU. For a local checkout,
install the optional GPU backends with:

```sh
pip install ".[portable-gpu]"  # Taichi Vulkan/Metal
pip install ".[cuda]"          # separate NVIDIA CUDA target for Numba
```

Use `raystrack[portable-gpu]` instead of `.[portable-gpu]` when installing a
published release that includes these features. Optional dependencies do not
change a CPU-only installation.

`device="auto"` compares warmed CPU/GPU work on unrestricted solves and reuses
the measured plan on the same prepared solver. Capped or cancellable solves
use a cached plan; without calibration, small batches use CPU and larger work
uses an available GPU. Call `session.warmup()` before interactive deadlines to
measure the choice and compile kernels. Set `auto_tune=False` for the original
CUDA, then Taichi, then CPU availability preference.
`device="gpu"` requires a GPU; `"cuda"`, `"taichi"`, `"vulkan"`, and `"metal"`
request a specific backend and raise when unavailable. Vulkan targets compatible
AMD/Intel/NVIDIA devices; Metal targets compatible Macs. Drivers determine
availability. Intersections use compute kernels, without hardware ray-tracing
extensions.

```python
from raystrack import available_devices
print(available_devices())
```

Taichi owns one runtime per process. Raystrack reuses a compatible FP32 GPU
runtime and rejects incompatible runtimes instead of resetting their data.
For adapter selection, set Taichi's `TI_VISIBLE_DEVICE` environment variable
before first GPU use. Backend kernel/buffer access is serialized.

### Rhino 8 / Grasshopper
Raystrack is also available through the Rhino 8 Package Manager for use in
Rhino and Grasshopper.

In Rhino 8, run the `PackageManager` command, search for `Raystrack`, install
the package, and restart Rhino if prompted.

Rhino 8 embeds Python 3.9. Check optional wheel availability for its platform:
Taichi 1.7.4 ships Python 3.9 wheels for Windows/Linux, while its Apple Silicon
wheels start at Python 3.10. Metal therefore requires a compatible external
Python environment on those Macs; this repository does not bundle a
Rhino-to-external-process bridge. Vulkan has been tested on an AMD Radeon 860M;
Metal and Intel hardware require device-specific validation.

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

## Moving scenes and interactive previews

Prepare geometry once and update it explicitly. Prepared solvers own copied,
read-only geometry. After an update, pass `prepared.meshes` to low-level
functions; mismatched external meshes raise instead of using stale data.

```python
import numpy as np
from raystrack import PreparedSolver, PreviewSession, MatrixParams, SkyParams

prepared = PreparedSolver(meshes, acceleration="instanced")
common = dict(samples=4, rays=32, device="auto", bvh="builtin", sampling_mode="adaptive")
with PreviewSession(prepared, MatrixParams(**common), SkyParams(**common)) as session:
    plan = session.warmup()  # diagnostic rays, separate from view-factor estimates
    frame = session.preview(
        emitter_names=[meshes[0][0]],
        max_total_rays=8192,
        max_time_ms=None,  # warm up before interactive deadlines
    )
    transform = np.eye(4)
    transform[0, 3] = 2.0
    session.update_transform(meshes[-1][0], transform)
    frame = session.preview(max_total_rays=8192, max_time_ms=50)
    refined = session.refine(factor=4)  # continue stationary samples to 4x total
    print(frame.scene_version, frame.rays_used, frame.elapsed_ms)
    print(refined.rays_used, refined.cumulative_rays)  # additional versus retained rays
```

- `update_transform(name_or_index, matrix)` applies an **absolute** proper rigid
  4×4 transform relative to local vertices (`world = R @ local + translation`).
  Scale, shear, reflection, projective, and non-finite transforms fail.
- `update_vertices(name_or_index, vertices)` keeps faces and replaces world
  vertices, resetting the local transform basis.
- `update_mesh(name_or_index, vertices, faces=None)` permits topology changes.
  Changed faces rebuild the BVH; stable topology refits bounding boxes.
- `rebuild_bvh()` restores partitions after large motion. A refit remains
  correct but its original partition can become inefficient.

Changed emitter frames, areas, sampling distributions, bounds, and visibility
masks refresh together. Unchanged emitters retain their preparation. Compatible
CUDA allocations update in place, with changed triangle-range uploads where
practical. CPU and portable buffers also persist across solves.

`preview()` defaults to a **global** cap of 65,536 rays and a soft 50 ms deadline.
Selected emitters retain every other mesh as an occluder: a moving blocker can
change factors between stationary surfaces. Cold compilation/preparation can
consume a frame's deadline before any samples are traced. The deadline is
checked between chunks, so an already running chunk can exceed it.
`ray_batch_size` controls that latency tradeoff (65,536 normally; 8,192 in
preview sessions).

Both parameter classes expose `emitter_names`, `max_total_rays`, `max_time_ms`,
and `ray_batch_size`. Zero budgets perform no tracing. Untraced rows are omitted;
residual `Rest` factors appear only for rows with scene and sky estimates.
`sampling_mode="fair"` rotates chunks across requested emitter rows. Initial
chunks grow gradually, so small budgets distribute work before large batches
improve throughput. A cap smaller than the number of live emitters can still
leave rows untraced. `sampling_mode="adaptive"` first explores every row with
complete randomized replicates, then gives more work to rows with higher
sampling error while retaining regular exploration. Partial replicates do not
certify convergence.
Compatible outside-workflow settings share rays and one budget; incompatible
settings spend the remaining budget on the second pass.

Low-level solves accept `cancel=lambda: ...` and `progress=lambda ray_count: ...`.
Progress reports additional rays per completed chunk. Budgeted/selected solves
trace direct sender rows; shortcut reciprocity does not invent untraced rows.
`reciprocity_mode="bidirectional"` can average sampled front-side pairs.
Preview results do not normalize incomplete data to enforce row sums.

`session.submit(...)` returns a `Future` from one preview worker. New requests
cancel queued work and supersede running work at its next chunk. Running stale
requests return `cancelled=True` with empty result mappings; queued futures may
raise `CancelledError`. Session update methods cancel old work before mutation.
Shared prepared solvers serialize low-level solves and updates with a lock.

`PreviewResult` contains immutable `scene`, `sky`, and `rest` mappings,
`scene_version`, `rays_used`, `cumulative_rays`, `elapsed_ms`, `completed`,
`cancelled`, `status`, `converged`, and `statistics`.
`completed` means a usable current-scene preview, **not** statistical convergence.
`status` distinguishes sampling, convergence, iteration caps, and cancellation.
Statistics include completed-replicate sampling error and per-emitter ray counts;
they do not establish formal confidence-interval coverage. `as_dict()` produces
ordinary dictionaries for JSON export.

Refinement retains raw counts and running statistics for the same scene and
continues at the saved seed and intra-iteration ray offset. `rays_used` counts
only newly traced rays; `cumulative_rays` includes retained work. By default,
`refine(factor=2)` targets twice the current cumulative ray count. An explicit
`refine(max_total_rays=...)` requests that many **additional** rays. Convergence
or iteration limits can stop before the target. Changing geometry, sampling,
outputs, or selection requires a fresh preview. Cancelling or superseding work
discards its resumable state. Shared scene/sky accumulation retains both kinds
of counts while either needs more work, so later stricter refinement can use
intersections already computed for the other output.

Low-level matrix/sky/outside calls also accept `accumulator=SolveAccumulator()`
with a persistent prepared solver. Each call's ray budget is additional; returned
estimates include prior calls. Incompatible outside-workflow sampling uses
separate matrix and sky accumulators and spends the remaining global budget on
the second pass.

`session.warmup()` or `warmup(prepared, matrix_params, sky_params)` measures
synchronized ray generation, tracing, and reduction on representative work.
The returned `ExecutionPlan` exposes the backend, chunk size, timings, failures,
and diagnostic ray count. Explicit GPU requests preserve their backend.
`tune_budget_ms` is a soft calibration limit: compilation and an initial warm
measurement per backend can exceed it. Plans persist through rigid motion and
are separated by workload size, acceleration mode, and output configuration.

`PreparedSolver(..., acceleration="instanced")` keeps a local BVH per object
and a top-level BVH of transformed object bounds. Rigid updates change transforms
and the top-level hierarchy; local tracing geometry remains reusable. World
emitter preparation refreshes when needed. To share a prototype across multiple
objects, use `PreparedSolver.from_instances(prototypes, instances)`, where
`prototypes` is the usual mesh list and each instance is
`(name, prototype_name_or_index, proper_rigid_4x4_transform)`. Deformation and
topology edits detach the changed instance's local geometry. This is software
two-level traversal, using the existing compute backends.
Instanced acceleration always uses its two-level BVHs; this constructor choice
takes precedence over the flat-scene `bvh` parameter. The top-level tree
automatically rebuilds when its bounding-box quality measure degrades, and
`rebuild_bvh()` remains available for explicit rebuilding.

Unrestricted shared CUDA and portable solves can enqueue replicates up to a
convergence checkpoint and read back compact count arrays together. FP32
geometry and bounded portable int32 counters feed host int64 totals and float64
statistics. Set `convergence_interval` to reduce readback frequency; interactive
cancellation uses smaller chunks instead. CPU can be faster for small scenes.

Run the moving-blocker example and timing/consistency benchmark:

```sh
python examples/ex06_dynamic_preview.py --device cpu
python validation/benchmark_dynamic.py --device cpu --frames 3
python validation/benchmark_dynamic.py --device vulkan --json dynamic-results.json
python validation/benchmark_dynamic.py --device vulkan --acceleration instanced --json instanced-results.json
python validation/validate_dynamic_backends.py --devices cpu vulkan --accelerations flat instanced --json accuracy-results.json
```

The benchmark separates preparation, cold/warm solves, updates/refits, and fresh
preparation, and checks updates against fresh seeded solves. Agreement checks
cache correctness, rather than Monte Carlo error against an analytic reference.
Log messages use standard output; set `RAYSTRACK_LOG_CONSOLE=1` to explicitly
open the legacy log console.

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
