# Raystrack

<img src="https://raw.githubusercontent.com/philip-ba/raystrack/main/raystrack_icon.svg" alt="Raystrack" width="160">

Raystrack computes radiative view factors between triangulated surfaces using
quasi Monte Carlo ray tracing. Version 2.0.0 has a unified API built
around `Scene`, `Solver`, `Query`, `Run`, and immutable `Result` snapshots.
The v1 calculation functions have been replaced. See the
[migration guide](https://github.com/philip-ba/raystrack/blob/main/docs/v2-migration.md) before upgrading an existing script.

The [Grasshopper integration](https://github.com/philip-ba/raystrack/blob/main/docs/grasshopper.md) lives in this repository too.
It provides compiled **RS** components, live background solves, a Raystrack
ribbon icon, and a bundled Python runtime through Yak or a standalone ZIP.

## Installation

```sh
pip install "raystrack==2.0.0"
pip install "raystrack[portable-gpu]==2.0.0"  # optional Taichi Vulkan/Metal
pip install "raystrack[cuda]==2.0.0"          # optional NVIDIA CUDA
```

In Rhino 8.35 or later on Windows, open **PackageManager**, search for
**raystrack**, and install version **2.0.0**. Restart Rhino, then open
Grasshopper's **Raystrack** tab. The Yak package includes its Python runtime.
The [Grasshopper guide](https://github.com/philip-ba/raystrack/blob/main/docs/grasshopper.md) covers components and examples.

To install from a source checkout:

```sh
pip install .
pip install ".[portable-gpu]"  # optional Taichi Vulkan/Metal
pip install ".[cuda]"          # optional NVIDIA CUDA target for Numba
```

The package requires Python 3.9–3.13, NumPy, and Numba. CPU tracing works without
optional GPU dependencies. `device="cpu"`, `"cuda"`, `"vulkan"`, or `"metal"`
chooses a backend explicitly; unavailable GPU requests raise availability errors.
`"gpu"` requires an available GPU, and `"taichi"` chooses a portable GPU runtime.
`"auto"` can select CPU or GPU. These are compute tracers with software BVHs;
they do not use hardware ray tracing extensions.

```python
from raystrack import available_devices
print(available_devices())
```

Vulkan has been tested on an AMD Radeon 860M. Metal, Intel GPUs, and physical
CUDA hardware need their own validation; CUDA simulator checks verify kernel
behavior rather than performance. Taichi reuses one compatible FP32 runtime per
process and serializes access. Set `TI_VISIBLE_DEVICE` before first GPU use when
choosing a Taichi adapter.

## Quick start

```python
import numpy as np
from raystrack import (
    Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy,
    Budget, Channel,
)

vertices = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
faces = np.array([[0,1,2], [0,2,3]], np.int32)
scene = Scene.from_meshes({
    "lower": Mesh(vertices, faces),
    "upper": Mesh(vertices + [0,0,1], faces[:, ::-1]),
})
options = SolveOptions(
    sampling=Sampling(density=16, rays_per_cell=128, seed=11),
    accuracy=Accuracy(max_replicates=20, min_replicates=5, tolerance=1e-4),
    batch_size=65536,
)
with Solver(scene, device="cpu", bvh="builtin") as solver:
    result = solver.solve(Query.row("lower", sky="merged"), options,
                          Budget(rays=65536))

print(result.value("lower", Channel("surface", "upper", "front")))
print(result.value("lower", Channel("sky")))
print(result.status, result.rays_used, result.cumulative_rays)
```

`Mesh` owns immutable float32 vertices and int32 triangle faces. Winding defines
the front side. Surface IDs are unique strings independent of display labels;
IDs containing `_front`, spaces, or Unicode stay unambiguous. `Scene` owns
ordered surfaces and rigid instance transforms. Editing a label leaves the
geometry revision unchanged.

## Choose outputs and sampling

`Query.matrix()` requests all surface rows. `Query.row("wall")` requests one
sender, and `Query.pair("wall", "roof")` selects one receiver. Receiver selection
filters outputs; every surface can still occlude rays. `receiver_sides` can
select `("front",)`, `("back",)`, or both.

Add `sky="merged"` or `sky="tregenza145"` to a scene query to trace sky with the
same rays. `Query.sky(senders=["wall"], discrete=True)` requests only sky outputs.
Sky patches are indexed `0..144` through `Channel("sky", patch=index)`.

`SolveOptions` holds one immutable configuration for a query:

- `Sampling`: emission density, rays per cell, seed, face reversal, fair/adaptive
  allocation, and estimator strategy.
- `Accuracy`: replicate cap, minimum replicate/ray floors, tolerance, and
  convergence check interval.
- `Postprocessing`: explicit reciprocity transformation after complete solves.
- `batch_size`: a cap for bounded trace chunks.

The default cosine strategy uses independently shifted Halton replicates.
`mode="fair"` serves each selected sender; `"adaptive"` gives more work to rows
with larger uncertainty after initial exploration. A minimum ray floor reduces
zero-hit early stopping but does not prove that rare receivers were observed.

For a small selected receiver, use `Sampling(strategy="area_pair",
pair_samples=8192, sequence="shifted_halton")` with `Query.pair(...)`.
Unsupported estimator/output combinations raise `CapabilityError`. Area-pair
sampling weights uniformly sampled surface pairs by their cosine and
inverse-distance factors and checks visibility against all occluders. It supports
CPU only; unsupported GPU, sky, or reciprocity combinations raise explicitly.
`sequence="random"` is also available for this estimator.

Meshes can see and occlude themselves: other faces of the emitting surface
participate in tracing, and `Query.pair("box", "box")` selects its self view
factor. Matrix/row queries include the same channels. For a closed box with
outward mesh normals, use `Sampling(flip_faces=True)` to emit into its interior;
the self factor is 1 on `Channel("surface", "box", "back")`, with zero escape.
With inward normals, use the front channel and leave `flip_faces=False`.
Receiver side labels always follow the stored mesh winding.

Reciprocity is opt-in through `Postprocessing("bidirectional")` or
`Postprocessing("shortcut")`, both requiring complete sender/receiver tables.
The shortcut transformation preserves the v1 upper-direction estimate but v2
still traces every requested row; it does not halve the workload. Transformed
uncertainties are unknown and the transformation is recorded in provenance.
`"rowsum"` additionally requires a closed enclosure without escape or sky.
Raw previews use `Postprocessing("none")`, the default.

## Moving scenes and continued work

```python
with Solver(scene, device="cpu", acceleration="instanced") as solver:
    query = Query.row("lower", sky="merged")
    run = solver.start(query, options)
    first = run.advance(Budget(rays=4096))
    refined = run.advance(Budget(rays=12288))  # only additional rays
    assert refined.cumulative_rays == 16384

    transform = np.eye(4)
    transform[0, 3] = 2
    scene.update_transform("upper", transform)
    new_frame = solver.start(query, options).advance(Budget(rays=4096))
```

Transforms are absolute proper rotations and translations relative to the local
mesh; scale, shear, reflection, invalid matrices, and float32 overflow are
rejected. `scene.update_vertices(id, world_vertices)` retains topology and resets
the transform. `scene.update_mesh(id, Mesh(...))` replaces geometry and resets
the transform. Updates invalidate existing runs before waiting for active work.
A new run starts a new stream for the new `scene.revision`; old results remain
immutable snapshots labeled with their original revision.

Flat acceleration refits packed triangle bounds when topology is unchanged.
`acceleration="instanced"` keeps local object BVHs and updates instance transforms
and a top-level hierarchy. Its hierarchy refits and rebuilds when its measured
bound cost deteriorates. World emitter frames are prepared lazily; emitter work
can still scale with changed triangle counts. Share local geometry explicitly:

```python
prototype = Mesh(vertices, faces)
instances = Scene.from_instances({"panel": prototype}, [
    ("panel0", "panel", np.eye(4)),
    ("panel1", "panel", transform),
])
```

Rigid updates retain local geometry/BVH buffers on CPU, Vulkan/Metal, and CUDA.
Deformation refreshes the affected object, and topology changes rebuild it.
Instancing reduces update/storage work; tracing speed depends on the scene.

`Run.advance` preserves exact ray offsets and completed replicate statistics.
Accuracy, postprocessing, and batch size can change on a stationary run; sampling
settings require a new run. Solver configuration and run queries are read-only;
create a new solver to change the backend. Use `advance(options=...)` to adjust
accuracy. Ray budgets are additional work. Time budgets are
soft deadlines checked between chunks and include preparation; cold compilation
can exceed them. `Budget(rays=0)` returns the current estimate without tracing.

`Run.submit(...)` returns a future. Repeated submissions queue additive advances
through one solver worker. Cancel an obsolete run explicitly before replacing
it. `Run.cancel()` is terminal: later advances return a cancelled snapshot with
zero additional rays. Cancellation retains partial estimates. A geometry change
marks the run `invalidated`; later advances raise and require a new run. Solver
and run context managers close resources.

## Read results and uncertainty

`Result.row(sender)` maps typed channels to estimates. Surface, sky, `rest`, and
`unrequested` channels avoid suffix parsing. Rest contains escaped rays outside
requested sky; unrequested contains hits omitted by receiver/side selection.
For cosine queries these channels preserve raw energy accounting.

`coverage` aligns with `sender_ids`: `0` means not started, `1` means sampled,
and `-1` means coverage is unknown in imported data. A requested zero in a
sampled row is known zero; an unsampled value is `None`. Sparse arrays hold
estimates and errors, while `.dense()` exports a mutable copy.

Estimates include partial replicates. `Result.error(...)` is `None` until there
are at least two complete randomized replicates and no pending partial replicate.
`statistics["emitters"]` retains completed-replicate standard errors, sample
counts, pending rays, and convergence details. Partial work cannot claim
convergence. Error estimates describe sampling variation, not mesh or floating
point error, and zero observed hits can underestimate rare-event uncertainty.

## Calibration and storage

`solver.warmup(query, options)` compiles and measures representative synchronized
CPU/GPU tracing. Probe rays are diagnostic and do not enter results or ray
budgets. An unrestricted `Solver.solve` may calibrate automatically; bounded
runs use cached/provisional plans. Resumed runs retain their numerical backend.
Warm up before interactive deadlines, or use an explicit device with
`auto_tune=False` for a fixed execution choice.

```python
from raystrack.io import save, load, import_v1_json

save("case.raystrack", scene, new_frame, metadata={"case": "moving wall"})
stored = load("case.raystrack")
legacy = import_v1_json("examples/street_canyon.json", "examples/vf_matrix.json")
```

Stores are versioned directories with chunked arrays and a manifest published
last. Existing stores are never overwritten. V2 preserves stable IDs, shared
geometry, transforms, sparse channels, statistics, execution, and provenance.
`load` also reads v1 stores. Legacy imports preserve unknown sampling coverage
and uncertainty instead of inventing counts. V2 currently loads a full immutable
snapshot; selective on-disk row loading and appendable writers are unavailable.

Rhino conversion is isolated in `raystrack.integrations.rhino` through
`from_rhino_mesh` and `from_rhino_scene`; importing the numerical package does
not require Rhino. The converter triangulates quads and copies vertices.

## Examples and verification

Run examples directly from this checkout, beginning with
`python examples/ex00_street_canyon_geometry.py`. Examples also build the canyon
in memory, so generated geometry files are optional. `ex06_dynamic_preview.py`
accepts `--device`, `--acceleration`, `--budget`, `--frames`, and `--warmup`.

```sh
PYTHONPATH=src python -m pytest tests -q
python validation/validate_dynamic_backends.py --devices cpu vulkan --accelerations flat instanced
python validation/benchmark_v2_vs_v1.py --gpu-device vulkan
```

The [v2 accuracy report](validation/results/v2_dynamic_accuracy.json) passes 72
closed-form comparisons across CPU/Vulkan and flat/instanced traversal. The
[validation guide](validation/readme.md) describes larger analytical checks,
optional benchmarks and their limits. Performance timing validation was skipped
at the user's request because background simulations made measurements
unreliable; this change makes no speedup claim. The [migration guide](https://github.com/philip-ba/raystrack/blob/main/docs/v2-migration.md)
shows v1-to-v2 replacements.
