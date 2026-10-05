# Migrating to the v2 source API

The v1 reference checkpoint is
`1757748cea4f523324f6ba44235699390b9a826c`. The current source tree uses a unified
v2 API; package metadata stays `1.0.2` as requested. This work does not publish a
release, create a release tag, or push the repository.

## Objects and ownership

V1 passed lists of `(name, vertices, faces)` and separate mutable parameter
records to calculation functions. V2 uses owned immutable `Mesh` values, a
mutable revisioned `Scene`, a reusable `Solver`, immutable `Query` and
`SolveOptions` values, and persistent `Run` instances.

```python
from raystrack import Mesh, Scene, Solver, Query, Budget

scene = Scene.from_meshes({name: Mesh(vertices, faces)
                          for name, vertices, faces in old_meshes})
with Solver(scene, device="cpu") as solver:
    result = solver.solve(Query.matrix(), budget=Budget(rays=65536))
```

Stable IDs and labels have separate roles. IDs identify surfaces and channels;
labels are display metadata. No new result parser interprets `_front` or `_back`
inside an ID. Arrays and result metadata have immutable backing storage.

## Entry point mapping

| V1 entry point | V2 operation |
| --- | --- |
| `view_factor_matrix(meshes, params)` | `Solver(scene).solve(Query.matrix(), options)` |
| selected `emitter_names` | `Query.matrix(senders=[...])` or `Query.row(id)` |
| selected receiver calculation | `Query.pair(sender, receiver)` |
| `view_factor_to_tregenza_sky` | `Query.sky(discrete=False/True)` |
| `view_factor_outside_workflow` | `Query.matrix(sky="merged"/"tregenza145")` |
| `view_factor_targeted` | `Query.pair(...)` with `Sampling(strategy="area_pair")` |
| reusable `PreparedSolver` | public reusable `Solver(scene)` |
| `PreviewSession.preview` | `solver.start(query, options).advance(Budget(...))` |
| `PreviewSession.refine` | `run.advance(Budget(rays=additional))` |
| `PreviewSession.submit` | `run.submit(Budget(...))` |
| `save_run` / mesh/result JSON writers | `raystrack.io.save(path, scene, result)` |
| `open_store` | `raystrack.io.load(path)` returning `StoredRun` |
| legacy mesh/result JSON readers | isolated `raystrack.io.import_v1_json(...)` |

The v1 calculation functions, public `MatrixParams`, `SkyParams`,
`PreviewSession`, `SolveAccumulator`, and mutable/appendable `RunStore` are
removed. There are no production compatibility aliases. Legacy data readers
are isolated in `raystrack.io`. Geometry kernels and prepared acceleration
remain internal implementation details; public callers should use `Scene` and
`Solver`.

## Controls

```python
from raystrack import SolveOptions, Sampling, Accuracy, Postprocessing

options = SolveOptions(
    sampling=Sampling(density=16, rays_per_cell=128, seed=11, mode="fair"),
    accuracy=Accuracy(max_replicates=100, min_replicates=5,
                      tolerance=1e-4, mode="stderr", check_interval=1),
    postprocessing=Postprocessing(reciprocity="none"),
    batch_size=65536,
)
```

| V1 setting | V2 setting |
| --- | --- |
| `samples`, `rays`, `seed`, `flip_faces` | `Sampling.density`, `rays_per_cell`, `seed`, `flip_faces` |
| `sampling_mode` | `Sampling.mode` |
| targeted `samples`, `sequence` | `Sampling.pair_samples`, `sequence` |
| `max_iters`, `min_iters` | `Accuracy.max_replicates`, `min_replicates` |
| `tol`, `tol_mode`, `convergence_interval` | `Accuracy.tolerance`, `mode`, `check_interval` |
| `min_total_rays` | `Accuracy.min_rays` |
| `max_total_rays`, `max_time_ms` | additional `Budget.rays`, soft `Budget.time_ms` |
| `ray_batch_size` | `SolveOptions.batch_size` |
| device/BVH/GPU/tuning controls | `Solver(...)` construction |
| reciprocity settings | explicit `Postprocessing(reciprocity=...)` |

Matrix and sky outputs share one configuration and one ray stream in a combined
query. If they require different seeds or samplers, start two independent runs
and allocate each an explicit budget. Their results have separate cumulative
counts. There are no hidden child accumulators.

Postprocessing runs after complete sampling. `"bidirectional"` averages
exchanged front-side area; `"shortcut"` derives the reverse direction from the
v1 upper-direction estimate. Both trace all requested rows in v2. `"rowsum"`
requires a closed enclosure with no escape or sky. Incomplete previews use
`"none"`. Transformed errors remain unknown; provenance records the method.

## Results and continuation

```python
from raystrack import Channel, Budget

run = solver.start(Query.row("wall", sky="merged"), options)
first = run.advance(Budget(rays=1000))
second = run.advance(Budget(rays=3000))
value = second.value("wall", Channel("surface", "roof", "front"))
assert first.cumulative_rays == 1000
assert second.rays_used == 3000 and second.cumulative_rays == 4000
```

Surface, sky, rest, and omitted-hit channels are typed values. `Result.row`
includes requested sampled zeros. `coverage=0` means not started and values are
`None`; `coverage=1` means sampled. Legacy data can have `coverage=-1` when it
cannot establish sampling history.

Estimates include every accepted ray. Complete independently shifted
replicates determine standard errors. `Result.error` stays `None` during a
partial replicate and until two complete replicates exist. Raw completed
standard errors and pending/completed ray counts remain in `statistics` with
their basis stated. A pending partial replicate cannot claim convergence.
Statistics and execution metadata are immutable snapshots, distinct from the
live run.

Accuracy, batch size, and postprocessing can change between advances. Sampling
settings and query selections are fixed for a run. Changing the seed, density,
ray count, estimator, or selected channels requires a fresh run.

`Run.submit` queues additive work on one solver worker. It no longer coalesces
frames or replaces queued work implicitly. Call `old_run.cancel()` and start a
new run when replacing a UI frame. Cancellation is terminal but retains partial
results; later advances add zero rays and return `status="cancelled"`.

Geometry edits invalidate runs before waiting for active chunks. An in-flight
result can finish as an old-revision `invalidated` snapshot; subsequent advances
raise `RuntimeError` and require a new run. Old snapshots remain usable as
historical values. Explicitly close runs/solvers or use context managers.

## Dynamic geometry and instances

`scene.update_transform(id, rigid_4x4)` sets an absolute local-to-world transform.
Only proper rotations and translations are accepted. `update_vertices` accepts
world coordinates and retains faces; `update_mesh(id, Mesh(...))` replaces
geometry. Both reset the transform basis to identity. All successful geometry
edits increment `scene.revision`; label edits do not.

Choose `Solver(scene, acceleration="instanced")` for local object BVHs plus a
refittable top-level hierarchy. `Scene.from_instances` shares `Mesh` identity
between repeated instances. Rigid transforms retain local BLAS/triangle buffers;
world emitter distributions still need lazy refresh for changed emitters.
Deformation/topology changes refresh or rebuild affected object acceleration.
All supported tracers implement nearest-hit/front-back/occlusion semantics on
this representation. This is software instancing, not native hardware RT.

## Storage

```python
from raystrack.io import save, load, import_v1_json

save("new-case.raystrack", scene, result, metadata={"description": "case"})
stored = load("new-case.raystrack")
legacy = import_v1_json("meshes.json", "matrix.json", "sky.json", "rest.json")
```

V2 stores one complete snapshot in chunked arrays and writes the manifest last.
Existing paths are rejected. Geometry is deduplicated by owned `Mesh` identity;
transforms, labels, channels, coverage, raw statistics, execution, and provenance
round-trip. A result must match the saved scene revision.

`load` imports v1 directory stores as well as v2. Missing v1 uncertainty and ray
counts remain unknown. V2 currently loads the full immutable scene/result;
selective disk reads and appendable frame writers are deferred. V2 validates and copies owned geometry during full reads and deduplicates shared
instances. These are storage tradeoffs; no timing or size benchmark is claimed
in this migration.

## Verification and measurements

The original 111 numerical/lifecycle/backend behaviors were migrated without
production wrappers. Numeric workload adapters live only under `tests/`; direct
v2 tests cover public lifecycles, model ownership, channels, budgets, IO,
capability errors, and kernel parity. The complete hardware suite passed 202 tests on the tested environment,
including CPU, real AMD Vulkan, and CUDA simulator paths.

[72 analytical comparisons](../validation/results/v2_dynamic_accuracy.json)
cover six cases, three disjoint seeds, CPU/Vulkan, and flat/instanced acceleration.
Every comparison passed the `1e-3` per-seed absolute tolerance; the largest error
was approximately `2.62e-4`. This checks fixed-work estimates, not standard-error
coverage.

Performance timing validation was skipped at the user's request because other
background simulations made measurements unreliable. All newly measured timing
reports were excluded. The optional [comparison tool](../validation/benchmark_v2_vs_v1.py)
exports the fixed v1 SHA to TEMP and runs equivalent workloads in separate
processes without changing the working tree. It can measure small CPU work,
large GPU work, and moving shared instances on a quiet system. Benchmark tools
remain available for future use; this migration makes no speedup claim.
