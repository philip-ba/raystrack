# Validation and benchmarks

The analytical helpers now construct public v2 `Mesh`/`Scene`/`Solver` objects
and read replicate counts from `Result.statistics`. Legacy result files remain
historical artifacts. New v2 reports have `v2_` prefixes.

## Accuracy

```sh
python validation/validate_dynamic_backends.py --devices cpu vulkan --accelerations flat instanced --json validation/results/v2_dynamic_accuracy.json
```

The fixed-work suite checks moving parallel squares at gaps 0.5, 1, and 2;
parallel 2×1 rectangles; coaxial discs approximated by 256 segments; and an
adjacent perpendicular square/rectangle. Defaults use density 256,
256 rays/cell, 64 complete replicates, and seeds 11/1009/10007. These seed ranges are
disjoint. CPU and Vulkan plus flat and instanced acceleration produce 72
comparisons. All passed the per-seed absolute tolerance 0.001 on the tested AMD
Radeon 860M; the maximum error was about 0.000262. This is an estimate-accuracy
check, not an uncertainty-coverage study or validation of every GPU.

Run the larger existing analytic and View3D checks with:

```sh
python validation/run_all.py
```

The five radiation-reference scripts use standard-error tolerance 1e-4,
minimum 40 and maximum 500 replicates. The canyon comparison reads the saved
View3D reference under `validation/view3d_reference/`. `run_all` regenerates the
traditional report files; the v2 fixed-work suite writes a separate report.

## Equal-workload v1/v2 comparison

```sh
python validation/benchmark_v2_vs_v1.py --gpu-device vulkan --json validation/results/v2_baseline_comparison.json
```

The baseline is `1757748cea4f523324f6ba44235699390b9a826c`. The script uses
`git archive` into TEMP and separate subprocesses with an explicit source path.
It never checks out or modifies the working tree. The v1 worker branch is an
isolated benchmark fixture; the package has no old calculation wrappers.

Workloads have equal geometry, seed 13, ray budgets, and batch sizes:

| Workload | Geometry | Rays | Backend |
| --- | --- | ---: | --- |
| small | 4 triangles | 512 | CPU |
| large | 24,576triangles | 262,144 | Vulkan |
| moving shared instances | 90,112world triangles /16,384 unique local triangles | 4,096 per frame | Vulkan |

By default, three independent process trials alternate API order; each trial
uses five untimed warmups and fifteen measured repeats.
Moving instances include transform commits and lazy scene/emitter preparation.
When run, the report preserves each measurement, cold costs, exact ray counts, and
per-channel estimate differences. Medians are within-run repeats rather than
independent timing trials. Timings vary with drivers, cache state, scheduling,
and workload; report slowdowns as well as improvements.

## Dynamic update costs

```sh
python validation/benchmark_dynamic.py --device cpu --acceleration flat --subdivisions 64 --frames 3 --budget 512 --json validation/results/v2_dynamic_flat_cpu_24576.json
python validation/benchmark_dynamic.py --device cpu --acceleration instanced --subdivisions 64 --frames 3 --budget 512 --json validation/results/v2_dynamic_instanced_cpu_24576.json
```

The tool separates transform commits, lazy acceleration/emitter refresh, warmed
solve time, and a fresh solver preparation at the same world position. Internal
cache refresh instrumentation is identical for reused and fresh solvers. Seeded
updated/fresh results must agree and have the same ray count. A rigid transform
can preserve BLAS storage while lazy emitter work still scales with geometry.
`--warmup` explicitly measures a device plan; probe rays are excluded from result
budgets.

## Sampling and storage

```sh
PYTHONPATH=src python validation/benchmark_sampling.py
PYTHONPATH=src python validation/benchmark_store.py
```

The sampling benchmark compares cosine rays with random and shifted-Halton
area-pair estimates of a rare receiver, then compares equal 1024-ray one-way and
bidirectional estimates. Area-pair is CPU-only. Its small-receiver reference is
a center-point limit, not an exact finite-rectangle integral.

The storage tool measures a deterministic synthetic fixture. V2 validates and
owns immutable geometry and deduplicates shared Mesh identities. It currently
reads complete snapshots, without selective on-disk row loading.

Performance timings were skipped at the user's request because other background
simulations made measurements unreliable. Newly measured timing reports were
excluded and no speed or size comparison is claimed. The optional benchmark
scripts remain available for a future quiet run.

## Tests and hardware limits

```sh
PYTHONPATH=src python -m pytest tests -q
```

The suite includes CPU numerical regression, dynamic/instancing invariants,
exact split-budget continuation, cancellation/queued async work, query/model
validation, IO security/roundtrip, CUDA simulator kernels, and optional Taichi
hardware tests. Missing optional dependencies skip portable tests. GPU tests
must report the actual initialized backend; simulator runs do not measure GPU
throughput. Physical CUDA, Metal, and Intel GPU behavior require independent
hardware verification.
