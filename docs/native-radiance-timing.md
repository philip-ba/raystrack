# Raystrack versus Radiance: matched wall timings

These measurements describe the earlier timing session. The subsequent
[BSDF milestone report](native-bsdf-20261111.md#verification-and-performance)
adds fresh preserved-DLL/candidate matched comparisons after renderer changes,
with separate first/warm/edit costs and explicit single-trial canopy limits.

The new measurements use the existing working tree on
`dev/native-renderer-20261111`, HEAD `6cadade`. Renderer implementation was not
changed. The current combined DLL was rebuilt before benchmarking; its SHA-256
is recorded in each raw report. All jobs ran sequentially, without validation
layers or traversal profiling. These are workload-specific measurements, not a
claim that one renderer is always faster.

On the annual cube, the first result took 218.65 s with Radiance's coefficient
workflow, 7.57 s with Embree and 13.93 s with Vulkan RT. Native first-result
ratios were 28.9x and 15.7x. Reusing Radiance coefficients reversed that result:
0.190 s for full-year application versus native retracing medians of 8.08 s and
12.18 s. Thus Radiance's cached evaluation was about 42x and 64x faster.

## Comparable result timings

Seconds below are medians across independent process trials. Warm values are
the median of each trial's three-repeat median. The annual first result has
only one fresh process sample, explicitly marked N=1; there is no independently
repeated annual cold median. Ratio = Radiance time / native time. Values above
1 favor native; values below 1 favor Radiance.

| Workload and operation | Radiance s | Embree s | Vulkan RT s | Rad/Embree | Rad/Vulkan | Repetitions |
|---|---:|---:|---:|---:|---:|---|
| Cube, all 8760 hours, first result | 218.648 | 7.573 | 13.928 | 28.87 | 15.70 | 1 process sample |
| Cube, all 8760 hours, warm evaluation | 0.190354 | 8.077 | 12.183 | 0.02357 | 0.01562 | 3 repeats in 1 process |
| Cube, 12 selected hours, first result | 0.561487 | 0.138062 | 3.480511 | 4.07 | 0.1613 | 3 process trials |
| Cube, 12 selected hours, warm evaluation | 0.0001572 | 0.075744 | 0.123974 | 0.002075 | 0.001268 | 3 x 3 repeats |
| Cube, 12 selected hours, edit plus result | 0.692944 | 0.068532 | 0.135192 | 10.11 | 5.13 | 3 process trials |
| Canopy, 24 steps, first result | 7.914063 | 0.525753 | 4.375450 | 15.05 | 1.81 | 3 process trials |
| Canopy, 24 steps, warm evaluation | 0.0014011 | 0.458006 | 0.538330 | 0.003059 | 0.002603 | 3 x 3 repeats |
| Canopy, 24 steps, edit plus result | 7.591110 | 0.502200 | 0.578050 | 15.12 | 13.13 | 3 process trials |

Radiance first and edited results include geometry source/octree preparation,
sky and exact-sun coefficient construction, saved coefficient I/O and combined
matrix application. The existing reference routine also performs a primary-sun
diagnostic and one internal diffuse application; these small extra costs remain
visible rather than being silently subtracted. For the annual run they took
0.265 s and 0.109 s respectively. Its sun-coefficient trace dominated at
217.530 s; diffuse coefficient tracing took 0.343 s.

Native first results include ShortwaveSolver construction, DLL/device/geometry
initialization and the combined solve. Native warm evaluation calls solve()
again on the retained geometry and retraces transport; it does not reuse a
weather-transfer matrix. The solver returns immutable component snapshots;
the total-property sum used for comparison is outside the native call timer.
Radiance warm evaluation applies resident sun and sky coefficients and allocates
the combined hourly result. Neither method merely returns a previously cached
final output. Warm repeats use the original forcing, not newly measured weather
variants. Common forcing load/save, Python startup/imports and final benchmark
JSON output are excluded from the table and separately visible in process logs.

Edited native times include Scene.update_transform commits plus the subsequent
solve, including lazy world-geometry preparation, backend update and transport.
Radiance rebuilds coefficients for the edited geometry. Cube edits translate the
six cube surfaces and their 20 attached sensors by 0.2 m in +X; ground and open
sensors stay fixed. Canopy edits translate the canopy by 0.375 m while all 64
sensors stay fixed. Every edited native solve matched a fresh build.

## Setup phases, not interchangeable speed ratios

| Workload | Radiance two octree subprocesses s | Radiance coefficient build s, including octrees | Embree native initialization s | Vulkan native initialization s |
|---|---:|---:|---:|---:|
| Annual cube, N=1 | 0.052283 | 218.462441 | 0.060030 | 2.776263 |
| 12 selected cube hours, median of 3 | 0.029946 | 0.561259 | 0.061188 | 3.355709 |
| 24-step canopy, median of 3 | 0.915075 | 7.912404 | 0.085763 | 3.770318 |

These phases do different work. Radiance's coefficient build includes tracing;
native initialization does not. Do not divide these columns to claim a renderer
speedup. Native initialization is instrumented around the existing NativeGeometry
constructor; world-scene preparation outside that constructor is included in
the first-result timer. Cold means a fresh child process/solver, with OS and
driver caches left intact, not a rebooted system or flushed disk cache.
Within each worker, the order is Radiance, Embree, then Vulkan. Vulkan's first
result includes its first device/pipeline setup but shares the DLL already
loaded for Embree; it is not an independently measured cold DLL load.

## Workloads, accuracy and machine

- Hardware: AMD Ryzen AI 7 350, Radeon 860M, 16 logical CPUs, Windows 11.
  Python 3.13.9, existing C++20 Release build, Embree 4.4.1; Vulkan hardware ray
  queries are on the device while transport/material/reduction work stays on host.
- Radiance: pinned `C:/Radiance/bin`, 5.4 `5.4.4ee32974b1`, one process (`-n 1`).
  Native host transport: one thread. Embree acceleration construction: SDK
  default, not pinned; the benchmark does not claim to know its actual worker
  count. OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and OMP_NUM_THREADS were set to 1.
  The Vulkan shader has 64 invocations per workgroup; GPU invocations are not
  equivalent to a CPU host-thread count.
- Cube: 1 m cube and 20 x 20 m finite ground, 14 triangles, reflectance 0.2,
  black lower environment, 23 sensors (4 equal-area samples on each exposed face,
  2 ground, 1 open). Annual Berlin2025 synthetic forcing is loaded from the
  validated saved arrays; no weather substitution. The quick case selects 12
  hours across four seasons, including 10 sun-up hours, not 12 contiguous hours.
- Canopy: existing grid-256 geometry, 131072 triangles, 64 sensors, 24 solar
  directions, DNI=700*sun_Z and DHI=100 W/m2, no ground. Thin diffuse material
  reflectance 0.35 and transmittance 0.10 is matched to Radiance trans. Both receive
  the uniform sky as a common 145-patch Reinhart matrix. Sensor positions reuse
  the existing canopy RNG seed 11. This differs from the old native-only canopy
  timing, which used a one-row isotropic sky and only one replicate.
- New timing runs: native 2048 cosine samples/sensor x 2 independent replicates,
  seed 17, maximum one material interaction; Radiance 4096 ambient samples,
  sky `-ab 2`, sun `-ab 1`, `-aa 0 -as 0 -u- -lw 1e-8`, exact source testing.
  Direct sun uses a separate 0.01-degree source per direction (4409 annual,
  10 quick, 24 canopy), matching the native point-sun direction. No lux conversion.
- All timed cases passed the comparison gates. The annual combined exposed-face
  maximum exposure error was 0.4514%, daylight NRMSE 0.6270%; the earlier separate
  direct/diffuse/combined full validation remains the source for all 30 gates.
  The new runner times combined transport only. Canopy comparisons report the
  means of the 32 lower and 32 upper sensors, and also preserve individual-sensor
  error statistics. Quick/canopy exposure sums cover only their actual steps,
  despite the inherited metric field name `annual_relative_error`.

## Evidence and reproduction

The benchmark-only runner is `validation/benchmark_radiance_matched.py`. It does
not modify renderer algorithms. All raw samples, commands, phase costs, forcing
hashes, library hashes, gate results and reference stderr are retained under:

- `build/annual-validation/timing-investigation-annual/`
- `build/annual-validation/timing-investigation-quick/`
- `build/annual-validation/timing-investigation-canopy/`
- Combined summary: `build/annual-validation/timing-investigation-summary.json`

Each directory has `trials.json`, per-trial `report.json`, `summary.json` and
`reference/radiance_commands.json`; edited runs also have `reference-edited/`.
`trial-*.stderr` and `trial-*.stdout` preserve process output. The earlier smoke
run is excluded from medians. The common forcing is copied into each trial.

```powershell
Set-Location -LiteralPath 'C:\Users\Philip\Desktop\raystrack'
$env:PYTHONPATH = 'src'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
Remove-Item Env:VK_INSTANCE_LAYERS,Env:RAYSTRACK_VULKAN_PROFILE -ErrorAction SilentlyContinue
cmake --build build/raystrack-native-final --config Release
.local/release-venv/Scripts/python.exe validation/benchmark_radiance_matched.py --scope quick --edited --trials 3 --repeats 3 --output build/annual-validation/timing-investigation-quick
.local/release-venv/Scripts/python.exe validation/benchmark_radiance_matched.py --scope annual --trials 1 --repeats 3 --output build/annual-validation/timing-investigation-annual
.local/release-venv/Scripts/python.exe validation/benchmark_radiance_matched.py --scope canopy --edited --trials 3 --repeats 3 --output build/annual-validation/timing-investigation-canopy
```

Output directories must not already exist; use new names when reproducing.
Reusing the input sky/forcing does not reuse the Radiance coefficients in these
new cold runs: each builds and traces its own reference from scratch.

## Missing data and prior evidence

There is no full-year edited-scene comparison or three-process annual cold
median from this investigation. The 12-hour and 24-step edited measurements
are not annual estimates and must not be extrapolated. To obtain the missing
full-year cold/edited medians with three independent processes and five warm
repeats each, run sequentially in the environment above:

```powershell
.local/release-venv/Scripts/python.exe validation/benchmark_radiance_matched.py --scope annual --edited --trials 3 --repeats 5 --output build/annual-validation/timing-full-year-edited-3trials
```

This rebuilds annual Radiance coefficients both before and after an edit in
each trial, so it performs six full coefficient setups. It has not been run.
No scaling claim is made for more complex BSDFs, delta transport or other GPUs.

The older annual artifacts had one Radiance coefficient setup (about 244 s)
and isolated native calls with unclear repeated cold/warm boundaries. Their
90.75 s candidate and 320.04 s baseline total elapsed values include both
backends, isolated modes, repeats, diagnostics and I/O; neither is a single
renderer annual runtime. The older 0.119 s Radiance figure times diffuse matrix
application only, not complete sun+sky output. Geometry-update-only improvements
and the old native-only canopy figures cannot supply Radiance speed ratios.
The new measured rows above address those gaps without rewriting old artifacts.

## Checks for this investigation

The rebuilt binary, all seven measured process trials (three quick, one annual,
three canopy), numerical comparison gates, and all edited/fresh checks completed
successfully. The exploratory smoke trial is additional and excluded from the
tables. Existing annual fixture/reference tests passed 5/5 in 0.83 s, the new
benchmark module passed py_compile, and git diff --check passed. Renderer
implementation and the existing test sources were not edited; the full renderer
suite was not rerun for these benchmark/documentation additions.

The next implementation prompt is `docs/next-bsdf-renderer-session.txt`.
The launch command was syntax-checked with PowerShell and the installed
Codex CLI `0.167.0-alpha.4.1`; it was not executed, so no next-session renderer
work was started. The available local model cache includes gpt-6.1-sol and medium
effort. The command explicitly sets lead and default subagent model/effort and
enables multi-agent execution, consistent with the
[official CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
