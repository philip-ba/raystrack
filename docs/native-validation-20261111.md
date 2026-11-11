# Native renderer validation, 2026-11-11

The subsequent [BSDF milestone report](native-bsdf-20261111.md) adds clear-plate
delta transport, sided diffuse sheets and fresh verification. This document's
test counts and unimplemented-glazing statements describe the earlier session.

This session used the existing dirty working tree on
`dev/native-renderer-20261111`, HEAD `6cadade`. No dependency was installed.
The machine is an AMD Ryzen AI 7 350 with Radeon 860M and 16 logical CPUs,
Windows 11, Python 3.13.9. Builds use the existing CMake/MSVC toolchain,
Embree 4.4.1 and Vulkan SDK 1.4.350.0 documented in the backend guides.

## Reproduce

From repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
$env:RAYSTRACK_RADIANCE_BIN = 'C:\Radiance\bin'
New-Item -ItemType Directory -Force build/test-temp | Out-Null
$env:TEMP = (Resolve-Path build/test-temp).Path
$env:TMP = $env:TEMP
cmake --build build/raystrack-native-final --config Release
.local/release-venv/Scripts/python.exe validation/annual_radiance.py --samples 2048 --replicates 2 --radiance-samples 4096 --output build/annual-validation/annual
.local/release-venv/Scripts/python.exe validation/validate_shortwave_panels.py
.local/release-venv/Scripts/python.exe validation/validate_clear_glazing.py
.local/release-venv/Scripts/python.exe validation/benchmark_native_updates.py --grid 256 --rays 65536 --repeats 3 --json build/annual-validation/updates.json
.local/release-venv/Scripts/python.exe validation/benchmark_native_shortwave.py --grid 256 --samples 2048 --repeats 5 --embree-dll-dir .local/embree-4.4.1/embree-4.4.1.x64.windows/bin
.local/release-venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
$env:VK_LAYER_PATH = (Resolve-Path build/vulkan-deps/sdk/Bin).Path
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
ctest --test-dir build/raystrack-native-final -C Release --output-on-failure
git diff --check
```

The configured build directory was reused. A new build can be configured using
the SDK locations and commands in [Embree](native-embree.md) and
[Vulkan](native-vulkan.md). Copy the Embree runtime DLLs beside the native DLL.
All result directories above are ignored by Git. Run timing tools sequentially;
unset `VK_INSTANCE_LAYERS` and profiler variables for the performance comparison,
then explicitly enable validation for correctness checks.

The actual candidate annual command additionally used
`--reuse-reference build/annual-validation/baseline-annual --output build/annual-validation/candidate-annual`.
The runner checks all forcing/sensor arrays and geometry before reuse and
records the original Radiance commands. This preserves the already executed
full-year reference while avoiding another unchanged coefficient setup.

## Annual and optical evidence

The [annual report](../validation/annual-radiance.md#executed-2026-11-11) records
all 8760 hours, neutral W/m2 and Wh/m2, finite ground, matched sky, exact sun
directions and exposed-face results. Both native backends passed the 2% annual
and 5% daylight NRMSE gates in direct, diffuse and combined modes. Results,
worst hourly errors, shadow/night diagnostics, convergence, raw coefficients,
forcing and commands are under `build/annual-validation/candidate-annual/`.
The original rebuilt DLL and full reference are under `baseline/` and
`baseline-annual/` respectively.

`validate_shortwave_panels.py` checks large-sheet analytic behavior, off-axis
transmission, asymmetric front/back lobes, reversed winding, material-only edits
and finite overhead panels on the cube. It stores commands and arrays under
`build/annual-validation/panels/`. Radiance 5.4 empirically selects the Visible
XML block: changing only Visible transmission from .35 to .65 changes measured
transmission by .65/.35. Matched comparisons therefore duplicate controlled
Solar coefficients into Visible and explicitly label this equivalence.
The directional model is reciprocal, including the orientation fixture with
unequal front/back reflection and equal reciprocal transmission. Klems projected weights, energy, sampling PDF and
reciprocity have separate Python regression coverage.

A first overhead panel put a ground sensor exactly on its shadow silhouette.
Native point-sun and Radiance finite-source edge classifications differed by a
full direct beam there. The final scattering fixture uses halfextent 1.1 m to
avoid that ambiguity and retains the policy in its report. Coincident-triangle
hit filtering in native longwave remains documented in the solver guide.

An initial orientation fixture also assigned unequal forward/reverse diffuse
transmission. Radiance reversed the incident/outgoing pair during adjoint
evaluation and produced the opposite coefficient; that nonreciprocal fixture
was not an equivalent reference. The final check varies front/back reflection
while retaining reciprocal transmission, consistent with the
[Radiance shading contract](https://github.com/LBNL-ETA/Radiance/blob/master/src/rt/m_bsdf.c).

## Remaining scope

Geometry updates still pack world-space arrays. Embree refits fixed topology;
Vulkan retains device/pipeline/buffers but rebuilds its acceleration structures.
Shared prototypes, native transform-only instancing, GPU transport/reduction,
clear glazing/delta events and wavelength-resolved transport remain open.
No OptiX claim is made without compatible NVIDIA hardware. The panel reference
validates finite-width scattering and does not justify exposing delta glazing.

The first glazing reference gate is now implemented in
`validation/validate_clear_glazing.py`, with six analytic/invalid-input tests.
It defines a smooth, nonabsorbing, unpolarized parallel plate in air, index 1.52,
zero geometric thickness and incoherent internal reflections. It sums the
internal-reflection series separately for s and p polarization before averaging.
The direct transmitted beam and mirror-reflected sun were checked against
Radiance `glass` at incidence angles 0, 20, 45, 60, 75 and 85 degrees. The
maximum irradiance error was 0.0000654 W/m2; R+T energy residual was zero.
The reference traces beam-center radiance and converts using the known source
projected solid angle and receiver cosine. Commands, per-angle values and
assumptions are in `build/annual-validation/clear-glazing/`. The
[Radiance glass implementation](https://github.com/LBNL-ETA/Radiance/blob/master/src/rt/glass.c)
provides the independent comparison. Absorption, dispersion, displacement,
interference and roughness are outside this model. Native delta path and
light-connection handling still need implementation before a public material.

## Geometry timing evidence

`updates-baseline-quiet.json` and `updates-candidate-quiet.json` in
`build/annual-validation/` retain all samples, library hashes and parity checks.
The shared runner measured 131072 triangles, 66049 vertices, 65536 rays, seed 11
and three sequential revisions per edit type. Values below are median wall
milliseconds for the update alone; world-array preparation and subsequent
tracing are separately recorded. Timing excludes validation layers and profiling.

| Backend | Edit | Baseline ms | Candidate ms |
|---|---|---:|---:|
| Embree | Rigid world vertices | 6.668 | 0.994 |
| Embree | Vertex deformation | 6.882 | 0.861 |
| Embree | Changed winding/topology | 6.359 | 6.405 |
| Vulkan RT | Rigid world vertices | 3024.554 | 26.405 |
| Vulkan RT | Vertex deformation | 1842.887 | 28.769 |
| Vulkan RT | Changed winding/topology | 1812.064 | 25.227 |

Every update matched fresh-build surface IDs, sidedness and occlusion. Topology
changes intentionally rebuild Embree; Vulkan still rebuilds its acceleration
structures, but no longer recreates the device and pipeline. Stationary trace
medians were 49.56 versus 49.52 ms for Embree and 55.99 versus 53.16 ms for Vulkan.
Three repeats do not support a general stationary tracing speedup claim.
Whole-process peak working sets, including simultaneous updated/fresh scenes
and both sequential backends, were 586.0 and 587.1 MB. This is not GPU VRAM use.
An earlier five-repeat file without `-quiet` was retained but excluded after an
unidentified Python process overlapped part of that run.

The canopy end-to-end solve benchmark uses 131072 triangles, 64 sensors,
24 timesteps, 2048 samples per sensor and five warm repeats in each of three
independent process trials. Median edited solves changed from 159.56 to
129.36 ms on Embree and 2487.33 to 307.85 ms on Vulkan RT (1.23x and 8.08x).
All CPU/GPU initial and moved irradiance arrays agreed in every trial. Vulkan
stationary trial ranges overlap; no stationary throughput improvement is
established. Raw commands, hashes and all samples are in
`build/annual-validation/canopy-comparison.json`; aggregate values are in
`canopy-summary.json`. The [Vulkan guide](native-vulkan.md) includes both current
measurements and explicitly labeled historical results.

## Executed checks

The native-enabled full suite passed **327 tests, 24 skipped** in 163.62 s.
The six subsequently added glazing reference tests passed separately in 0.75 s;
they were added after full-suite collection, so they are not included in 327.
The final annual fixture/reporting tests passed 5/5 after formatting edits.
With `RAYSTRACK_NATIVE_LIBRARY` absent, the model, v2 solver, self-viewing,
Grasshopper contract/packaging/worker and native capability tests passed
**137 tests, 12 skipped** in 36.60 s. This separate compatibility check used:

```powershell
Remove-Item Env:RAYSTRACK_NATIVE_LIBRARY -ErrorAction SilentlyContinue
.local/release-venv/Scripts/python.exe -m pytest tests/test_model_v2.py tests/test_solver_v2.py tests/test_self_viewing.py tests/test_grasshopper_contract.py tests/test_grasshopper_packaging.py tests/test_grasshopper_worker.py tests/test_native_longwave.py -q -p no:cacheprovider
.local/release-venv/Scripts/python.exe -m pytest tests/test_clear_glazing_reference.py -q -p no:cacheprovider
```

The rebuilt combined-library Vulkan probe passed CTest **1/1** with Khronos
validation on the actual Radeon 860M (6.10 s). Final panel comparisons also ran
with Khronos validation. `git diff --check` passed. Logs are
`final-tests.txt`, `native-absent-tests.txt`, `ctest-final.txt`,
`panels-final-console.txt`, `clear-glazing-console.txt` and
`annual-fixture-tests.log` under `build/annual-validation/`. There is no configured
repository lint/type gate or embedded toolkit quality runner; these checks do
not imply coverage of unavailable GPUs or confidence calibration in all scenes.
