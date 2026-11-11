# File-driven moving district

Edit [`examples/dynamic_district/project.json`](../examples/dynamic_district/project.json),
then run it from the checkout. Python uses the existing NumPy/Numba environment;
there are no additional packages, binary downloads or heavy storage formats.
The native extension is optional for existing v2/Grasshopper use and required
for this shortwave workflow. Build it with the pinned
[Embree](native-embree.md) / [Vulkan](native-vulkan.md) instructions. A C++20
reference-only build also works with `--backend native-builtin`.
The checked-in JSON is 18.4 KB; its held motion lives in the separate editable
24.0 KB `poses.csv`, using UTC boundary, stable instance ID, XYZ metres and yaw
degrees about +Z. Inline 4x4 poses remain supported for arbitrary rotations.

For a fresh source checkout, install the existing base package with
`python -m pip install .`, build the optional native reference library using
`cmake -S native -B build/native-project` and
`cmake --build build/native-project --config Release`, then select that build's
DLL with `RAYSTRACK_NATIVE_LIBRARY`. Run
`python -m raystrack.project_run run examples/dynamic_district/project.json --backend native-builtin --output build/my-district`.
Embree and actual Vulkan builds use the pinned SDK guides above.
The commands below select this machine's existing environment and combined DLL.

```powershell
$env:PYTHONPATH = 'src;.'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
.local/release-venv/Scripts/python.exe -m raystrack.project_run run examples/dynamic_district/project.json --backend embree --output build/my-district
```

Use a new output directory for a changed project. Repeat the same command with
`--resume` after an interrupted run. `--max-blocks 3` creates a deliberately
partial run. Resume checks every completed chunk, source asset, solver/runner
source hash, settings and native-library hash; it recomputes exposure from
verified numeric blocks. It resumes completed intervals, not in-flight rays.

Read selected outputs without loading the year:

```powershell
.local/release-venv/Scripts/python.exe -m raystrack.project_run inspect build/my-district
.local/release-venv/Scripts/python.exe -m raystrack.project_run export build/my-district --start 4100 --stop 4104 --sensors t0/glass t0/person --output build/selected.csv
```

```python
from raystrack.project_store import ResultReader
reader = ResultReader('build/my-district')
window = reader.read_times('2025-06-21T10:00:00Z', '2025-06-21T14:00:00Z',
                         sensor_ids=['t0/glass'], components=['transmitted'],
                         bands=['neutral_R'])
print(window['values'], window['errors'])
```

The [schema guide](project-schema.md) describes the JSON, typed NPY meshes,
sensor/weather CSV and output axes. Geometry uses metres and east/north/up.
The manifest includes 5 shared mesh prototypes, 32 instances (18 pedestrian box
proxies, 8 shade slats, pavement, two facades and three canopies), 244 initial
triangles and 7 upward sensors, including sheltered, pedestrian and elevated
open reference positions. Materials include opaque pavement/facades, an ideal
clear plate, a sided diffuse transmitting sheet and the unchanged licensed
[CGDB Solar Full Klems fixture](../tests/fixtures/bsdf/ATTRIBUTION.md).
Neutral RGB stores three copies of broadband solar irradiance; these are not
three measured spectral bands. The library also accepts an explicitly synthetic
finite [`rough_transmission`](district-optics.md) material.

Weather midpoints and state boundaries are separate UTC instants. The fallback
is deterministic synthetic Berlin forcing in fixed UTC+01:00 standard time,
covering all 8,760 one-hour intervals of 2025 without DST. Longitude is positive
east; the standard meridian is 15 degrees east. DNI is direct normal W/m2;
DHI is horizontal diffuse W/m2, converted to an isotropic upper sky with a black
lower hemisphere. It is not historical weather. No local measured EPW/WEA was
verified for this project. The supported production interchange is explicit
UTC midpoint/duration/DNI/DHI CSV; EPW/WEA must be converted with its timestamp,
longitude/meridian and units conventions recorded by the caller.

Sampled absolute poses apply to local column-vector geometry as `R*x+t`.
They are held from an interval's start until replaced; there is no interpolation
or motion blur. Every event must coincide with an interval start. All edits
at a boundary occur before its solve, and all pedestrian/shade transforms commit
as one atomic batch. Unchanged matrices avoid a commit. Deformation replaces
world vertices and resets the transform; topology replacement also resets it.
The fixture freezes shade 7 after its topology event to preserve this world basis.
The annual fixture has 12 pose boundaries, an optical change at interval 4320,
facade deformation at 6480 and one removed slat triangle at 7200.

Generate a busy 15-minute day or a larger repeated district:

```powershell
.local/release-venv/Scripts/python.exe -m validation.create_district_project --busy --output build/busy-input
.local/release-venv/Scripts/python.exe -m raystrack.project_run run build/busy-input/project.json --backend embree --output build/busy-results
.local/release-venv/Scripts/python.exe -m validation.create_district_project --busy --tiles 16 --output build/scale-input
.local/release-venv/Scripts/python.exe -m validation.benchmark_dynamic_project build/scale-input/project.json --backend embree --output build/scale-results
```

The busy day covers June 21 UTC with 96 actual 15-minute intervals and a pose
boundary at every interval. Motion/material/deformation/topology replay and
resume, duration-correct exposure, selected-window export, time partition
invariance, actual person/shade shadow changes and immutable prior snapshots
are regression-tested. Native random streams are stable under time partitioning
for these contracts. Sensor partitioning would change native sample indices;
this runner instead enforces a configurable `max_sensors` admission limit.
`max_result_bytes` caps each output array by reducing time block size, with
32 MiB default. Several arrays coexist with geometry and native buffers; this is
not a total-process memory cap. Native primary path chunks are bounded; delta
branch capability limits remain explicit. Recorded peak resident memory includes
Python and native allocations; device memory is not inferred from host RAM.

| Capability | Embree | Vulkan RT / builtin |
| --- | --- | --- |
| Shared local prototype acceleration | Verified true instances | Explicit world-geometry fallback |
| Atomic poses / no-op suppression | Verified | Verified Python state semantics |
| Deformation / topology | Prototype refit / rebuild | Retained Vulkan device with AS rebuild / CPU rebuild |
| Opaque, sheet, plate, Solar Full Klems | Host transport | Same host transport; actual Vulkan hardware traversal |
| Solid dielectric sky | Existing convex/separated limits | Same limits |
| Solid dielectric DNI | Explicit capability error | Explicit capability error |
| Persistent transfer cache / GPU transport | Not implemented | Not implemented |
| Hourly / exposure retention, selective reads | NPY default, NPZ option | Same storage |

Each native solve recomputes transport from the captured geometry/material and
forcing. No persistent transfer cache or cache-hit performance claim is made.
Existing material CDF reuse is keyed by immutable optical-object identity;
geometry updates do not reuse stale lighting contributions. Existing v2 stores
and readers remain unchanged.

Independent accuracy remains the separate
[full-year opaque Radiance cube](../validation/annual-radiance.md) and
[controlled constituent optical references](district-optics.md), including
reflective/transmitting sensors, low sun, independent sun/sky and transformed
measured BSDF frames. The elevated district sensor also checks analytic open
sun/sky throughout each evidence run. These checks do not establish independent
annual accuracy of the full mixed-material district. Radiance 5.4 selects the
fixture's different Visible data; no matched measured-Solar Radiance claim is
made. Radiance reference commands follow the official
[LBNL repository](https://github.com/LBNL-ETA/Radiance) and
[rtrace manual](https://radsite.lbl.gov/radiance/man_html/rtrace.1.html).

Actual execution, storage and before/after measurements are recorded in
[`validation/evidence/dynamic-district-20261111.json`](../validation/evidence/dynamic-district-20261111.json).
Raw reports, arrays and build/test logs remain under ignored `build/`.

## Executed evidence

The 8,760-interval case used 7 sensors, 32 instances and 244 initial triangles
(243 after topology replacement), with 64 samples, two replicates, three
interactions and seed 17. Three fresh processes per backend ran sequentially,
alternating backend order. All agent/test workloads were idle. An unrelated
active Python process was observed and left untouched, so system isolation is
not established. Timings are host wall-clock measurements without validation
layers. Values below are medians, with trial ranges in the JSON evidence.
Solve/publication timers exclude fresh-reference probes. The first trial per
backend included those probes; the default CLI includes them at state boundaries,
so its total wall time also includes their extra backend builds.

| Case | Backend | Sum of solves (s) | Publication (s) | Peak host RAM (MB) |
| --- | --- | ---: | ---: | ---: |
| Full year | Embree | 1.708 | 3.541 | 99.0 |
| Full year | Vulkan RT | 6.847 | 3.821 | 260.7 |
| Busy day: 96 intervals, 32 instances, 7 sensors | Embree | 0.312 | 5.832 | 97.4 |
| Same busy day | Vulkan RT | 4.501 | 4.511 | 249.3 |
| Busy 16-cell district: 512 instances, 112 sensors, 3,904 initial triangles | Embree | 2.545 | 7.644 | 152.2 |
| Same 16-cell district | Vulkan RT | 10.457 | 7.697 | 306.9 |

The full-year first block took 0.086 s on Embree and 2.588 s on Vulkan at the
median, including lazy backend setup; later block timings are in each raw
report. Blocks differ in duration and scene state, so they are not equal-work
stationary comparisons. The largest annual result array was 112,896 bytes.
Four intervals/one sensor/one channel requested 128 value bytes; median verified
selected-read times were 11.49 ms for the Embree output and 7.19 ms for the Vulkan
output. These measure files and OS cache variation, not tracing backends.
Full hourly output occupied about 12.06 MB; a complete exposure-only run used
375,411 bytes and integrated identically. Its process peak was 100.5 MB.

All 12 annual daylight fresh-build probes per backend matched exactly; the
least probe forcing was 418.49 W/m2. The elevated open sensor's annual direct
projection residual was at most 0.0000188 W/m2 and its isotropic sky residual
below 1e-12 W/m2. The moving day ran 96 pose boundaries and all scheduled edit
types. The native instance and project regressions prove person and shade
shadow changes at fixed forcing, immutable prior results, periodic deformation
refresh, selected CSV export, hourly partition invariance and resume across
motion/material/deformation/topology boundaries.

Annual Embree/Vulkan component differences were at most 0.28925 W/m2 and
exposure differences 0.28923 Wh/m2; this is a consistency diagnostic, not an
independent accuracy gate. At the largest difference, the summed two-backend
component standard error was 14.95 W/m2; two replicates give weak precision.
The original flat DLL versus shared Embree differed
by at most 0.00000739 W/m2. The old small-fixture annual median solve was 1.615 s
(1.522–2.058 s); the shared path was 1.708 s (1.663–1.987 s). These ranges overlap,
and the new median is slower. No annual end-to-end speedup is established.

The district's deliberately low sampling has weak uncertainty: the largest
conservative component exposure one-SE bound was 26.29 kWh/m2. Multiplying by
Student-t(1) 12.7062 gives an approximate 334 kWh/m2 95% half-width. Hourly samples
are correlated, so errors are summed linearly. This is sampling uncertainty,
not a geometry/material/weather error bound. Quantitative district predictions
need their own refinement and independent accuracy budget.

The equal-input geometry benchmark used 256 moving panel instances plus one
static surface, 65,536 diagnostic rays, 32 sensors, 256 samples and five warm/edit
repeats in each of three alternating fresh processes. The repeated-mesh case
had 163,840 world triangles but 33,280 unique prototype triangles. The
static-heavy case had 131,584 world triangles and 131,074 unique triangles.

| Median metric (ms except memory) | Repeated: old flat / shared | Static-heavy: old flat / shared |
| --- | ---: | ---: |
| Native creation | 60.55 / 54.47 | 59.52 / 55.43 |
| Transform geometry packing | 4.210 / 0.003 | 3.257 / 0.003 |
| Native transform commit | 2.701 / 1.178 | 1.753 / 1.586 |
| Warm solve | 3.169 / 1.917 | 1.706 / 1.823 |
| Peak host MB, including updated+fresh handles | 137.85 / 101.88 | 124.76 / 128.99 |

Both paths produced the same output hash and all updated/fresh hit/solve
comparisons passed. Transform-only shared counters retained two prototype
builds, zero prototype refits and 1,280 transform commits across five frames.
Python scene-edit costs are reported separately; reducing native commit cost
does not eliminate them. Static-heavy warm solves regressed 6.8% and peak RAM
increased 3.4%; these are retained alongside the repeated-geometry gains.

A separate controlled material-edit benchmark kept geometry unchanged and
verified changed optics against fresh solves. Across three fresh processes,
median assignment/warm solve/fresh-build-and-solve times were 0.0109 / 4.263 / 8.097 ms
on shared Embree and 0.0144 / 16.115 / 3802.842 ms on Vulkan. Each cohort performed 15
successful geometry-reuse and fresh-equivalence checks. These edits use the
existing optical CDF cache; no persistent transport cache was introduced.

The profiled busy Vulkan run submitted 191,540 rays over 955 dispatches and 96
geometry commits on the Radeon 860M. Host phases were context 173 ms,
pipeline 578 ms, geometry packing 5.58 ms, acceleration builds 3202 ms, mapped upload
packing 0.72 ms, dispatch/wait 185 ms and readback/decode 121 ms. Total solves were
4835 ms, leaving 570 ms for host orchestration/transport and uninstrumented work.
These are host phase boundaries, not GPU timestamps or discrete PCIe copies.
The measured rebuild cost supports prioritizing shared Vulkan BLAS/TLAS before
selecting a GPU transport kernel. VRAM was not measured.

Reproduce geometry, storage, material and workflow measurements in separate
processes with the other workloads idle. Use the preserved original DLL for
`--mode flat`, then the rebuilt DLL for `--mode shared`:

```powershell
.local/release-venv/Scripts/python.exe -m validation.benchmark_native_instances --mode shared --instances 256 --grid 16 --static-grid 128 --rays 65536 --samples 256 --repeats 5 --json build/instances.json
.local/release-venv/Scripts/python.exe -m validation.benchmark_material_updates --backend embree --output build/material-edits.json
.local/release-venv/Scripts/python.exe -m validation.benchmark_project_store --output build/store-trial --storage npy
.local/release-venv/Scripts/python.exe -m validation.benchmark_dynamic_project examples/dynamic_district/project.json --backend embree --probes --output build/annual-trial
```

Repeat three times with new outputs, alternating baseline/candidate ordering.
Use `--grid 1 --static-grid 256` for the static-heavy case, `--storage npz` for
compressed storage, and `--backend vulkan-rt` for real hardware traversal.
Current raw cohorts are under `build/dynamic-district-20261111/`.
The timing inputs used inline pose matrices. Generate them exactly with
`--inline-states`; their project SHA256 reproduces the saved original. The final
CSV form was checked for identical float32 transforms at all 12 boundaries,
then run for a full year on both backends. This input-layout change does not
change the held motion model.
Those later CSV singleton runs recorded 2.513 / 11.872 s of Embree/Vulkan solves,
4.677 / 4.930 s of publication and 99.2 / 314.9 MB peak RAM. They are retained
as final workflow verification, not a paired input-format timing experiment.

Verification executed on this checkout: focused baseline 52 passed; native full
suite 545 passed/26 skipped; native-absent v2/Grasshopper 147 passed/24 skipped;
final project/storage/physics/instance/CSV/CLI contracts 71 passed; Release rebuild and
Khronos CTest 2 / 2 passed. The full suite preceded final replay/provenance
hardening and CSV interchange; the final 71-test run and CSV CLI partial/resume cover those changes.
Optional skips are Taichi, licensed headless Rhino and non-Vulkan invocations
of the slender-facet hardware regression. The independently reconstructed
annual cube retained 45 / 45 gates (max exposure 1.38842%, daylight NRMSE 1.71544%).
The final measured Solar checker passed all six backend/side groups; the
16,384-sample/four-replicate controlled mixed references passed on all three
native backends. Their largest absolute mixed residual was 0.92006 W/m2, within
the fixed 5 SE+0.15 W/m2 gate. Blocked low-sun error was exactly zero.

Final commands and raw scoped logs:

```powershell
$env:RAYSTRACK_RADIANCE_BIN = 'C:/Radiance/bin'
.local/release-venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=build/dynamic-district-20261111/pytest-full --junitxml=build/dynamic-district-20261111/full-suite.xml
.local/release-venv/Scripts/python.exe -m pytest tests/test_project.py tests/test_project_run.py tests/test_district_physics.py tests/test_native_instances.py -q -p no:cacheprovider --basetemp=build/dynamic-district-20261111/pytest-final-csv --junitxml=build/dynamic-district-20261111/final-csv.xml
.local/release-venv/Scripts/python.exe validation/annual_radiance.py --samples 2048 --replicates 2 --radiance-samples 4096 --backends native-builtin embree vulkan-rt --reuse-reference build/glass-urban-20261111/annual-reference-verified --output build/dynamic-district-20261111/annual-reference
.local/release-venv/Scripts/python.exe -m validation.district_physics --backend native-builtin --backend embree --backend vulkan-rt --output build/dynamic-district-20261111/physics-final.json
.local/release-venv/Scripts/python.exe validation/validate_measured_bsdf.py --backend native-builtin --backend embree --backend vulkan-rt --output build/dynamic-district-20261111/measured-final.json
```

The native-disabled compatibility command and build/validation-layer setup are
unchanged from the [preceding milestone](native-glass-urban-20261111.md).
No toolkit quality runner is embedded and no linter/type checker is configured;
executed tests/build/CTest and `git diff --check` are the deterministic gates.
