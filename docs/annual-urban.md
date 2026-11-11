# Bounded annual urban workflow

`python -m validation.annual_urban` evaluates all **8,760 actual hourly intervals**
of the deterministic Berlin 2025 forcing used by `annual_radiance.py`. It uses
fixed UTC+01:00 standard-time interval midpoints, geometric sun directions, an
isotropic upper sky, black lower environment, metre geometry, W/m2 irradiance
and Wh/m2 exposure. This is synthetic weather, not an EPW or measured year.
The existing [annual cube comparison](../validation/annual-radiance.md) remains
the independent Radiance accuracy evidence; the urban run establishes workflow,
revision correctness and measured capacity, not independent annual accuracy.

The default fixture is one 12 m courtyard cell, 24 triangles and five upward
point sensors. Repeated cells contain opaque ground and two walls, a clear
plate canopy, a sided diffuse transmitting canopy, a finite Full Klems canopy,
and six staggered shade slats. Sensors sample each canopy, the shade and an open
part of the courtyard. Three surface interactions are enabled by default.
Clear glazing uses the existing lossless zero-thickness plate; it does not
model solid glass or refraction. Default Klems coefficients are synthetic,
reciprocal finite lobes with bounded energy, not measured materials.

For measurement-based data, supply `--bsdf-xml
tests/fixtures/bsdf/2011-SA1.XML`. This loads the unchanged original Solar
Full Klems matrix with its distinct front/back lobes. Its original SHA256 is
`5f398e08a81fecd57f9e566fa2a16f08bb0f2ca2a22fdba04d5d949c818b3336`.
Retain the fixture's accompanying provenance/license files and attribution.
The original is measurement-based CGDB product 6006 under CC BY-ND 4.0, with
Solar ASTM891 direct1.5/Flat spectral labels; no matrix symmetrization or
Solar/Visible substitution occurs. The fixture's approximation to reciprocity
is not asserted as exact. The [CGDB license](https://windows.lbl.gov/cgdb-license)
governs that original data. Supplying it does not establish annual Radiance
material parity: the installed Radiance reference path selects Visible data.
Only that exact pinned content hash receives the measurement-based label;
other supplied XML is labeled user supplied with measurement provenance
unverified, and the default remains explicitly synthetic.

Four absolute scheduled changes affect every cell: shade translation at hour
2190, diffuse-sheet optical coefficients at 4380, wall vertex deformation at
6570 and removal of one shade triangle at 7300. Each event is a block boundary.
Geometry and material revisions are recorded per block. A material edit keeps
the acceleration structure eligible for reuse. First and post-edit three-step
probes compare the live solver with a fresh solver at identical sampling and
forcing; completed result snapshots remain immutable. Regression tests also
compare an uninterrupted solve with a checkpoint resumed after translation and
continued through material, deformation and topology edits.

## Reproduce

Use the existing build/dependency setup in [native-shortwave.md](native-shortwave.md).
No extra dependencies or binaries are required. From the repository root:

```powershell
$env:PYTHONPATH = 'src;.'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.local/release-venv/Scripts/python.exe -m validation.annual_urban --backend embree --tiles 4 --bsdf-xml tests/fixtures/bsdf/2011-SA1.XML --output build/glass-bsdf-urban-20261111/annual-urban-final-embree-trial1 --sweep
.local/release-venv/Scripts/python.exe -m validation.annual_urban --backend vulkan-rt --tiles 4 --bsdf-xml tests/fixtures/bsdf/2011-SA1.XML --output build/glass-bsdf-urban-20261111/annual-urban-final-vulkan-rt-trial1
```

The final evidence repeats these commands in three fresh processes per backend,
alternating Embree/Vulkan sequentially. Replace `trial1` with `trial2`/`trial3`
and omit `--sweep` after the first Embree process. The saved orchestration body
is `build/glass-bsdf-urban-20261111/run-annual-final.ps1`; its commands were
executed directly because this machine disables PowerShell script-file
execution. No execution policy was changed. Validation layers and
`RAYSTRACK_VULKAN_PROFILE` were unset for these runs.

The four-cell annual case has 96 initial triangles, 48 surfaces and 20 sensors.
Defaults are 64 samples, two randomized independent replicates, seed17 and
three surface interactions. This deliberately modest sampling demonstrates a
complete mixed-material annual workflow. It does not guarantee small hourly
Monte Carlo error; increase samples/replicates for quantitative decisions.
Reports retain maximum component standard errors per block and conservative
integrated standard-error bounds. Hourly reuse induces covariance, so annual
errors are summed linearly, not in quadrature. For two replicates the report
also supplies approximate 95% Student-t(1) half-widths; their precision is weak
and they describe sampling error, not geometry/material/model error.

## Batching and restart

The workflow sends at most 168 timesteps per solve by default. It writes each
component irradiance and standard error, timestamps and durations to a separate
compressed NPZ before atomically replacing `report.json`. Exposure is accumulated
from explicit `irradiance * duration_hours`; completed output blocks are never
loaded together in memory. Native and NumPy intermediate arrays still exist
inside each solve; the reported `largest_result_array_bytes` bounds one output
array, not total solver memory. Sensor count and samples are not independently
batched by this workflow.

Resume the exact command with `--resume`. Existing reports otherwise raise an
error. A checkpoint compares forcing, fixture/sensors, parameters, workflow,
solver/optical source and compiled library hashes, then verifies all committed
block checksums and replays scene edits. Changed provenance fails explicitly.
A crash before report publication recomputes the uncommitted block. Restart is
at completed block boundaries, not inside native path tracing; atomic rename is
process-interruption protection, not a power-loss durability guarantee. Disk
capacity remains proportional to the number of retained hourly outputs.
Use one writer per output directory; concurrent processes targeting the same
checkpoint are unsupported and no interprocess lock is provided.

## Capacity, timing and limits

`--sweep` records separate measured cases for 1/4/16/64 cells, 1/4/16/64 panel
subdivisions, 5/20/80 sensors per cell, 24/72/168 timesteps and 1/2/4 material
families. Other dimensions remain fixed in each case. The largest geometry case
has 49,164 triangles and five sensors; the largest district has 1,536 triangles,
768 surfaces and 320 sensors. These are actual selected January-hour runs,
not full-year runtime or maximum-capacity projections. All cases retain a fresh
solver parity probe. `scale-sweep.json` keeps every dimension and actual timing.

The report separates full-block solve and compressed-output write time from
three-step cold/post-edit, repeated and fresh-build probes. Probe workloads
match each other; the first full block also has an equal-size repeated solve.
Do not divide a three-step probe time by a 168-hour block time to claim speedup.
The OS resident high-water mark includes native allocations and is a process
lifetime maximum. Sweep cases run sequentially in one process, so their peaks
are cumulative and cannot be read as isolated per-case allocation costs. No
VRAM meter is supplied. Timing depends on competing work, driver caches and
thermals; a single run establishes execution cost, not a stable speed ratio.

This workflow measures broadband neutral channels: three representations of
the same radiometric quantity. It does not treat RGB as wavelength bands or
claim multi-band capacity. Independent spectral transport, measured fixture
validation and annual reference accuracy require their own gates. Native
transport still executes on the host for Embree and Vulkan hardware traversal.
There is no persistent revision-keyed transfer cache or native shared-prototype
acceleration here. Native sky reduction reuses paths within a call, and the
solver reuses material CDFs; repeated workflow calls retrace. Block solve/write
phases expose the current costs, but do not isolate GPU dispatch, host transport
or driver time sufficiently to justify a GPU-resident implementation by themselves.

## Verification

```powershell
New-Item -ItemType Directory -Force build/annual-urban-temp | Out-Null
$env:TEMP = (Resolve-Path build/annual-urban-temp).Path
$env:TMP = $env:TEMP
.local/release-venv/Scripts/python.exe -m pytest tests/test_annual_urban.py tests/test_annual_radiance.py tests/test_shortwave_transport.py -q -p no:cacheprovider
```

The tests exercise actual compiled tracing for blocked/unblocked equivalence,
restart including all four revisions, checksum and configuration rejection,
fresh-build edit parity and immutable snapshots. Fixture/time contracts run
without the optional native library. Backend parity is a conformance check,
not independent model validation or a replacement for stochastic convergence.

## Executed 2026-11-11

All six final processes completed every hour with the original measurement-based
CGDB Solar fixture. All 30 initial/post-edit fresh probes matched exactly;
annual exposure repeated exactly across the three processes of each backend.
The existing Ryzen AI 7 350 / Radeon 860M host, Embree 4.4.1 and native ABI4 build
were used with other agent workloads idle during timing. The report records
the exact library, Python workflow, optics, fixture and forcing hashes.

| Measured quantity | Embree | Vulkan hardware traversal |
|---|---:|---:|
| Sum of 8760-hour block solves, median [range] s | 4.383 [4.286, 4.653] | 11.825 [11.442, 11.983] |
| Initial three-step cold solve, median [range] s | .0593 [.0555, .0727] | 2.165 [1.882, 2.301] |
| Same three-step repeat, median [range] s | .00303 [.00268, .00323] | .01541 [.01507, .01621] |
| First 168-hour repeated block, median [range] s | .0466 [.0401, .0484] | .1621 [.1467, .1859] |
| Process peak resident RAM range, decimal MB | 103.694–104.006 | 283.242–285.495 |

The block solve total excludes cold probes, edited probes, fresh rebuilds and
output writes. In trial1, compressed block writes took 1.303s Embree/1.393s
Vulkan. The five fresh-build correctness probes additionally took .0343s and
14.623s respectively. Those rebuilds are validation overhead, not recurring
annual solver work. Matched three-step edit-commit-plus-solve medians, seconds:

| Edit | Embree | Vulkan |
|---|---:|---:|
| Shade translation | .00678 | .02750 |
| Sheet optics | .00832 | .01817 |
| Wall deformation | .00784 | .01578 |
| Shade topology | .00548 | .01219 |

Every CPU/Vulkan trial compared all hourly arrays. Maximum difference was
.01977W/m2 in the transmitted component, with other components identical;
maximum annual component exposure difference was .01977Wh/m2. These common-host,
same-seed comparisons demonstrate conformance only. Maximum reported component
standard error was 50.282W/m2 at this low sample count; no annual material
accuracy tolerance is inferred from this workflow run.

The larger actual selected 24-hour scale cases completed at 64 cells/1536 triangles/
320 sensors in .2260s of block solves, and 49,164 triangles/5 sensors in .01779s.
Those dimensions differ and their solve totals exclude fresh-build probes;
neither case establishes a capacity ceiling or predicts larger annual runtime.

Raw arrays, probes, hashes and per-trial ranges are in
`build/glass-bsdf-urban-20261111/annual-urban-final-{embree,vulkan-rt}-trial{1,2,3}/`.
The first Embree directory holds all 17 measured scale cases and
`scale-sweep.json`. The common summary is
`build/glass-bsdf-urban-20261111/annual-urban-final-summary.json`, generated by
the accompanying `summarize-annual-final.py`. No general speedup is claimed.
