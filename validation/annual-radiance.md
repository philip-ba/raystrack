# Annual neutral solar comparison

The subsequent [BSDF milestone](../docs/native-bsdf-20261111.md) reran all8760
hours after ABI3/delta transport integration. All original30 face/mode gates
passed; an additional native-builtin comparison brings this to45/45. New evidence
is `build/bsdf-milestone-20261111/annual/`, preserving old reports. A separate
full8760 direct analytic clear-plate check exists under `clear-glazing/` there;
it does not imply fullannual Radiance material coverage for glass.

Run from the existing development checkout in PowerShell. No extra Python
package or downloaded executable is required. The runner uses NumPy, the
optional native renderer, and the existing Radiance installation. It pins
`C:/Radiance/bin`; it never chooses a different Radiance from PATH.

```powershell
$env:PYTHONPATH = 'src'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
.local/release-venv/Scripts/python.exe validation/annual_radiance.py --quick --output build/annual-validation/quick
.local/release-venv/Scripts/python.exe validation/annual_radiance.py --samples 4096 --radiance-samples 4096 --output build/annual-validation/annual
.local/release-venv/Scripts/python.exe -m pytest tests/test_annual_radiance.py -q -o cache_dir=build/pytest-cache
```

The default is the full 8760-hour year. `--quick` selects twelve representative
timestamps for development. `--backends embree` restricts the native comparison;
default backends are Embree and Vulkan RT. `--report-only` retains an unsuccessful
numerical comparison without exit status 2. Errors always remain in the report.
`--skip-spots` disables independent high quality rtrace diagnostics.

The selected library must be rebuilt from source before verification. Preserve
pre-change DLLs separately when comparing implementations. Run final timing
samples without other benchmark processes. The script runs backends sequentially,
but cannot stop unrelated system workloads.

## Matched physical problem

The 1 m cube occupies `[-.5,.5] x [-.5,.5] x [0,1]` metres. World +Z is up,
+Y is north and +X east. The ground is a finite 20 x 20 m plane at Z=0.
Cube and ground use neutral Lambertian reflectance 0.2 on both sides.
The lower environment is black. The bottom face seals the cube and is excluded
from measurements because it lies in contact with the explicit ground.

Each exposed face uses four equal-area midpoint samples on a 2 x 2 subdivision.
Both engines use the identical stored sensor coordinates and outward normals,
offset 0.1 mm from the face. Raystrack adds its documented 0.01 mm ray offset.
Two upward ground sensors at `(0,-1,.0001)` and `(0,1,.0001)` capture sun and
shadow conditions. An open upward sensor at `(0,0,3)` checks direct DNI
projection. The fixture's ground contact introduces coincident triangles only
on the buried face; outgoing exterior sensor rays never need to distinguish
these coincident intersections.

Forcing is **synthetic**, not historical Berlin weather. The tracked function
`synthetic_forcing()` defines smooth seasonal and multiday clear/cloud modulation
of attenuated DNI and nonnegative DHI, with exact night zeros. Save its complete
WEA and NumPy arrays with each run. Berlin coordinates are 52.52 N, 13.405 E,
year 2025, fixed standard time UTC+01:00, one-hour intervals sampled at their
midpoints. UTC timestamps and actual durations are saved. Neither DST nor an
EPW year substitution is applied.

Radiance `gendaymtx -s -O1 -m 1 -c 1 1 1 -g 0 0 0` supplies the common diffuse
Perez/Reinhart sky matrix. Its ground row is dropped. The diffuse matrix's
geometric solar model is internal to Perez generation; both renderers receive
the same saved diffuse matrix, while the separately controlled direct sun uses
Raystrack's geometric solar directions in both engines. The Reinhart patch
integral can differ slightly from the supplied DHI; report that discretization
residual rather than renormalizing one renderer alone.

## Radiometric equivalence and bounce counts

Use neutral equal RGB channels as three copies of the same broadband reference
quantity. Radiance `rcontrib -I+ -V+` gives irradiance contributions in each channel.
Unit-radiance sky coefficients are multiplied by W/(m2 sr), producing W/m2.
For an unoccluded upward sensor, a uniform sky of radiance 1 integrates to pi.
The executable-backed test verifies this and verifies a unit-DNI source with
sun-direction Z=.8 gives irradiance .8. No 179 lm/W factor or lux conversion is
used anywhere.

Direct sun is never reconstructed from a coarse sky patch. Each actual sun-up
hour has an independently named distant source, oriented along the exact common
sun vector. Source diameter is 0.01 degrees and radiance is the reciprocal of
its projected solid angle `pi*sin(diameter/2)^2`, giving unit DNI. Multiply that
source's coefficient by its own hour's DNI. This tiny finite disk approximates
Raystrack's point sun and can differ at a shadow boundary; it is not the physical
0.533-degree solar disk. The source file and modifier order are retained.

The default includes at most one surface scattering event. The virtual Radiance
irradiance receiver consumes one ambient level for sky sampling, so sky uses
`-ab 2` while explicitly sampled sun uses `-ab 1`. More generally these are
`bounces+1` and `bounces`. Direct-only, diffuse-only, and combined runs refer to
the forcing component and retain this common maximum material interaction count.
Native combined minus native isolated sums is reported to check linearity.

Radiance uses one process, zero ambient caching error, no ambient supersamples,
deterministic sampling, `-lw 1e-8`, exact source testing (`-dt 0 -dc 1`), zero
source jitter, and a fixed depth cap. Every invoked command, exit status,
stderr, wall time and input hash is retained in `radiance_commands.json`.
The tested installation is Radiance 5.4, `5.4.4ee32974b1`, 2023-11-05 LBNL.

## Acceptance and uncertainty

Before tuning, targets were declared as 2% annual exposure error and 5% daylight
normalized RMSE for **each exposed face and each forcing mode**. Daylight is
defined by the common geometric sun being above the horizon. NRMSE divides
daylight RMSE by mean daylight reference irradiance, with a 1 W/m2 floor.
Annual relative errors use a 1 Wh/m2 denominator floor. Below 1 W/m2, report
absolute error rather than a misleading relative percentage. Deterministic
analytic conversion tests use 1e-5 relative tolerance.

`report.json` contains per-face and aggregate MAE, RMSE, bias, exposure, explicit
pass/fail gates, twenty worst timestamp/group errors per mode, ground shadow
diagnostics and night residuals. Independent native replicates provide component
standard errors. Hourly transfer samples are reused, so annual uncertainty is
bounded with a linear sum of standard errors; independent-hour quadrature would
be unjustified. A separate seed with four times the samples at low sun, high sun
and a ground shadow transition checks native convergence. High-quality `rtrace`
spot runs independently reconstruct the actual piecewise sky and single sun,
checking coefficient reconstruction and Radiance sample convergence. These
checks do not establish universal confidence coverage for all scenes.

Files `forcing.npz`, `radiance_hourly.npz`, `embree_hourly.npz`, and
`vulkan-rt_hourly.npz` share the same timestep and sensor order. Each engine's
hourly file contains `direct`, `diffuse`, and `combined` arrays with axes
`(timestep,sensor,RGB)`. `forcing.npz` stores labels, points, normals, timestamps,
durations, sun, DNI, DHI, and patch radiance. Coefficient `.npy` files preserve
the expensive Radiance setup separately from matrix application. `report.json`
records cold/steady native calls and coefficient command timings; these are
different algorithms and setup/trace/application costs must remain visible.

Primary references:
[rcontrib manual](https://radsite.lbl.gov/radiance/man_html/rcontrib.1.html),
[rtrace manual](https://radsite.lbl.gov/radiance/man_html/rtrace.1.html),
[Radiance source](https://github.com/LBNL-ETA/Radiance), and
[Reinhart binning](https://github.com/LBNL-ETA/Radiance/blob/master/src/cal/cal/reinhart.cal).
Radiance is separately distributed under its upstream BSD-style license.
No Radiance binaries are redistributed by this validation runner.

## Executed 2026-11-11

The pre-change DLL was rebuilt and saved under
`build/annual-validation/baseline/`. The full annual baseline is in
`build/annual-validation/baseline-annual/`; the integrated candidate is in
`build/annual-validation/candidate-annual/`. The candidate validates every
forcing/sensor array, geometry text and available saved hashes before reusing
the unchanged Radiance coefficient result. Original missing baseline metadata
is listed explicitly in its reference provenance. Baseline acceptance was
calculated separately in `acceptance-supplement.json` because its run began
before the report gate was implemented.

Both backends passed all 30 exposed-face/mode gates over the full 8760 hours.
Native sampling was 2048 per sensor and two independent replicates; Radiance
coefficients used 4096 ambient samples. Combined annual neutral exposure:

| Face | Radiance kWh/m2 | Embree kWh/m2 | Embree error |
|---|---:|---:|---:|
| Top | 905.932 | 905.710 | -0.0245% |
| North | 203.885 | 202.965 | -0.4514% |
| East | 615.523 | 613.982 | -0.2504% |
| South | 891.093 | 888.615 | -0.2782% |
| West | 616.328 | 616.755 | +0.0692% |

Vulkan produced effectively the same face metrics. The maximum exposed-face
annual errors by forcing mode were 0.116% direct, 1.388% diffuse, and 0.451%
combined. Maximum daylight NRMSE was respectively 0.250%, 1.715%, and 0.627%.
This includes hourly errors rather than just annual cancellation. A deliberately
smaller 512-sample quick case had diffuse exposure errors up to 3.77%; increasing
samples and checking an independent seed resolved that sampling discrepancy.
No material coefficient was fitted to the reference.

Independent Radiance rtrace spot runs used 32768 ambient samples at low sun
on October 24, a ground shadow transition on June 24, and maximum sun elevation
on June 21. Maximum coefficient reconstruction discrepancies over all sensors
were 0.218 W/m2 for sun and 0.353 W/m2 for diffuse sky. Independent native
8192-sample refinement changed the exposed-face combined spot estimates by at
most 0.673 W/m2. The open direct projection residual was 0.0000531 W/m2.
The Reinhart integral's DHI residual was at most 0.489 W/m2, shared by both
engines. These are sample/discretization checks, not a universal error bound.

Radiance coefficient setup took 0.393 s for diffuse sky, 243.315 s for 4409
independent sun sources, and 0.292 s for the primary direct diagnostic matrix;
two octree compilations took 0.060 s. Sky generation took 0.637 s and diffuse
annual matrix application 0.119 s. The candidate annual combined solve took
9.934 s on Embree and 13.193 s on Vulkan, followed by warmed calls of 10.691 s
and 13.975 s. These are individual end-to-end samples on this laptop, not a
stationary speedup claim. The candidate run, including separate forcing modes,
spot checks, refinements, repeated calls and I/O, took 90.751 s while reusing the
reference coefficients. The original coefficient run occurred during parallel
development, so its timings retain that qualification.

Radiance amortizes its coefficient setup over later weather variants; native
transport currently repeats direct shadow connections on each solve. Compare
first-run setup and warmed application separately. A revision-keyed native
transfer cache is a remaining performance opportunity.

`candidate-annual/validation-supplement.json` adds conservative annual 95%
Student-t intervals, complete hourly backend differences, and coefficient/forcing
hashes without rewriting the original run. With only two independent replicates,
the t multiplier is 12.7062 and interval precision is weak; the independent-seed
convergence checks are retained alongside these intervals. The full-array maximum
Embree/Vulkan difference was 0.024401 W/m2 for direct and combined forcing and
zero for diffuse forcing. Exact CPU/GPU sample equality is not an acceptance gate.

The reporting supplement can be reproduced without retracing:

```powershell
.local/release-venv/Scripts/python.exe -c "from validation.annual_radiance import supplement_saved_run; supplement_saved_run('build/annual-validation/candidate-annual')"
```
