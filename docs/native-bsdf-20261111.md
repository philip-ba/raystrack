# BSDF transport milestone, 2026-11-11

The later [glass and urban implementation report](native-glass-urban-20261111.md)
supersedes this session's ABI3, missing-measured-fixture and missing-solid-glass
status. The evidence and counts below remain the historical milestone record;
current builds require ABI4.

This work extends the existing dirty development tree at `6cadade` on
`dev/native-renderer-20261111`. It preserves the v2 and Grasshopper APIs and
introduces no dependency, toolchain or binary download. Existing versions and
licenses remain those in [Embree](native-embree.md) and [Vulkan](native-vulkan.md).
Raw evidence belongs to ignored `build/bsdf-milestone-20261111/`; its `baseline/`
contains the pre-change combined DLL and runtime dependencies with SHA-256 hashes.

## Numerical contract and implementation

`ClearGlazing(refractive_index=1.52)` is an achromatic, nonabsorbing,
zero-thickness parallel plate in air. Each polarization's incoherent internal
reflection series is summed before averaging unpolarized light. Both external
media are air, so no external eta-squared radiance correction is needed. Net
transmission continues straight; reflection follows the geometric normal.
The model excludes thickness, displacement, absorption, dispersion, coherent
interference, polarization state transport, volumetric Snell refraction and TIR.
Each sheet event uses scalar unpolarized coefficients; polarization correlations
between successive oblique panes are not transported. The normal-incidence
multi-plate energy gate has equal s/p coefficients and avoids that limitation.
Its Python `DeltaSample` reports dimensionless event coefficients and probability
masses separately from finite BSDF densities and solid-angle PDFs.

`DiffuseSheet(front_reflectance, back_reflectance, transmittance)` permits
different Lambertian reflection on winding sides and reciprocal shared RGB
transmission. Each side obeys rho+tau<=1, with the remainder absorbed.
`ThinDiffuse` retains its symmetric meaning. Coefficients are dimensionless,
finite BSDFs are sr^-1, radiance W/(m2 sr), irradiance W/m2 and exposure Wh/m2.
Neutral channels repeat one broadband radiometric quantity; colored inputs
continue to require a channel-basis label.

Native sampling, finite evaluation, delta continuation and reductions run in
host C++ for all backends. Vulkan RT runs hardware traversal on the Radeon 860M;
`device="vulkan"` continues to mean Taichi software traversal. The builtin CPU
traversal is separate from the legacy v2 CPU/Numba solver.

Sun transport partitions deterministic all-delta connections from sensors and
delta connections following finite scattering events. Random path escapes add
sky only. This avoids counting the same solar energy in NEE and sampled delta
paths. Image-sun directions are generated from actual clear-surface world normals;
geometry traversal verifies each path, including blockers. Every physical surface
event consumes one of `bounces` (0..32); a direct unscattered beam consumes none.
Output reflection/transmission components retain the first interaction at the
sensor. `samples>0` remains required when `bounces>0`, even for deterministic sun.

Exact direction algebra uses doubles, geometry rays use floats. Algebraic image
identity uses a 1e-12 per-coordinate tolerance. Distinct images closer than 2e-6
per coordinate fail explicitly rather than being merged. Limits are 4096 image
directions per sun/depth, 4096 distinct mirror normals, 1048576 image construction
steps per sun/depth, 1048576 cached directions per solve, 65536 visited branches
per connection, and 1048576 active branches.
Exceeding them raises `CapabilityError`, with no partial result or fallback.
Near-ambiguous facet normals likewise require explicit capability failure.
These conservative bounds limit complex faceted/nonparallel glazing scenes.

The C++/ctypes material interface is deliberately ABI **3**. ABI2 libraries are
rejected before scene calls; they cannot interpret a new plate as diffuse.
Optical edits retain acceleration, geometry edits refresh normal metadata, and
completed result arrays and revision stamps remain immutable.

## Formats and provenance

WINDOW XML remains one optical layer, Full Klems 145x145, Rows/Columns,
explicit Solar/Visible selection, four front/back lobes in sr^-1 indexed
[incident,outgoing]. Missing reverse transmission is derived by the existing
reciprocity orientation permutation. Python forward/reverse sampling and PDF,
native reverse sampling and transformed material tangent retain their contracts.
These are finite angular lobes, including directional/anisotropic transmission;
they are not analytic glossy models or specular deltas.

Imports now attach frozen `KlemsSource` metadata: original-byte SHA-256, filename,
band, order, winding mapping, supplied/derived lobes, Material XML and spectral
labels. Explicit unsupported units, axes, structures and data types are rejected.
Text-stream hashes identify their UTF-8 encoding; file/binary imports hash exact bytes.
Missing unit/type declarations retain the documented implicit WINDOW density
convention. Keep the original licensed document alongside metadata: arbitrary XML
comments and license text are not copied into the metadata object. The
[LBNL format manual](https://windows.lbl.gov/sites/default/files/Downloads/Solar%20BSDF%20documentation.pdf)
is the format authority. No licensed measured fixture was found locally; the
existing optical matrices and new loader tests are controlled synthetic data.
There is no new measured-data format, Half/Quarter Klems, custom basis or tensor
tree support in this milestone.

## Coverage matrix

All listed native paths use the same host transport. B=builtin CPU traversal,
E=Embree CPU traversal, V=Vulkan RT hardware traversal. Each backend cell is
tracked separately in the [machine-readable coverage matrix](../build/bsdf-milestone-20261111/capability-matrix.json).

| Material / capability | B | E | V | Loader, evaluation / sampling | Sun / sky / mixed | Edits | Annual evidence |
|---|---|---|---|---|---|---|---|
| [Opaque front/back RGB](../tests/test_shortwave_transport.py) | supported | supported | supported | Python value, native cosine | direct, sky, finite/delta mixtures | material + geometry/fresh | [full cube B/E/V](../build/bsdf-milestone-20261111/annual/report.json) |
| [ThinDiffuse symmetric RGB](../tests/test_shortwave_transport.py) | supported | supported | supported | Python value, native two hemispheres | sun, sky, two panels | material + geometry/fresh | [selected panels](../build/bsdf-milestone-20261111/panels/report.json); no full-year material gate |
| [DiffuseSheet reciprocal sided RGB](../tests/test_shortwave_milestone.py) | supported | supported | supported | new value, same native finite kernel | sun, sky, mixed with glass | material + winding + geometry/fresh | no full-year material reference |
| [Full Klems Solar, finite directional](../tests/test_shortwave_klems.py) | supported | supported | supported | XML/in-memory; eval/forward+reverse PDF Python; native reverse CDF | sun, sky, rotation, glass plus finite event | material + tangent transform | selected panels; no full-year material gate |
| Full Klems Visible | inspect only | inspect only | inspect only | Python import/eval/sampling | solver rejects non-Solar | n/a | unsupported native forcing |
| [ClearGlazing plate delta](../tests/test_shortwave_glazing.py) | supported, bounded | supported, bounded | supported, bounded | analytic coefficients/discrete mass, native events | beam/reflected sun, sky, nonparallel and multiple glass, finite mixtures | material + geometry/fresh | [full8760 direct analytic plate](../build/bsdf-milestone-20261111/clear-glazing/report.json); selected Radiance/diffuse, no fullannual Radiance glass |
| Licensed measured fixture / new basis | unsupported | unsupported | unsupported | no fixture/new basis | unsupported | n/a | none |
| Analytic glossy / solid refraction | unsupported | unsupported | unsupported | no model | unsupported | n/a | plate reference does not establish these |

Evidence paths: `tests/test_shortwave_transport.py`, `test_shortwave_glazing.py`,
`test_shortwave_milestone.py`, `test_shortwave_materials.py`, `test_shortwave_klems.py`,
`test_bsdf_vulkan.py`, `test_glazing_radiance.py`, and native geometry/update/longwave
tests. Reference runners are `validation/validate_clear_glazing.py`,
`validate_shortwave_panels.py` and `annual_radiance.py`.

## Verification and performance

The reproducible command record and machine-readable result manifest are
`commands.ps1` and `session-summary.json` under the evidence directory. Test logs,
JUnit XML, full annual arrays/reports, reference commands and raw timing trials
remain there. Executed checks:

- Fresh combined Release source build succeeded; saved ABI2 DLL rejection passed.
- Full native-enabled suite: **419 passed, 24 skipped** in 244.22 s.
- Native-absent v2/model/self-view/Grasshopper suite: **137 passed, 12 skipped**
  in 39.46 s.
- New plate transport regressions: **36 passed** across all three backends.
- Glazing/annual executable reference tests: **14 passed** in 9.59 s.
- Final explicit sided-sheet sun+sky and tangent-rotation checks: **6 passed**
  in 9.12 s after strengthening the sheet fixture's diffuse forcing.
- Khronos hardware selection: **21 passed, 38 deselected**; real Radeon 860M
  CTest **1/1 passed**, with no validation-error/VUID text.
- Original 30 full8760 face/mode gates passed again on Embree/Vulkan; adding
  builtin gives **45/45 gates passed**. All direct/diffuse/combined modes retain
  worst timestamps, low/high-sun/shadow-transition and independent-seed diagnostics
  in `annual/report.json`. Exact forcing/geometry/coefficient provenance checks
  passed before reusing unchanged Radiance coefficients.
  `annual/validation-supplement.json` preserves conservative annual 95% Student-t
  intervals (two independent replicates, weak interval precision), exact input
  hashes and hourly backend differences; independent-seed refinements remain
  separate convergence evidence. Maximum annual errors were .1159% direct,
  1.3884% diffuse and .4515% combined; maximum daylight NRMSE .2498%, 1.7155%
  and .6270%, respectively.
- Clear plate six-angle native/analytic maximum error 9.72e-6 W/m2 and
  native/Radiance maximum 6.54e-5 W/m2. Diffuse hemispherical quadrature error
  .0504 W/m2, combined linearity residual 5.69e-14 W/m2. Separate full8760 direct
  analytic plate execution on all three backends has maximum error 2.58e-5 W/m2,
  exposure relative error 1.52e-10 and exact night zero. This is not a fullannual
  Radiance glass material validation.

Cross-review caught native overflow for extreme finite refractive indices and
ambiguous close facet-normal matching; both now have regressions. A depth32
test expectation was corrected to honor the existing first-event component
contract (T-R-T belongs to transmitted). No acceptance tolerance was widened.

Acceptance was fixed before checking: existing beam rtol=2e-5, atol=.002 W/m2;
full annual per-face exposure<=2%, daylight NRMSE<=5%, with 1 Wh/m2 and 1 W/m2
floors. Stochastic mixed tests use independent replicates and analytic quadrature,
with declared component standard-error gates. No material fitting or post-failure
tolerance inflation is used. Point-sun silhouette differences from the finite
Radiance disk remain explicit; controlled scattering sensors avoid boundaries.

Pinned Radiance is C:/Radiance/bin 5.4 (5.4.4ee32974b1). The installed matrix path
selects Visible empirically; panel parity duplicates controlled Solar coefficients
into Visible and labels equivalent coefficients. Unequal front/back reflection
retains reciprocal transmission for valid adjoint reference comparisons.

Final timing runs were sequential, with all agents idle, validation/profiling
disabled and OPENBLAS/MKL/OMP thread counts set to one. Native host transport
uses one thread; Embree construction keeps the SDK default worker policy. The
machine remains Ryzen AI 7 350 / Radeon 860M, 16 logical CPUs, Windows 11,
Python3.13.9, Embree4.4.1 and the documented Vulkan SDK/driver.

The unchanged `benchmark_radiance_matched.py` used 2048 samples, 2 independent
replicates, seed17, maximum one material interaction and Radiance4096 samples.
Both versions passed accuracy gates and every edited native result matched a
fresh build. The saved ABI2 DLL comparison uses an ignored Python package copy
with only its ctypes layout/version gate adapted; `baseline-adapter.json` records
the exact adapter hashes. Timed materials are Opaque/ThinDiffuse, not a glazing
throughput benchmark. This preserves a finite-path regression comparison.

Seconds, first result / warmed evaluation / edit plus result:

| Case | Radiance | Embree candidate | Vulkan RT candidate | Process trials x warm repeats |
|---|---|---|---|---|
| Cube, 12 selected hours | .68380 / .0001798 / .59935 | .10808 / .05873 / .06328 | 2.69536 / .11579 / .13232 | 3 x 3 |
| Canopy, 24 steps, 131072 triangles | 5.87822 / .0007982 / 6.06330 | .37329 / .31614 / .32334 | 2.78963 / .61151 / .58974 | 1 x 3 |

The preserved-DLL warmed times were .08122/.13200s for cube Embree/Vulkan and
.29524/.58471s for canopy. Candidate canopy warmed calls were about7.1%/4.6%
slower in these single-process samples. Quick medians improved, but the tests
do not establish a general speedup or a stable performance regression. Versions
ran sequentially baseline then candidate; OS/driver caches and thermal variation
were not reset or counterbalanced. All raw repetitions, ranges, commands, library
hashes and ratios remain in `timing-{quick,canopy}-{baseline,candidate}/` and
`performance-summary.json`. The update runner's raw refit/rebuild/fresh parity
measurements are `updates.json` and are not renderer speed ratios.

Radiance warmed evaluation applies cached coefficients; native calls retrace.
First and edited Radiance results include coefficient construction and octrees.
The native first result includes geometry/device setup. The fullannual cube gate
reused exact-checked Radiance coefficients, so its reported prior coefficient
setup cost is historical, not a newly measured cold reference. The 12-hour and
24-step edit timings cannot be extrapolated into annual edit runtime. GPU-resident
BSDF transport, native shared prototypes and revision-keyed transfer caching
remain separate performance work.

Four child roles were launched explicitly with `gpt-6.1-sol`, medium reasoning.
The harness permits three simultaneous children, so the initial Vulkan inventory
released its slot for the fourth Radiance role, then resumed for hardware checks.
This session exposes no tool to change or independently verify the lead model;
no claim of verified lead model identity is made.
