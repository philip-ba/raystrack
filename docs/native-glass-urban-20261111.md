# Glass, BSDF and annual urban implementation

This integrates the existing renderer working tree on
`dev/native-renderer-20261111`, starting at `6cadade8ea29a83744191fcff05062e7b713f196`.
Public v2 Python and Grasshopper behavior is preserved. Native ABI4 adds Beer
absorption to the optional material layout; actual saved ABI2/3 libraries were
rejected before scene calls. No package, SDK, toolchain or binary download was
added. The unchanged optical fixture has a separate
[CC BY-ND 4.0 attribution](../tests/fixtures/bsdf/ATTRIBUTION.md).

## Implemented slices and explicit blockers

- `SolidDielectric(refractive_index=1.52, absorption=0)` models a finite closed,
  connected, outward convex boundary in air. Geometry is in metres; n is in
  [1,10], and scalar/RGB Beer coefficients are in m^-1. Sky paths implement
  Fresnel, entrance/exit Snell refraction, TIR, distance absorption and reverse
  radiance `(eta_current/eta_next)^2` factors. Each boundary consumes an
  interaction, up to 32. Independent slab finite-series and convex-plane wedge
  quadratures check absorption and multiple events. Mixed finite/plate/solid
  sky paths and revision/fresh-build tests retain immutable snapshots.
- Solids require exterior sensors and bounding boxes disjoint from **every**
  other surface. Nonconvexity, nesting, overlaps and interior objects are
  rejected. Normal offsets perturb geometry by epsilon; omitted distance to
  the previous boundary plane is restored for Beer attenuation. Unresolved
  thickness/clearance fails explicitly. Each boundary uses unpolarized scalar
  Fresnel coefficients; polarization correlations are not transported.
- **Solid sun is unsupported:** refractive light connections need independently
  verified angular Jacobians and caustic handling. Any nonzero DNI with a solid
  raises `CapabilityError`. The prior clear-plate sun contract is unchanged.
- The [unchanged LBNL CGDB Solar Full Klems fixture](measured-bsdf-reference.md)
  adds measured-product evidence, original hash/license/conventions, both-side
  direct/diffuse/combined integral checks and sampling/evaluation/PDF diagnostics.
  Source reciprocity residuals are retained without repair. This is processed
  measurement-based product data, not raw calibrated instrument measurements.
  The installed Radiance 5.4 panel path selects Visible and the original Solar
  values differ, blocking a matched measured-Solar Radiance claim.
- [Independent wavelength intervals](native-wavelength-bands.md) require explicit
  nonoverlapping nm edges, scalar per-band optics, band-integrated DNI/DHI and
  provenance. Caller arrays are captured before solving bands. Synthetic
  two-band energy/integration checks run on all traversals. RGB and separate
  Solar/Visible blocks do not supply spectral data. Cross-band effects,
  intra-band dispersion and production spectral weather/optics are unsupported.
- [The bounded annual district workflow](annual-urban.md) executes all 8,760
  standard-time hourly intervals with mixed materials, shade, three interactions
  and scheduled transform/material/deformation/topology edits. It checks fresh
  builds, output hashes and restart at completed blocks. The separate annual
  Radiance cube remains the independent annual accuracy gate. Analytic rough
  dielectric remains unsupported; measured directional finite lobes are not
  presented as a microfacet model.

## Capabilities

V=verified by linked tests, L=limited as stated, U=unsupported. Backend columns
are builtin CPU / Embree CPU / Vulkan RT on Radeon 860M. Other hardware is
unverified. All Vulkan material/transport/reduction work remains on the host.

| Material / operation | B / E / V | Paths and bands | Edits / reproduction |
|---|---|---|---|
| Opaque, ThinDiffuse, DiffuseSheet | V / V / V | sun/sky, finite and plate mixtures, multiple events; neutral broadband/labeled RGB | material/transform/vertices/topology; [transport](../tests/test_shortwave_transport.py), [sided sheet](../tests/test_shortwave_milestone.py) |
| ClearGlazing thin plate | L / L / L | reflected/transmitted sun and sky; mixed paths; image/branch limits; achromatic in air | material/geometry; [plate](../tests/test_shortwave_glazing.py), [prior references](native-bsdf-20261111.md) |
| SolidDielectric finite solid | L / L / L | sky Snell/Fresnel/TIR/Beer, finite/plate mixtures; DNI U; n[1,10], convex separated solids | material/transform tested, geometry revalidated; [solid](../tests/test_shortwave_solid_glass.py), [native algebra](../native/scattering_test.cpp) |
| Synthetic Solar Full Klems | V / V / V | directional finite reflection/transmission, sun/sky and mixtures; analytic glossy/delta U | material, transformed tangent and geometry; [Klems](../tests/test_shortwave_klems.py), [milestone](../tests/test_shortwave_milestone.py) |
| Licensed CGDB Solar Full Klems | V / V / V | both sides, three angles, three forcing modes; annual execution L, independent annual material reference U; original broadband Solar | original hash; [checker](../validation/validate_measured_bsdf.py), [tests](../tests/test_measured_bsdf_reference.py) |
| Visible / other BSDF bases | U / U / U | Python Visible inspection only; native rejects; Half/Quarter/tensor tree U | [loader contract](native-bsdf-20261111.md) |
| Independent top-hat bands | V / V / V | explicit scalar optics and integrated W/m², isotropic DHI; inherited path restrictions | one geometry revision and captured forcing frame; [tests](../tests/test_shortwave_spectral.py), [example](../examples/ex09_independent_bands.py) |
| Annual blocks/restart | V / V / V | full-year measured urban E/V; builtin regressions; demonstrated broadband only | all four scheduled edit types; [workflow tests](../tests/test_annual_urban.py) |
| Persistent transfer cache / native prototypes / GPU transport | U / U / U | per-call sky reduction and material CDF reuse already exist | work below |

A slender prism exposed a Radeon 860M entrance-hit miss. The stress regression
requires accurate oracle agreement or explicit inconsistent-medium failure;
the moderate prism passes independent quadrature on all traversals. This
geometry-conditioning limit remains; no tolerance was widened to clear it.

## Acceptance and reproduction

The baseline corrected for the source checkout (`PYTHONPATH=src;.`) passed
110 focused tests. Pre-change DLLs and tracked patch/status were preserved in
`build/glass-urban-20261111/baseline`. The first attempt without PYTHONPATH failed
collection against an older installed wheel and is not counted as passing.
Acceptance before implementation: preserve plate/v2 behavior; slab and wedge
error within four replicate standard errors plus 0.15 W/m² and converged
quadrature; measured sheet within five standard errors plus 0.1 W/m²;
unchanged annual cube exposure error<=2% and daylight NRMSE<=5%
with existing floors; edited/fresh and restart parity at existing strict gates.
No acceptance tolerance was widened. Two/four-replicate uncertainty is limited
and covers sampling variation, not instrument or model uncertainty.

Use the pinned existing [Embree](native-embree.md) and [Vulkan](native-vulkan.md)
build instructions. The combined build enables both backend options. The local
verification commands are:

```powershell
$env:PYTHONPATH = 'src;.'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
$env:RAYSTRACK_RADIANCE_BIN = 'C:/Radiance/bin'
& 'C:/Program Files/CMake/bin/cmake.exe' --build build/raystrack-native-final --config Release
.local/release-venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=build/glass-urban-20261111/pytest-full-final --junitxml=build/glass-urban-20261111/full-suite-final.xml
.local/release-venv/Scripts/python.exe validation/validate_measured_bsdf.py --backend native-builtin --backend embree --backend vulkan-rt --output build/glass-urban-20261111/measured-bsdf.json
.local/release-venv/Scripts/python.exe examples/ex09_independent_bands.py --backend embree
$env:VK_LAYER_PATH = (Resolve-Path build/vulkan-deps/sdk/Bin).Path
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
& 'C:/Program Files/CMake/bin/ctest.exe' --test-dir build/raystrack-native-final -C Release --output-on-failure
git diff --check
```

The [annual guide](annual-urban.md) gives exact district/scale commands. Rerun
the unchanged full-year cube with the following command. Omit `--reuse-reference`
to rebuild Radiance coefficients on another machine; reuse checks exact inputs.

```powershell
.local/release-venv/Scripts/python.exe validation/annual_radiance.py --samples 2048 --replicates 2 --radiance-samples 4096 --backends native-builtin embree vulkan-rt --reuse-reference build/bsdf-milestone-20261111/annual --output build/glass-urban-20261111/annual-reference-verified
```

Final review hardened the existing reference runner: cache reuse now requires
source/coefficient hashes, reconstructs sun/sky results from verified
coefficients and recomputes the inexpensive zero-bounce primary diagnostic.
Unchecked cached hourly arrays cannot define expected irradiance. New reports
hash primary coefficients and hourly output. Acceptance requires all requested
backends, all three forcing modes and all five exposed faces; partial evidence
cannot pass. Red/green regressions and another full-year 45-gate execution
validated these changes without altering numerical tolerances.

Native-absent v2/Grasshopper and optional-interface regression command:

```powershell
Remove-Item Env:RAYSTRACK_NATIVE_LIBRARY -ErrorAction SilentlyContinue
.local/release-venv/Scripts/python.exe -m pytest tests/test_model_v2.py tests/test_solver_v2.py tests/test_self_viewing.py tests/test_grasshopper_contract.py tests/test_grasshopper_packaging.py tests/test_grasshopper_worker.py tests/test_native_longwave.py tests/test_shortwave_spectral.py tests/test_native_geometry.py -q -p no:cacheprovider --basetemp=build/glass-urban-20261111/pytest-native-absent-final --junitxml=build/glass-urban-20261111/native-absent-final.xml
```

Raw arrays/builds remain ignored. The toolkit quality runner is not embedded;
no linter/type gate is configured. Actual tests, CMake/CTest and whitespace
checks provide deterministic verification. All three implementation agent
launches explicitly requested `gpt-6.1-sol`, medium reasoning, and were accepted
by the harness; the BSDF agent then performed read-only integration review.
No available tool independently confirms served model identity. Lead model
identity/reasoning is also not introspectable.

## Executed verification

The [tracked evidence summary](../validation/evidence/native-glass-urban-20261111.json)
records source/library/fixture hashes, scoped test counts and measured capacity.
Executed on the actual checkout:

- Combined Release rebuild and CTest **2/2 passed**.
- Initial native-enabled full suite **480 passed, 26 skipped**. After the
  reference-cache fix, full rerun **484 passed, 26 skipped** in 415.20 s.
  The last requested-backend regression was added after that run's collection;
  the final focused annual checker suite passed **10/10** in 0.92 s.
- Native-absent v2/Grasshopper/interface suite **147 passed, 24 skipped**.
- Khronos-enabled real hardware checks **30 passed, 2 skipped, 63 deselected**;
  no validation-error/VUID text. Full-suite skips cover optional Taichi,
  licensed headless Rhino and two non-Vulkan instances of a hardware-only test.
- Final independently reconstructed full-year reference: **45/45 gates passed**
  across builtin, Embree and Vulkan; maximum annual exposure error 1.38842%,
  maximum daylight NRMSE 1.71544%. Original reference evidence was preserved.
- Final measured-BSDF report passed all six backend/side groups. Six quiet
  mixed-material annual urban processes completed all 8,760 intervals and
  30 fresh-build probes matched exactly. [The annual guide](annual-urban.md)
  records timing trials/ranges, RAM, uncertainty and all 17 scale cases.
- Exact staged fixture SHA-256, staged path/artifact review and whitespace
  checks passed. Licensed XML is treated as binary source data for Git diffs
  to preserve original whitespace; it is reviewed by its original hash.

Raw evidence is under `build/glass-urban-20261111/` and
`build/glass-bsdf-urban-20261111/`, excluded from the commit. Final correctness
reruns overlapped; only the explicitly quiet annual trials support timings.

## Production-scale follow-up

1. Validate refracted sun and caustics against independent slab/prism and
   matched Radiance point/disk sources before annual solid glazing. Replace
   AABB exclusions with validated geometry/medium rules before nesting or
   interior objects. Preserve separate plate/solid/finite-BSDF semantics.
2. Import production geometry/sensors and measured weather with fixed time
   conventions. Obtain licensed per-band optical/forcing data and response
   functions; keep calibration separate from independent validation.
3. Set an annual urban accuracy budget, refine independent samples/replicates
   against holdout direct/diffuse/reference hours and edits, and add a Solar-
   capable measured-material reference. Validate a rough dielectric model
   separately, including sampling/evaluation/PDF, energy and reciprocity.
4. Profile realistic geometry/sensor sizes with isolated process RAM/device
   memory. Add revision-keyed sun/sky transfer reuse with material/geometry,
   sampling, sky basis and source-direction invalidation tests; retain bounded
   outputs/checkpoint semantics.
5. Add native prototypes and measured transform-only acceleration updates,
   then GPU-resident transport where profiles justify it. Batch sensors and
   time, verify deterministic streams, and measure multiple sizes/bands before
   projecting larger districts. Small full-year runs and selected-hour sweeps
   do not establish a production runtime or a maximum machine capacity.
