# Native renderer roadmap

Branch `dev/native-renderer-20261111`. Planning baseline `6cadade`.

The [native solver guide](native-shortwave.md) describes the implemented API,
its validation and the remaining performance and material work. The sections
below retain the original implementation plan and preflight evidence.

## File-driven moving district, 2026-11-11

The [moving district workflow](dynamic-district.md) adds editable versioned
projects, typed mesh assets, sensor/weather/held-pose CSV tracks, atomic transform
batches, selected-window readers, exposure retention and strict block resume.
Embree shortwave now shares prototype acceleration with true rigid instances;
deformation refits where prototype topology is retained and topology changes
rebuild. Vulkan RT retains its tested device/pipeline reuse with explicit flat
geometry fallback. The new synthetic finite Klems rough transmission sheet is
reciprocal and energy bounded; it does not implement microfacet glass.

Next priorities, in dependency order:

1. Convert verified production EPW/WEA/geometry into the documented interchange,
   define district error budgets and validate hold/interpolation against finer
   motion samples. Current synthetic annual execution is not annual accuracy.
2. Add tested Vulkan shared BLAS/TLAS instancing and update flags, then compare
   device build/transfer/dispatch/readback on realistic repeated geometry.
   Keep host wall-clock profiles distinct from GPU timestamp measurements.
3. Design reusable transport keys including geometry/material revisions,
   sensor identities/frames, sample streams, sky basis and sun directions.
   Prove motion and material invalidation before caching across solves. Native
   per-call transfer reduction and optical CDF reuse are not persistent caches.
4. Validate refractive sun connections, spatial focusing and angular Jacobians
   with independent slab/prism references before enabling solid-glass DNI.
   Preserve explicit capability errors and separate delta/finite densities.
5. Add sensor-partition-stable random streams and measured total-memory
   admission before large sensor grids; profile host transport before choosing
   any GPU-resident kernel. Current time chunks and sensor limits are explicit.

## Glass and annual urban implementation, 2026-11-11

The [current implementation and capability matrix](native-glass-urban-20261111.md)
supersede historical open-item claims below. Native ABI4 adds smooth convex
solid boundaries with sky-path Snell refraction, Fresnel events, TIR and Beer
absorption. Refracted sun connections remain unsupported and fail explicitly.
The prior clear-plate sun contract is preserved.

An unchanged licensed measurement-based CGDB Solar Full Klems fixture now has
independent side-specific integral/PDF checks. Its original reciprocity
residuals are retained; the installed Radiance Visible selection prevents a
matched measured-Solar Radiance claim. No analytic rough dielectric was added:
finite directional measured lobes are validated without claiming a microfacet
model or raw instrument uncertainty.

The [annual district runner](annual-urban.md) executes all 8,760 hours with
mixed materials, multiple interactions and four kinds of scheduled scene edits,
using bounded blocks and hash-validated restart. Its scale sweep reports actual
geometry/sensor/time/material sizes. [Independent wavelength bands](native-wavelength-bands.md)
require explicit per-band optics and integrated forcing; production spectral
weather and optical datasets remain input work. GPU-resident transport,
native prototype instancing and revision-keyed persistent transfer caching
remain performance work, with measurements preceding implementation.

## BSDF transport milestone, 2026-11-11

The [BSDF coverage/evidence report](native-bsdf-20261111.md) supersedes the
historical glazing status below. Native ABI3 adds a lossless clear-plate delta
material, deterministic transmitted/reflected sun connections, finite/delta
mixtures and multiple plates. It also adds reciprocal sided `DiffuseSheet`,
strict XML unit/structure checks and immutable import provenance. Finite Klems
lobes retain their own evaluation/sampling contract. Vulkan still performs only
hardware traversal, with transport on the host. Solid refraction, analytic glossy,
new measured formats/bases and licensed measured fixtures remain open.

## Next milestone: glass, measured BSDFs and annual urban scenes

The current `ClearGlazing` model covers a lossless thin plate in air. Native
ABI3 handles its transmitted and reflected delta paths alongside finite
scattering. Use the BSDF milestone report above for current validation scope.

1. **Solid glass.** Add a finite-thickness dielectric path with explicit medium
   boundaries, Fresnel reflection, Snell refraction, total internal reflection,
   and distance-dependent absorption. Validate analytic slabs and a wedge or
   prism, then check mixed sun and sky scenes against Radiance where comparable.
   State nesting, overlap, absorption, and caustic limits by backend.
2. **Real translucent assemblies.** Validate a redistributable measured Solar
   Full Klems fixture end to end and add a full-year material gate. Add a tested
   rough dielectric model and expand measured bases as fixtures require. Verify
   orientation, sidedness, energy, and sampling/evaluation/PDF agreement.
3. **Wavelength bands.** Define band-specific forcing, material optics, units,
   and integration. Prove independent-band transport on an analytic fixture
   before claiming spectral results. Track dispersion separately.
4. **Annual urban workflow.** Demonstrate a reproducible 8,760-hour district
   scene with glazing, BSDF panels, shading, multiple interactions, and dynamic
   scene edits. Measure scale across geometry, sensors, time steps, materials,
   and bands. Prioritize revision-keyed transfer reuse, bounded batches and
   restart, and shared prototype acceleration. Use profiles to scope GPU-resident
   transport. Report measured capacity and larger-scene projections separately.

The ordered production-scale follow-up and remaining blockers are recorded in
the current implementation report linked above.

## Annual validation and update milestone, 2026-11-11

The full 8760-hour Berlin2025 synthetic cube comparison now runs reproducibly
against pinned Radiance 5.4, Embree and real AMD Vulkan RT. All exposed-face
direct/diffuse/combined gates passed: maximum annual errors 0.116%, 1.388%, 0.451%,
respectively, and maximum daylight NRMSE 0.250%, 1.715%, 0.627%. Independent-seed
native refinement and high-quality rtrace spot checks support the comparison.
See the [annual report](../validation/annual-radiance.md) and
[commands and measured update results](native-validation-20261111.md).

Fixed-topology Embree vertex-buffer refits and Vulkan device/pipeline reuse
are implemented and checked against fresh builds. On the 131072-triangle update
fixture, rigid-update medians changed from 6.67 to 0.99 ms for Embree and
3025 to 26.4 ms for Vulkan. Stationary tracing is not materially improved by this
slice. Shared native prototype instancing remains unimplemented.

Diffuse sheets and directional Full Klems panels have controlled equivalent
Radiance coefficient checks, explicit Visible-block detection, front/back and
winding checks, and material-only edits. These are finite-width scattering
models. A separate analytic nonabsorbing thin-plate Fresnel reference now passes
six-angle transmitted/reflected-sun comparisons with Radiance (maximum error
0.0000654 W/m2). Native thin-plate delta paths now ship in ABI3. Solid refraction and
wavelength-resolved transport remain open.

The next priorities follow the measurements:

1. Cache native sun/sky transfer by geometry, material, sampling and sun-direction
   revisions. Current annual native solves repeat direct shadow connections;
   Radiance pays a large one-time coefficient setup and then applies weather
   matrices cheaply. Compare cache setup, memory and forcing-only application.
2. Move host hit decoding and transport sampling/material/reduction work into
   deeper native batches or a validated GPU-resident slice. Traversal profiling
   separates mapped input/output processing from dispatch/wait; it does not
   separately measure host irradiance reduction or GPU timestamps.
3. Add shared prototype acceleration and transform-only TLAS updates, with
   deformation and topology fallback and larger-motion BVH quality evidence.
4. Extend the verified thin-plate model to solid glass and measured assemblies.

## First implementation session

Deliver a reproducible shortwave irradiance example from Python that runs its
ray batches and transport in C++20 with Embree. Use a small scene with upward
and tilted sensors, a movable shade, an opaque reflector, and a thin translucent
panel. Generate a year of solar directions from site and time inputs. Solve a
few representative steps through the full scene, and run the direct-only
annual schedule on the small fixture. Accept supplied weather inputs and
include a synthetic forcing example with its provenance recorded.

Build direct sun and isotropic sky first. Then add one diffuse interaction for
reflection and transmission. Run a separate, time-boxed Vulkan ray-query
probe on the laptop in parallel. The first simulation's acceptance depends
on Embree and the reference fixtures. Follow it with opt-in Embree longwave
view factors, then integrate Vulkan and optimize both solvers using profiles.

### Time, sky and outputs

- Accept latitude north-positive, longitude east-positive, an explicit year,
  interval and site time convention. Define the annual interval using local
  standard time with a fixed UTC offset and sample each interval at its
  midpoint. Store explicit UTC instants and interval durations. Include leap
  day when present. Specify timestamp conversion before importing weather
  records. Use geometric solar position and record the horizon policy.
- Return unit vectors toward the sun in east-north-up coordinates, with an
  explicit rotation from east-north-up to scene coordinates. Use a verified
  solar-position provider and compare against published SPA values. Evaluate
  pvlib as an optional provider and check its supported Python versions before
  pinning it. Keep the core able to accept precomputed sun vectors. Solar
  position is independent of the
  supplied direct normal irradiance (DNI) and diffuse horizontal irradiance
  (DHI). In the first model, a sun center at or below the horizon contributes
  zero direct irradiance.
- Use a directional sun and a uniform upper-hemisphere sky of radiance
  `DHI / pi`, with a black lower environment. Keep sun and sky distinct. Report
  direct, diffuse sky, and first-interaction irradiance with component and
  total values in W/m². Sum irradiance times actual interval hours for Wh/m².
  Store the source, time convention, coordinate frame, backend and revisions
  with the results. Label neutral RGB as three equal representations of the
  same reference quantity. Carry the channel basis for colored results and
  require an explicit convention for converting them to broadband solar power.

### Scene, materials and traversal

- Reuse v2 `Scene`, stable surface IDs, shared local `Mesh` prototypes and
  winding. For the first shortwave API, attach a revisioned optical material
  table by surface ID alongside the geometry. An opaque assignment has front
  and back Lambertian reflectances. A thin translucent assignment has one
  symmetric diffuse reflection/transmission model shared by both sides. Check
  each channel for finite nonnegative values and `reflectance + transmittance
  <= 1`. Lambertian BRDF and diffuse BTDF are respectively `rho / pi` and
  `tau / pi` in sr⁻¹ over their corresponding hemispheres. Capture the shading
  frame from the geometric winding. Model scattered light through the panel
  as a diffuse interaction. Clear glazing and specular transport follow later.
- Capture geometry revision, material revision, and immutable forcing at run
  start. Publish results from one consistent snapshot. After a commit, a fresh
  solve must see the edits, and earlier results must retain their values and
  revisions. Material edits invalidate shortwave work that would otherwise be
  resumed on stale properties and leave geometry acceleration reusable. Keep
  existing longwave geometry revisions and observers intact. Define how stale
  bindings are handled on surface removal and ID reuse before exposing either
  edit through the shortwave API.
- Share a small internal geometry query contract across the two integrators.
  Batch closest hit and bounded occlusion on committed geometry. A hit reports
  stable surface/primitive IDs, distance along a unit direction, barycentrics
  and front/back from stored winding. Agree on scene units, ray intervals,
  origin offsets, self-hits and ambiguous triangle edges. Use opaque visibility
  along segments with no modeled scattering event. The thin panel participates
  in intersections and contributes through its transmission lobe when sampled
  as the first interaction. Keep material evaluation, light sampling and
  output reduction with the integrator. Cross the Python boundary in batches.
- Serialize Embree scene commits against tracing on the same handle. Start with
  correct refresh/rebuild on rigid motion, deformation and topology edits. Use
  update benchmarks to decide when to introduce shared prototype acceleration,
  instance updates and vertex refits. A material-only edit must leave the
  acceleration structure untouched. Keep backend selection explicit and record
  the backend actually used. The existing `device="vulkan"` uses Taichi's
  software BVH and retains that meaning.

### Parallel work and integration gates

| Workstream | Deliverable |
| --- | --- |
| Solar and forcing subagent | Annual timeline, SPA comparisons, DNI/DHI validation, deterministic example data |
| Native CPU subagent | C++20 build and binding, Embree ray queries, geometry commit and update tests |
| Vulkan probe subagent | Real BLAS/TLAS build and compute ray-query dispatch with readback, adapter and feature report |
| Integration lead | C++ shortwave transport, analytic fixtures, pinned Radiance comparison, baseline LW measurements |

Agree on direction, hit, revision and unit conventions before merging the
work. Integrate Embree and the direct case first, then diffuse sky and the
one-interaction case. Test from the public Python entry point as each slice
lands. Document exact versions, licenses, supported platforms and reproducible
build steps before introducing dependencies or downloading binaries.

- Check published solar vectors, cardinal orientation, timezone conversion,
  interval counts, leap day and night. Validate annual energy against an
  explicit sum of irradiance times interval hours on synthetic forcing.
- For unobstructed sun, recover `DNI × max(dot(normal, sun), 0)` on horizontal
  and tilted planes. Check zero under a full opaque shade. A uniform sky over
  black ground gives DHI on an open horizontal sensor and DHI/2 on an open
  vertical sensor. Use analytic diffuse fixtures and a matched Radiance scene
  to test front/back reflection and transmission. Include zero transmission,
  the reflectance/transmittance energy bound, and a check that diffuse
  transmission illuminates off-axis receivers.
- Move, deform and replace a blocker, then compare each result against a
  fresh scene at the same geometry. Change only a material and verify changed
  transport with no acceleration rebuild. Keep a result from each revision and
  check its values after subsequent edits. Run the existing v2 and Grasshopper
  contract tests with the native extension absent as well as present.
- Record ray counts, seeds, tolerances and the reference setup. Use absolute
  and relative tolerances for deterministic checks and independent replicate
  intervals for stochastic estimates. Fix an error budget and sample count per
  fixture before accepting parity. Compare Radiance at matching source,
  radiometric convention and bounce depth. Use a fully open or fully shaded
  sensor for point-sun comparisons against the finite Radiance sun disk.

## Follow-on milestones

1. **Native longwave on Embree.** Adapt traversal behind the v2 solver. Retain
   the estimators, selected-receiver semantics, occlusion by the full scene,
   mesh self-viewing, sidedness, sky/escape and partial-run accounting. Gate
   on analytic enclosures, reciprocity, moving scenes, split budgets and
   existing regression fixtures. Benchmark equal work against the existing
   CPU and Vulkan compute backends. Measure cold builds, warm throughput,
   geometry update latency and memory independently. Move sampling and
   reduction into native batches where profiling identifies a bottleneck.
2. **Vulkan hardware traversal.** Carry over the query conformance suite,
   then keep ray generation, transport and reduction GPU-resident for a
   representative solver workload. Profile build, dispatch, transfer and
   synchronization. Add TLAS transform updates and BLAS refit where supported
   and measured. Report capability failures explicitly. Test the AMD Radeon
   860M before broadening the supported adapter matrix.
3. **Discretized sky and annual transport.** Introduce explicit world-space
   patch directions, boundaries and solid angles. Audit compatibility before
   sharing the existing LW Tregenza classifier. Its azimuth offsets differ
   from Radiance's, which also has a ground row. Use an explicit SW basis for
   Radiance import if the bin boundaries differ. Validate a diffuse-only
   Perez/Tregenza or Reinhart sky against `gendaymtx -s -O1`, keep the solar
   direction separate, and test orientation and integrated irradiance. Cache
   sky transfer across timesteps when geometry and materials are stationary.
4. **BSDF, glazing and spectral transport.** Add specular and multiple-bounce
   transport. For Klems, define local frames, incident/outgoing directions,
   front/back lobes and projected-solid-angle weights. Treat Solar and Visible
   data blocks as distinct inputs. Validate BSDF sampling/PDF consistency,
   reciprocity where applicable, and energy conservation before adding
   spectral bands and additional material models. Evaluate OptiX once a
   supported NVIDIA test environment is available.

## Local preflight and references

- `vulkaninfo` reports AMD Radeon 860M, AMD driver 26.10.1, and
  `rayQuery`, `accelerationStructure` and `bufferDeviceAddress` enabled as
  advertised device features. A hardware dispatch and native performance
  measurements are pending. CMake 4.3.2 and Visual Studio C++ Build Tools are
  installed. The current shell exposes neither `VULKAN_SDK` nor `glslc`.
- The PATH-resolved Ladybug Radiance installation reports LBNL 5.4a
  `5.4.b268408ad5`. A temporary `oconv` and `rtrace -I+ -h -ab 0 -av 0 0 0`
  test with an overhead disk normalized to 800 W/m² returned 800.0044 in
  each neutral channel for the open sensor and zero behind a black panel.
  This checks the reference setup. Raystrack shortwave comparison is pending.
- A second reference run used `C:\Radiance\bin` build `5.4.4ee32974b1` and a
  large thin sheet with neutral diffuse reflectance 0.3 and transmittance 0.4.
  Sensors facing the sheet from either side received about 240 W/m² reflected
  and 320 W/m² transmitted per channel under the same 800 W/m² source, using
  one ambient bounce. This gives the first transport comparison two separate
  observables.
- The focused pre-change suite `test_model_v2.py`, `test_solver_v2.py` and
  `test_self_viewing.py` passed 71 tests with five GPU cases deselected using
  the existing local release environment. Native parity and performance checks
  belong to the implementation milestones.
- [Existing Raystrack validation commands](../validation/readme.md),
  [Solar Position Algorithm report](https://docs.nrel.gov/docs/fy08osti/34302.pdf),
  [pvlib solar position implementation](https://github.com/pvlib/pvlib-python/blob/main/pvlib/solarposition.py)
- [Embree](https://github.com/RenderKit/embree),
  [Embree scene commits](https://github.com/RenderKit/embree/blob/master/doc/src/api/rtcCommitScene.md),
  [Vulkan ray queries](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_ray_query.html),
  [Vulkan acceleration structures](https://docs.vulkan.org/spec/latest/chapters/accelstructures.html)
- [LBNL Radiance repository](https://github.com/LBNL-ETA/Radiance),
  [rtrace manual](https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/rtrace.pdf),
  [gendaymtx manual](https://floyd.lbl.gov/radiance/man_html/gendaymtx.1.html),
  [Reinhart patch definitions](https://github.com/LBNL-ETA/Radiance/blob/master/src/cal/cal/reinhart.cal),
  [LBNL solar BSDF conventions](https://windows.lbl.gov/sites/default/files/Downloads/Solar%20BSDF%20documentation.pdf)
