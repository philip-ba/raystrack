# Independent wavelength intervals

`solve_independent_bands` runs the existing native scalar transport once for each
explicit `WavelengthBand(name, lower_nm, upper_nm)`. Each interval has a unit
top-hat response on `[lower_nm, upper_nm)`; intervals cannot overlap. The sum
covers only the supplied intervals, which may have gaps. This is independent
band transport, without a continuous wavelength sampler.

Every `BandInput` requires a complete surface-to-material mapping, optical
provenance, band-integrated DNI and DHI, and forcing provenance. DNI and DHI are
in W/m² **after** integration over that interval, not W/(m² nm). DHI drives a
uniform upper-hemisphere sky, with radiance DHI/π W/(m² sr). Inputs can be scalars
or hourly one-dimensional arrays. No code splits broadband weather into bands.
For spectral irradiance density Eλ supplied per nm, callers must integrate
`∫ Eλ dλ` over the stated edges using the source's wavelength grid and record
the quadrature, units, source, and any weighting in the provenance.

Each optical model must be scalar within its band (equal internal neutral RGB
channels). Coefficients describe a constant or explicitly documented effective
band value. A weighted effective coefficient can depend on the illuminating
spectrum and direction; it is an approximation whose validity belongs to the
input dataset. RGB triples are rejected. The broadband CGDB Solar fixture and
the separate WINDOW Visible block do not supply wavelength-resolved optics.
Do not copy those values into bands and call them measured spectral data.

The wrapper retains one scene revision across the solves. Each band owns its
native solver/geometry; there is no geometry-sharing or transfer cache between
bands. Material and backend capability errors propagate, including the
solid-glass restriction on sun forcing. Results preserve all band inputs and
per-band immutable snapshots. `irradiance` has axes band, time, sensor, component
in W/m²; `total` sums bands and components. `exposure()` sums interval-duration
weighted bands in Wh/m² by sensor/component. Individual standard errors remain
in `results`; no combined error is inferred because reused sampling seeds can
correlate bands.

No fluorescence, wavelength redistribution, coherent interference,
polarization-state transport, response-weighted photometry, or spectral sky
generation is implemented. Scalar refractive index is constant per band; this
does not integrate dispersion within an interval or validate dispersive caustics.
Annual urban runs remain broadband unless the caller provides an independent
complete band dataset and executes the workflow separately for those inputs.

Reproduce the synthetic acceptance case after building the optional native
library as described in [the native guide](native-shortwave.md):

```powershell
$env:PYTHONPATH = 'src;.'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
.local/release-venv/Scripts/python.exe examples/ex09_independent_bands.py --backend embree
.local/release-venv/Scripts/python.exe -m pytest tests/test_shortwave_spectral.py -q -p no:cacheprovider --basetemp=build/spectral-tests
```

The example supplies synthetic 400–700 nm and 700–2500 nm forcing independently,
with transmittances 0.2 and 0.8. Expected transmitted irradiances are 24 and
272 W/m², total 296 W/m² and two-hour exposure 592 Wh/m². The regression checks
band-specific optics, direct/diffuse linearity, night, duration integration,
immutable inputs/results, overlap rejection and rejection of RGB spectral claims
on builtin CPU, Embree, and Vulkan RT traversal.
