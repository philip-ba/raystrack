# District optical references and finite rough transmission

The file-driven district preserves `Opaque`, sided `DiffuseSheet`, ideal
`ClearGlazing` plate and original Solar Full Klems transport. Units are metres,
BSDF sr^-1, irradiance W/m2 and interval-integrated exposure Wh/m2. All optional
native backends use the same C++ host transport; Vulkan RT accelerates traversal.

`KlemsBSDF.rough_transmission(.55, 8, reflectance=.15)` adds an explicitly
**synthetic finite directional sheet**, with identical front/back Lambertian
reflection and reciprocal transmission. It is a controlled scattering model,
not measured rough glass or a microfacet dielectric. The JSON material type is
`rough_transmission`, with scalar `transmittance`, `exponent`, and optional
`reflectance` (default zero). No extra dependency or native ABI change is needed.

For Full Klems polar/azimuth cell-center unit directions c_i and c_o on one
hemisphere (the central cell uses the zenith), the transmission density is
`C * max(dot(c_i,c_o),0)^exponent`. The existing incident XY reversal places
straight-through transmission on the matrix diagonal. One global C makes the
largest directional-hemispherical transmittance equal to the requested value.
Other directions can transmit less and absorb more. Global normalization and
the half-turn transpose symmetry preserve reciprocity; independently normalizing
each row would generally break it. The density is constant inside each Klems
cell and remains finite. Exponent zero is Lambertian, and exponents up to 256
concentrate this coarse angular lobe. This basis cannot resolve a delta limit.
Reflection plus maximum transmission must be at most one.

The solid-glass DNI investigation keeps the existing capability error. At one
Snell interface, the angular Jacobian is
`dOmega_t/dOmega_i = (n_i/n_t)^2 * cos(theta_i)/cos(theta_t)`.
Finite differences check this separately from Fresnel power and the adjoint
radiance factor `(n_i/n_t)^2`. Parallel slab entrance/exit angular maps cancel;
an arbitrary convex solid still needs a refractive boundary-value connection,
spatial focusing/caustic treatment, branch accounting and independent reference
tests. The angular identity alone does not supply that solver. Solid sky
transport retains its existing convexity, separation and exterior-sensor limits.

Run the controlled mixed-path checks from the checkout:

```powershell
$env:PYTHONPATH = 'src;.'
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/raystrack-native-final/Release/raystrack_native.dll).Path
.local/release-venv/Scripts/python.exe -m validation.district_physics --backend native-builtin --backend embree --backend vulkan-rt --output build/dynamic-district/physics.json
.local/release-venv/Scripts/python.exe -m pytest tests/test_district_physics.py tests/test_shortwave_klems.py tests/test_measured_bsdf_reference.py -q -p no:cacheprovider --basetemp=build/dynamic-district/pytest-optics
```

The runner compares selected high, oblique and low-sun directions for a finite
sheet below an ideal clear plate. Two sensors inspect reflection above the
sheet and transmission below it. Independent projected-angle integration
computes each Full Klems integral; Gauss-Legendre quadrature integrates the
plate's continuously varying Fresnel transmission inside each incident cell.
Separate sun, sky and combined runs check linearity. Rotating the fixture and
sun checks transformed BSDF frames, and edited/fresh results must match exactly.
A separate moving opaque proxy checks an 80 W/m2 low-sun open sensor against
an exactly zero blocked sensor and preserves the original result snapshot.

The fixed stochastic gate is five replicate standard errors plus 0.15 W/m2;
component errors are summed conservatively. Four replicates also report a
Student-t(3) 95% half-width (3.182446305 times that bound). These estimates cover
sampling variation and have weak precision at four replicates. Absolute errors
are retained, including at zero. No tolerance depends on a measured residual.

The measured case imports the unchanged licensed
[`2011-SA1.XML`](../tests/fixtures/bsdf/ATTRIBUTION.md), checks its original SHA256
and preserves its CGDB Solar spectral labels and source metadata. The installed
Radiance 5.4 reference path selects different **Visible** data, so this report
uses independent matrix quadrature and makes no measured-Solar Radiance parity
claim. Existing [equivalent-material Radiance checks](native-bsdf-20261111.md)
and the [full-year opaque cube](../validation/annual-radiance.md) remain separate
accuracy gates. These controlled constituent tests do not establish independent
accuracy for the district's complete annual mixed-material prediction.
