# Measurement-based Solar Full Klems reference

The unchanged [CGDB woven-shade fixture](../tests/fixtures/bsdf/ATTRIBUTION.md)
adds measured-product evidence alongside the controlled synthetic matrices.
Its exact bytes are pinned by SHA-256; no spectrum relabeling, reciprocity
correction, energy normalization, fitting, or matrix repair is applied.
The dataset's [initial release notes](https://windows.lbl.gov/sites/default/files/CGDB-1-2-ReleaseNotes.pdf)
identify LBNL as the measurement laboratory and product 6006 as this file.
The [CGDB license](https://windows.lbl.gov/cgdb-license) permits unchanged
redistribution under CC BY-ND 4.0. See the fixture attribution for full credit,
license links, source URL and exact digest; the XML has a separate data license.

Reproduce import and density checks without a native library:

```powershell
$env:PYTHONPATH='src;.'
.local/release-venv/Scripts/python.exe validation/validate_measured_bsdf.py --output build/measured-bsdf-reference/import-report.json
```

Reproduce the original download (do not overwrite the pinned fixture):

```powershell
Invoke-WebRequest -Uri 'https://windows.lbl.gov/sites/default/files/2011-SA1.XML' -OutFile build/measured-bsdf-reference/2011-SA1.XML
Get-FileHash build/measured-bsdf-reference/2011-SA1.XML -Algorithm SHA256
```

The script accepts `--xml FILE --sha256 DIGEST` for a separately obtained Solar
Full Klems file, retaining the loader's existing capability and energy checks.
Supply its license/provenance separately; the script cannot establish either.

Independent quadrature computes each patch's projected solid angle as
`pi*(sin(theta_high)^2-sin(theta_low)^2)/nphi`; the sum is pi sr.
Multiplying a BSDF density in sr^-1 by this weight yields dimensionless
directional hemispherical reflection/transmission. The checks compare explicit
evaluation integrals, normalized forward/reverse PDFs, and sampled evaluation
and PDFs on both sides and several incident/outgoing patches.

Solar normal-incidence front R/T are .465439518/.117909197; back R/T are
.525789513/.116344097. Maximum directional R+T is .698611137. These agree with
rounded [CGDB release-note values](https://windows.lbl.gov/sites/default/files/cgdb-8-0-release-notes.pdf).
The finite transmission changes with incidence and includes off-specular energy.
The largest absolute reciprocity residuals are .050214, .048300 and .019976
sr^-1 for front reflection, back reflection and transmission. They are reported
as source-data properties, not treated as exact physical symmetry or silently
removed. This establishes import/transport behavior on a real processed product,
not its original instrument calibration, uncertainty or manufacturing variation.

With an explicitly selected ABI-compatible library, add native transport:

```powershell
$env:RAYSTRACK_NATIVE_LIBRARY='C:/Users/Philip/Desktop/raystrack/build/raystrack-native-final/Release/raystrack_native.dll'
.local/release-venv/Scripts/python.exe validation/validate_measured_bsdf.py --backend native-builtin --backend embree --backend vulkan-rt --output build/measured-bsdf-reference/report.json
.local/release-venv/Scripts/python.exe -m pytest tests/test_measured_bsdf_reference.py -q -p no:cacheprovider --basetemp=build/measured-bsdf-reference/pytest-tmp
```

This gate uses a horizontal sheet of half-width 10,000 scene units, sensors one
unit above/below, three incident patch centers, 800 W/m2 DNI and 100 W/m2
isotropic DHI. All quantities are neutral broadband Solar radiometric values,
with identical RGB representations. Native output irradiance is W/m2. One
surface interaction is compared separately for direct, diffuse and combined
forcing against independent matrix integrals for each winding side. Four randomized replicates,
32,768 samples each, seed 671, use a fixed five-standard-error plus .1 W/m2
finite-sheet/geometry allowance. Reports also contain descriptive 95% Student-t
interval half-widths (three degrees of freedom; only four replicates). This is a large-sheet limit test, not a
full-year reference or evidence for an analytic rough dielectric model.

Pinned Radiance 5.4 in the existing panel workflow selects Visible blocks.
Solar and Visible differ in this unchanged measured file, so an unqualified
Radiance/neutral-Solar equality claim would compare different optics. The
native gates use exact Solar integrals instead; no duplicated or relabeled
Visible block is distributed. A full-year measured-product Radiance gate and
raw measurement uncertainty remain separate work.

Executed 2026-11-11 with the rebuilt ABI4 combined library: the focused new
fixture and existing Klems suite passed **23 tests** in 19.36 s. The standalone
report at `build/measured-bsdf-reference/report.json` passed all six backend/side
groups (builtin, Embree and Radeon Vulkan RT; front and back), each with three
forcing modes and three solar directions. Largest absolute irradiance residual
was 1.02676 W/m2, within its declared stochastic gate. Import-only PDF integral
residual was 2.78e-15; sample/PDF residual was 1.51e-14. These are numerical
transport checks, not measured instrument accuracy claims.
