# Raystrack for Grasshopper

The compiled components and their Python worker live in this repository. See
[the Grasshopper guide](../docs/grasshopper.md) for the component workflow.

Typed outputs show their actual geometry, query, sampling, accuracy or result
settings in a Panel. Port tooltips and component help explain accepted values;
**RS Inspect** adds a usage description; set Details true for the complete JSON
snapshot. Invalid data reports a useful reason
instead of breaking the display, and result readers wait while a solver prepares
its first snapshot.

For **Transform**, connect a native Grasshopper transformation, such as the
**Transform (X)** output of Move or Rotate. Feed the original Rhino mesh/Brep
into Move's Geometry (G) input and a vector into its Motion (T) input;
do not connect the vector or 16 raw matrix numbers directly. Translations and
rotations are supported. Leave the port empty if the input mesh has already been
transformed, so the placement is applied once. Use a separate **RS Surface**
with its own ID for each repeated placement. See the guide's [Transform instructions](../docs/grasshopper.md#connecting-transform).

## Build and package on Windows

Run from the repository root, using a build Python with pip and setuptools:

```powershell
python tools/grasshopper/build.py
```

Rhino 8 must be installed for its assembly references and Yak executable. The
builder uses Windows' .NET Framework C# compiler without requiring a dotnet SDK.
References to RhinoCommon, Grasshopper, GH_IO and Newtonsoft.Json are resolved
from Rhino; those Rhino assemblies are never distributed in the package.

The full build writes these artifacts under `dist/grasshopper/`:

- `raystrack-2.0.0/Raystrack.Components.gha`, its XML documentation, and adjacent `runtime/`.
- `raystrack-2.0.0-win-x64.zip`, a complete portable manual installation.
- `raystrack-2.0.0-rh8_35-win.yak` for Rhino Package Manager installation.

The local build targets Rhino 8.35 and later on Windows. Yak infers the minimum
Rhino version from the assembly references, so another installed Rhino version
may produce a different target filename. The builder reports the exact filename.
The bundle includes its guides and the portable
`examples/grasshopper/boxes.gh` definition when present in the checkout.
Install the local package explicitly with:

```powershell
python tools/grasshopper/build.py --install
```

This installs version 2.0.0 in Rhino's versioned package directory;
it does not publish to the public Yak server. The old `raystrack_grasshopper`
checkout is the previous v1 integration and remains separate. Rhino should load one active
version of Raystrack at a time; restart Rhino after changing installed packages.

For manual installation, extract the entire zip into a Grasshopper library
directory. Keep `runtime/` beside the `.gha`; copying the assembly alone cannot
run simulations. Use only one installation location to avoid duplicate component
registrations. `--gha-only --skip-yak` produces an assembly-only zip with this
runtime requirement recorded inside it. `--skip-yak` builds the full zip without
building a Yak artifact.

## Shipped runtime

The distribution includes the official [CPython 3.12.10 Windows AMD64 embeddable
package](https://www.python.org/downloads/release/python-31210/), verified against
the SHA-256 digest published by Python.org. This is the final 3.12 release with
official Windows binary installers; later 3.12 security releases are source-only.
The runtime contains the current local Raystrack source, NumPy 1.26.4, Numba
0.64.0, llvmlite 0.46.0 and Taichi 1.7.4, with all transitive dependencies pinned
in `runtime-requirements.txt`. CPU and portable Vulkan support are available
without a user Python installation. CUDA requires the corresponding installed
driver/toolchain capabilities; physical CUDA validation remains outstanding.

Downloads and wheel installation happen at **build time**. A simulation never
downloads packages or uses Rhino's Python environment. `python312._pth` isolates
imports to relative paths in the shipped runtime. Python and dependency licenses
are included in `runtime/licenses/`, and `runtime/runtime-manifest.json` records
archive/wheel provenance, SHA-256 hashes and all shipped runtime file hashes.
No pip console launchers or absolute build-machine installation paths are shipped.
`Raystrack.Components.xml` contains assembly documentation. The separate
`package-manifest.json` records source and non-runtime payload hashes and the
runtime manifest hash, identifying the code and files in a particular build.

```powershell
# Rebuild after the download cache is populated, without network requests:
python tools/grasshopper/build.py --offline
# Check the shipped interpreter against a real local Vulkan device as well:
python tools/grasshopper/build.py --gpu-smoke
```

The default build runs an independent CPU correctness smoke test with exact
resumed sample prefixes using the shipped Python. `--cache` selects the download
cache; `--rhino`, `--compiler`, `--yak` and `--output` override build locations.
Generated artifacts are ignored by Git. Both the Python package and the
Grasshopper package are version `2.0.0`. See the
[release workflow](../docs/releasing.md) for validation and publication.

## Validation scope

The earlier native acceptance report applies to its recorded `2.0.0-dev.2`
source hashes. The earlier dev.4 version passed 81 targeted tests, with one
platform-specific skip, including Save/Load, readable runtime fields, Sky
summaries, native Breps and Grasshopper component connections. The headless
geometry checks verified all 145 sky patches and 1,305 native interior directions
against the solver. See `validation/results/grasshopper_dev4_native.json`.
Run the current host script for complete installed-canvas acceptance.

`python tools/grasshopper/assembly_contract.py` provides a separate managed
contract check for value summaries, malformed payload messages and rendered icon
pixels. It compiles only temporary test assemblies, opens no Rhino window, and
does not build or install a distribution. A passing managed check does not replace
the native host validation.

Use `--native` with the managed contract tool to also check actual Rhino Breps,
world placement, all 145 spherical patch areas and native text tags in a headless
Rhino geometry core. This does not install a package.

**RS To Brep** replaces RS Instance. It converts an RS Surface or complete RS
Scene back to triangulated native Breps, with IDs and Labels in matching order.
**RS Sky** holds all sky settings and connects to RS Query's Sky input. Its
Patches, Labels, Label points and Tags show the exact solver numbering on a dome;
Patches and Tags can be baked directly. **RS Runtime** now returns a readable
Devices list describing each reported field.

Version **2.0.0** enables self-viewing within a single mesh, including its
own receiver channels and its occlusion of sky/other receivers, and refreshes
the supplied icons. For a closed box with outward mesh normals, set **RS
Sampling → Flip** true and read the box's **back** channel; with inward mesh
normals, leave Flip false and read **front**. Use the same ID for both sender
and receiver in a pair query, or use matrix/row. CPU, CUDA and portable GPU
traversal support self-viewing, as does the CPU area-pair estimator.

The dev.5 regression suite passed 254 tests, with 23 optional/platform skips.
The actual packaged interpreter also passed the CPU and physical Vulkan enclosure
checks. CUDA self-viewing was checked in simulation with both ray generation
paths; physical CUDA hardware was not tested.
See [the self-viewing report](../validation/results/grasshopper_dev5_self_viewing.json).
