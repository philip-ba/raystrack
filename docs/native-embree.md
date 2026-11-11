# Optional Embree CPU traversal

Raystrack's native CPU backend uses [Intel Embree 4.4.1](https://github.com/RenderKit/embree/releases/tag/v4.4.1)
under the [Apache License 2.0](https://github.com/RenderKit/embree/blob/v4.4.1/LICENSE.txt).
The upstream Windows binary includes oneTBB 2022.3.0 under the Apache
License 2.0 and other third-party components with their own notices. Keep
the archive's license and notices with any redistribution. Raystrack keeps
Embree out of the base Python installation and CMake does not fetch it.

The setup below targets Windows x64 using CMake 4.3.2 and the Visual Studio
18 Build Tools with the MSVC 14.50.35717 x64 compiler. Embree also publishes
CPU builds for Linux x86-64 and macOS x86-64/arm64. The same CMake package
discovery applies with a platform-matched Embree installation and a C++20
compiler.

From the repository root in PowerShell, download the pinned Windows x64 CPU
archive, verify the [upstream SHA-256](https://github.com/RenderKit/embree/releases/expanded_assets/v4.4.1),
and expand it under the ignored `.local` directory.

```powershell
$embreeRoot = Join-Path $PWD '.local/embree-4.4.1'
$embreeArchive = "$embreeRoot.zip"
New-Item -ItemType Directory -Path (Split-Path $embreeRoot) -Force | Out-Null
curl.exe --fail --location --output $embreeArchive 'https://github.com/RenderKit/embree/releases/download/v4.4.1/embree-4.4.1.x64.windows.zip'
if ($LASTEXITCODE -ne 0) { throw 'Embree download failed' }
$embreeSHA = '5aa1fa3161a720f9c610b06c652b80f421cb2b1fe8221fdf92b74157336fdf8f'
if ((Get-FileHash -Algorithm SHA256 $embreeArchive).Hash -ne $embreeSHA) { throw 'Unexpected Embree SHA-256' }
Expand-Archive -LiteralPath $embreeArchive -DestinationPath $embreeRoot
$embreeConfig = Get-ChildItem -LiteralPath $embreeRoot -Recurse -Filter embree-config.cmake |
    Select-Object -First 1 -ExpandProperty DirectoryName
$embreeRuntime = Get-ChildItem -LiteralPath $embreeRoot -Recurse -Filter embree4.dll |
    Select-Object -First 1 -ExpandProperty DirectoryName
if (!$embreeConfig -or !$embreeRuntime) { throw 'Missing Embree CMake config or runtime DLL' }
$env:PATH = "$embreeRuntime;$env:PATH"

cmake -S native -B build/native-embree -G 'Visual Studio 18 2026' -A x64 -DRAYSTRACK_EMBREE=ON "-Dembree_DIR=$embreeConfig"
cmake --build build/native-embree --config Release

$nativeDir = Resolve-Path 'build/native-embree/Release'
Copy-Item -LiteralPath "$embreeRuntime/embree4.dll", "$embreeRuntime/tbb12.dll", "$embreeRuntime/tbbmalloc.dll" -Destination $nativeDir
$env:RAYSTRACK_NATIVE_LIBRARY = Join-Path $nativeDir 'raystrack_native.dll'
$env:PYTHONPATH = 'src'
python -m pytest tests/test_native_longwave.py -q
```

For a shared SDK installation on Windows, Python callers can register the
SDK's DLL directory with `os.add_dll_directory(embree_runtime)` before loading
the native library and keep the returned handle alive while using it.

Request the backend with `Solver(scene, device="embree")` for cosine view
factors. This uses `acceleration="flat"` and `bvh="auto"`, the default
acceleration settings. Results record `"embree"` as their execution backend.
The area-pair estimator uses `device="cpu"` or `device="auto"`.

The flat adapter commits world-space triangles. The shortwave scene adapter
prefers shared local prototypes and native instances when the loaded DLL
exports the additive ABI4 instance extension. Stable surface slots and each
surface's primitive indices are returned independently of Embree's internal
BVH ordering. Material values belong to transport and are independent of
Embree geometry.

## Shared rigid instances

`Scene.from_instances()` shares immutable `Mesh` identity. The shortwave adapter
deduplicates those identities into one Embree child scene (BLAS) per prototype
and one top-level geometry instance per ordered surface. Absolute transforms
use `world = rotation @ local + translation`; winding is retained. Hits include
world-space points and normals, the ordered instance slot and the prototype's
original primitive index. Bounded occlusion traverses the same scene.

`scene.update_transforms({id: matrix, ...})` validates an entire batch and
publishes one scene revision. Empty and identical batches leave revision and
active work unchanged. Changed batches invalidate active runs before waiting
for the execution lock, then revalidate current meshes and commit one snapshot.
Completed result snapshots remain immutable. Poses are absolute, never
incremental. Transform-only native commits retain all prototype acceleration;
changed instance transforms and the top-level scene are committed together
while tracing is idle. Native `instance_statistics` exposes prototype builds,
vertex refits, changed transform commits and top-level commits for verification.

Same-topology vertex changes refit affected prototypes and propagate their new
bounds through the instance scene. Topology, prototype count/binding changes
and the conservative 32-deformation refresh rebuild. A deformation of only one
member of a shared group splits its mesh identity and can therefore rebuild;
subsequent same-topology deformations of that unique mesh refit. Transform-only
edits do not consume the deformation refresh count. Empty prototypes retain
their instance slots. All IDs, indices, coordinates, bounds and proper-rigid
transforms are checked before any committed buffers change. Invalid preflight
edits retain the previous scene. Resource/device failures during commits are
reported; recovery transactions for those failures are not promised.

`NativeGeometry.geometry_mode` explicitly reports `shared-prototype-instances`
or `flat-world`. Vulkan RT, native builtin, legacy native longwave workspace
adapters and older compatible ABI4 DLLs retain the flat path. Vulkan shared
BLAS/TLAS instances and update-in-place acceleration are not implemented here.
Existing Taichi `device="vulkan"` remains a separate software tracer.

Regression and sequential performance runner (no additional dependencies):

```powershell
.local/release-venv/Scripts/python.exe -m pytest tests/test_native_instances.py tests/test_native_embree_updates.py -q
.local/release-venv/Scripts/python.exe -m validation.benchmark_native_instances --mode flat --instances 256 --grid 16 --static-grid 128 --json build/dynamic-district/instances-flat.json
.local/release-venv/Scripts/python.exe -m validation.benchmark_native_instances --mode shared --instances 256 --grid 16 --static-grid 128 --json build/dynamic-district/instances-shared.json
```

Select the preserved original DLL for flat timing and the rebuilt DLL for shared
timing; repeat in separate processes with other agents idle. The runner separates
Python edit, world packing, native commit, traversal, first/warm transport and
fresh rebuild costs. It checks update/fresh hit and irradiance equality at each
pose. Peak working set includes comparison rebuilds; host phase clocks are not
GPU timestamps. These cases measure execution, not independent material accuracy
or universal speedups.

## Fixed-topology updates

The native update entry point retains the Embree device, scene, geometry and
buffers when ordered surface slots, vertex counts and exact triangle indices
are unchanged. It skips identical vertex buffers and uses
[`rtcUpdateGeometryBuffer`](https://github.com/RenderKit/embree/blob/v4.4.1/doc/src/api/rtcUpdateGeometryBuffer.md)
followed by geometry and scene commits for changed buffers. The scene uses
dynamic, robust traversal and low scene build quality; per-geometry
[`RTC_BUILD_QUALITY_REFIT`](https://github.com/RenderKit/embree/blob/v4.4.1/doc/src/api/rtcSetGeometryBuildQuality.md)
requests vertex-only BVH refitting. Exact connectivity and winding checks prevent
an equal-sized topology edit from incorrectly taking this path. All matching
surfaces are validated before buffers are changed.

Changed topology or surface/vertex counts rebuild acceleration. After 32 changed
refits the next commit also rebuilds, bounding how long a deformed hierarchy is
retained. This periodic refresh is conservative; it is not a measured BVH cost
criterion. This legacy flat path still prepares world-space vertices in Python;
the shared shortwave path above avoids that packing for transform-only edits.

Measure isolated commit and trace costs, including updated/fresh-build hit and
winding parity, from the repository root:

```powershell
$env:PYTHONPATH = 'src'
python validation/benchmark_native_updates.py --backends embree --grid 256 --rays 65536 --repeats 5 --json build/native-annual/embree-updates.json
python -m pytest tests/test_native_embree_updates.py tests/test_native_longwave.py -q
```

The benchmark reports raw repetitions and preparation separately, uses seed 11,
and records the SDK's automatic thread policy and input buffer sizes. It does
not measure process peak memory or prove a speedup for every scene.
