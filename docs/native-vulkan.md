# Native Vulkan ray queries

The [BSDF coverage report](native-bsdf-20261111.md) includes clear-plate and mixed
finite/delta transport verified on this traversal. Material evaluation, light
connections and reductions continue to run in host C++; the shader is unchanged.

The experimental Vulkan traversal uses the device's Vulkan ray-query extension.
The existing Python `device="vulkan"` option still selects the Taichi backend.
Use `ShortwaveSolver(..., backend="vulkan-rt")` or
`Solver(..., device="vulkan-rt")` with a build that enables `RAYSTRACK_VULKAN`.

## Build

Use a C++20 compiler, CMake, the Vulkan headers and loader import library, and
`glslc` to compile the GLSL compute shader to SPIR-V. The Windows x64 reference
toolchain is Visual Studio Build Tools 18 with MSVC 14.50, CMake 4.3.2, and
[LunarG Vulkan SDK 1.4.350.0](https://vulkan.lunarg.com/doc/view/1.4.350.0/windows/getting_started.html).
The [official x64 SDK download](https://vulkan.lunarg.com/sdk/home/) lists SHA-256
`855b27ba05d2d8119c5114c5d4ff870ca38f2c632b11e1bb9923b9b7e6ecfe7b`.
See the [SDK component licenses](https://vulkan.lunarg.com/license/home) for
redistribution terms. Vulkan-Headers uses Apache-2.0 and MIT, Vulkan-Loader is
primarily Apache-2.0, and shaderc is Apache-2.0 with third-party notices. The
Vulkan SDK is a build dependency. The runtime uses the platform Vulkan loader
and vendor GPU driver. Raystrack does not distribute the SDK or driver.

With the SDK installed and `VULKAN_SDK` set to its root, run from the repository:

```powershell
cmake -S native/vulkan -B build/vulkan -A x64
cmake --build build/vulkan --config Release
ctest --test-dir build/vulkan -C Release --output-on-failure
```

The standalone build tests the same traversal used by the Python API. To use
the Khronos validation layer, point `VK_LAYER_PATH` at the SDK's `Bin`
directory and set `VK_INSTANCE_LAYERS` to `VK_LAYER_KHRONOS_validation` before
running CTest.

Build the Raystrack shared library and select it in the same shell:

```powershell
cmake -S native -B build/native-vulkan -A x64 -DRAYSTRACK_VULKAN=ON
cmake --build build/native-vulkan --config Release
$env:RAYSTRACK_NATIVE_LIBRARY = (Resolve-Path build/native-vulkan/Release/raystrack_native.dll).Path
```

CMake discovers the SDK from the build environment. On Linux, use a
platform-matched SDK and omit the Visual Studio `-A x64` argument.

## Runtime requirements and scope

The Vulkan device must expose a compute queue, Vulkan 1.2 or higher,
`VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, and
`VK_KHR_ray_query` with their required feature bits. The probe performs a real
compute dispatch against a committed triangle scene. An unavailable physical
device or missing feature is an explicit error.

The Windows x64 probe passed on the AMD Radeon 860M with driver 26.10.1,
with Khronos validation enabled. It checks closest-hit metadata, bounded
occlusion, splitting a batch across dispatches, and a geometry rebuild.

The implementation builds world-space triangle geometry into a BLAS and one
identity instance in a TLAS. A geometry edit builds replacement BLAS/TLAS
objects on the retained Vulkan device. The compute pipeline, descriptor objects
and mapped ray/hit buffers remain allocated. Successful commits replace the
host hit metadata and acceleration structures together, then rebind the TLAS.
Invalid vertices or indices leave the previous scene usable. Topology edits,
surface count changes, and empty/refilled scenes use the same update path.
This is device/pipeline reuse; geometry still gets a full acceleration build.
Shared prototype BLAS instances and AS refits remain future work.

Native ray input and hit output are transferred in batches using persistent
mapped buffers. The occlusion query transfers a single 32-bit result per ray.
Sampling, materials and result accumulation run on the host. `update` and trace
operations serialize on one traversal handle. Different handles own independent
devices and do not share a process-global Vulkan context.

## Profiling and update regression

Set `RAYSTRACK_VULKAN_PROFILE=1` before creating a traversal to emit one aggregate
JSON object to stderr when it is destroyed. Fields are milliseconds:

- `context_ms`: instance/device/queue/command-pool setup.
- `pipeline_ms`: shader, descriptor and compute pipeline setup.
- `geometry_pack_ms`: validation, flat geometry packing and winding normals.
- `geometry_build_ms`: geometry-buffer allocation/upload, BLAS/TLAS build and
  synchronized completion; includes driver first-use costs.
- `upload_pack_ms`: host ray packing into mapped coherent input memory.
- `dispatch_wait_ms`: command recording, submission and queue-idle wait.
- `readback_decode_ms`: coherent output reads and host hit/occlusion decoding.

The record includes the device name and Vulkan's raw API/driver version and
vendor/device identifiers. Counters include successful geometry commits
(initial creation included), dispatches, and submitted rays. These are host wall-clock phase boundaries,
not GPU timestamp queries. On the AMD integrated GPU, mapped coherent memory
does not provide separately timed discrete PCIe copies. Transport sampling and
irradiance reductions are outside traversal and outside these counters.

```powershell
$env:VK_LAYER_PATH = (Resolve-Path build/vulkan-deps/sdk/Bin).Path
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$env:RAYSTRACK_VULKAN_PROFILE = '1'
cmake --build build/vulkan --config Release
ctest --test-dir build/vulkan -C Release --output-on-failure
python -m pytest tests/test_native_vulkan_updates.py -q
```

The standalone probe checks update versus fresh-build hits, stable surface and
primitive slots, changed topology/winding, invalid-edit rollback, and empty to
nonempty transitions. CTest fails on validation error/VUID text. Its captured
output is `build/vulkan/Testing/Temporary/LastTest.log`.

## Shortwave benchmark

Build the shared library with both `RAYSTRACK_VULKAN=ON` and
`RAYSTRACK_EMBREE=ON` and select it via `RAYSTRACK_NATIVE_LIBRARY`. On Windows,
pass the absolute Embree runtime directory to the benchmark:

```powershell
python validation/benchmark_native_shortwave.py --grid 256 --samples 2048 --repeats 5 --embree-dll-dir $embreeRuntime
```

In the previous implementation session this 131,072-triangle canopy, with 64 sensors and 24
timesteps, took 159 ms on Embree and 265 ms on Vulkan at the median of five
steady solves. The first solve took 263 ms and 2,650 ms, respectively.
Solving after a geometry move took 205 ms and 421 ms. The two backends produced
the same irradiance arrays for this seed in both geometry revisions. These are
end-to-end timings for this scene. The cold result includes device, pipeline,
and acceleration structure setup as applicable.

### Geometry-update implementation comparison (2026-11-11)

The same tracked runner was executed with the preserved pre-change DLL and the
updated DLL, in three independent subprocess trials per version. Version order
alternated between trials; each process ran Embree and then Vulkan sequentially.
All other agent workloads were idle during these samples. Khronos validation
and traversal profiling were disabled for timing and checked separately.

The fixture has 131,072 triangles, 64 sensors, 24 timesteps, 2,048 cosine
samples per sensor, seed 11, a diffuse thin-sheet canopy, and five stationary
repeats. Values below are medians across three trial values; stationary values
are medians of each trial's five repeats. A moved solve includes lazy geometry
preparation and transport at the edited position.

| Backend | Version | First solve (ms) | Stationary (ms) | Moved solve (ms) |
| --- | --- | ---: | ---: | ---: |
| Embree | Before | 196.15 | 137.25 | 159.56 |
| Embree | Updated | 182.96 | 128.70 | 129.36 |
| Vulkan RT | Before | 2227.02 | 276.44 | 2487.33 |
| Vulkan RT | Updated | 2100.69 | 271.10 | 307.85 |

The measured Vulkan moved-solve ratio is 8.08, from retaining the device and
pipeline across edits. Stationary trial medians overlap: before 260.94–316.09 ms,
updated 249.85–301.49 ms. These samples do not establish a stationary throughput
improvement. First-use driver costs also vary substantially. Embree remains
faster on this host-transport workload. Both backends produced identical
irradiance arrays before and after the move in all six runs.

Raw commands, DLL SHA-256 hashes, every timing sample and backend differences
are saved under ignored `build/annual-validation/canopy-comparison.json`;
`canopy-summary.json` contains the aggregation above. To reproduce each version,
select its DLL in `RAYSTRACK_NATIVE_LIBRARY` and run the command shown above with
`--grid 256 --samples 2048 --repeats 5`. Separate geometry preparation/update and
trace measurements use `validation/benchmark_native_updates.py`.
