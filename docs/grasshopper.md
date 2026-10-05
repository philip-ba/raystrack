# Raystrack for Grasshopper

Raystrack's Grasshopper plugin now lives in this repository. It is a compiled
Rhino 8 GHA with its own Python runtime. Geometry, requests, options and results
follow the same v2 object model as the Python API. The previous
`raystrack_grasshopper` repository is kept as a reference; this migration does
not delete or change it.

## Install

The current Windows 64-bit build supports Rhino 8.35 or newer and Grasshopper.
Yak derives this minimum from the Rhino assemblies used to compile the plugin.
The Yak package
and the standalone ZIP contain the same `Raystrack.Components.gha` and adjacent
`runtime/` directory. Yak remains the Package Manager installation route. The
standalone distribution allows direct GHA installation without Python or pip on
the target computer.

For a local build:

```powershell
python tools/grasshopper/build.py
```

The build produces a Yak package and a standalone ZIP under `dist/grasshopper/`.
The development tools target `raystrack-2.0.0-dev.3-rh8_35-win.yak`.
The package version is `2.0.0-dev.3`; the numerical Python package retains its
current development version `1.0.2`.

Install the local Yak package with the build tool's `--install` option, then
restart Rhino so it loads the new assembly:

```powershell
python tools/grasshopper/build.py --install
```

For manual installation, extract the standalone ZIP into a Grasshopper Libraries
folder. Keep `Raystrack.Components.gha` and `runtime/`
together. Copying the GHA by itself is insufficient: the plugin intentionally
uses its packaged Python interpreter instead of Rhino's Python environment.

The runtime is the official Python 3.12.10 embeddable 64-bit distribution with
pinned NumPy, Numba, llvmlite and Taichi dependencies. Installing and starting the
plugin requires no first-launch pip operation. Vulkan support is included;
hardware/backend availability still depends on the local driver. Use **RS
Runtime** to inspect the interpreter and available devices. A requested backend
that cannot execute the selected strategy produces an explicit error.

The build needs network access to obtain the pinned Python distribution and
dependency wheels. Runtime execution does not download dependencies. `--gha-only`
builds the assembly for development; a runnable installation still needs its
matching runtime directory. `--skip-yak` can produce the standalone artifact
without Yak packaging.

The bundle includes `Raystrack.Components.xml` with the assembly's XML
documentation, plus `package-manifest.json` recording source and payload hashes.
The bundled runtime has its own `runtime/runtime-manifest.json`. These identify
the code behind a package and keep validation reports tied to the tested build.

## Canvas structure

The **Raystrack** ribbon has a category icon and numbered sections. Existing
component artwork is reused to retain the original visual style. Every component
has an icon and descriptions on its ports.

Hover over a port for its accepted value type and purpose. Component help explains
the workflow, defaults and execution triggers. Connect a typed output to a Panel
for a concise summary of its settings or contents. **RS Inspect** adds a usage
description; set its **Details** input true to append the complete JSON snapshot.
Passing a Sampling, Accuracy, Options or Query value now
shows its actual selections rather than just the name of the type.

| Section | Component | Purpose |
| --- | --- | --- |
|00 Setup | **RS Runtime** | **Python** identifies the shipped interpreter; **Devices** reports availability; **Status** reports the check's state. Pulse Refresh to check. |
|01 Scene | **RS Surface** | **Surface** carries the ID, optional label, triangle mesh and transform. Its summary reports geometry counts and placement. |
|01 Scene | **RS Instance** | **Surface** reuses the prototype geometry with a new ID and composed transform; feed it into the same Scene input. |
|01 Scene | **RS Scene** | **Scene** summarizes surface and shared-geometry counts; **IDs** lists the exact query identifiers in scene order. |
|02 Solve | **RS Query** | **Query** summarizes selected senders/receivers, scene or sky outputs, and receiver sides. |
|02 Solve | **RS Sampling** | **Sampling** summarizes the estimator, density, rays or pair samples, seed and allocation mode. |
|02 Solve | **RS Accuracy** | **Accuracy** summarizes the tolerance, convergence mode and replicate limits. |
|02 Solve | **RS Options** | **Options** combines sampling/accuracy selections, batch size and reciprocity; omitted inputs display their effective defaults. |
|02 Solve | **RS Solve** | **Result** is the latest snapshot; **Status**, **Progress**, **Rays** and **Info** describe execution and diagnostics. |
|03 Results | **RS Result Table** | **Senders** and **Channels** identify rows/columns; **Labels** names columns; **Values**/**Errors** are matching trees; **Coverage** marks sampled rows. |
|03 Results | **RS Result Value** | **Value**, **Error** and **Coverage** describe the single selected sender/channel entry. |
|03 Results | **RS Inspect** | **Text** gives a readable summary and usage description. Set **Details** true to append JSON with settings, execution, convergence and provenance. |
|04 Files | **RS Save** | **Path** names the saved v2 folder; **Status** reports completion or failure. Scene and optional Result are stored together. |
|04 Files | **RS Load** | **Scene** and optional **Result** are imported snapshots; **Status** reports the read operation. |

A typical definition is:

```text
Rhino meshes --> RS Surface --> RS Scene ----+
                                            |
RS Query -----------------------------------+--> RS Solve --> RS Result Table
                                            |          +--> RS Result Value
RS Sampling --+                             |          +--> RS Inspect
              +--> RS Options --------------+
RS Accuracy --+
```

Open [facing-plates.gh](../examples/grasshopper/facing-plates.gh) for a complete portable example. It
embeds a unit-square prototype and its opposing instance, selects their pair
query, and connects table/value inspection. **Run** and **Cancel** are false;
set **Run** true to solve. The example contains no file-operation paths.

Surface IDs identify senders and receivers. Labels are display text. Include
every blocker in **RS Scene**: filtering receivers in **RS Query** selects outputs
and retains all scene occluders. Mesh winding controls the emitting normal and
receiver front/back side. Instance transforms are rigid; use new mesh geometry
for deformation or scaling.

### Connecting Transform

The **Transform** input accepts a native Grasshopper transformation. Connect the
**Transform (X)** output of **Move**, **Rotate** or another rigid transform component.
For example, connect the original Rhino mesh/Brep to **Move**'s **Geometry (G)**
input and a translation vector to its **Motion (T)** input, then
connect Move's **Transform (X)** output to **RS Instance**. Feed the original **RS
Surface** to that instance and choose a new ID. A vector is the motion used to
construct a transform; the vector itself is not this input's value. Do not pass
a Panel of 16 matrix numbers.

Leaving Transform empty uses identity. Only translation and rotation are
supported; scale, shear, mirror and malformed transforms are rejected with an
explanation. For scaling or deformation, create the changed Rhino mesh upstream
and convert that geometry with **RS Surface**.

Choose one way to apply a placement. If you connect **Move** or **Rotate**'s
already transformed **Geometry** output to **RS Surface**, leave Surface's
Transform empty. Connecting that transformed geometry and the same transform
applies the placement twice.

An instance transform is applied **after** the prototype's current transform.
For local mesh point `p`, prototype placement `P` and instance transform `T`, the
instance position is `T * (P * p)`. Thus moving an already placed prototype moves
that placed copy; it does not reset the prototype to its original mesh position.

For **RS Query**, choose `matrix`, `row`, `pair` or `sky`. A row selects one sender;
a pair selects one sender and one receiver. Sky can be `merged` or
`tregenza145`. Result channels carry a kind, surface ID, side and optional sky
patch; front/back names are no longer parsed from `_front` or `_back` suffixes.

**RS Sampling** retains the validated cosine estimator and offers CPU-only
`area_pair` sampling for pair queries. The `fair` and `adaptive` modes allocate
work differently while preserving the selected estimator. **RS Accuracy**
controls convergence. Errors are unknown until sufficient completed replicates
exist, and pending partial replicates cannot claim convergence.

Typed result summaries include the status, sender/channel counts, ray counts and
coverage. A **Channels** value names its actual receiver ID/side or sky patch.
The ordinary **Labels** output provides the same column meanings as plain text.
In a table, branch `{0}` belongs to the first Sender and its items follow Channels
order. Errors share the same branches and columns as Values.

## Background execution and live progress

Set **Run** false and then true to submit a solve. The GHA sends the request from
a background task to a document-owned Python process. Grasshopper's UI thread
does not wait for ray tracing, compilation or IPC. The component schedules canvas
updates while it runs; its message and **Status**, **Progress** and **Rays** outputs
report live execution.

A solver reports `queued`, `preparing`, `running`, `paused`, `succeeded`,
`cancelled` or `failed`. **Info** reports diagnostics, including capability and
runtime errors. Preparation may include Numba compilation on the first solve;
the canvas remains interactive during that phase.

Progress is a percentage from 0 to 100. Rays is the cumulative count for the
current run, including earlier advances. A succeeded request may have exhausted
its replicate limit; inspect the Result's convergence status before treating it
as meeting the requested tolerance.

The **Ray budget** is additional work for a submitted advance. Zero selects an
unlimited advance. A finite budget produces a partial snapshot and can pause the
run. Rearming **Run** resumes the same compatible query and sampling prefix.
Changing the scene, query or sampling starts a new compatible run; prior results
remain snapshots rather than being silently reinterpreted.

Set **Cancel** true to request cancellation. The solver returns the completed
sample prefix as a partial result. Cancellation is checked between bounded
chunks, so a currently executing kernel may finish before the state changes.
The worker process remains available for subsequent operations in the same
document. Removing the document stops its owned process without terminating
other Python processes or simulations.

Definitions saved with **Run** true reopen with their execution triggers
disarmed. Set false and then true to execute again. The same rule applies to
file operations. Moving a slider or recomputing an unrelated part of the canvas
does not repeatedly submit the same enabled trigger.

Coverage distinguishes sampled rows from uncomputed rows and legacy unknown
coverage. A sampled zero is a known zero; an uncomputed or unknown entry must not
be used as zero. Reciprocity is explicit in **RS Options** and recorded in result
provenance. Incomplete calculations are not silently normalized.

### Correcting input errors

Input errors identify the value or field that needs correction. For example,
duplicate IDs ask you to give surfaces distinct IDs; an unsupported transform
asks for a rigid translation/rotation; a wrong Raystrack value type identifies
the expected connection. Malformed snapshots show **invalid data** with a reason
instead of failing during Panel display or preview.

Result-reading components wait quietly while a connected solver has not produced
its first snapshot. A waiting input is not a calculation failure. Once a result
exists, an unknown sender or channel reports that selection problem. Use **RS
Scene**'s IDs and **RS Result Table**'s Channels/Labels to choose valid names.
Missing errors or estimates remain unknown rather than being converted to zero.

For backend or runtime failures, read **RS Solve**'s Info output or the component's
runtime message. Choose a supported device/estimator or correct the installation
described in the message, then set Run false and true to retry. **RS Save** refuses
an existing target folder; choose a new folder instead of overwriting stored work.

## Old component migration

Old definitions require rewiring to the new scene/request/options workflow. The
GHA uses new stable component IDs; it does not masquerade as the old scripted
components or keep their duplicated calculation contracts.

| Previous component | New workflow |
| --- | --- |
| RaystrackComputeVF | RS Scene + RS Query (`pair`) + RS Solve |
| RaystrackComputeVFMatrix | RS Scene + RS Query (`matrix`) + RS Solve |
| RaystrackComputeSkyVF | RS Scene + RS Query (`sky`) + RS Solve |
| RaystrackMatrixParams / RaystrackSkyParams | RS Sampling + RS Accuracy + RS Options |
| RaystrackGetVF | RS Result Value |
| RaystrackGetTable | RS Result Table |
| RaystrackSaveMeshes / RaystrackSaveVFMatrix | RS Save, storing scene and optional result together |
| RaystrackLoadMeshes / RaystrackLoadVFMatrix | RS Load |
| Installer / outside workflow / sync helpers | Bundled runtime + RS Runtime + background RS Solve |

The numerical core can import v1 storage through its isolated import adapter.
Coverage and statistics absent from v1 data remain unknown. New writes use v2
only. See [the Python migration guide](v2-migration.md) for model, estimator and
storage details.

## Validation in Rhino and Grasshopper

After installing the built Yak package, run this command in Rhino, replacing
`<repository>` with the path to your Raystrack source checkout:

```text
_-RunPythonScript "<repository>\tools\grasshopper\host_smoke.py"
```

The script opens Grasshopper, builds a real definition using two facing unit
squares (one prototype and one rigidly transformed instance), runs the installed
components, and writes
`.local/host-proof/grasshopper-report.json`. It returns to Rhino immediately and
uses a Windows Forms timer on the UI thread to verify responsiveness while the
worker is active. The generated definition is saved as
`.local/host-proof/raystrack-plates-smoke.gh`.

The acceptance checks include the analytical pair factor `0.1998248957`, multiple
live ray updates, instance geometry/preview, component icons and port descriptions,
v2 Save/Load, active-run
cancellation, owned-worker cleanup when the document is removed, and disarmed
triggers after reopening the saved definition. A runtime/device check is queued
during the long solve to verify that an outstanding background request still
allows status updates and cancellation. The report includes images rendered by
the actual Grasshopper editor during progress and after reopening the definition.
The report must have
`"phase": "passed"`; a generated report or launched application alone is not a
successful test.

From the source checkout, `python tools/grasshopper/host_smoke.py --verify`
checks that the report contains every required piece of host evidence.

The previously installed `2.0.0-dev.2` package passed this acceptance test on 2026-10-05
in Rhino `8.35.26251.13001`, using its shipped Python `3.12.10`. The facing-square
calculation returned `0.199676513672` versus the analytical `0.1998248957`
(absolute error `0.000148382028`). The UI timer ticked 25 times during active
background execution, and the component reported distinct live ray counts.
Save/Load, cancellation, a queued runtime check, document-owned process cleanup,
and reopening with disarmed triggers all passed. See the saved
[acceptance report](../validation/results/grasshopper_v2_acceptance.json) for the
checks, source hashes and device information, and the
[portable example](../examples/grasshopper/facing-plates.gh) for the tested workflow.

That report applies to its recorded source hashes. The later `2.0.0-dev.3`
usability refinements receive separate managed and runtime checks. Native GUI
validation of those changes is left to your own testing after installation.
Test the changes in Grasshopper or run the host script before attributing the
earlier acceptance result to that build.

Developers can separately run `python tools/grasshopper/assembly_contract.py` to
check summaries, malformed snapshot diagnoses, rigid transforms, component help
and icon pixels against a temporary
managed assembly. This neither opens Rhino nor builds or installs a distribution;
it does not replace the native host test.

Background simulations can affect throughput, so this host proof checks
correctness and responsiveness without making performance comparisons. Real
Rhino acceptance covers Windows and Rhino 8. Physical CUDA and Metal execution
remain separate validation tasks; no claim about those devices follows from the
CPU host test.
