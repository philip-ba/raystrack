# Raystrack for Grasshopper

Raystrack's Grasshopper plugin now lives in this repository. It is a compiled
Rhino 8 GHA with its own Python runtime. Geometry, requests, options and results
follow the same v2 object model as the Python API. The previous
`raystrack_grasshopper` repository is kept as a reference; this migration does
not delete or change it.

## Install

The current Windows 64-bit build supports Rhino 8.35 or newer and Grasshopper.
Open Rhino's **PackageManager**, search for **raystrack**, and install version
**2.0.0**, then restart Rhino. The compiled components appear on the
**Raystrack** tab in Grasshopper.
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
The release tools target `raystrack-2.0.0-rh8_35-win.yak`.
The Grasshopper package and numerical Python package are both version `2.0.0`.

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

The **Raystrack** ribbon has a category icon and numbered sections. The supplied RS component artwork is embedded, including the Sky icon. Every component
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
|01 Scene | **RS To Brep** | Converts a Surface or Scene to native triangulated **Breps** at their saved world placement, with matching **IDs** and **Labels**. |
|01 Scene | **RS Scene** | **Scene** summarizes surface and shared-geometry counts; **IDs** lists the exact query identifiers in scene order. |
|02 Solve | **RS Query** | **Query** summarizes selected senders/receivers, scene or sky outputs, and receiver sides. Connect an RS Sky object to **Sky**. |
|02 Solve | **RS Sky** | Sets merged/discrete sky mode and dome display settings; outputs **Sky**, native **Patches**, matching **Labels**, **Label points**, and bakeable text **Tags**. |
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

Open [boxes.gh](../examples/grasshopper/boxes.gh) for a portable example. It
compares two outward-emitting closed boxes and the inward-emitting faces of a
box. Use the **Values** output of RS Result Table for view factors; **Errors**
contains standard errors, which can correctly be zero. Saved true Run inputs
are disarmed on reopen: set **Run** false, then true to launch. For the interior
case use **Flip** true and **back** receiver sides.

Surface IDs identify senders and receivers. Labels are display text. Include
every blocker in **RS Scene**: filtering receivers in **RS Query** selects outputs
and retains all scene occluders. Mesh winding controls the emitting normal and
receiver front/back side. Surface transforms are rigid; use new mesh geometry
for deformation or scaling. RS Instance has been removed; use native transforms
and a separate RS Surface with its own ID for each placement. Older definitions
containing RS Instance need that component replaced.

### Connecting Transform

The **Transform** input accepts a native Grasshopper transformation. Connect the
**Transform (X)** output of **Move**, **Rotate** or another rigid transform component.
For example, connect the original Rhino mesh/Brep to **Move**'s **Geometry (G)**
input and a translation vector to its **Motion (T)** input, then
connect Move's **Transform (X)** output to **RS Surface** with the original
geometry and choose a distinct ID for each placement. A vector is the motion used to
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

For **RS Query**, choose `matrix`, `row`, `pair` or `sky`. A row selects one sender;
a pair selects one sender and one receiver. Connect **RS Sky** to Query's **Sky**
input to include sky channels; this connection is required for Mode=`sky`.
Other query modes can request sky and scene contributions together.

### Sky dome and labels

**RS Sky** contains all sky settings. Mode=`merged` returns a single hemisphere
with label `Sky`; Mode=`tregenza145` returns 145 spherical Breps in patch order
`0..144`. **Labels** exactly matches RS Result Table's sky channel labels.
**Tags** displays each number just above its patch and can be baked directly
as native Rhino text, alongside **Patches**. **Label points** supports custom
Text Tag components. Connect **Sky** to a Panel or RS Inspect for a readable
representation of its mode and display settings.

**Center**, **Radius**, **Label size** and **Show labels** affect the display.
Label size `0` scales automatically with Radius. Hidden labels leave Tags empty
while still returning Labels and Label points. Display placement does not change
solver directions: +Z is up, azimuth starts at +X and increases toward +Y, and
alternate Tregenza rings are offset by half a sector. Patch 144 is the zenith cap.
The dome has no rotation setting, so its labels stay aligned with the solver.

### Converting back to Rhino

Connect an RS Surface or RS Scene, including RS Load's Scene, to **RS To Brep**.
The component returns one native Brep per surface, in scene order, and applies
each saved transform once. IDs and Labels follow the same order. These are the
stored triangle faces; the original smooth Brep is not retained after RS Surface
meshes it. Use or bake these Breps with ordinary Rhino/Grasshopper tools.

Result channels carry a kind, surface ID, side and optional sky patch; front/back
names are no longer parsed from `_front` or `_back` suffixes.

### Self-viewing within one mesh

Self-viewing is enabled by default. A surface can receive rays on other faces
of its own mesh, so a concave enclosure does not need to be split into separate
RS Surfaces. Matrix and row queries include the sender's receiver channels;
for a pair query, choose the same surface ID for Senders and Receivers.
Unrequested self-hits still block sky and other receivers.

For a box with outward mesh normals, set **RS Sampling → Flip** true to emit
into the interior, and read the box's **back** receiver channel (or keep Sides
set to both). A closed box then has self view factor 1 and zero sky/escape. With
inward mesh normals, leave Flip false and read the **front** receiver channel.
Flip changes emitting normals only; receiver side labels keep the stored mesh
winding. An outward-emitting convex mesh correctly has zero self view factor.

CPU, CUDA and portable GPU tracing share this behavior, including flat/BVH and
instanced traversal. The CPU area-pair estimator also accepts a same-ID pair.
Ray origins are offset from their starting triangle to avoid immediate numerical
self-intersections; other faces of the same mesh remain visible.

The dev.5 self-viewing checks cover closed/convex boxes on CPU and physical
Vulkan, including deferred replicates, plus an open-box analytical check
(measured self 0.79859375 and sky 0.20140625 versus expected 0.8 and 0.2).
See the [self-viewing report](../validation/results/grasshopper_dev5_self_viewing.json).

The dev.5 regression suite passed 254 tests, with 23 optional/platform skips.
The actual packaged interpreter also passed the CPU and physical Vulkan enclosure
checks. CUDA self-viewing was checked in simulation with both ray generation
paths; physical CUDA hardware was not tested.

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
squares (one lower surface and one rigidly transformed upper surface), runs the installed
components, and writes
`.local/host-proof/grasshopper-report.json`. It returns to Rhino immediately and
uses a Windows Forms timer on the UI thread to verify responsiveness while the
worker is active. The generated definition is saved as
`.local/host-proof/raystrack-plates-smoke.gh`.

The acceptance checks include the analytical pair factor `0.1998248957`, multiple
live ray updates, surface placement and To Brep, Sky patches/labels/tags, component icons and port descriptions,
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
[portable example](../examples/grasshopper/boxes.gh) for the tested workflow.

That report applies to its recorded source hashes. The earlier `2.0.0-dev.4`
changes passed 81 targeted tests (one platform-specific skip), including worker
Save/Load, managed summaries/icons, native surface/scene Breps and actual
Grasshopper connections for Sky/Query/To Brep. All 1,305 sampled native Brep
interior directions matched the numerical solver's patch IDs. See the
[current validation report](../validation/results/grasshopper_dev4_native.json).
These checks used a headless Rhino core. Run the host script for installed-canvas
and interactive viewport acceptance of this version.

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

`python tools/grasshopper/assembly_contract.py --native` also verifies native
surface/scene Breps, world transforms, every spherical sky patch's area and
text tag, and the merged hemisphere in a headless Rhino geometry core.
