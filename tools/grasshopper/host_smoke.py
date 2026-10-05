"""Exercise the installed GHA in real Rhino 8 and Grasshopper.

Run with Rhino's ``_-RunPythonScript`` (legacy IronPython is supported). The
script returns to Rhino immediately; a UI timer drives its acceptance checks.
Reports and the generated Grasshopper definition live in .local/host-proof.
Only documents and worker PIDs created by this script are removed.
"""
import json
import hashlib
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROOF = os.path.join(ROOT, ".local", "host-proof")
if not os.path.isdir(PROOF):
    os.makedirs(PROOF)


def write_report(report):
    """Persist the latest acceptance state so an external runner can inspect failures."""
    with open(os.path.join(PROOF, "grasshopper-report.json"), "w") as stream:
        json.dump(report, stream, indent=2)


def validate_report(report):
    """Require actual host evidence rather than accepting a passed label alone."""
    assert report.get("phase") == "passed", "Real-host test did not pass"
    assert report.get("rhino") and report.get("assembly"), "Installed host identity is missing"
    assert len(report.get("components", [])) == 14, "Installed component catalog is incomplete"
    assert report.get("active_heartbeats", 0) >= 2, "No UI heartbeat during background execution"
    live = report.get("live_ray_counts", [])
    assert len(set(live)) >= 2 and all(value > 0 for value in live), "No distinct live ray updates"
    assert max(report.get("result_rays", 0), report.get("cancelled_rays", 0)) >= max(live), "Final ray count is inconsistent"
    assert abs(report.get("pair_value", -1) - 0.1998248957) < 0.003, "Analytical check failed"
    for proof in ("icons_and_tooltips", "ribbon_icon", "instance_transform", "save_load_preserved",
                  "cancelled_partial", "runtime_multiplexing", "document_worker_stopped", "reopen_disarmed"):
        assert report.get(proof) is True, "Missing host evidence: " + proof
    assert report.get("owned_worker_pid", 0) > 0, "Owned worker was not observed"
    assert report.get("document"), "Generated Grasshopper document was not saved"
    return report


def run_host():
    """Build a test-owned GH document and verify it through asynchronous UI callbacks."""
    import clr
    import System
    import Rhino
    gh = Rhino.RhinoApp.GetPlugInObject("Grasshopper")
    gh.LoadEditor()
    clr.AddReference("Grasshopper")
    clr.AddReference("GH_IO")
    import Grasshopper
    from Grasshopper.Kernel import GH_Document, GH_RuntimeMessageLevel
    from Grasshopper.Kernel.Data import GH_Path
    from Grasshopper.Kernel.Types import GH_Boolean, GH_Integer, GH_Number, GH_String, GH_Mesh, GH_Transform
    from GH_IO.Serialization import GH_Archive
    from System.Windows.Forms import Timer

    Grasshopper.Instances.DocumentEditor.Show()
    Grasshopper.Instances.DocumentEditor.ClientSize = System.Drawing.Size(1440, 920)
    assembly = next(a for a in System.AppDomain.CurrentDomain.GetAssemblies()
                    if a.GetType("Raystrack.Grasshopper.SolveComponent") is not None)
    document = GH_Document()
    Grasshopper.Instances.DocumentServer.AddDocument(document)
    Grasshopper.Instances.ActiveCanvas.Document = document
    nodes = {}
    report = {"phase": "waiting", "rhino": str(Rhino.RhinoApp.Version),
              "rhino_process_id": int(System.Diagnostics.Process.GetCurrentProcess().Id),
              "assembly": assembly.Location, "states": [], "heartbeats": 0,
              "active_heartbeats": 0, "live_ray_counts": [], "checks": []}
    with open(assembly.Location, "rb") as stream:
        report["assembly_sha256"] = hashlib.sha256(stream.read()).hexdigest()
    started = time.time()
    state = {"phase": "solve", "phase_start": started, "saved": None,
             "cleanup_pid": None, "reopened": None, "runtime_submitted": False,
             "cancel_progress": []}

    def node(key, class_name, x, y):
        """Instantiate an installed component and position it in the test document."""
        component_type = assembly.GetType("Raystrack.Grasshopper." + class_name)
        if component_type is None:
            raise RuntimeError("Installed component unavailable: " + class_name)
        component = System.Activator.CreateInstance(component_type)
        component.CreateAttributes()
        component.Attributes.Pivot = System.Drawing.PointF(x, y)
        document.AddObject(component, False)
        nodes[key] = component
        return component

    def parameter(component, name, input_port=True):
        """Resolve a port by public name so API changes produce an explicit failure."""
        ports = component.Params.Input if input_port else component.Params.Output
        matches = [p for p in ports if str(p.Name).lower() == name.lower()]
        if len(matches) != 1:
            raise RuntimeError("Missing or ambiguous port " + component.Name + "." + name)
        return matches[0]

    def put(key, name, value):
        """Supply native persistent GH values, including Mesh and Transform goos."""
        port = parameter(nodes[key], name)
        port.PersistentData.Clear()
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, bool):
                goo = GH_Boolean(item)
            elif isinstance(item, int) and port.GetType().Name == "Param_Integer":
                goo = GH_Integer(item)
            elif isinstance(item, (int, float)):
                goo = GH_Number(item)
            elif isinstance(item, Rhino.Geometry.Mesh):
                goo = GH_Mesh(item)
            elif isinstance(item, Rhino.Geometry.Transform):
                goo = GH_Transform(item)
            else:
                goo = GH_String(item)
            port.PersistentData.Append(goo, GH_Path(0))
        port.ExpireSolution(False)

    def connect(source, output_name, target, input_name):
        """Wire actual component ports rather than emulating the model conversion."""
        parameter(nodes[target], input_name).AddSource(parameter(nodes[source], output_name, False))

    def output(key, name):
        """Read the first computed output value without submitting additional work."""
        data = list(parameter(nodes[key], name, False).VolatileData.AllData(True))
        return data[0] if data else None

    def text(key, name):
        """Extract a primitive output's readable value for status and progress checks."""
        value = output(key, name)
        return str(value.Value) if value is not None and hasattr(value, "Value") else str(value) if value is not None else None

    def payload(key, name):
        """Decode a portable snapshot through its public JSON representation."""
        value = output(key, name)
        if value is None:
            return None
        return json.loads(value.ToJson() if hasattr(value, "ToJson") else str(value.Value))

    def alive(pid):
        """Check one recorded process ID without touching unrelated user processes."""
        if not pid:
            return False
        try:
            process = System.Diagnostics.Process.GetProcessById(int(pid))
            running = not process.HasExited
            process.Dispose()
            return running
        except System.ArgumentException:
            return False

    def errors():
        """Collect real component errors so silent host failures cannot pass the proof."""
        return [key + ": " + str(message) for key, component in nodes.items()
                for message in component.RuntimeMessages(GH_RuntimeMessageLevel.Error)]

    def capture(filename):
        """Render the actual editor, framing only the active test-owned document."""
        # Render the application's own WinForms control after it returns to its
        # message loop. This captures the real ribbon and canvas, not a mockup.
        editor = Grasshopper.Instances.DocumentEditor
        def select_ribbon(control):
            """Select the Raystrack category through the native ribbon's public property."""
            if control.GetType().FullName == "Grasshopper.GUI.Ribbon.GH_Ribbon":
                control.ActiveTabName = "Raystrack"
            for child in control.Controls:
                select_ribbon(child)
        select_ribbon(editor)
        canvas = Grasshopper.Instances.ActiveCanvas
        current = canvas.Document
        if current is not None:
            bounds = current.BoundingBox(False)
            if bounds.Width > 0 and bounds.Height > 0:
                # Frame only this test-owned document, independent of the
                # user's previously selected zoom or viewport target.
                zoom = min(1.0, max(0.2, (canvas.ClientSize.Width - 80.0) / bounds.Width),
                           max(0.2, (canvas.ClientSize.Height - 80.0) / bounds.Height))
                canvas.Viewport.Zoom = zoom
                canvas.Viewport.MidPoint = System.Drawing.PointF(bounds.Left + bounds.Width / 2,
                                                                bounds.Top + bounds.Height / 2)
                canvas.Refresh()
        editor.Refresh()
        bitmap = System.Drawing.Bitmap(editor.ClientSize.Width, editor.ClientSize.Height)
        try:
            editor.DrawToBitmap(bitmap, System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height))
            path = os.path.join(PROOF, filename)
            bitmap.Save(path, System.Drawing.Imaging.ImageFormat.Png)
            return path
        finally:
            bitmap.Dispose()

    def export_basic_example():
        """Save the proven workflow with idle triggers and no machine-specific file paths."""
        archive = GH_Archive()
        assert archive.ReadFromFile(report["document"])
        example = GH_Document()
        assert archive.ExtractObject(example, "Definition")
        for component in list(example.Objects):
            if component.GetType().Name in ("SaveComponent", "LoadComponent", "RuntimeComponent"):
                example.RemoveObject(component, False)
            elif component.GetType().Name == "SolveComponent":
                for name in ("Run", "Cancel"):
                    port = parameter(component, name)
                    port.PersistentData.Clear()
                    port.PersistentData.Append(GH_Boolean(False), GH_Path(0))
        # Embed sample geometry but remove local file-operation paths. Opening
        # this portable graph cannot start a solver or write a result folder.
        assert not any(c.GetType().Name in ("SaveComponent", "LoadComponent") for c in example.Objects)
        example_folder = os.path.join(ROOT, "examples", "grasshopper")
        if not os.path.isdir(example_folder):
            os.makedirs(example_folder)
        path = os.path.join(example_folder, "facing-plates.gh")
        output_archive = GH_Archive()
        output_archive.AppendObject(example, "Definition")
        assert output_archive.WriteToFile(path, True, False)
        return path

    def validate_catalog():
        """Check installed component IDs, icons, category and all port descriptions."""
        types = [t for t in assembly.GetTypes() if not t.IsAbstract
                 and t.FullName.startswith("Raystrack.Grasshopper.")
                 and t.IsSubclassOf(Grasshopper.Kernel.GH_Component)]
        names = []
        guids = []
        for component_type in types:
            component = System.Activator.CreateInstance(component_type)
            icon = component.Icon_24x24
            assert icon is not None and icon.Width == 24 and icon.Height == 24, component.Name
            assert component.Name.startswith("RS "), component.Name
            assert component.Category == "Raystrack", component.Name
            assert all(str(p.Description).strip() for p in component.Params.Input), component.Name
            assert all(str(p.Description).strip() for p in component.Params.Output), component.Name
            names.append(str(component.Name))
            guids.append(str(component.ComponentGuid))
        assert len(names) == 14, names
        assert len(set(guids)) == len(guids)
        report["components"] = sorted(names)
        report["icons_and_tooltips"] = True
        assembly_info = System.Activator.CreateInstance(assembly.GetType("Raystrack.Grasshopper.AssemblyInfo"))
        report["assembly_version"] = str(assembly_info.Version)
        assert assembly_info.Icon is not None and assembly_info.Icon.Width == 24 and assembly_info.Icon.Height == 24
        report["ribbon_icon"] = True
        report["checks"].append("14 installed components have unique IDs, 24px icons and port descriptions")

    def square(z, upward):
        """Create a unit square with explicit winding for the analytical plate fixture."""
        mesh = Rhino.Geometry.Mesh()
        for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
            mesh.Vertices.Add(x, y, z)
        if upward:
            mesh.Faces.AddFace(0, 1, 2)
            mesh.Faces.AddFace(0, 2, 3)
        else:
            mesh.Faces.AddFace(0, 2, 1)
            mesh.Faces.AddFace(0, 3, 2)
        mesh.Normals.ComputeNormals()
        mesh.Compact()
        return mesh

    validate_catalog()
    node("a", "SurfaceComponent", 40, 70)
    node("b", "InstanceComponent", 40, 250)
    put("a", "Geometry", square(0, True))
    put("a", "ID", "A")
    connect("a", "Surface", "b", "Surface")
    rotation = Rhino.Geometry.Transform.Rotation(System.Math.PI, Rhino.Geometry.Vector3d.XAxis,
                                                Rhino.Geometry.Point3d.Origin)
    transform = Rhino.Geometry.Transform.Translation(0, 1, 1) * rotation
    put("b", "Transform", transform)
    put("b", "ID", "B")
    node("scene", "SceneComponent", 260, 140)
    connect("a", "Surface", "scene", "Surfaces")
    connect("b", "Surface", "scene", "Surfaces")
    node("query", "QueryComponent", 260, 360)
    put("query", "Mode", "pair")
    put("query", "Senders", ["A"])
    put("query", "Receivers", ["B"])
    node("sampling", "SamplingComponent", 40, 520)
    put("sampling", "Density", 64)
    put("sampling", "Rays", 512)
    put("sampling", "Seed", 1)
    node("accuracy", "AccuracyComponent", 260, 600)
    put("accuracy", "Max replicates", 12)
    put("accuracy", "Min replicates", 12)
    put("accuracy", "Tolerance", 0.0)
    node("options", "OptionsComponent", 480, 480)
    connect("sampling", "Sampling", "options", "Sampling")
    connect("accuracy", "Accuracy", "options", "Accuracy")
    put("options", "Batch size", 4096)
    node("solve", "SolveComponent", 690, 140)
    connect("scene", "Scene", "solve", "Scene")
    connect("query", "Query", "solve", "Query")
    connect("options", "Options", "solve", "Options")
    put("solve", "Device", "cpu")
    put("solve", "Acceleration", "instanced")
    put("solve", "Run", True)
    node("table", "ResultTableComponent", 920, 70)
    connect("solve", "Result", "table", "Result")
    node("value", "ResultValueComponent", 920, 300)
    connect("solve", "Result", "value", "Result")
    put("value", "Sender", "A")
    put("value", "Receiver", "B")
    node("inspect", "InspectComponent", 920, 510)
    connect("solve", "Result", "inspect", "Data")
    node("save", "SaveComponent", 1170, 70)
    connect("scene", "Scene", "save", "Scene")
    connect("solve", "Result", "save", "Result")
    store = os.path.join(PROOF, "plates-" + str(int(time.time())) + ".raystrack")
    put("save", "Folder", store)
    node("load", "LoadComponent", 1170, 350)
    put("load", "Folder", store)
    node("runtime", "RuntimeComponent", 1170, 600)
    document.NewSolution(False)
    timer = Timer()
    timer.Interval = 200

    def set_phase(phase):
        """Record the next acceptance stage and reset its deadline."""
        state["phase"] = phase
        state["phase_start"] = time.time()
        report["phase"] = phase

    def check_analytic():
        """Verify instance placement and the facing-square estimate against its exact value."""
        prototype = payload("a", "Surface")
        instance = payload("b", "Surface")
        assert prototype["mesh"] == instance["mesh"], "Instance changed the shared prototype mesh"
        bounds = output("b", "Surface").ClippingBox
        assert bounds.IsValid
        assert abs(bounds.Min.Z - 1) < 1e-6 and abs(bounds.Max.Z - 1) < 1e-6
        assert abs(bounds.Min.X) < 1e-6 and abs(bounds.Max.X - 1) < 1e-6
        assert abs(bounds.Min.Y) < 1e-6 and abs(bounds.Max.Y - 1) < 1e-6
        report["instance_transform"] = True
        report["instance_preview_bounds"] = str(bounds)
        report["checks"].append("Instance shares its prototype mesh and previews the rigidly transformed upper plate")
        value = float(text("value", "Value"))
        expected = 0.1998248957
        assert abs(value - expected) < 0.003, (value, expected)
        result = payload("solve", "Result")
        assert result is not None
        report["pair_value"] = value
        report["analytic_value"] = expected
        report["analytic_absolute_error"] = abs(value - expected)
        report["result_rays"] = int(float(text("solve", "Rays")))
        assert report["result_rays"] > 0
        assert len(list(parameter(nodes["table"], "Values", False).VolatileData.AllData(True))) > 0
        assert text("inspect", "Text")
        report["canvas_image"] = capture("grasshopper-canvas.png")
        report["checks"].append("Facing plates match analytic0.1998248957 within0.003")

    def start_long_run():
        """Rearm a long solve with the same sampling settings for live cancellation checks."""
        put("solve", "Cancel", False)
        put("solve", "Run", False)
        put("accuracy", "Max replicates", 100000)
        put("accuracy", "Min replicates", 100000)
        document.NewSolution(False)
        put("solve", "Run", True)
        document.NewSolution(False)

    def tick(sender, event):
        """Advance acceptance checks while Rhino's UI message loop remains available."""
        try:
            report["heartbeats"] += 1
            phase = state["phase"]
            if phase not in ("cleanup", "reopen"):
                messages = errors()
                if messages:
                    raise RuntimeError("; ".join(messages))
            status = text("solve", "Status")
            report["last_heartbeat"] = {
                "index": report["heartbeats"], "phase": phase, "status": status,
                "rays": text("solve", "Rays"), "progress": text("solve", "Progress"),
                "component_message": str(nodes["solve"].Message),
                "worker_process_id": int(nodes["solve"].WorkerProcessId)}
            if phase in ("solve", "cancel_wait", "remove_wait"):
                report["states"].append(status)
                if status in ("queued", "preparing", "running"):
                    report["active_heartbeats"] += 1
                rays = text("solve", "Rays")
                if rays is not None:
                    count = int(float(rays))
                    if phase in ("solve", "cancel_wait") and count > 0 and count not in report["live_ray_counts"]:
                        report["live_ray_counts"].append(count)
                        if status == "running" and "progress_image" not in report:
                            report["progress_image"] = capture("grasshopper-progress.png")
                progress = text("solve", "Progress")
                if progress is not None:
                    report["last_progress"] = float(progress)
                if status == "failed":
                    raise RuntimeError("RS Solve failed: " + str(text("solve", "Info")))
            if phase == "solve" and status == "succeeded":
                check_analytic()
                put("save", "Run", True)
                document.NewSolution(False)
                set_phase("save")
            elif phase == "save" and text("save", "Status") == "succeeded":
                assert os.path.isfile(os.path.join(store, "manifest.json"))
                put("load", "Run", True)
                document.NewSolution(False)
                set_phase("load")
            elif phase == "load" and text("load", "Status") == "succeeded":
                restored = payload("load", "Result")
                original = payload("solve", "Result")
                assert restored["cumulative_rays"] == original["cumulative_rays"]
                assert restored["values"] == original["values"]
                assert payload("load", "Scene") is not None
                report["checks"].append("Save/Load preserves v2 result values and ray count")
                report["save_load_preserved"] = True
                # Preserve true triggers, then verify they are disarmed on Read().
                archive = GH_Archive()
                archive.AppendObject(document, "Definition")
                saved_path = os.path.join(PROOF, "raystrack-plates-smoke.gh")
                assert archive.WriteToFile(saved_path, True, False)
                report["document"] = saved_path
                start_long_run()
                set_phase("cancel_wait")
            elif phase == "cancel_wait" and status == "running" and int(float(text("solve", "Rays") or "0")) > 0:
                count = int(float(text("solve", "Rays")))
                if count not in state["cancel_progress"]:
                    state["cancel_progress"].append(count)
                if not state["runtime_submitted"]:
                    # Runtime's Future queues behind the solve in Python. Its
                    # outstanding IPC request must not block Status or Cancel.
                    put("runtime", "Refresh", True)
                    document.NewSolution(False)
                    state["runtime_submitted"] = True
                if len(state["cancel_progress"]) >= 3:
                    assert report["active_heartbeats"] >= 2
                    report["checks"].append("UI timer ticks and live ray/progress outputs update during a background solve")
                    put("solve", "Cancel", True)
                    document.NewSolution(False)
                    set_phase("cancelling")
            elif phase == "cancelling" and text("solve", "Status") == "cancelled":
                assert not nodes["solve"].Busy
                report["checks"].append("Cancel stops the active solve and preserves partial output")
                assert payload("solve", "Result") is not None
                report["cancelled_partial"] = True
                report["cancelled_rays"] = int(float(text("solve", "Rays")))
                set_phase("runtime_wait")
            elif phase == "runtime_wait" and text("runtime", "Status") == "ready":
                python = text("runtime", "Python")
                devices = text("runtime", "Devices")
                assert "3.12.10" in python, python
                assert "runtime" in python.lower() and "python.exe" in python.lower(), python
                assert "cpu" in devices.lower(), devices
                report["runtime_python"] = python
                report["runtime_devices"] = json.loads(devices)
                report["runtime_multiplexing"] = True
                report["checks"].append("Queued Runtime check allows live Status/Cancel IPC and completes after cancellation")
                start_long_run()
                set_phase("remove_wait")
            elif phase == "remove_wait" and status in ("preparing", "running") and nodes["solve"].WorkerProcessId > 0:
                state["cleanup_pid"] = int(nodes["solve"].WorkerProcessId)
                report["owned_worker_pid"] = state["cleanup_pid"]
                assert alive(state["cleanup_pid"])
                Grasshopper.Instances.DocumentServer.RemoveDocument(document)
                set_phase("cleanup")
            elif phase == "cleanup" and not alive(state["cleanup_pid"]):
                report["checks"].append("Removing this document terminates its owned worker")
                report["document_worker_stopped"] = True
                reopened = GH_Document()
                archive = GH_Archive()
                assert archive.ReadFromFile(report["document"])
                assert archive.ExtractObject(reopened, "Definition")
                Grasshopper.Instances.DocumentServer.AddDocument(reopened)
                Grasshopper.Instances.ActiveCanvas.Document = reopened
                reopened.NewSolution(False)
                state["reopened"] = reopened
                set_phase("reopen")
            elif phase == "reopen" and time.time() - state["phase_start"] >= 1:
                reopened = state["reopened"]
                for component in reopened.Objects:
                    if component.GetType().Name in ("SolveComponent", "SaveComponent", "LoadComponent"):
                        assert not component.Busy, component.Name
                        assert str(component.Message) == "Set Run false, then true", (component.Name, component.Message)
                report["checks"].append("Saved true Run triggers are disarmed when the definition is reopened")
                report["reopen_disarmed"] = True
                report["reopened_image"] = capture("grasshopper-reopened.png")
                report["basic_example"] = export_basic_example()
                report["phase"] = "passed"
                report["elapsed_seconds"] = time.time() - started
                validate_report(report)
                timer.Stop()
                write_report(report)
                return
            if time.time() - started > 360:
                raise RuntimeError("Host proof exceeded360seconds in phase " + phase)
            if state["phase"] == "cleanup" and time.time() - state["phase_start"] > 15:
                raise RuntimeError("Document worker did not exit after removal")
            write_report(report)
        except Exception:
            timer.Stop()
            report["phase"] = "failed"
            report["error"] = traceback.format_exc()
            try:
                report["failed_image"] = capture("grasshopper-failed.png")
            except Exception:
                pass
            # A failed long-run check must not leave this test's solver working
            # while unrelated user simulations continue in the background.
            try:
                Grasshopper.Instances.DocumentServer.RemoveDocument(document)
                if state["reopened"] is not None:
                    Grasshopper.Instances.DocumentServer.RemoveDocument(state["reopened"])
            except Exception:
                pass
            write_report(report)

    timer.Tick += tick
    timer.Start()
    import scriptcontext
    scriptcontext.sticky["raystrack_host_smoke"] = (timer, tick, document, nodes, report, state)
    write_report(report)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        with open(os.path.join(PROOF, "grasshopper-report.json")) as stream:
            report = validate_report(json.load(stream))
        print("Native host acceptance passed for recorded assembly: " + report["assembly"])
    else:
        try:
            run_host()
        except Exception:
            write_report({"phase": "failed", "rhino_process_id": os.getpid(), "error": traceback.format_exc()})
