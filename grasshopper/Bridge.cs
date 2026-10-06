// Raystrack Grasshopper transport. Physics stays in the bundled Python worker.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GH_IO.Serialization;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Rhino.Geometry;

namespace Raystrack.Grasshopper
{
    /// <summary>Reuse the original embedded PNG artwork at Grasshopper's 24-pixel display size.</summary>
    internal static class Icons
    {
        /// <summary>Shared rendered original icons, loaded once per embedded resource.</summary>
        private static readonly Dictionary<string, Bitmap> cache = new Dictionary<string, Bitmap>();
        /// <summary>Load and cache the original icon using explicit pixel-space high-quality downsampling.</summary>
        internal static Bitmap Get(string name)
        {
            lock (cache)
            {
                Bitmap icon;
                if (cache.TryGetValue(name, out icon)) return icon;
                using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Raystrack.Icons." + name + ".png"))
                {
                    if (stream == null) return null;
                    using (var source = new Bitmap(stream))
                    {
                        // Preserve the original PNG artwork; rasterize once in pixel
                        // units so its 300-DPI metadata cannot affect GH's 24px icon.
                        icon = new Bitmap(24, 24, PixelFormat.Format32bppArgb);
                        icon.SetResolution(96, 96);
                        using (var graphics = Graphics.FromImage(icon))
                        using (var attributes = new ImageAttributes())
                        {
                            graphics.Clear(Color.Transparent);
                            graphics.CompositingMode = CompositingMode.SourceCopy;
                            graphics.CompositingQuality = CompositingQuality.HighQuality;
                            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                            graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                            attributes.SetWrapMode(WrapMode.TileFlipXY);
                            double scale = Math.Min(24.0 / source.Width, 24.0 / source.Height);
                            int width = Math.Max(1, (int)Math.Round(source.Width * scale));
                            int height = Math.Max(1, (int)Math.Round(source.Height * scale));
                            graphics.DrawImage(source, new Rectangle((24 - width) / 2, (24 - height) / 2, width, height),
                                0, 0, source.Width, source.Height, GraphicsUnit.Pixel, attributes);
                        }
                    }
                }
                cache.Add(name, icon);
                return icon;
            }
        }
    }

    /// <summary>Identify the compiled Raystrack library in Grasshopper and Rhino's package manager.</summary>
    public sealed class AssemblyInfo : GH_AssemblyInfo
    {
        /// <summary>Library name displayed by Grasshopper.</summary>
        public override string Name { get { return "Raystrack"; } }
        /// <summary>Original Raystrack artwork rendered once at 24 pixels and 96 DPI.</summary>
        public override Bitmap Icon { get { return Icons.Get("raystrack_icon"); } }
        /// <summary>Explain the library's reusable scenes and background radiative solving.</summary>
        public override string Description { get { return "Radiative view factors with reusable scenes and live background solves."; } }
        /// <summary>Stable assembly identity used by Grasshopper.</summary>
        public override Guid Id { get { return new Guid("d87b74f0-6832-48c2-8d02-337199d4b6dd"); } }
        /// <summary>Package author displayed in library information.</summary>
        public override string AuthorName { get { return "Philip Balizki"; } }
        /// <summary>Source repository for package help and issue reports.</summary>
        public override string AuthorContact { get { return "https://github.com/philip-ba/raystrack"; } }
        /// <summary>Development package version, matching the Yak manifest.</summary>
        public override string Version { get { return "2.0.0"; } }
    }

    /// <summary>Register Raystrack's ribbon category, original icon and compact category name.</summary>
    public sealed class AssemblyPriority : GH_AssemblyPriority
    {
        /// <summary>Install the Raystrack ribbon icon and category abbreviations before component registration.</summary>
        public override GH_LoadingInstruction PriorityLoad()
        {
            Instances.ComponentServer.AddCategoryIcon("Raystrack", Icons.Get("raystrack_icon"));
            Instances.ComponentServer.AddCategoryShortName("Raystrack", "RS");
            Instances.ComponentServer.AddCategorySymbolName("Raystrack", 'R');
            return GH_LoadingInstruction.Proceed;
        }
    }

    // These are portable snapshots, never Python objects or session handles.
    /// <summary>Portable scene/settings/result data with readable summaries and Rhino mesh previews; no Python handles are retained.</summary>
    public sealed class SnapshotGoo : GH_Goo<JObject>, IGH_PreviewData
    {
        /// <summary>Cached transformed viewport geometry for this snapshot.</summary>
        private List<Mesh> meshes;
        /// <summary>Create an empty snapshot placeholder.</summary>
        public SnapshotGoo() { Value = new JObject(); }
        /// <summary>Deep-copy portable JSON without retaining caller-owned mutable fields.</summary>
        public SnapshotGoo(JObject value) { Value = value == null ? new JObject() : (JObject)value.DeepClone(); }
        /// <summary>Whether the snapshot has a supported kind and valid geometry/settings/result field shapes.</summary>
        public override bool IsValid { get { return SnapshotValidation.Error(Value) == null; } }
        /// <summary>Actionable validation reason, or an empty string for valid data.</summary>
        public override string IsValidWhyNot { get { return SnapshotValidation.Error(Value) ?? ""; } }
        /// <summary>Readable type label even when a saved snapshot is incomplete.</summary>
        public override string TypeName { get { return "Raystrack " + (Value == null || Value["kind"] == null || Value["kind"].Type != JTokenType.String ? "data" : (string)Value["kind"]); } }
        /// <summary>Explain this data type and which component accepts its output.</summary>
        public override string TypeDescription { get { return SnapshotSummary.Description(Value); } }
        /// <summary>Deep-copy the snapshot so duplicated definitions never share mutable JSON fields.</summary>
        public override IGH_Goo Duplicate() { return new SnapshotGoo(Value); }
        /// <summary>Return detached JSON to scripting components without exposing the stored snapshot.</summary>
        public override object ScriptVariable() { return Value.DeepClone(); }
        /// <summary>Serialize portable fields as compact JSON for archive storage or the Python worker.</summary>
        public string ToJson() { return Value.ToString(Formatting.None); }
        /// <summary>Render counts, IDs, settings and progress for Panels instead of large arrays or a CLR type name.</summary>
        public override string ToString()
        {
            return SnapshotSummary.Format(Value);
        }
        /// <summary>Render an explicit receiver side, sky patch, escape or unrequested-hit channel.</summary>
        public static string ChannelLabel(JObject channel)
        {
            if (channel == null) return "Unknown channel";
            string kind = (string)channel["kind"];
            // Channel goos use a separate outer kind, so the channel kind is retained.
            if (kind == "channel") kind = (string)channel["channel_kind"];
            if (kind == "surface") return (string)channel["surface_id"] + " / " + ((string)channel["side"] ?? "unknown side");
            if (kind == "sky") return channel["patch"] == null || channel["patch"].Type == JTokenType.Null ? "Sky" : "Sky patch " + channel["patch"];
            return kind == "rest" ? "Escape" : "Unrequested hits";
        }
        /// <summary>Store the portable JSON snapshot in a Grasshopper archive.</summary>
        public override bool Write(GH_IWriter writer) { writer.SetString("RaystrackSnapshot", ToJson()); return true; }
        /// <summary>Read portable saved data with validation; report malformed archive data without throwing.</summary>
        public override bool Read(GH_IReader reader)
        {
            meshes = null;
            try { Value = JObject.Parse(reader.GetString("RaystrackSnapshot")); return IsValid; }
            catch (Exception ex) { Value = new JObject { { "kind", "invalid" }, { "reason", "Saved Raystrack data could not be read: " + ex.Message } }; return false; }
        }
        /// <summary>Accept another snapshot, JSON object or JSON text only when its Raystrack data validates.</summary>
        public override bool CastFrom(object source)
        {
            var goo = source as SnapshotGoo;
            var json = source as JObject;
            if (goo != null) Value = (JObject)goo.Value.DeepClone();
            else if (json != null) Value = (JObject)json.DeepClone();
            else if (source is string) { try { Value = JObject.Parse((string)source); } catch (JsonException) { return false; } }
            else return false;
            meshes = null;
            return IsValid;
        }
        /// <summary>Construct and cache transformed preview meshes; invalid snapshots produce no preview.</summary>
        private IEnumerable<Mesh> PreviewMeshes()
        {
            if (meshes != null) return meshes;
            meshes = new List<Mesh>();
            if (!IsValid) return meshes;
            IEnumerable<JObject> surfaces = new JObject[0];
            if ((string)Value["kind"] == "scene") surfaces = Value["surfaces"].Children<JObject>();
            if ((string)Value["kind"] == "surface") surfaces = new[] { Value };
            foreach (var surface in surfaces)
            {
                meshes.Add(SnapshotGeometry.SurfaceMesh(surface));
            }
            return meshes;
        }
        /// <summary>Bounds of transformed surface previews, used by Grasshopper viewport clipping.</summary>
        public BoundingBox ClippingBox { get { var box = BoundingBox.Empty; foreach (var m in PreviewMeshes()) box.Union(m.GetBoundingBox(false)); return box; } }
        /// <summary>Draw cached surface mesh wires using Grasshopper's requested preview color.</summary>
        public void DrawViewportWires(GH_PreviewWireArgs args) { foreach (var m in PreviewMeshes()) args.Pipeline.DrawMeshWires(m, args.Color); }
        /// <summary>Draw cached shaded surface meshes using Grasshopper's requested material.</summary>
        public void DrawViewportMeshes(GH_PreviewMeshArgs args) { foreach (var m in PreviewMeshes()) args.Pipeline.DrawMeshShaded(m, args.Material); }
    }

    /// <summary>Multiplex JSON requests to one document-owned, isolated Python process without UI-thread waits.</summary>
    internal sealed class DocumentWorker : IDisposable
    {
        /// <summary>The Python process owned by this document worker.</summary>
        private readonly Process process;
        /// <summary>Serialize protocol writes without serializing correlated response waits.</summary>
        private readonly object gate = new object();
        /// <summary>Pending replies indexed by unique request IDs.</summary>
        private readonly Dictionary<string, TaskCompletionSource<JObject>> requests = new Dictionary<string, TaskCompletionSource<JObject>>();
        /// <summary>Bounded recent worker stderr used in failure messages.</summary>
        private readonly Queue<string> errors = new Queue<string>();
        /// <summary>Resolved adjacent bundled interpreter, or an explicitly configured fallback.</summary>
        public readonly string Executable;
        /// <summary>Owned Python process ID, or zero when it is unavailable.</summary>
        public int ProcessId { get { try { return process.Id; } catch { return 0; } } }
        /// <summary>Whether the document's Python process is still running.</summary>
        public bool Alive { get { try { return !process.HasExited; } catch { return false; } } }
        /// <summary>Bounded diagnostic stderr tail for actionable worker exit messages.</summary>
        private string ErrorTail { get { lock (errors) return string.Join("\n", errors); } }
        /// <summary>Multiplex JSON requests to one document-owned, isolated Python process without UI-thread waits.</summary>
        public DocumentWorker()
        {
            string root = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            Executable = Path.Combine(root, "runtime", "python.exe");
            if (!File.Exists(Executable)) Executable = Environment.GetEnvironmentVariable("RAYSTRACK_PYTHON");
            if (string.IsNullOrWhiteSpace(Executable) || !File.Exists(Executable))
                throw new FileNotFoundException("Raystrack needs its runtime folder beside the GHA. Install the full Yak or standalone bundle and restart Rhino.");
            string cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Raystrack", "cache");
            Directory.CreateDirectory(cache);
            process = new Process();
            process.StartInfo = new ProcessStartInfo(Executable, "-I -B -u -m raystrack.integrations.grasshopper.worker")
            {
                UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = root,
                RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
            };
            process.StartInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            process.StartInfo.EnvironmentVariables["NUMBA_CACHE_DIR"] = Path.Combine(cache, "numba");
            process.StartInfo.EnvironmentVariables["TI_OFFLINE_CACHE_FILE_PATH"] = Path.Combine(cache, "taichi");
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("NUMBA_NUM_THREADS")))
                process.StartInfo.EnvironmentVariables["NUMBA_NUM_THREADS"] = "4";
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs args)
            {
                if (args.Data == null) return;
                lock (errors) { errors.Enqueue(args.Data); while (errors.Count > 12) errors.Dequeue(); }
            };
            process.Start();
            process.BeginErrorReadLine();
            Task.Run((Action)ReadResponses);
        }
        /// <summary>Read versioned replies continuously and correlate each response with its request ID.</summary>
        private void ReadResponses()
        {
            try
            {
                string line;
                while ((line = process.StandardOutput.ReadLine()) != null)
                {
                    var response = JObject.Parse(line);
                    if ((int?)response["protocol"] != 2) throw new InvalidOperationException("Raystrack worker protocol mismatch.");
                    string id = (string)response["id"];
                    TaskCompletionSource<JObject> request;
                    lock (requests) { if (id == null || !requests.TryGetValue(id, out request)) continue; requests.Remove(id); }
                    if ((bool?)response["ok"] != true) request.TrySetException(new InvalidOperationException((string)response["error"]["message"]));
                    else request.TrySetResult((JObject)response["result"]);
                }
                FailRequests(new InvalidOperationException("Raystrack worker closed its output. " + ErrorTail));
            }
            catch (Exception ex) { FailRequests(ex); }
        }
        /// <summary>Fail and remove pending requests when the worker output closes or the document is disposed.</summary>
        private void FailRequests(Exception error)
        {
            lock (requests) { foreach (var request in requests.Values) request.TrySetException(error); requests.Clear(); }
        }
        // Only called from Task.Run. No process startup, import, read or wait on the UI thread.
        /// <summary>Write one versioned request and wait on its correlated reply from a background task; other requests remain independent.</summary>
        public JObject Call(string operation, JObject arguments)
        {
            string id = Guid.NewGuid().ToString();
            var completion = new TaskCompletionSource<JObject>();
            lock (requests) requests.Add(id, completion);
            try
            {
                // Serialize writes only. Queued Save/Runtime replies must not block
                // independent progress polling and cancellation in this document.
                lock (gate)
                {
                    if (!Alive) throw new InvalidOperationException("Raystrack Python worker exited. " + ErrorTail);
                    var request = new JObject { { "protocol", 2 }, { "id", id }, { "operation", operation }, { "arguments", arguments } };
                    process.StandardInput.WriteLine(JsonConvert.SerializeObject(request,
                        new JsonSerializerSettings { StringEscapeHandling = StringEscapeHandling.EscapeNonAscii }));
                    process.StandardInput.Flush();
                }
                if (!completion.Task.Wait(TimeSpan.FromSeconds(60))) throw new TimeoutException("Raystrack " + operation + " request did not respond within 60 seconds. Other progress and cancellation requests remain available.");
                return completion.Task.Result;
            }
            finally { lock (requests) requests.Remove(id); }
        }
        /// <summary>Close input, allow worker shutdown, then stop only this owned process and fail pending requests.</summary>
        public void Dispose()
        {
            try
            {
                if (!process.HasExited) { process.StandardInput.Close(); if (!process.WaitForExit(750)) process.Kill(); }
            }
            catch (InvalidOperationException) { }
            finally { FailRequests(new InvalidOperationException("Grasshopper document worker was closed.")); }
        }
    }

    /// <summary>Manage one worker per Grasshopper document and release only resources owned by that document.</summary>
    internal static class Workers
    {
        /// <summary>Live workers indexed by Grasshopper document ID.</summary>
        private static readonly Dictionary<Guid, DocumentWorker> clients = new Dictionary<Guid, DocumentWorker>();
        /// <summary>Document IDs forbidden from starting late background workers.</summary>
        private static readonly HashSet<Guid> closed = new HashSet<Guid>();
        /// <summary>Observe document removal and process shutdown to release only owned workers.</summary>
        static Workers()
        {
            Instances.DocumentServer.DocumentRemoved += delegate(GH_DocumentServer sender, GH_Document document) { Close(document.DocumentID); };
            AppDomain.CurrentDomain.ProcessExit += delegate { lock (clients) foreach (var worker in clients.Values) worker.Dispose(); };
        }
        /// <summary>Permit worker creation for a newly attached or reopened Grasshopper document.</summary>
        public static void Register(Guid id) { lock (clients) closed.Remove(id); }
        /// <summary>Reuse a live document worker or launch its isolated process from a background task.</summary>
        public static DocumentWorker Get(Guid id)
        {
            lock (clients)
            {
                if (closed.Contains(id)) throw new InvalidOperationException("Grasshopper document is closed.");
                DocumentWorker worker;
                if (clients.TryGetValue(id, out worker) && worker.Alive) return worker;
                if (worker != null) worker.Dispose();
                clients[id] = worker = new DocumentWorker();
                return worker;
            }
        }
        /// <summary>Return the worker ID for a known document without launching a process.</summary>
        public static int Pid(Guid id) { lock (clients) { DocumentWorker w; return clients.TryGetValue(id, out w) ? w.ProcessId : 0; } }
        /// <summary>Mark a document closed and dispose its worker outside the UI thread.</summary>
        public static void Close(Guid id)
        {
            DocumentWorker worker;
            lock (clients) { closed.Add(id); if (!clients.TryGetValue(id, out worker)) return; clients.Remove(id); }
            Task.Run((Action)worker.Dispose);
        }
        /// <summary>Queue per-component resource release without affecting another document or component.</summary>
        public static void Release(Guid id, string key)
        {
            DocumentWorker worker;
            lock (clients) if (!clients.TryGetValue(id, out worker)) return;
            Task.Run(delegate { try { worker.Call("release", new JObject { { "key", key } }); } catch { } });
        }
    }

    /// <summary>Common component icons, offline help, typed input validation and actionable runtime errors.</summary>
    public abstract class RsComponent : GH_Component
    {
        /// <summary>Signal that a connected background result has not produced its first snapshot yet.</summary>
        private sealed class PendingDataException : Exception { }
        /// <summary>Embedded original PNG resource name for this component.</summary>
        private readonly string icon;
        /// <summary>Ribbon placement of this component.</summary>
        private readonly GH_Exposure exposure;
        /// <summary>Common component icons, offline help, typed input validation and actionable runtime errors.</summary>
        protected RsComponent(string name, string nickname, string description, string group, string iconName, GH_Exposure exposure = GH_Exposure.primary)
            : base("RS " + name, nickname, description, "Raystrack", group) { icon = iconName; this.exposure = exposure; }
        /// <summary>Original Raystrack artwork rendered once at 24 pixels and 96 DPI.</summary>
        protected override Bitmap Icon { get { return Icons.Get(icon); } }
        /// <summary>Ribbon placement priority for this component within its workflow group.</summary>
        public override GH_Exposure Exposure { get { return exposure; } }
        /// <summary>Provide offline workflow examples, native Transform instructions and parameter conventions in component Help.</summary>
        protected override string HelpDescription { get { return Description + ComponentHelp.For(GetType().Name); } }
        /// <summary>Evaluate the component, wait quietly for pending result inputs and turn failures into readable runtime messages.</summary>
        protected override void SolveInstance(IGH_DataAccess da)
        {
            try { Evaluate(da); }
            catch (PendingDataException) { }
            catch (Exception ex) { ReportError(da, ErrorMessage(ex)); }
        }
        /// <summary>Implement this component's input-to-output behavior without blocking on worker requests.</summary>
        protected abstract void Evaluate(IGH_DataAccess da);
        /// <summary>Read and validate a portable input using its visible port name in any error.</summary>
        protected JObject Data(IGH_DataAccess da, int index, string kind, bool required)
        {
            object value = null;
            if (!da.GetData(index, ref value))
            {
                if (!required) return null;
                // A connected background solver may not have its first snapshot
                // yet. Result readers should wait quietly, then recompute.
                if (kind == "result" || kind == null) throw new PendingDataException();
                throw new ArgumentException("Input '" + Params.Input[index].Name + "': connect an RS " + CultureKind(kind) + " output.");
            }
            try { return ObjectData(value, kind); }
            catch (ArgumentException ex) { throw new ArgumentException("Input '" + Params.Input[index].Name + "': " + ex.Message); }
        }
        /// <summary>Unwrap and deep-copy a snapshot of the expected kind, rejecting malformed or mismatched data.</summary>
        protected static JObject ObjectData(object value, string kind)
        {
            var wrapper = value as GH_ObjectWrapper;
            if (wrapper != null) value = wrapper.Value;
            var goo = value as SnapshotGoo;
            var json = goo != null ? goo.Value : value as JObject;
            if (json == null || (kind != null && (json["kind"] == null || json["kind"].Type != JTokenType.String || (string)json["kind"] != kind))) throw new ArgumentException("Expected an RS " + CultureKind(kind) + " output; received " + (goo != null ? goo.TypeName : value == null ? "empty data" : value.GetType().Name) + ". Connect the matching component output.");
            string error = SnapshotValidation.Error(json);
            if (error != null) throw new ArgumentException(error);
            return (JObject)json.DeepClone();
        }
        /// <summary>Publish a component-scoped error and an explicit failed state where async status outputs are available.</summary>
        protected virtual void ReportError(IGH_DataAccess da, string message)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, Name + ": " + message); Message = "Check inputs";
        }
        /// <summary>Unwrap task failures and explain file-access or malformed-input problems in user terms.</summary>
        protected static string ErrorMessage(Exception error)
        {
            var aggregate = error as AggregateException; if (aggregate != null) error = aggregate.GetBaseException();
            if (error is UnauthorizedAccessException) return "The file or folder is not writable. Choose a folder you can access. " + error.Message;
            if (error is NullReferenceException || error is IndexOutOfRangeException) return "Input data is incomplete or malformed. Reconnect the originating Raystrack output and check the input tooltips.";
            return error.Message;
        }
        /// <summary>Capitalize a snapshot kind for readable component input messages.</summary>
        private static string CultureKind(string kind) { return kind == null ? "Raystrack data" : System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(kind); }
        /// <summary>Read a native Grasshopper Transform, default empty inputs to identity and reject scale/shear/mirror.</summary>
        protected Transform TransformInput(IGH_DataAccess da, int index)
        {
            var value = Transform.Identity;
            bool found = da.GetData(index, ref value);
            Require(found || Params.Input[index].VolatileDataCount == 0,
                "Input 'Transform': connect a Grasshopper Transform (X) output from Move or Rotate, or leave empty for identity. A vector or a list of 16 numbers is not a Transform.");
            SurfaceComponent.ValidateRigid(value);
            return value;
        }
        /// <summary>Read a text parameter with its documented default when no value is available.</summary>
        protected static string Text(IGH_DataAccess da, int index, string fallback) { string value = fallback; da.GetData(index, ref value); return value; }
        /// <summary>Read an integer parameter with its documented default when no value is available.</summary>
        protected static int Integer(IGH_DataAccess da, int index, int fallback) { int value = fallback; da.GetData(index, ref value); return value; }
        /// <summary>Read a numeric parameter with its documented default when no value is available.</summary>
        protected static double Number(IGH_DataAccess da, int index, double fallback) { double value = fallback; da.GetData(index, ref value); return value; }
        /// <summary>Read a Boolean trigger or setting, defaulting to false.</summary>
        protected static bool Boolean(IGH_DataAccess da, int index) { bool value = false; da.GetData(index, ref value); return value; }
        /// <summary>Serialize a Rhino Transform into four row-major rows; it acts on column-vector points.</summary>
        protected static JArray Matrix(Transform t)
        {
            var value = new JArray();
            for (int r = 0; r < 4; r++) { var row = new JArray(); for (int c = 0; c < 4; c++) row.Add(t[r, c]); value.Add(row); }
            return value;
        }
        /// <summary>Reject an invalid input with a repair instruction shown in the component runtime messages.</summary>
        protected static void Require(bool condition, string message) { if (!condition) throw new ArgumentException(message); }
        /// <summary>Register the containing document without starting numerical work or a Python process.</summary>
        public override void AddedToDocument(GH_Document document) { base.AddedToDocument(document); Workers.Register(document.DocumentID); }
    }

    /// <summary>Schedule background RPC requests, collect completed responses and disarm saved triggers on reopen.</summary>
    public abstract class AsyncComponent : RsComponent
    {
        /// <summary>Background RPC task; its incomplete result is never waited on by SolveInstance.</summary>
        protected Task<JObject> pending;
        /// <summary>Latest detached response used for status and output snapshots.</summary>
        protected JObject response;
        /// <summary>Previous trigger value used to require a deliberate false-to-true launch.</summary>
        protected bool previousTrigger;
        /// <summary>Whether this component may launch after the saved-trigger reopen safeguard.</summary>
        protected bool armed = true;
        /// <summary>Owner document whose worker executes this component's requests.</summary>
        protected Guid documentId;
        /// <summary>Whether a background request is incomplete; never waits for completion.</summary>
        public bool Busy { get { return pending != null && !pending.IsCompleted; } }
        /// <summary>Document-owned worker identity for diagnostics without starting a process.</summary>
        public int WorkerProcessId { get { return Workers.Pid(documentId); } }
        /// <summary>Schedule background RPC requests, collect completed responses and disarm saved triggers on reopen.</summary>
        protected AsyncComponent(string name, string nickname, string description, string group, string iconName)
            : base(name, nickname, description, group, iconName) { }
        /// <summary>Detect false-to-true trigger edges; reopening a saved true value requires a new false edge first.</summary>
        protected bool Rising(bool trigger)
        {
            if (!trigger) armed = true;
            bool launch = trigger && !previousTrigger && armed;
            previousTrigger = trigger;
            return launch;
        }
        /// <summary>Capture detached request arguments and start process/import/RPC work in Task.Run.</summary>
        protected void Begin(string operation, JObject args)
        {
            var document = OnPingDocument();
            if (document == null) return;
            documentId = document.DocumentID;
            var id = documentId;
            var copied = (JObject)args.DeepClone();
            pending = Task.Run(delegate { return Workers.Get(id).Call(operation, copied); });
            PollAgain();
        }
        /// <summary>Publish a completed reply or task error without waiting on an incomplete request.</summary>
        protected bool Collect()
        {
            if (pending == null) return false;
            if (!pending.IsCompleted) { PollAgain(); return false; }
            var finished = pending;
            pending = null;
            if (finished.IsFaulted) throw finished.Exception.GetBaseException();
            if (finished.IsCanceled) throw new InvalidOperationException("Raystrack request cancelled.");
            response = finished.Result;
            return true;
        }
        /// <summary>Schedule a lightweight UI solution to refresh progress while the document is still attached.</summary>
        protected void PollAgain()
        {
            var document = OnPingDocument();
            if (document != null) document.ScheduleSolution(200, delegate(GH_Document d) { if (OnPingDocument() == d) ExpireSolution(false); });
        }
        /// <summary>Restore archived component state and disarm all launch/write triggers until they return to false.</summary>
        public override bool Read(GH_IReader reader)
        {
            // Never re-launch a simulation or file write just by opening a saved graph.
            armed = false; previousTrigger = true; response = null; pending = null;
            return base.Read(reader);
        }
    }
}
