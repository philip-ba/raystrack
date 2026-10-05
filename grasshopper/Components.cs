using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Rhino.Geometry;

namespace Raystrack.Grasshopper
{
    /// <summary>Create a stable-ID triangle surface from a Mesh/Brep and an optional native rigid Transform.</summary>
    public sealed class SurfaceComponent : RsComponent
    {
        /// <summary>Create a stable-ID triangle surface from a Mesh/Brep and an optional native rigid Transform.</summary>
        public SurfaceComponent() : base("Surface", "Surface", "Give a mesh or Brep a stable surface ID and optional rigid transform. Mesh Breps once upstream for large scenes.", "01 Scene", "RaystrackLoadMeshes") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17401"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGeometryParameter("Geometry", "G", "Mesh or Brep. Face winding defines the front side. Coordinates use the Rhino document units.", GH_ParamAccess.item);
            p.AddTextParameter("ID", "ID", "Stable unique surface ID, used in queries and results.", GH_ParamAccess.item, "surface");
            p.AddTextParameter("Label", "L", "Optional display label; changing labels does not change geometry.", GH_ParamAccess.item, "");
            p.AddTransformParameter("Transform", "T", "Optional Grasshopper Transform: connect the X/Transform output of Move or Rotate. Leave empty for identity. Rotation/translation only; no scale, shear or mirror. Use original Geometry, not already moved geometry plus the same transform.", GH_ParamAccess.item); p[3].Optional = true;
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddGenericParameter("Surface", "S", "Portable surface snapshot; connect to RS Scene or RS Instance.", GH_ParamAccess.item); }
        /// <summary>Create a stable-ID triangle surface from a Mesh/Brep and an optional native rigid Transform.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            IGH_GeometricGoo input = null;
            if (!da.GetData(0, ref input)) return;
            object geometry = input.ScriptVariable();
            Mesh mesh = geometry as Mesh;
            bool ownMesh = false;
            if (mesh == null)
            {
                var brep = geometry as Brep;
                Require(brep != null, "Input 'Geometry': connect a Mesh or Brep. Convert curves/points to a meshed surface first.");
                mesh = new Mesh(); ownMesh = true;
                var parts = Mesh.CreateFromBrep(brep, MeshingParameters.Default);
                Require(parts != null && parts.Length > 0, "Input 'Geometry': this Brep could not be meshed. Use Grasshopper Mesh Brep upstream and inspect its output.");
                foreach (var part in parts) { mesh.Append(part); part.Dispose(); }
            }
            try
            {
                string id = Text(da, 1, "surface");
                Require(!string.IsNullOrWhiteSpace(id), "ID must be nonempty and unique in the scene.");
                var transform = TransformInput(da, 3);
                var vertices = new JArray(); var faces = new JArray();
                foreach (var v in mesh.Vertices) vertices.Add(new JArray(v.X, v.Y, v.Z));
                foreach (var f in mesh.Faces)
                {
                    faces.Add(new JArray(f.A, f.B, f.C));
                    if (f.IsQuad) faces.Add(new JArray(f.A, f.C, f.D));
                }
                Require(vertices.Count > 0 && faces.Count > 0, "Input 'Geometry': mesh has no faces. Connect a nonempty surface mesh or Brep.");
                Require(mesh.IsValid, "Input 'Geometry': mesh is invalid. Check vertex coordinates and face indices, or remesh the Brep upstream.");
                da.SetData(0, new SnapshotGoo(new JObject { { "kind", "surface" }, { "id", id }, { "label", Text(da, 2, "") },
                    { "mesh", new JObject { { "vertices", vertices }, { "faces", faces } } }, { "transform", Matrix(transform) } }));
            }
            finally { if (ownMesh) mesh.Dispose(); }
        }
        /// <summary>Reject nonfinite, nonaffine, scaled, sheared or mirrored transforms and explain native Move/Rotate wiring.</summary>
        internal static void ValidateRigid(Transform t)
        {
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
                Require(!double.IsNaN(t[r, c]) && !double.IsInfinity(t[r, c]), "Transform contains invalid numbers. Connect a Move/Rotate Transform (X) output or leave it empty.");
            Require(Math.Abs(t[3, 0]) + Math.Abs(t[3, 1]) + Math.Abs(t[3, 2]) + Math.Abs(t[3, 3] - 1) < 1e-7, "Transform must be affine. Use the Transform (X) output of Move or Rotate.");
            for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++)
            {
                double dot = 0; for (int r = 0; r < 3; r++) dot += t[r, a] * t[r, b];
                Require(Math.Abs(dot - (a == b ? 1 : 0)) < 1e-6, "Transform cannot contain scale or shear. Scale the geometry upstream before RS Surface, then use Move/Rotate for instances.");
            }
            double determinant = t[0, 0] * (t[1, 1] * t[2, 2] - t[1, 2] * t[2, 1]) -
                t[0, 1] * (t[1, 0] * t[2, 2] - t[1, 2] * t[2, 0]) + t[0, 2] * (t[1, 0] * t[2, 1] - t[1, 1] * t[2, 0]);
            Require(Math.Abs(determinant - 1) < 1e-6, "Transform cannot mirror geometry. Mirror the mesh upstream before RS Surface; use RS Sampling Flip to reverse only emitting normals.");
        }
    }

    /// <summary>Share prototype triangle geometry while assigning a distinct ID and composing a new rigid placement.</summary>
    public sealed class InstanceComponent : RsComponent
    {
        /// <summary>Share prototype triangle geometry while assigning a distinct ID and composing a new rigid placement.</summary>
        public InstanceComponent() : base("Instance", "Instance", "Reuse a surface's triangle geometry with a new ID and rigid transform. Geometry is shared in the solver and saved files.", "01 Scene", "RaystrackSync", GH_Exposure.secondary) { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17402"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Surface", "S", "Prototype surface from RS Surface or RS Instance.", GH_ParamAccess.item);
            p.AddTextParameter("ID", "ID", "Unique stable ID for the new instance.", GH_ParamAccess.item, "instance");
            p.AddTransformParameter("Transform", "T", "Grasshopper Transform from Move/Rotate's X output; empty means identity. Applied AFTER the prototype transform: new = T * prototype. Rotation/translation only, in document units.", GH_ParamAccess.item); p[2].Optional = true;
            p.AddTextParameter("Label", "L", "Optional instance display label.", GH_ParamAccess.item, "");
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddGenericParameter("Surface", "S", "Instance snapshot for RS Scene.", GH_ParamAccess.item); }
        /// <summary>Share prototype triangle geometry while assigning a distinct ID and composing a new rigid placement.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            var surface = Data(da, 0, "surface", true);
            string id = Text(da, 1, "instance"); Require(!string.IsNullOrWhiteSpace(id), "ID must be nonempty.");
            var t = TransformInput(da, 2);
            var old = Transform.Identity; var rows = surface["transform"] as JArray;
            if (rows != null) for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) old[r, c] = (double)rows[r][c];
            surface["id"] = id; surface["label"] = Text(da, 3, ""); surface["transform"] = Matrix(t * old);
            da.SetData(0, new SnapshotGoo(surface));
        }
    }

    /// <summary>Collect all surfaces and blockers into a complete scene with unique IDs.</summary>
    public sealed class SceneComponent : RsComponent
    {
        /// <summary>Collect all surfaces and blockers into a complete scene with unique IDs.</summary>
        public SceneComponent() : base("Scene", "Scene", "Collect all emitting, receiving and blocking surfaces. Queries select outputs without removing any occluders.", "01 Scene", "RaystrackLoadMeshes") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17403"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p) { p.AddGenericParameter("Surfaces", "S", "All RS Surface/Instance snapshots, including blockers. IDs must be unique.", GH_ParamAccess.list); }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("Scene", "S", "Reusable complete scene for RS Solve or RS Save.", GH_ParamAccess.item);
            p.AddTextParameter("IDs", "ID", "Surface IDs in scene order.", GH_ParamAccess.list);
        }
        /// <summary>Collect all surfaces and blockers into a complete scene with unique IDs.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            var input = new List<object>(); if (!da.GetDataList(0, input)) return;
            var surfaces = new JArray(); var ids = new HashSet<string>();
            foreach (var item in input)
            {
                var surface = ObjectData(item, "surface"); string id = (string)surface["id"];
                Require(ids.Add(id), "Duplicate surface ID '" + id + "'. Give each surface or instance a distinct ID.");
                surfaces.Add(surface);
            }
            Require(surfaces.Count > 0, "Surfaces must contain at least one surface.");
            da.SetData(0, new SnapshotGoo(new JObject { { "kind", "scene" }, { "surfaces", surfaces } }));
            da.SetDataList(1, surfaces.Select(s => (string)s["id"]));
        }
    }

    /// <summary>Construct matrix, row, pair or sky selections without removing scene occluders.</summary>
    public sealed class QueryComponent : RsComponent
    {
        /// <summary>Construct matrix, row, pair or sky selections without removing scene occluders.</summary>
        public QueryComponent() : base("Query", "Query", "Select matrix, row, pair or sky outputs through the same solver. Scene occluders are always retained.", "02 Solve", "RaystrackComputeVFMatrix") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17404"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddTextParameter("Mode", "M", "matrix, row, pair or sky.", GH_ParamAccess.item, "matrix");
            p.AddTextParameter("Senders", "S", "Sender IDs; empty means all. row/pair require one ID.", GH_ParamAccess.list); p[1].Optional = true;
            p.AddTextParameter("Receivers", "R", "Receiver IDs; empty means all. pair requires one ID. Does not remove blockers.", GH_ParamAccess.list); p[2].Optional = true;
            p.AddTextParameter("Sky", "Sky", "none, merged or tregenza145. Sky mode defaults to merged when Mode is sky.", GH_ParamAccess.item, "none");
            p.AddTextParameter("Sides", "Side", "both, front or back. Receiver sides are explicit channels.", GH_ParamAccess.item, "both");
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddGenericParameter("Query", "Q", "Query snapshot for RS Solve.", GH_ParamAccess.item); }
        /// <summary>Construct matrix, row, pair or sky selections without removing scene occluders.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            string mode = Text(da, 0, "matrix").Trim().ToLowerInvariant(), sky = Text(da, 3, "none").Trim().ToLowerInvariant(), sides = Text(da, 4, "both").Trim().ToLowerInvariant();
            Require(new[] { "matrix", "row", "pair", "sky" }.Contains(mode), "Mode must be matrix, row, pair or sky.");
            Require(new[] { "none", "merged", "tregenza145" }.Contains(sky), "Sky must be none, merged or tregenza145.");
            Require(new[] { "both", "front", "back" }.Contains(sides), "Sides must be both, front or back.");
            var senders = new List<string>(); var receivers = new List<string>(); da.GetDataList(1, senders); da.GetDataList(2, receivers);
            Require(senders.Distinct().Count() == senders.Count && receivers.Distinct().Count() == receivers.Count, "Sender and receiver IDs must be unique.");
            Require(senders.All(s => !string.IsNullOrWhiteSpace(s)) && receivers.All(s => !string.IsNullOrWhiteSpace(s)), "Sender and receiver IDs must be nonempty.");
            if (mode == "row" || mode == "pair") Require(senders.Count == 1, "row/pair requires one sender ID.");
            if (mode == "pair") Require(receivers.Count == 1, "pair requires one receiver ID.");
            if (mode == "sky" && sky == "none") sky = "merged";
            da.SetData(0, new SnapshotGoo(new JObject { { "kind", "query" }, { "senders", senders.Count == 0 ? null : new JArray(senders) },
                { "receivers", receivers.Count == 0 ? null : new JArray(receivers) }, { "scene", mode != "sky" },
                { "sky_mode", sky == "none" ? null : sky }, { "receiver_sides", sides == "both" ? new JArray("front", "back") : new JArray(sides) } }));
        }
    }

    /// <summary>Construct immutable cosine/area-pair sampling controls for RS Options.</summary>
    public sealed class SamplingComponent : RsComponent
    {
        /// <summary>Construct immutable cosine/area-pair sampling controls for RS Options.</summary>
        public SamplingComponent() : base("Sampling", "Sampling", "Control the validated cosine-ray or CPU area-pair estimator. Separate sampling from convergence and execution settings.", "02 Solve", "RaystrackMatrixParams", GH_Exposure.secondary) { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17405"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddIntegerParameter("Density", "D", "Emitter sampling density; positive integer. Default 16.", GH_ParamAccess.item, 16);
            p.AddIntegerParameter("Rays", "R", "Cosine rays per emitter cell; positive integer. Default 128.", GH_ParamAccess.item, 128);
            p.AddIntegerParameter("Seed", "Seed", "Nonnegative deterministic sample seed.", GH_ParamAccess.item, 1);
            p.AddTextParameter("Mode", "M", "fair or adaptive ray allocation across senders.", GH_ParamAccess.item, "fair");
            p.AddTextParameter("Strategy", "S", "cosine (CPU/GPU) or area_pair (CPU pair queries only).", GH_ParamAccess.item, "cosine");
            p.AddIntegerParameter("Pair samples", "P", "Area-pair samples per replicate. Default 8192.", GH_ParamAccess.item, 8192);
            p.AddBooleanParameter("Flip", "F", "Reverse emitting normals without changing receiver-side labels.", GH_ParamAccess.item, false);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddGenericParameter("Sampling", "S", "Sampling snapshot for RS Options.", GH_ParamAccess.item); }
        /// <summary>Construct immutable cosine/area-pair sampling controls for RS Options.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            int density = Integer(da, 0, 16), rays = Integer(da, 1, 128), seed = Integer(da, 2, 1), pairs = Integer(da, 5, 8192);
            string mode = Text(da, 3, "fair").Trim().ToLowerInvariant(), strategy = Text(da, 4, "cosine").Trim().ToLowerInvariant();
            Require(density > 0, "Input 'Density': use an integer greater than 0 (default 16).");
            Require(rays > 0, "Input 'Rays': use an integer greater than 0 (default 128).");
            Require(pairs > 0, "Input 'Pair samples': use an integer greater than 0 (default 8192).");
            Require(seed >= 0, "Input 'Seed': use a nonnegative integer (default 1).");
            Require(mode == "fair" || mode == "adaptive", "Mode must be fair or adaptive."); Require(strategy == "cosine" || strategy == "area_pair", "Strategy must be cosine or area_pair.");
            da.SetData(0, new SnapshotGoo(new JObject { { "kind", "sampling" }, { "density", density }, { "rays_per_cell", rays }, { "seed", seed },
                { "mode", mode }, { "strategy", strategy }, { "pair_samples", pairs }, { "flip_faces", Boolean(da, 6) }, { "sequence", "shifted_halton" } }));
        }
    }

    /// <summary>Construct convergence tolerances and replicate limits for RS Options.</summary>
    public sealed class AccuracyComponent : RsComponent
    {
        /// <summary>Construct convergence tolerances and replicate limits for RS Options.</summary>
        public AccuracyComponent() : base("Accuracy", "Accuracy", "Set replicate limits and numerical convergence. Unknown uncertainty in a partial replicate remains unknown.", "02 Solve", "RaystrackSkyParams", GH_Exposure.secondary) { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17406"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddIntegerParameter("Max replicates", "Max", "Maximum completed replicates per sender. Default 100.", GH_ParamAccess.item, 100);
            p.AddIntegerParameter("Min replicates", "Min", "Minimum completed replicates before checking convergence. Default 5.", GH_ParamAccess.item, 5);
            p.AddNumberParameter("Tolerance", "Tol", "Nonnegative tolerance. Zero forces the replicate limit.", GH_ParamAccess.item, 0.0001);
            p.AddTextParameter("Mode", "M", "stderr or delta convergence test.", GH_ParamAccess.item, "stderr");
            p.AddIntegerParameter("Min rays", "R", "Nonnegative minimum cumulative rays before convergence.", GH_ParamAccess.item, 0);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddGenericParameter("Accuracy", "A", "Accuracy snapshot for RS Options.", GH_ParamAccess.item); }
        /// <summary>Construct convergence tolerances and replicate limits for RS Options.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            int max = Integer(da, 0, 100), min = Integer(da, 1, 5), rays = Integer(da, 4, 0); double tolerance = Number(da, 2, 0.0001); string mode = Text(da, 3, "stderr").Trim().ToLowerInvariant();
            Require(max > 0, "Input 'Max replicates': use an integer greater than 0 (default 100).");
            Require(min > 0, "Input 'Min replicates': use an integer greater than 0 (default 5).");
            Require(rays >= 0, "Input 'Min rays': use a nonnegative integer (default 0).");
            Require(tolerance >= 0 && !double.IsNaN(tolerance) && !double.IsInfinity(tolerance), "Input 'Tolerance': use a finite nonnegative number (default 0.0001); 0 forces the replicate limit.");
            Require(mode == "stderr" || mode == "delta", "Mode must be stderr or delta.");
            da.SetData(0, new SnapshotGoo(new JObject { { "kind", "accuracy" }, { "max_replicates", max }, { "min_replicates", min },
                { "tolerance", tolerance }, { "mode", mode }, { "min_rays", rays }, { "check_interval", 1 } }));
        }
    }

    /// <summary>Combine optional sampling/accuracy with bounded batches and explicit final reciprocity.</summary>
    public sealed class OptionsComponent : RsComponent
    {
        /// <summary>Combine optional sampling/accuracy with bounded batches and explicit final reciprocity.</summary>
        public OptionsComponent() : base("Options", "Options", "Combine sampling, convergence, bounded execution chunks and explicit reciprocity. Empty inputs use Python defaults.", "02 Solve", "RaystrackMatrixParams", GH_Exposure.secondary) { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17407"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Sampling", "S", "Optional RS Sampling; empty uses density 16 and 128 rays per cell.", GH_ParamAccess.item); p[0].Optional = true;
            p.AddGenericParameter("Accuracy", "A", "Optional RS Accuracy; empty uses 100 maximum replicates and tolerance 0.0001.", GH_ParamAccess.item); p[1].Optional = true;
            p.AddIntegerParameter("Batch size", "B", "Maximum rays in a kernel chunk. Smaller chunks improve cancellation responsiveness; default 65536.", GH_ParamAccess.item, 65536);
            p.AddTextParameter("Reciprocity", "R", "none, shortcut, bidirectional or rowsum. Applied only to completed full-scene solves. rowsum requires a closed scene.", GH_ParamAccess.item, "none");
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddGenericParameter("Options", "O", "Unified options snapshot for any RS Solve query.", GH_ParamAccess.item); }
        /// <summary>Combine optional sampling/accuracy with bounded batches and explicit final reciprocity.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            var sampling = Data(da, 0, "sampling", false) ?? new JObject(); sampling.Remove("kind");
            var accuracy = Data(da, 1, "accuracy", false) ?? new JObject(); accuracy.Remove("kind");
            int batch = Integer(da, 2, 65536); string mode = Text(da, 3, "none").Trim().ToLowerInvariant();
            Require(batch > 0, "Batch size must be positive."); Require(new[] { "none", "shortcut", "bidirectional", "rowsum" }.Contains(mode), "Reciprocity must be none, shortcut, bidirectional or rowsum.");
            da.SetData(0, new SnapshotGoo(new JObject { { "kind", "options" }, { "sampling", sampling }, { "accuracy", accuracy },
                { "batch_size", batch }, { "postprocessing", new JObject { { "reciprocity", mode } } } }));
        }
    }

    /// <summary>Execute and resume numerical work in the document worker while publishing progress and partial snapshots.</summary>
    public sealed class SolveComponent : AsyncComponent
    {
        /// <summary>Detached launch arguments captured at the Run edge and submitted when ready.</summary>
        private JObject queuedLaunch;
        /// <summary>Current worker launch identity for correlated progress and cancellation.</summary>
        private string launch;
        /// <summary>Previous Cancel value used to detect a deliberate cancellation pulse.</summary>
        private bool previousCancel;
        /// <summary>Cancellation pulse awaiting an independent background protocol request.</summary>
        private bool cancelRequested;
        /// <summary>Latest actionable failure shown in status and runtime error messages.</summary>
        private string fault;
        /// <summary>Execute and resume numerical work in the document worker while publishing progress and partial snapshots.</summary>
        public SolveComponent() : base("Solve", "Solve", "Run in a separate Python process while Grasshopper stays interactive. Progress and partial results refresh automatically. Run is a rising trigger; no solve launches just by opening a saved graph.", "02 Solve", "RaystrackComputeVFMatrix") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17408"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Scene", "S", "Complete scene, including blockers. Inputs are captured when Run rises.", GH_ParamAccess.item);
            p.AddGenericParameter("Query", "Q", "Optional RS Query; empty selects a scene matrix.", GH_ParamAccess.item); p[1].Optional = true;
            p.AddGenericParameter("Options", "O", "Optional RS Options; empty uses shared Python defaults.", GH_ParamAccess.item); p[2].Optional = true;
            p.AddTextParameter("Device", "D", "auto, cpu, cuda, gpu, taichi, vulkan or metal (Metal requires a supported platform). Use RS Runtime to inspect local availability. Explicit unavailable devices report an error.", GH_ParamAccess.item, "auto");
            p.AddTextParameter("Acceleration", "A", "instanced (reuse moving geometry) or flat.", GH_ParamAccess.item, "instanced");
            p.AddIntegerParameter("Ray budget", "B", "Additional rays for this launch; 0 runs until convergence/replicate limit. A paused run resumes on the next Run pulse with unchanged inputs.", GH_ParamAccess.item, 0);
            p.AddBooleanParameter("Run", "Run", "Button or false-to-true transition to launch/resume. Changing inputs alone never launches. Saved true values are disarmed on reopen.", GH_ParamAccess.item, false);
            p.AddBooleanParameter("Cancel", "X", "False-to-true transition requests cancellation and keeps the partial result. Checked between kernel chunks; initial JIT compilation may finish first.", GH_ParamAccess.item, false);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("Result", "R", "Latest immutable partial/final snapshot. Connect to RS Result Table/Value, Inspect or Save.", GH_ParamAccess.item);
            p.AddTextParameter("Status", "S", "idle, queued, preparing, running, paused, succeeded, cancelled or failed. Succeeded may still have reached the replicate limit; inspect convergence in Result.", GH_ParamAccess.item);
            p.AddNumberParameter("Progress", "%", "Progress percentage toward the ray budget or replicate ceiling. Early convergence ends at 100%.", GH_ParamAccess.item);
            p.AddNumberParameter("Rays", "N", "Cumulative rays in this run, including earlier resumed launches.", GH_ParamAccess.item);
            p.AddTextParameter("Info", "I", "Current stage, convergence details or actionable worker error.", GH_ParamAccess.item);
        }
        /// <summary>Execute and resume numerical work in the document worker while publishing progress and partial snapshots.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            try { if (Collect()) fault = null; } catch (Exception ex) { fault = ErrorMessage(ex); response = null; launch = null; }
            bool run = Boolean(da, 6), cancel = Boolean(da, 7);
            if (cancel && !previousCancel) cancelRequested = true;
            previousCancel = cancel;
            if (Rising(run))
            {
                int budget = Integer(da, 5, 0); Require(budget >= 0, "Ray budget must be nonnegative.");
                string device = Text(da, 3, "auto").Trim().ToLowerInvariant(), acceleration = Text(da, 4, "instanced").Trim().ToLowerInvariant();
                Require(new[] { "auto", "cpu", "gpu", "cuda", "taichi", "vulkan", "metal" }.Contains(device), "Input 'Device': choose auto, cpu, gpu, cuda, taichi, vulkan or metal. RS Runtime lists available devices.");
                Require(acceleration == "flat" || acceleration == "instanced", "Acceleration must be flat or instanced.");
                queuedLaunch = new JObject { { "key", InstanceGuid.ToString() }, { "launch", Guid.NewGuid().ToString() },
                    { "scene", Data(da, 0, "scene", true) }, { "query", Data(da, 1, "query", false) ?? new JObject { { "kind", "query" } } },
                    { "options", Data(da, 2, "options", false) ?? new JObject { { "kind", "options" } } },
                    { "device", device }, { "acceleration", acceleration }, { "ray_budget", budget } };
                fault = null;
            }
            if (pending == null && queuedLaunch != null)
            {
                var args = queuedLaunch; queuedLaunch = null; launch = (string)args["launch"]; response = null;
                Begin("solve", args);
            }
            else if (pending == null && cancelRequested && launch != null)
            {
                cancelRequested = false; Begin("cancel", new JObject { { "key", InstanceGuid.ToString() }, { "launch", launch } });
            }
            else if (pending == null && Active(response)) Begin("status", new JObject { { "key", InstanceGuid.ToString() }, { "launch", launch } });
            string status = fault != null ? "failed" : response != null ? (string)response["status"] : pending != null ? "queued" : "idle";
            string info = fault ?? (response == null ? (armed ? "Press Run to solve." : "Set Run false, then true") : (string)response["message"] ?? status);
            double progress = response == null ? 0 : (double?)response["progress"] ?? 0;
            double rays = response == null ? 0 : (double?)response["cumulative_rays"] ?? 0;
            if (response != null && response["result"] is JObject) da.SetData(0, new SnapshotGoo((JObject)response["result"]));
            if (fault != null) AddRuntimeMessage(GH_RuntimeMessageLevel.Error, fault);
            if (response != null && response["error"] != null && response["error"].Type != JTokenType.Null) AddRuntimeMessage(GH_RuntimeMessageLevel.Error, response["error"].ToString());
            da.SetData(1, status); da.SetData(2, progress); da.SetData(3, rays); da.SetData(4, info);
            Message = !armed && status == "idle" ? "Set Run false, then true" : status == "running" ? progress.ToString("0.0") + "% | " + rays.ToString("N0") + " rays" : status;
            if (pending != null || Active(response) || queuedLaunch != null) PollAgain();
        }
        /// <summary>Whether a worker reply represents queued, preparing or running work that needs further polling.</summary>
        private static bool Active(JObject value) { return value != null && new[] { "queued", "preparing", "running" }.Contains((string)value["status"]); }
        /// <summary>Publish a component-scoped error and an explicit failed state where async status outputs are available.</summary>
        protected override void ReportError(IGH_DataAccess da, string message)
        {
            base.ReportError(da, message);
            if (pending == null && !Active(response)) { fault = message; response = null; }
            queuedLaunch = null;
            da.SetData(1, "failed"); da.SetData(2, 0.0); da.SetData(3, response == null ? 0.0 : (double?)response["cumulative_rays"] ?? 0.0); da.SetData(4, message);
            if (pending != null || Active(response)) PollAgain();
        }
        /// <summary>Release this solver's cached worker state when its component is removed.</summary>
        public override void RemovedFromDocument(GH_Document document) { Workers.Release(document.DocumentID, InstanceGuid.ToString()); base.RemovedFromDocument(document); }
    }

    /// <summary>Read sender rows and structured channels as value/error trees with explicit coverage.</summary>
    public sealed class ResultTableComponent : RsComponent
    {
        /// <summary>Read sender rows and structured channels as value/error trees with explicit coverage.</summary>
        public ResultTableComponent() : base("Result Table", "Table", "Read one row per sender and structured receiver channels. NaN means uncomputed/unknown; sampled zeros remain zero.", "03 Results", "RaystrackGetTable") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17409"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p) { p.AddGenericParameter("Result", "R", "Partial/final result from RS Solve or RS Load.", GH_ParamAccess.item); }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddTextParameter("Senders", "S", "Sender IDs in row order.", GH_ParamAccess.list);
            p.AddGenericParameter("Channels", "C", "Structured surface-side, sky-patch, escape and unrequested-hit channels.", GH_ParamAccess.list);
            p.AddTextParameter("Labels", "L", "Human-readable channel headings, in column order.", GH_ParamAccess.list);
            p.AddNumberParameter("Values", "V", "Tree with branch {sender index}, items in channel order. NaN means unknown.", GH_ParamAccess.tree);
            p.AddNumberParameter("Errors", "E", "Sampling standard errors; NaN when unavailable, partial or postprocessed.", GH_ParamAccess.tree);
            p.AddIntegerParameter("Coverage", "Cvg", "Per sender: 1 sampled, 0 uncomputed, -1 unknown after legacy import.", GH_ParamAccess.list);
        }
        /// <summary>Read sender rows and structured channels as value/error trees with explicit coverage.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            var result = Data(da, 0, "result", true); var channels = result["channels"].Children<JObject>().ToArray();
            da.SetDataList(0, result["sender_ids"].Values<string>());
            da.SetDataList(1, channels.Select(c => { var channel = (JObject)c.DeepClone(); channel["channel_kind"] = channel["kind"]; channel["kind"] = "channel"; return new SnapshotGoo(channel); }));
            da.SetDataList(2, channels.Select(SnapshotGoo.ChannelLabel));
            da.SetDataTree(3, Tree((JArray)result["values"])); da.SetDataTree(4, Tree((JArray)result["errors"])); da.SetDataList(5, result["coverage"].Values<int>());
        }
        /// <summary>Convert sender rows to GH branches, preserving unavailable entries as NaN and sampled zeros as zero.</summary>
        private static GH_Structure<GH_Number> Tree(JArray rows)
        {
            var tree = new GH_Structure<GH_Number>();
            for (int r = 0; r < rows.Count; r++)
            {
                var path = new GH_Path(r); tree.EnsurePath(path);
                foreach (var value in rows[r]) tree.Append(new GH_Number(value.Type == JTokenType.Null ? double.NaN : (double)value), path);
            }
            return tree;
        }
    }

    /// <summary>Read one estimate/error by sender ID and structured receiver-side or sky-patch selection.</summary>
    public sealed class ResultValueComponent : RsComponent
    {
        /// <summary>Read one estimate/error by sender ID and structured receiver-side or sky-patch selection.</summary>
        public ResultValueComponent() : base("Result Value", "Value", "Read a single structured channel without suffix parsing. Unknown entries and standard errors are NaN.", "03 Results", "RaystrackGetVF") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17410"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Result", "R", "Result from RS Solve or RS Load.", GH_ParamAccess.item);
            p.AddTextParameter("Sender", "S", "Sender surface ID.", GH_ParamAccess.item);
            p.AddTextParameter("Receiver", "Rcv", "Receiver ID for surface channels; unused for sky, rest and unrequested.", GH_ParamAccess.item, "");
            p.AddTextParameter("Kind", "K", "surface, sky, rest or unrequested.", GH_ParamAccess.item, "surface");
            p.AddTextParameter("Side", "Side", "front or back for surface channels.", GH_ParamAccess.item, "front");
            p.AddIntegerParameter("Patch", "P", "Sky patch 0..144; -1 selects merged sky.", GH_ParamAccess.item, -1);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddNumberParameter("Value", "V", "View factor, sampled zero, or NaN for unknown.", GH_ParamAccess.item);
            p.AddNumberParameter("Error", "E", "Sampling standard error, or NaN when unavailable.", GH_ParamAccess.item);
            p.AddIntegerParameter("Coverage", "C", "1 sampled, 0 uncomputed, -1 unknown.", GH_ParamAccess.item);
        }
        /// <summary>Read one estimate/error by sender ID and structured receiver-side or sky-patch selection.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            var result = Data(da, 0, "result", true); string sender = Text(da, 1, ""), receiver = Text(da, 2, ""), kind = Text(da, 3, "surface").Trim().ToLowerInvariant(), side = Text(da, 4, "front").Trim().ToLowerInvariant(); int patch = Integer(da, 5, -1);
            Require(new[] { "surface", "sky", "rest", "unrequested" }.Contains(kind), "Input 'Kind': choose surface, sky, rest or unrequested.");
            Require(kind != "surface" || side == "front" || side == "back", "Input 'Side': choose front or back. Use RS Result Table for imported channels whose side is unknown.");
            Require(kind != "surface" || !string.IsNullOrWhiteSpace(receiver), "Input 'Receiver': connect the receiving surface ID from RS Scene (not its Label).");
            Require(kind != "sky" || patch >= -1 && patch < 145, "Input 'Patch': use -1 for merged sky or 0 through 144 for Tregenza patches.");
            var senders = result["sender_ids"].Values<string>().ToList(); int r = senders.IndexOf(sender); Require(r >= 0, "Input 'Sender': '" + sender + "' is not in this result. Use the IDs from RS Result Table's Senders output.");
            var channels = result["channels"].Children<JObject>().ToArray(); int c = Array.FindIndex(channels, channel =>
                (string)channel["kind"] == kind && (kind != "surface" || (string)channel["surface_id"] == receiver && (string)channel["side"] == side) &&
                (kind != "sky" || ((int?)channel["patch"] ?? -1) == patch));
            Require(c >= 0, "Requested channel is not in this result; inspect RS Result Table's channel labels.");
            var value = result["values"][r][c]; var error = result["errors"][r][c];
            da.SetData(0, value.Type == JTokenType.Null ? double.NaN : (double)value); da.SetData(1, error.Type == JTokenType.Null ? double.NaN : (double)error); da.SetData(2, (int)result["coverage"][r]);
        }
    }

    /// <summary>Show a concise data summary, usage description and optionally complete JSON in a Panel.</summary>
    public sealed class InspectComponent : RsComponent
    {
        /// <summary>Show a concise data summary, usage description and optionally complete JSON in a Panel.</summary>
        public InspectComponent() : base("Inspect", "Inspect", "Inspect a portable scene, options, query or result snapshot, including execution metadata and convergence provenance.", "03 Results", "RaystrackGetTable", GH_Exposure.secondary) { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17411"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Data", "D", "Any Raystrack snapshot; connect this output directly to a Panel for its compact summary.", GH_ParamAccess.item);
            p.AddBooleanParameter("Details", "J", "False shows a readable summary and usage description. True appends complete JSON, including geometry arrays or result statistics.", GH_ParamAccess.item, false);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p) { p.AddTextParameter("Text", "T", "Readable counts/settings and usage description for a Panel. Set Details true to append full JSON.", GH_ParamAccess.item); }
        /// <summary>Show a concise data summary, usage description and optionally complete JSON in a Panel.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            var value = Data(da, 0, null, true);
            da.SetData(0, SnapshotSummary.Format(value) + "\n" + SnapshotSummary.Description(value) + (Boolean(da, 1) ? "\n\n" + value.ToString(Formatting.Indented) : ""));
        }
    }

    /// <summary>Write a new v2 scene/result folder asynchronously without overwriting existing data.</summary>
    public sealed class SaveComponent : AsyncComponent
    {
        /// <summary>Latest actionable failure shown in status and runtime error messages.</summary>
        private string fault;
        /// <summary>Write a new v2 scene/result folder asynchronously without overwriting existing data.</summary>
        public SaveComponent() : base("Save", "Save", "Write a new version-2 .raystrack folder in the background. Existing folders are never overwritten. A saved true trigger does not write on reopen.", "04 Files", "RaystrackSaveVFMatrix") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17412"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Scene", "S", "Complete scene matching the result's geometry.", GH_ParamAccess.item);
            p.AddGenericParameter("Result", "R", "Optional partial/final result to save with geometry.", GH_ParamAccess.item); p[1].Optional = true;
            p.AddTextParameter("Folder", "F", "New .raystrack folder path. Existing folders cannot be overwritten.", GH_ParamAccess.item);
            p.AddBooleanParameter("Run", "Run", "False-to-true transition writes once.", GH_ParamAccess.item, false);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddTextParameter("Path", "P", "Saved folder path.", GH_ParamAccess.item);
            p.AddTextParameter("Status", "S", "idle, working, succeeded or failed.", GH_ParamAccess.item);
        }
        /// <summary>Write a new v2 scene/result folder asynchronously without overwriting existing data.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            try { if (Collect()) fault = null; } catch (Exception ex) { fault = ErrorMessage(ex); response = null; }
            if (Rising(Boolean(da, 3)))
            {
                Require(pending == null, "Wait for the current file request to finish."); fault = null; response = null;
                string path = Text(da, 2, ""); Require(!string.IsNullOrWhiteSpace(path), "Input 'Folder': choose a new .raystrack folder path, for example C:\\Results\\case.raystrack.");
                Begin("save", new JObject { { "scene", Data(da, 0, "scene", true) }, { "result", Data(da, 1, "result", false) }, { "path", path } });
            }
            string status = fault != null ? "failed" : pending != null ? "working" : response != null ? "succeeded" : "idle";
            if (fault != null) AddRuntimeMessage(GH_RuntimeMessageLevel.Error, fault);
            if (response != null) da.SetData(0, (string)response["path"]);
            da.SetData(1, status); Message = !armed && status == "idle" ? "Set Run false, then true" : status;
        }
        /// <summary>Publish a component-scoped error and an explicit failed state where async status outputs are available.</summary>
        protected override void ReportError(IGH_DataAccess da, string message) { base.ReportError(da, message); fault = message; response = null; da.SetData(1, "failed"); }
    }

    /// <summary>Load v2 folders or isolated v1 imports asynchronously while preserving unknown statistics.</summary>
    public sealed class LoadComponent : AsyncComponent
    {
        /// <summary>Latest actionable failure shown in status and runtime error messages.</summary>
        private string fault;
        /// <summary>Load v2 folders or isolated v1 imports asynchronously while preserving unknown statistics.</summary>
        public LoadComponent() : base("Load", "Load", "Read a .raystrack folder in the background. Legacy v1 files import through the isolated adapter; unavailable coverage/errors remain unknown.", "04 Files", "RaystrackLoadVFMatrix") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17413"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddTextParameter("Folder", "F", "Versioned .raystrack folder or legacy v1 JSON path.", GH_ParamAccess.item);
            p.AddBooleanParameter("Run", "Run", "False-to-true transition reads the file once.", GH_ParamAccess.item, false);
        }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("Scene", "S", "Imported scene snapshot. Incomplete legacy geometry cannot be saved as v2 until supplied.", GH_ParamAccess.item);
            p.AddGenericParameter("Result", "R", "Imported result, with unknown statistics preserved.", GH_ParamAccess.item);
            p.AddTextParameter("Status", "S", "idle, working, succeeded or failed.", GH_ParamAccess.item);
        }
        /// <summary>Load v2 folders or isolated v1 imports asynchronously while preserving unknown statistics.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            try { if (Collect()) fault = null; } catch (Exception ex) { fault = ErrorMessage(ex); response = null; }
            if (Rising(Boolean(da, 1)))
            {
                Require(pending == null, "Wait for the current file request to finish."); fault = null; response = null;
                string path = Text(da, 0, ""); Require(!string.IsNullOrWhiteSpace(path), "Input 'Folder': connect the .raystrack folder or legacy JSON path to load.");
                Begin("load", new JObject { { "path", path } });
            }
            string status = fault != null ? "failed" : pending != null ? "working" : response != null ? "succeeded" : "idle";
            if (fault != null) AddRuntimeMessage(GH_RuntimeMessageLevel.Error, fault);
            if (response != null)
            {
                if (response["scene"] is JObject) da.SetData(0, new SnapshotGoo((JObject)response["scene"]));
                if (response["result"] is JObject) da.SetData(1, new SnapshotGoo((JObject)response["result"]));
            }
            da.SetData(2, status); Message = !armed && status == "idle" ? "Set Run false, then true" : status;
        }
        /// <summary>Publish a component-scoped error and an explicit failed state where async status outputs are available.</summary>
        protected override void ReportError(IGH_DataAccess da, string message) { base.ReportError(da, message); fault = message; response = null; da.SetData(2, "failed"); }
    }

    /// <summary>Inspect bundled Python and local backend capabilities in the background on Refresh.</summary>
    public sealed class RuntimeComponent : AsyncComponent
    {
        /// <summary>Latest actionable failure shown in status and runtime error messages.</summary>
        private string fault;
        /// <summary>Inspect bundled Python and local backend capabilities in the background on Refresh.</summary>
        public RuntimeComponent() : base("Runtime", "Runtime", "Check the bundled Python worker and CPU/GPU availability without running a simulation. An explicit refresh keeps driver probing off the UI thread.", "00 Setup", "RaystrackInstaller") { }
        /// <summary>Stable component identity, retained across development package updates and saved definitions.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17414"); } }
        /// <summary>Define input types, defaults and port help for this workflow step.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p) { p.AddBooleanParameter("Refresh", "R", "False-to-true transition checks the installed runtime and devices.", GH_ParamAccess.item, false); }
        /// <summary>Define documented snapshot/status/numeric output ports for this workflow step.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddTextParameter("Python", "P", "Bundled Python executable and version.", GH_ParamAccess.item);
            p.AddTextParameter("Devices", "D", "JSON device availability and failure reasons.", GH_ParamAccess.item);
            p.AddTextParameter("Status", "S", "idle, working, ready or failed.", GH_ParamAccess.item);
        }
        /// <summary>Inspect bundled Python and local backend capabilities in the background on Refresh.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            try { if (Collect()) fault = null; } catch (Exception ex) { fault = ErrorMessage(ex); response = null; }
            if (Rising(Boolean(da, 0))) { Require(pending == null, "Wait for the runtime check to finish."); fault = null; response = null; Begin("runtime", new JObject()); }
            string status = fault != null ? "failed" : pending != null ? "working" : response != null ? "ready" : "idle";
            if (fault != null) AddRuntimeMessage(GH_RuntimeMessageLevel.Error, fault);
            if (response != null) { da.SetData(0, (string)response["python"] + "\nPython " + (string)response["python_version"] + " | Raystrack " + (string)response["version"]); da.SetData(1, response["devices"].ToString(Formatting.Indented)); }
            da.SetData(2, status); Message = status;
        }
        /// <summary>Publish a component-scoped error and an explicit failed state where async status outputs are available.</summary>
        protected override void ReportError(IGH_DataAccess da, string message) { base.ReportError(da, message); fault = message; response = null; da.SetData(2, "failed"); }
    }
}
