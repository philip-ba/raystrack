using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Newtonsoft.Json.Linq;
using Rhino.Geometry;

namespace Raystrack.Grasshopper
{
    /// <summary>Decode stored triangles and apply their saved world placement exactly once.</summary>
    internal static class SnapshotGeometry
    {
        internal static IEnumerable<JObject> Surfaces(JObject value)
        {
            string kind = (string)value["kind"];
            if (kind == "surface") return new[] { value };
            if (kind == "scene") return value["surfaces"].Children<JObject>();
            throw new ArgumentException("Connect an RS Surface or RS Scene to Data, including the Scene output of RS Load.");
        }

        internal static Mesh SurfaceMesh(JObject surface)
        {
            var mesh = new Mesh();
            foreach (var v in surface["mesh"]["vertices"].Children<JArray>()) mesh.Vertices.Add((double)v[0], (double)v[1], (double)v[2]);
            foreach (var f in surface["mesh"]["faces"].Children<JArray>()) mesh.Faces.AddFace((int)f[0], (int)f[1], (int)f[2]);
            var rows = surface["transform"] as JArray;
            if (rows != null)
            {
                var transform = Transform.Identity;
                for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) transform[r, c] = (double)rows[r][c];
                mesh.Transform(transform);
            }
            mesh.Normals.ComputeNormals();
            return mesh;
        }

        internal static List<Brep> ToBreps(JObject value)
        {
            var output = new List<Brep>();
            try
            {
                foreach (var surface in Surfaces(value))
                {
                    using (var mesh = SurfaceMesh(surface))
                    {
                        var brep = Brep.CreateFromMesh(mesh, false);
                        if (brep == null || !brep.IsValid)
                        {
                            if (brep != null) brep.Dispose();
                            throw new ArgumentException("Surface '" + surface["id"] + "' could not be converted to a valid Brep. Check its triangle mesh.");
                        }
                        output.Add(brep);
                    }
                }
                return output;
            }
            catch { foreach (var brep in output) brep.Dispose(); throw; }
        }
    }

    /// <summary>Angular bounds matching the solver's zero-based, staggered Tregenza bins.</summary>
    internal sealed class SkyPatch
    {
        internal int Index;
        internal double Lower, Upper, Start, End;
        internal Vector3d Direction
        {
            get
            {
                // A full azimuth cap's label belongs at the zenith.
                double elevation = End - Start > 6 ? Math.PI / 2 : (Lower + Upper) / 2;
                double azimuth = (Start + End) / 2;
                return new Vector3d(Math.Cos(elevation) * Math.Cos(azimuth), Math.Cos(elevation) * Math.Sin(azimuth), Math.Sin(elevation));
            }
        }
        internal string Label { get { return Index < 0 ? "Sky" : "Sky patch " + Index; } }
    }

    /// <summary>Exact spherical Breps with aligned labels; display placement never rotates solver bins.</summary>
    internal static class SkyGeometry
    {
        internal static SkyPatch[] Patches(string mode)
        {
            if (mode == "merged") return new[] { new SkyPatch { Index = -1, Lower = 0, Upper = Math.PI / 2, Start = 0, End = 2 * Math.PI } };
            if (mode != "tregenza145") throw new ArgumentException("Sky Mode must be merged or tregenza145.");
            int[] counts = { 30, 30, 24, 24, 18, 12, 6, 1 };
            var output = new List<SkyPatch>();
            for (int ring = 0; ring < counts.Length; ring++)
            {
                double width = 2 * Math.PI / counts[ring];
                double offset = ring % 2 == 1 && counts[ring] != 1 ? width / 2 : 0;
                for (int sector = 0; sector < counts[ring]; sector++)
                    output.Add(new SkyPatch { Index = output.Count, Lower = ring * 12 * Math.PI / 180,
                        Upper = Math.Min(90, (ring + 1) * 12) * Math.PI / 180,
                        Start = offset + sector * width, End = offset + (sector + 1) * width });
            }
            return output.ToArray();
        }

        internal static Brep ToBrep(SkyPatch patch, Point3d center, double radius)
        {
            // Revolve a meridian arc. This also handles the full zenith cap without a degenerate four-corner surface.
            var arc = new Arc(center + radius * new Vector3d(Math.Cos(patch.Lower), 0, Math.Sin(patch.Lower)),
                center + radius * new Vector3d(Math.Cos((patch.Lower + patch.Upper) / 2), 0, Math.Sin((patch.Lower + patch.Upper) / 2)),
                center + radius * new Vector3d(Math.Cos(patch.Upper), 0, Math.Sin(patch.Upper)));
            using (var curve = arc.ToNurbsCurve())
            using (var surface = RevSurface.Create(curve, new Line(center, center + Vector3d.ZAxis), patch.Start, patch.End))
            {
                if (surface == null) throw new ArgumentException("The sky patch could not be created. Check Center and Radius.");
                var brep = surface.ToBrep();
                if (brep == null || !brep.IsValid) throw new ArgumentException("The sky patch could not be converted to a valid Brep.");
                return brep;
            }
        }

        internal static TextEntity Tag(SkyPatch patch, Point3d center, double radius, double height)
        {
            var direction = patch.Direction;
            var point = center + direction * (radius + height);
            var x = new Vector3d(-direction.Y, direction.X, 0);
            if (!x.Unitize()) x = Vector3d.XAxis;
            var plane = new Plane(point, x, Vector3d.CrossProduct(direction, x));
            return new TextEntity { PlainText = patch.Index < 0 ? "Sky" : patch.Index.ToString(), Plane = plane,
                TextHeight = height, Justification = TextJustification.MiddleCenter };
        }
    }

    /// <summary>Configure sky channels and output a labeled dome as native Rhino geometry.</summary>
    public sealed class SkyComponent : RsComponent
    {
        /// <summary>Build merged or discrete sky settings for RS Query and matching viewport geometry.</summary>
        public SkyComponent() : base("Sky", "Sky", "Configure sky channels and show their exact patch numbering on a dome. Connect Sky to RS Query; patches and tags are native Rhino geometry.", "02 Solve", "RS_Sky") { }
        /// <summary>Stable identity for sky configuration and visualization.</summary>
        public override Guid ComponentGuid { get { return new Guid("a240573a-a62c-4ab8-960f-8a621fd17416"); } }
        /// <summary>Sky mode and display settings live together here.</summary>
        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddTextParameter("Mode", "M", "merged (one hemisphere channel) or tregenza145 (patches 0..144).", GH_ParamAccess.item, "tregenza145");
            p.AddPointParameter("Center", "C", "Dome display center in Rhino document units. Does not change solver directions.", GH_ParamAccess.item, Point3d.Origin);
            p.AddNumberParameter("Radius", "R", "Positive display radius in Rhino document units. Does not change view factors.", GH_ParamAccess.item, 10);
            p.AddNumberParameter("Label size", "H", "Positive label height in document units; 0 uses 1.2% of Radius.", GH_ParamAccess.item, 0);
            p.AddBooleanParameter("Show labels", "L", "Preview and output bakeable patch number tags. Labels and Label points are always returned.", GH_ParamAccess.item, true);
        }
        /// <summary>Aligned geometry and text use the solver's exact channel order.</summary>
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("Sky", "S", "Sky object with a readable Panel representation; connect to RS Query's Sky input.", GH_ParamAccess.item);
            p.AddBrepParameter("Patches", "P", "Native spherical patch Breps in Labels order. 0..144 for Tregenza; one hemisphere for merged. Bake or use directly in Rhino.", GH_ParamAccess.list);
            p.AddTextParameter("Labels", "L", "Exact RS Result Table channel labels in patch order: Sky patch 0..144, or Sky for merged.", GH_ParamAccess.list);
            p.AddPointParameter("Label points", "C", "Patch centers on the dome in matching Patches/Labels order, for custom text tags.", GH_ParamAccess.list);
            p.AddGeometryParameter("Tags", "T", "Native text entities showing each patch number, placed just above its patch. Preview or bake directly. Empty when Show labels is false.", GH_ParamAccess.list);
        }
        /// <summary>Create settings and native geometry without contacting Python or changing query selection.</summary>
        protected override void Evaluate(IGH_DataAccess da)
        {
            string mode = Text(da, 0, "tregenza145").Trim().ToLowerInvariant();
            var center = Point3d.Origin; da.GetData(1, ref center);
            double radius = Number(da, 2, 10), height = Number(da, 3, 0);
            Require(center.IsValid, "Input 'Center': use a point with finite coordinates.");
            Require(!double.IsNaN(radius) && !double.IsInfinity(radius) && radius > 0, "Input 'Radius': use a finite number greater than 0.");
            Require(!double.IsNaN(height) && !double.IsInfinity(height) && height >= 0, "Input 'Label size': use 0 for automatic sizing or a positive finite height.");
            if (height == 0) height = radius * 0.012;
            bool show = Boolean(da, 4);
            var patches = SkyGeometry.Patches(mode);
            var breps = new List<Brep>();
            var tags = new List<GH_TextEntity>();
            try
            {
                foreach (var patch in patches)
                {
                    breps.Add(SkyGeometry.ToBrep(patch, center, radius));
                    if (show) tags.Add(new GH_TextEntity(SkyGeometry.Tag(patch, center, radius, height)));
                }
            }
            catch { foreach (var brep in breps) brep.Dispose(); foreach (var tag in tags) tag.Value.Dispose(); throw; }
            da.SetData(0, new SnapshotGoo(new JObject { { "kind", "sky" }, { "mode", mode },
                { "center", new JArray(center.X, center.Y, center.Z) }, { "radius", radius }, { "label_size", height }, { "show_labels", show } }));
            da.SetDataList(1, breps);
            da.SetDataList(2, patches.Select(p => p.Label));
            da.SetDataList(3, patches.Select(p => center + radius * p.Direction));
            da.SetDataList(4, tags);
        }
    }
}
