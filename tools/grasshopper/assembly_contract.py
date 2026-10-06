"""Test the C# value and icon contracts without opening Rhino or packaging a GHA.

The local Rhino assemblies provide references for a temporary managed test
assembly. Nothing is installed, no distribution is built, and the old native
host acceptance report is not modified. This checks summaries, malformed values
and actual 24px icon pixels; it does not replace an interactive Grasshopper test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RHINO = Path("C:/Program Files/Rhino 8")
DEFAULT_COMPILER = Path(os.environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"

PROBE = r'''
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public static class RaystrackContractProbe
{
    private static Type gooType;
    private static readonly JObject summaries = new JObject();
    private static readonly JArray invalidCases = new JArray();
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
    private static object Goo(string json)
    {
        return Activator.CreateInstance(gooType, new object[] { JObject.Parse(json) });
    }
    private static string Valid(string name, string json, params string[] tokens)
    {
        var value = Goo(json);
        Require((bool)gooType.GetProperty("IsValid").GetValue(value, null), name + " was rejected: " + gooType.GetProperty("IsValidWhyNot").GetValue(value, null));
        string summary = value.ToString();
        Require(!string.IsNullOrWhiteSpace(summary) && !summary.Contains("\n") && !summary.Contains("\r"), name + " needs a concise one-line summary");
        if (!name.EndsWith("channel")) Require(summary.StartsWith("RS "), name + " needs its Raystrack type prefix");
        foreach (var token in tokens) Require(summary.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0, name + " summary omitted '" + token + "': " + summary);
        summaries[name] = summary;
        return summary;
    }
    private static void Invalid(string name, string json)
    {
        var value = Goo(json);
        Require(!(bool)gooType.GetProperty("IsValid").GetValue(value, null), name + " malformed value was marked valid");
        string reason = (string)gooType.GetProperty("IsValidWhyNot").GetValue(value, null);
        string summary = value.ToString();
        Require(!string.IsNullOrWhiteSpace(reason), name + " malformed value has no useful diagnosis");
        Require(summary.IndexOf("invalid", StringComparison.OrdinalIgnoreCase) >= 0, name + " summary does not identify invalid data");
        Require(!summary.Contains("NullReferenceException") && !summary.Contains("InvalidCastException"), name + " leaked an implementation exception");
        invalidCases.Add(new JObject { { "case", name }, { "reason", reason }, { "summary", summary } });
    }
    private static string Surface(string id, string transform)
    {
        return "{\"kind\":\"surface\",\"id\":\"" + id + "\",\"label\":\"Plate " + id + "\",\"mesh\":{\"vertices\":[[0,0,0],[1,0,0],[0,1,0]],\"faces\":[[0,1,2]]},\"transform\":" + transform + "}";
    }
    private static void Values()
    {
        string identity = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]";
        string translated = "[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]]";
        string a = Surface("A", identity), b = Surface("B", translated);
        Valid("surface", a, "A", "3", "1", "identity");
        Valid("scene", "{\"kind\":\"scene\",\"surfaces\":[" + a + "," + b + "]}", "2", "1");
        Valid("query", "{\"kind\":\"query\",\"senders\":[\"A\"],\"receivers\":[\"B\"],\"scene\":true,\"sky_mode\":\"tregenza145\",\"receiver_sides\":[\"front\"]}", "A", "B", "tregenza145", "front");
        Valid("sky", "{\"kind\":\"sky\",\"mode\":\"tregenza145\",\"center\":[1,2,3],\"radius\":20,\"label_size\":0.24,\"show_labels\":true}", "145", "0..144", "20", "0.24", "shown", "+Z");
        Valid("merged_sky", "{\"kind\":\"sky\",\"mode\":\"merged\",\"center\":[0,0,0],\"radius\":10,\"label_size\":0.12,\"show_labels\":false}", "1 hemisphere", "hidden");
        string sampling = "{\"density\":3,\"rays_per_cell\":64,\"seed\":7,\"mode\":\"adaptive\",\"strategy\":\"cosine\",\"pair_samples\":1024,\"flip_faces\":false,\"sequence\":\"shifted_halton\"}";
        string accuracy = "{\"max_replicates\":20,\"min_replicates\":3,\"tolerance\":0.005,\"mode\":\"stderr\",\"check_interval\":1,\"min_rays\":0}";
        Valid("sampling", sampling.Insert(1, "\"kind\":\"sampling\","), "cosine", "3", "64", "7", "adaptive");
        Valid("accuracy", accuracy.Insert(1, "\"kind\":\"accuracy\","), "20", "3", "stderr");
        Valid("options", "{\"kind\":\"options\",\"sampling\":" + sampling + ",\"accuracy\":" + accuracy + ",\"batch_size\":2048,\"postprocessing\":{\"reciprocity\":\"bidirectional\"}}", "2048", "bidirectional", "cosine");
        Valid("default_options", "{\"kind\":\"options\"}", "16", "128", "65536", "none");
        Valid("result", "{\"kind\":\"result\",\"sender_ids\":[\"A\"],\"channels\":[{\"kind\":\"surface\",\"surface_id\":\"B\",\"side\":\"front\",\"patch\":null},{\"kind\":\"rest\",\"surface_id\":null,\"side\":null,\"patch\":null}],\"values\":[[0.2,0.8]],\"errors\":[[null,null]],\"coverage\":[1],\"statistics\":{},\"execution\":{},\"scene_revision\":0,\"rays_used\":128,\"cumulative_rays\":128,\"status\":\"max_iters\",\"converged\":false,\"elapsed_ms\":0,\"provenance\":[]}", "128", "1", "2");
        Valid("surface_channel", "{\"kind\":\"channel\",\"channel_kind\":\"surface\",\"surface_id\":\"B\",\"side\":\"front\",\"patch\":null}", "B", "front");
        Valid("sky_channel", "{\"kind\":\"channel\",\"channel_kind\":\"sky\",\"surface_id\":null,\"side\":null,\"patch\":33}", "sky", "33");
        Valid("merged_sky_channel", "{\"kind\":\"channel\",\"channel_kind\":\"sky\",\"surface_id\":null,\"side\":null,\"patch\":null}", "sky");
        Valid("escape_channel", "{\"kind\":\"channel\",\"channel_kind\":\"rest\",\"surface_id\":null,\"side\":null,\"patch\":null}", "escape");
        Valid("unrequested_channel", "{\"kind\":\"channel\",\"channel_kind\":\"unrequested\",\"surface_id\":null,\"side\":null,\"patch\":null}", "unrequested");
        Invalid("empty", "{}");
        Invalid("unknown_kind", "{\"kind\":\"mystery\"}");
        Invalid("surface_missing_geometry", "{\"kind\":\"surface\",\"id\":\"A\"}");
        Invalid("scene_wrong_shape", "{\"kind\":\"scene\",\"surfaces\":\"wrong\"}");
        Invalid("query_bad_side", "{\"kind\":\"query\",\"receiver_sides\":[\"left\"]}");
        Invalid("sampling_bad_density", "{\"kind\":\"sampling\",\"density\":0}");
        Invalid("accuracy_bad_limit", "{\"kind\":\"accuracy\",\"max_replicates\":0}");
        Invalid("options_bad_batch", "{\"kind\":\"options\",\"batch_size\":0}");
        Invalid("result_missing_grid", "{\"kind\":\"result\",\"sender_ids\":[\"A\"],\"channels\":[],\"coverage\":[1]}");
        Invalid("channel_bad_side", "{\"kind\":\"channel\",\"channel_kind\":\"surface\",\"surface_id\":\"B\",\"side\":\"left\"}");
        Invalid("sky_bad_mode", "{\"kind\":\"sky\",\"mode\":\"random\"}");
        Invalid("sky_bad_radius", "{\"kind\":\"sky\",\"mode\":\"merged\",\"center\":[0,0,0],\"radius\":0,\"label_size\":0.1,\"show_labels\":true}");
        Invalid("sky_bad_center", "{\"kind\":\"sky\",\"mode\":\"merged\",\"center\":[0,0],\"radius\":10,\"label_size\":0.1,\"show_labels\":true}");
    }
    private static JObject Icons(Assembly assembly)
    {
        var icons = new JObject();
        var getter = assembly.GetType("Raystrack.Grasshopper.Icons").GetMethod("Get", BindingFlags.Static | BindingFlags.NonPublic);
        string prefix = "Raystrack.Icons.";
        foreach (var resource in assembly.GetManifestResourceNames().Where(n => n.StartsWith(prefix) && n.EndsWith(".png")))
        {
            string name = resource.Substring(prefix.Length, resource.Length - prefix.Length - 4);
            var image = (Bitmap)getter.Invoke(null, new object[] { name });
            Require(image != null && image.Width == 24 && image.Height == 24, name + " needs a24px icon");
            Require(Math.Abs(image.HorizontalResolution - 96) < 0.1 && Math.Abs(image.VerticalResolution - 96) < 0.1, name + " needs96DPI rendering");
            int ink = 0, light = 0, antialias = 0;
            for (int y = 0; y < image.Height; y++) for (int x = 0; x < image.Width; x++)
            {
                Color pixel = image.GetPixel(x, y);
                // The category glyph has a transparent background, whereas
                // component artwork has a white background. Measure their
                // displayed contrast on white so both original styles count.
                double alpha = pixel.A / 255.0;
                double shade = pixel.R * alpha + 255 * (1 - alpha);
                if (shade < 130) ink++;
                if (shade > 230) light++;
                if (shade > 5 && shade < 250 && Math.Abs(pixel.R - pixel.G) < 3 && Math.Abs(pixel.R - pixel.B) < 3) antialias++;
            }
            Require(ink >= 8 && light >= 50, name + " lost its original dark/light artwork");
            Require(antialias >= 10, name + " needs antialiased resampling");
            icons[name] = new JObject { { "width", image.Width }, { "height", image.Height }, { "dpi", image.HorizontalResolution }, { "ink_pixels", ink }, { "light_pixels", light }, { "antialiased_pixels", antialias } };
        }
        Require(icons.Count >= 14, "Component icon resources are missing");
        return icons;
    }
    private static JObject Help(Assembly assembly)
    {
        var descriptions = new JObject();
        var method = assembly.GetType("Raystrack.Grasshopper.ComponentHelp").GetMethod("For", BindingFlags.Static | BindingFlags.NonPublic);
        var cases = new Dictionary<string, string[]> {
            { "SurfaceComponent", new[] { "Move", "Geometry", "Motion", "X", "ID" } },
            { "ToBrepComponent", new[] { "RS Surface", "RS Scene", "Transform", "triangulated", "IDs" } },
            { "SkyComponent", new[] { "RS Query", "0..144", "Tags", "Radius", "+Z", "144" } },
            { "SceneComponent", new[] { "blockers", "IDs", "occluders" } },
            { "QueryComponent", new[] { "pair", "Senders", "Receivers", "sky" } },
            { "SamplingComponent", new[] { "density", "seed", "area_pair" } },
            { "AccuracyComponent", new[] { "tolerance", "replicates" } },
            { "OptionsComponent", new[] { "defaults", "Batch", "Reciprocity" } },
            { "SolveComponent", new[] { "Run", "Cancel", "ADDITIONAL", "paused" } },
            { "ResultTableComponent", new[] { "Coverage", "NaN", "branch" } },
            { "ResultValueComponent", new[] { "Sender", "Kind=sky", "Patch=-1" } },
            { "InspectComponent", new[] { "Details", "JSON", "summary" } },
            { "SaveComponent", new[] { "folder", "never overwritten" } },
            { "LoadComponent", new[] { "legacy", "unknown" } },
            { "RuntimeComponent", new[] { "Refresh", "runtime" } }
        };
        foreach (var item in cases)
        {
            Require(assembly.GetType("Raystrack.Grasshopper." + item.Key) != null, "Missing component " + item.Key);
            string text = (string)method.Invoke(null, new object[] { item.Key });
            Require(!string.IsNullOrWhiteSpace(text) && text.Length >= 80, item.Key + " needs useful component help");
            foreach (var token in item.Value) Require(text.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0, item.Key + " help omitted " + token);
            descriptions[item.Key] = text;
        }
        return descriptions;
    }
    private static JObject SkyAndDevices(Assembly assembly)
    {
        Require(assembly.GetType("Raystrack.Grasshopper.InstanceComponent") == null, "Removed Instance component is still registered");
        var patchType = assembly.GetType("Raystrack.Grasshopper.SkyPatch");
        var method = assembly.GetType("Raystrack.Grasshopper.SkyGeometry").GetMethod("Patches", BindingFlags.Static | BindingFlags.NonPublic);
        var patches = (Array)method.Invoke(null, new object[] { "tregenza145" });
        Require(patches.Length == 145, "Tregenza needs 145 patches");
        var bounds = new JArray();
        foreach (var patch in patches)
        {
            var row = new JObject();
            foreach (var field in new[] { "Index", "Lower", "Upper", "Start", "End" })
                row[field.ToLowerInvariant()] = JToken.FromObject(patchType.GetField(field, BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch));
            var direction = (Rhino.Geometry.Vector3d)patchType.GetProperty("Direction", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch, null);
            row["direction"] = new JArray(direction.X, direction.Y, direction.Z);
            string label = (string)patchType.GetProperty("Label", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch, null);
            Require(label == "Sky patch " + bounds.Count, "Sky label order is inconsistent");
            row["label"] = label;
            bounds.Add(row);
        }
        Require(((Array)method.Invoke(null, new object[] { "merged" })).Length == 1, "Merged sky needs one hemisphere");
        var deviceMethod = assembly.GetType("Raystrack.Grasshopper.DeviceDescriptions").GetMethod("Format", BindingFlags.Static | BindingFlags.NonPublic);
        var devices = JObject.Parse("{\"cpu\":{\"available\":true},\"cuda\":{\"available\":false,\"name\":\"Example GPU\",\"id\":0,\"simulated\":false,\"reason\":\"Driver unavailable\"},\"taichi\":{\"installed\":true,\"available\":true,\"architectures\":[\"vulkan\"],\"version\":[1,7,4],\"runtime_arch\":null,\"adapter\":null,\"extra_capability\":\"future field\"}}");
        var lines = (string[])deviceMethod.Invoke(null, new object[] { devices });
        Require(lines.Length == 13, "Runtime must describe every backend field");
        string all = string.Join("\n", lines);
        foreach (var token in new[] { "cpu: available", "cuda: unavailable", "Example GPU", "Device index", "simulator", "Driver unavailable", "installed", "vulkan", "1.7.4", "not initialized", "automatic selection", "future field" })
            Require(all.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0, "Device report omitted " + token);
        Require(!all.Contains("{") && !all.Contains("["), "Device report still dumps JSON");
        return new JObject { { "patch_bounds", bounds }, { "device_lines", new JArray(lines) } };
    }
    private static JArray Transforms(Assembly assembly)
    {
        var method = assembly.GetType("Raystrack.Grasshopper.SurfaceComponent").GetMethod("ValidateRigid", BindingFlags.Static | BindingFlags.NonPublic);
        var cases = new List<Tuple<string, Rhino.Geometry.Transform, bool>>();
        var identity = Rhino.Geometry.Transform.Identity;
        cases.Add(Tuple.Create("identity", identity, true));
        var translated = identity; translated[0,3] = 4; translated[1,3] = -2; translated[2,3] = 1;
        cases.Add(Tuple.Create("translated", translated, true));
        var rotated = identity; rotated[0,0] = 0; rotated[0,1] = -1; rotated[1,0] = 1; rotated[1,1] = 0;
        cases.Add(Tuple.Create("rotated", rotated, true));
        var scaled = identity; scaled[0,0] = 2; cases.Add(Tuple.Create("scaled", scaled, false));
        var sheared = identity; sheared[0,1] = 0.2; cases.Add(Tuple.Create("sheared", sheared, false));
        var reflected = identity; reflected[0,0] = -1; cases.Add(Tuple.Create("mirrored", reflected, false));
        var nonAffine = identity; nonAffine[3,0] = 0.1; cases.Add(Tuple.Create("non_affine", nonAffine, false));
        var nonFinite = identity; nonFinite[0,3] = double.NaN; cases.Add(Tuple.Create("nonfinite", nonFinite, false));
        var report = new JArray();
        foreach (var item in cases)
        {
            string error = null;
            try { method.Invoke(null, new object[] { item.Item2 }); }
            catch (TargetInvocationException failure)
            {
                Require(failure.InnerException is ArgumentException, item.Item1 + " transform needs a friendly validation error");
                error = failure.InnerException.Message;
            }
            Require((error == null) == item.Item3, item.Item1 + " transform had the wrong acceptance result: " + error);
            report.Add(new JObject { { "case", item.Item1 }, { "accepted", error == null }, { "reason", error } });
        }
        return report;
    }
    private static JObject NativeGeometry(Assembly assembly)
    {
        gooType = assembly.GetType("Raystrack.Grasshopper.SnapshotGoo");
        var flags = BindingFlags.Static | BindingFlags.NonPublic;
        var geometry = assembly.GetType("Raystrack.Grasshopper.SnapshotGeometry");
        string placed = "[[0,-1,0,4],[1,0,0,-2],[0,0,1,3],[0,0,0,1]]";
        string identity = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]";
        string a = Surface("A", identity), b = Surface("B", placed);
        var convert = geometry.GetMethod("ToBreps", flags);
        var breps = (List<Rhino.Geometry.Brep>)convert.Invoke(null, new object[] { JObject.Parse("{\"kind\":\"scene\",\"surfaces\":[" + a + "," + b + "]}") });
        Require(breps.Count == 2 && breps.All(x => x.IsValid && x.Faces.Count == 1), "Scene To Brep did not preserve surfaces/triangles");
        var bounds = breps[1].GetBoundingBox(true);
        Require(Math.Abs(bounds.Min.X - 3) < 1e-6 && Math.Abs(bounds.Max.X - 4) < 1e-6 && Math.Abs(bounds.Min.Y + 2) < 1e-6 && Math.Abs(bounds.Max.Y + 1) < 1e-6 && Math.Abs(bounds.Min.Z - 3) < 1e-6, "To Brep did not apply world transform exactly once");
        var single = (List<Rhino.Geometry.Brep>)convert.Invoke(null, new object[] { JObject.Parse(b) });
        Require(single.Count == 1 && single[0].IsValid, "Single Surface To Brep failed");
        var sky = assembly.GetType("Raystrack.Grasshopper.SkyGeometry");
        var patchType = assembly.GetType("Raystrack.Grasshopper.SkyPatch");
        var patches = (Array)sky.GetMethod("Patches", flags).Invoke(null, new object[] { "tregenza145" });
        var create = sky.GetMethod("ToBrep", flags);
        var tagMethod = sky.GetMethod("Tag", flags);
        var center = new Rhino.Geometry.Point3d(4, -2, 3);
        double radius = 7, area = 0;
        var interiorSamples = new JArray();
        foreach (var patch in patches)
        {
            using (var brep = (Rhino.Geometry.Brep)create.Invoke(null, new object[] { patch, center, radius }))
            using (var mass = Rhino.Geometry.AreaMassProperties.Compute(brep))
            using (var tag = (Rhino.Geometry.TextEntity)tagMethod.Invoke(null, new object[] { patch, center, radius, 0.1 }))
            {
                Require(brep.IsValid && brep.Faces.Count == 1 && mass != null, "Invalid native sky patch");
                area += mass.Area;
                double lower = (double)patchType.GetField("Lower", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch);
                double upper = (double)patchType.GetField("Upper", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch);
                double start = (double)patchType.GetField("Start", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch);
                double end = (double)patchType.GetField("End", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch);
                double expected = radius * radius * (end - start) * (Math.Sin(upper) - Math.Sin(lower));
                Require(Math.Abs(mass.Area - expected) < 1e-5, "Native patch area differs from solver angular sector");
                var face = brep.Faces[0];
                var point = face.PointAt(face.Domain(0).Mid, face.Domain(1).Mid);
                Require(Math.Abs(point.DistanceTo(center) - radius) < 1e-6 && point.Z >= center.Z, "Patch is not on the upper sphere");
                var samples = new JArray();
                foreach (double u in new[] { 0.05, 0.5, 0.95 }) foreach (double v in new[] { 0.05, 0.5, 0.95 })
                {
                    var radial = face.PointAt(face.Domain(0).ParameterAt(u), face.Domain(1).ParameterAt(v)) - center;
                    radial.Unitize(); samples.Add(new JArray(radial.X, radial.Y, radial.Z));
                }
                interiorSamples.Add(samples);
                Require(tag.PlainText == ((int)patchType.GetField("Index", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(patch)).ToString(), "Native tag has wrong patch number");
            }
        }
        Require(Math.Abs(area - 2 * Math.PI * radius * radius) < 1e-4, "Native patches do not cover a hemisphere");
        var merged = (Array)sky.GetMethod("Patches", flags).Invoke(null, new object[] { "merged" });
        using (var brep = (Rhino.Geometry.Brep)create.Invoke(null, new object[] { merged.GetValue(0), center, radius }))
        using (var mass = Rhino.Geometry.AreaMassProperties.Compute(brep))
            Require(brep.IsValid && Math.Abs(mass.Area - area) < 1e-4, "Merged hemisphere differs from discrete dome");
        foreach (var brep in breps.Concat(single)) brep.Dispose();
        return new JObject { { "surface_and_scene_breps", true }, { "world_transform_once", true }, { "valid_spherical_patches", patches.Length }, { "interior_directions", interiorSamples },
            { "spherical_area", area }, { "expected_area", 2 * Math.PI * radius * radius }, { "native_text_tags", true }, { "merged_hemisphere", true }, { "components", NativeComponents(assembly) } };
    }
    private static Grasshopper.Kernel.GH_Component Node(Assembly assembly, Grasshopper.Kernel.GH_Document document, string name, int x, int y)
    {
        var node = (Grasshopper.Kernel.GH_Component)Activator.CreateInstance(assembly.GetType("Raystrack.Grasshopper." + name + "Component"));
        node.CreateAttributes(); node.Attributes.Pivot = new PointF(x, y); document.AddObject(node, false); return node;
    }
    private static void Set(Grasshopper.Kernel.GH_Component node, int port, object value)
    {
        var flags = BindingFlags.Instance | BindingFlags.Public;
        var parameter = node.Params.Input[port];
        var data = parameter.GetType().GetProperty("PersistentData", flags).GetValue(parameter, null);
        data.GetType().GetMethod("Clear").Invoke(data, null);
        var append = data.GetType().GetMethods().Single(m => m.Name == "Append" && m.GetParameters().Length == 2);
        append.Invoke(data, new object[] { value, new Grasshopper.Kernel.Data.GH_Path(0) });
        parameter.ExpireSolution(false);
    }
    private static object[] Output(Grasshopper.Kernel.GH_Component node, int port)
    {
        return node.Params.Output[port].VolatileData.AllData(true).Cast<object>().ToArray();
    }
    private static JObject NativeComponents(Assembly assembly)
    {
        using (var document = new Grasshopper.Kernel.GH_Document())
        {
            Grasshopper.Kernel.GH_Document.EnableSolutions = true;
            document.Enabled = true;
            var a = Node(assembly, document, "Surface", 40, 60);
            var b = Node(assembly, document, "Surface", 40, 240);
            var mesh = new Rhino.Geometry.Mesh();
            mesh.Vertices.Add(0,0,0); mesh.Vertices.Add(1,0,0); mesh.Vertices.Add(1,1,0); mesh.Vertices.Add(0,1,0);
            mesh.Faces.AddFace(0,1,2); mesh.Faces.AddFace(0,2,3); mesh.Normals.ComputeNormals();
            Set(a, 0, new Grasshopper.Kernel.Types.GH_Mesh(mesh));
            Set(b, 0, new Grasshopper.Kernel.Types.GH_Mesh(mesh.DuplicateMesh()));
            Set(a, 1, new Grasshopper.Kernel.Types.GH_String("A"));
            Set(b, 1, new Grasshopper.Kernel.Types.GH_String("B"));
            var transform = Rhino.Geometry.Transform.Translation(0,1,1) * Rhino.Geometry.Transform.Rotation(Math.PI, Rhino.Geometry.Vector3d.XAxis, Rhino.Geometry.Point3d.Origin);
            Set(b, 3, new Grasshopper.Kernel.Types.GH_Transform(transform));
            var scene = Node(assembly, document, "Scene", 270, 140);
            scene.Params.Input[0].AddSource(a.Params.Output[0]); scene.Params.Input[0].AddSource(b.Params.Output[0]);
            var brep = Node(assembly, document, "ToBrep", 490, 50); brep.Params.Input[0].AddSource(scene.Params.Output[0]);
            var sky = Node(assembly, document, "Sky", 270, 510);
            var query = Node(assembly, document, "Query", 500, 360);
            Set(query, 0, new Grasshopper.Kernel.Types.GH_String("pair"));
            Set(query, 1, new Grasshopper.Kernel.Types.GH_String("A"));
            Set(query, 2, new Grasshopper.Kernel.Types.GH_String("B"));
            query.Params.Input[3].AddSource(sky.Params.Output[0]);
            document.NewSolution(false);
            Require(Output(brep, 0).Length == 2, "Native To Brep component did not output both scene surfaces: " + string.Join("; ", new[] { a, b, scene, brep }.Select(n => n.Name + "=" + Output(n,0).Length + " " + string.Join(", ", n.RuntimeMessages(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error)))));
            Require(((Grasshopper.Kernel.Types.GH_Brep)Output(brep, 0)[1]).Value.GetBoundingBox(true).Min.Z > 0.9999, "Native To Brep component placement failed");
            Require(Output(sky, 1).Length == 145 && Output(sky, 2).Length == 145 && Output(sky, 3).Length == 145 && Output(sky, 4).Length == 145, "Native Sky outputs are not aligned");
            for (int i = 0; i < 145; i++)
            {
                Require(((Grasshopper.Kernel.Types.GH_Brep)Output(sky, 1)[i]).Value.IsValid, "Native Sky component emitted invalid Brep");
                Require(((Grasshopper.Kernel.Types.GH_String)Output(sky, 2)[i]).Value == "Sky patch " + i, "Native Labels order differs from channels");
                Require(Output(sky, 4)[i] is Grasshopper.Kernel.Types.GH_TextEntity, "Tags cannot be previewed/baked as native GH text");
            }
            var skyValue = JObject.Parse((string)gooType.GetMethod("ToJson").Invoke(Output(sky, 0)[0], null));
            Require((string)skyValue["mode"] == "tregenza145", "Native Sky snapshot has wrong mode");
            var queryValue = JObject.Parse((string)gooType.GetMethod("ToJson").Invoke(Output(query, 0)[0], null));
            Require((string)queryValue["sky_mode"] == "tregenza145" && (bool)queryValue["scene"], "Query did not accept Sky alongside pair receivers");
            Set(sky, 1, new Grasshopper.Kernel.Types.GH_Point(new Rhino.Geometry.Point3d(4,-2,3)));
            Set(sky, 2, new Grasshopper.Kernel.Types.GH_Number(7));
            Set(sky, 3, new Grasshopper.Kernel.Types.GH_Number(0.3));
            Set(sky, 0, new Grasshopper.Kernel.Types.GH_String("merged"));
            Set(sky, 4, new Grasshopper.Kernel.Types.GH_Boolean(false));
            Set(query, 0, new Grasshopper.Kernel.Types.GH_String("sky"));
            document.NewSolution(false);
            Require(Output(sky, 1).Length == 1 && Output(sky, 2).Length == 1 && Output(sky, 4).Length == 0, "Merged/hidden labels outputs failed");
            var zenith = ((Grasshopper.Kernel.Types.GH_Point)Output(sky, 3)[0]).Value;
            Require(zenith.DistanceTo(new Rhino.Geometry.Point3d(4,-2,10)) < 1e-6, "Native Center/Radius inputs did not place sky labels correctly");
            queryValue = JObject.Parse((string)gooType.GetMethod("ToJson").Invoke(Output(query, 0)[0], null));
            Require((string)queryValue["sky_mode"] == "merged" && !(bool)queryValue["scene"], "Sky-only Query did not use Sky mode");
            Set(sky, 0, new Grasshopper.Kernel.Types.GH_String("tregenza145"));
            Set(sky, 4, new Grasshopper.Kernel.Types.GH_Boolean(true));
            Set(sky, 1, new Grasshopper.Kernel.Types.GH_Point(Rhino.Geometry.Point3d.Origin));
            Set(sky, 2, new Grasshopper.Kernel.Types.GH_Number(10));
            Set(sky, 3, new Grasshopper.Kernel.Types.GH_Number(0));
            Set(query, 0, new Grasshopper.Kernel.Types.GH_String("pair"));
            var sampling = Node(assembly, document, "Sampling", 40, 490);
            var accuracy = Node(assembly, document, "Accuracy", 40, 750);
            var options = Node(assembly, document, "Options", 500, 700);
            options.Params.Input[0].AddSource(sampling.Params.Output[0]); options.Params.Input[1].AddSource(accuracy.Params.Output[0]);
            var solve = Node(assembly, document, "Solve", 760, 200);
            solve.Params.Input[0].AddSource(scene.Params.Output[0]); solve.Params.Input[1].AddSource(query.Params.Output[0]); solve.Params.Input[2].AddSource(options.Params.Output[0]);
            var table = Node(assembly, document, "ResultTable", 1000, 100); table.Params.Input[0].AddSource(solve.Params.Output[0]);
            var value = Node(assembly, document, "ResultValue", 1000, 400); value.Params.Input[0].AddSource(solve.Params.Output[0]);
            Set(value, 1, new Grasshopper.Kernel.Types.GH_String("A")); Set(value, 2, new Grasshopper.Kernel.Types.GH_String("B"));
            var inspect = Node(assembly, document, "Inspect", 1000, 670); inspect.Params.Input[0].AddSource(solve.Params.Output[0]);
            document.NewSolution(false);
            var archive = new GH_IO.Serialization.GH_Archive(); archive.AppendObject(document, "Definition");
            var archivedSky = new GH_IO.Serialization.GH_Archive(); archivedSky.AppendObject((GH_IO.GH_ISerializable)Output(sky,0)[0], "Sky");
            var restoredSky = Activator.CreateInstance(gooType);
            Require(archivedSky.ExtractObject((GH_IO.GH_ISerializable)restoredSky, "Sky") && restoredSky.ToString() == Output(sky,0)[0].ToString(), "Sky snapshot archive did not round-trip");
            string example = Environment.GetEnvironmentVariable("RAYSTRACK_GH_EXAMPLE_OUTPUT");
            if (!string.IsNullOrEmpty(example))
            {
                Require(archive.WriteToFile(example, true, false), "Could not save portable example");
            }
            return new JObject { { "surface_scene_round_trip", true }, { "query_accepts_sky_object", true }, { "aligned_brep_text_outputs", true }, { "merged_and_hidden_labels", true }, { "example", example } };
        }
    }
    [STAThread]
    public static int Main(string[] args)
    {
        Console.OutputEncoding = new System.Text.UTF8Encoding(false);
        AppDomain.CurrentDomain.AssemblyResolve += delegate(object sender, ResolveEventArgs request) {
            string path = Path.Combine(Environment.GetEnvironmentVariable("RAYSTRACK_RHINO_SYSTEM"), new AssemblyName(request.Name).Name + ".dll");
            return File.Exists(path) ? Assembly.LoadFrom(path) : null;
        };
        try
        {
            var assembly = Assembly.LoadFrom(args[0]);
            gooType = assembly.GetType("Raystrack.Grasshopper.SnapshotGoo", true);
            Values();
            var report = new JObject { { "phase", "passed" }, { "validation", "managed_assembly_contract" },
                { "host_gui_opened", false }, { "distribution_built", false }, { "installed", false },
                { "summaries", summaries }, { "invalid_cases", invalidCases }, { "icons", Icons(assembly) },
                { "component_help", Help(assembly) }, { "transforms", Transforms(assembly) }, { "sky_and_devices", SkyAndDevices(assembly) } };
            Console.WriteLine(report.ToString(Formatting.None));
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.ToString());
            return 1;
        }
    }
}
'''


def compile_managed(compiler, output, sources, references, resources=()):
    """Compile test-only CLR assemblies in a caller-owned temporary directory."""
    options = ["/nologo", "/utf8output", "/target:exe" if output.suffix == ".exe" else "/target:library",
               "/optimize+", '/out:"' + str(output) + '"', "/reference:System.dll",
               "/reference:System.Core.dll", "/reference:System.Drawing.dll",
               "/reference:System.Windows.Forms.dll"]
    options.extend('/reference:"' + str(path) + '"' for path in references)
    options.extend('/resource:"' + str(path) + '",Raystrack.Icons.' + path.name for path in resources)
    options.extend('"' + str(path) + '"' for path in sources)
    response = output.with_suffix(".rsp")
    response.write_text("\n".join(options) + "\n", encoding="utf-8-sig")
    result = subprocess.run([str(compiler), "@" + str(response)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("Managed test compilation failed:\n" + result.stdout + result.stderr)


def native_host_contract(folder, assembly, executable, rhino, compiler):
    """Bootstrap RhinoCommon from its installation, then test with a headless core.

    Rhino's platform services must remain beside the original RhinoCommon;
    initializing a copied reference DLL breaks native startup.
    """
    bootstrap = folder / "native-bootstrap.cs"
    bootstrap.write_text('''using System; using System.IO; using System.Reflection;
class Bootstrap {
    [STAThread] static int Main(string[] args) {
        try {
            var rhino = Assembly.LoadFrom(args[0]);
            using (var core = (IDisposable)Activator.CreateInstance(rhino.GetType("Rhino.Runtime.InProcess.RhinoCore"))) {
                var probe = Assembly.LoadFrom(args[1]);
                var subject = Assembly.LoadFrom(args[2]);
                var method = probe.GetType("RaystrackContractProbe").GetMethod("NativeGeometry", BindingFlags.Static | BindingFlags.NonPublic);
                Console.WriteLine(method.Invoke(null, new object[] { subject }).ToString());
            }
            return 0;
        } catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
''', encoding="utf-8")
    runner = folder / "native-bootstrap.exe"
    compile_managed(compiler, runner, [bootstrap], [])
    env = os.environ.copy()
    env["PATH"] = str(rhino / "System") + os.pathsep + env.get("PATH", "")
    result = subprocess.run([str(runner), str(rhino / "System" / "RhinoCommon.dll"), str(executable), str(assembly)],
                            cwd=folder, env=env, timeout=90, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("Native Rhino geometry check failed:\n" + result.stdout + result.stderr)
    return json.loads(result.stdout)


def run_contract(rhino=DEFAULT_RHINO, compiler=DEFAULT_COMPILER, *, native=False):
    """Run actual C# contracts without installing a plugin or launching a GUI."""
    references = [rhino / "System" / "RhinoCommon.dll", rhino / "System" / "Newtonsoft.Json.dll",
                  rhino / "Plug-ins" / "Grasshopper" / "Grasshopper.dll",
                  rhino / "Plug-ins" / "Grasshopper" / "GH_IO.dll"]
    for path in [compiler] + references:
        if not path.is_file():
            raise FileNotFoundError("Managed contract prerequisite is missing: " + str(path))
    sources = sorted((ROOT / "grasshopper").glob("*.cs"))
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    with tempfile.TemporaryDirectory(prefix="raystrack-gh-contract-") as temporary:
        folder = Path(temporary)
        for path in references:
            shutil.copy2(path, folder / path.name)
        assembly = folder / "Raystrack.ContractSubject.dll"
        compile_managed(compiler, assembly, sources, references,
                        sorted((ROOT / "grasshopper" / "icons").glob("*.png")))
        probe = folder / "probe.cs"
        probe.write_text(PROBE, encoding="utf-8")
        executable = folder / "probe.exe"
        compile_managed(compiler, executable, [probe], references)
        env = os.environ.copy()
        env["RAYSTRACK_RHINO_SYSTEM"] = str(rhino / "System")
        env["PATH"] = str(rhino / "System") + os.pathsep + env.get("PATH", "")
        command = [str(executable), str(assembly)]
        result = subprocess.run(command, cwd=folder, timeout=60, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("Managed contract probe failed:\n" + result.stdout + result.stderr)
        report = json.loads(result.stdout)
        if native:
            report["native_geometry"] = native_host_contract(folder, assembly, executable, rhino, compiler)
            report["validation"] = "managed_and_native_geometry_contract"
    after = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if source_hashes != after:
        raise RuntimeError("Product sources changed during the managed contract test; rerun on a stable source snapshot")
    report["source_sha256"] = source_hashes
    return report


def main(argv=None):
    """Print the independent report or save it to an explicitly selected path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhino", type=Path, default=DEFAULT_RHINO)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--native", action="store_true", help="Also check Breps and text with a headless Rhino geometry core")
    args = parser.parse_args(argv)
    report = run_contract(args.rhino, args.compiler, native=args.native)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
