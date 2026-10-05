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
            { "InstanceComponent", new[] { "prototype", "AFTER", "Transform" } },
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
    public static int Main(string[] args)
    {
        Console.OutputEncoding = new System.Text.UTF8Encoding(false);
        try
        {
            var assembly = Assembly.LoadFrom(args[0]);
            gooType = assembly.GetType("Raystrack.Grasshopper.SnapshotGoo", true);
            Values();
            var report = new JObject { { "phase", "passed" }, { "validation", "managed_assembly_contract" },
                { "host_gui_opened", false }, { "distribution_built", false }, { "installed", false },
                { "summaries", summaries }, { "invalid_cases", invalidCases }, { "icons", Icons(assembly) },
                { "component_help", Help(assembly) }, { "transforms", Transforms(assembly) } };
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


def run_contract(rhino=DEFAULT_RHINO, compiler=DEFAULT_COMPILER):
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
        result = subprocess.run([str(executable), str(assembly)], cwd=folder, timeout=60,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("Managed contract probe failed:\n" + result.stdout + result.stderr)
        report = json.loads(result.stdout)
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
    args = parser.parse_args(argv)
    report = run_contract(args.rhino, args.compiler)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
