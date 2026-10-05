using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Raystrack.Grasshopper
{
    /// <summary>Validate portable input shapes before components index their fields.</summary>
    internal static class SnapshotValidation
    {
        /// <summary>Return an actionable first error, or null for a supported snapshot.</summary>
        internal static string Error(JObject value)
        {
            try
            {
                if (value == null) return "Connect a Raystrack component output.";
                string kind = (string)value["kind"];
                switch (kind)
                {
                    case "surface": return Surface(value);
                    case "scene":
                        var surfaces = value["surfaces"] as JArray;
                        if (surfaces == null) return "Scene Surfaces must be a list of RS Surface outputs.";
                        var ids = new HashSet<string>();
                        foreach (var entry in surfaces)
                        {
                            var surface = entry as JObject;
                            if (surface == null) return "Each Scene item must be an RS Surface or Instance.";
                            string error = Surface(surface);
                            if (error != null) return error;
                            if (!ids.Add((string)surface["id"])) return "Duplicate surface ID '" + surface["id"] + "'; give every instance a unique ID.";
                        }
                        return null;
                    case "query":
                        string selection = IDs(value["senders"], "Senders") ?? IDs(value["receivers"], "Receivers");
                        if (selection != null) return selection;
                        if (Present(value["scene"]) && value["scene"].Type != JTokenType.Boolean) return "Query Scene must be true or false.";
                        string sky = Choice(value, "sky_mode", new[] { "merged", "tregenza145" });
                        if (sky != null) return sky;
                        if ((bool?)value["scene"] == false && !Present(value["sky_mode"])) return "A Query must request scene or sky outputs.";
                        var sides = value["receiver_sides"];
                        if (Present(sides) && (!(sides is JArray) || !sides.Any() || sides.Values<string>().Distinct().Count() != sides.Count() || sides.Values<string>().Any(s => s != "front" && s != "back")))
                            return "Receiver sides must select front and/or back once each.";
                        return null;
                    case "sampling": return Sampling(value);
                    case "accuracy": return Accuracy(value);
                    case "options":
                        if (Present(value["sampling"]) && !(value["sampling"] is JObject)) return "Options Sampling must come from RS Sampling.";
                        if (Present(value["accuracy"]) && !(value["accuracy"] is JObject)) return "Options Accuracy must come from RS Accuracy.";
                        if (Present(value["postprocessing"]) && !(value["postprocessing"] is JObject)) return "Options Postprocessing must be a settings object.";
                        return Sampling(value["sampling"] as JObject ?? new JObject()) ?? Accuracy(value["accuracy"] as JObject ?? new JObject()) ??
                            Integer(value, "batch_size", 1) ?? Choice(value["postprocessing"] as JObject ?? new JObject(), "reciprocity", new[] { "none", "shortcut", "bidirectional", "rowsum" });
                    case "result": return Result(value);
                    case "channel": return Channel(value, "channel_kind");
                    default: return (string)value["reason"] ?? "Expected an RS Surface, Scene, Query, Sampling, Accuracy, Options, Result or Channel output.";
                }
            }
            catch (Exception) { return "Raystrack data has an invalid field type; reconnect the originating component output."; }
        }

        /// <summary>Check geometry array shapes, index bounds and the optional rigid matrix.</summary>
        private static string Surface(JObject value)
        {
            if (value["id"] == null || value["id"].Type != JTokenType.String || string.IsNullOrWhiteSpace((string)value["id"])) return "Surface ID must be nonempty text.";
            string prefix = "Surface '" + value["id"] + "': ";
            var mesh = value["mesh"] as JObject;
            if (mesh == null) return prefix + "missing mesh; connect RS Surface or RS Instance.";
            var vertices = mesh["vertices"] as JArray;
            var faces = mesh["faces"] as JArray;
            if (vertices == null || faces == null) return prefix + "mesh needs vertex and triangle lists.";
            foreach (var vertex in vertices)
                if (!(vertex is JArray) || vertex.Count() != 3 || vertex.Any(v => !Finite(v))) return prefix + "each vertex needs three finite coordinates (X, Y, Z).";
            foreach (var face in faces)
                if (!(face is JArray) || face.Count() != 3 || face.Any(v => v.Type != JTokenType.Integer || (long)v < 0 || (long)v >= vertices.Count)) return prefix + "each triangle needs three valid zero-based vertex indices.";
            if (Present(value["transform"]))
            {
                var rows = value["transform"] as JArray;
                if (rows == null || rows.Count != 4 || rows.Any(r => !(r is JArray) || r.Count() != 4 || r.Any(v => !Finite(v))))
                    return prefix + "Transform must be a finite 4 by 4 matrix; in Grasshopper connect a Move/Rotate Transform (X) output.";
                var transform = Rhino.Geometry.Transform.Identity;
                for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) transform[r, c] = (double)rows[r][c];
                try { SurfaceComponent.ValidateRigid(transform); } catch (ArgumentException ex) { return prefix + ex.Message; }
            }
            return null;
        }

        /// <summary>Validate optional IDs without interpreting display labels or side suffixes.</summary>
        private static string IDs(JToken value, string name)
        {
            if (!Present(value)) return null;
            var ids = value as JArray;
            if (ids == null || ids.Any(id => id.Type != JTokenType.String || string.IsNullOrWhiteSpace((string)id))) return name + " must be a list of nonempty surface IDs from RS Scene's IDs output.";
            if (ids.Values<string>().Distinct().Count() != ids.Count) return name + " contains duplicate surface IDs.";
            return null;
        }

        /// <summary>Check scalar sampling settings while retaining Python defaults for omitted fields.</summary>
        private static string Sampling(JObject value)
        {
            string error = Integer(value, "density", 1) ?? Integer(value, "rays_per_cell", 1) ?? Integer(value, "pair_samples", 1) ?? Integer(value, "seed", 0) ??
                Choice(value, "mode", new[] { "fair", "adaptive" }) ?? Choice(value, "strategy", new[] { "cosine", "area_pair" }) ?? Choice(value, "sequence", new[] { "shifted_halton", "random" });
            if (error != null) return "Sampling: " + error;
            if (Present(value["flip_faces"]) && value["flip_faces"].Type != JTokenType.Boolean) return "Sampling Flip must be true or false.";
            if (((string)value["strategy"] ?? "cosine") == "cosine" && Present(value["sequence"]) && (string)value["sequence"] != "shifted_halton") return "Cosine sampling requires shifted_halton sequence.";
            return null;
        }

        /// <summary>Check finite tolerances and replicate limits without changing convergence rules.</summary>
        private static string Accuracy(JObject value)
        {
            string error = Integer(value, "max_replicates", 1) ?? Integer(value, "min_replicates", 1) ?? Integer(value, "check_interval", 1) ?? Integer(value, "min_rays", 0) ?? Choice(value, "mode", new[] { "stderr", "delta" });
            if (error != null) return "Accuracy: " + error;
            if (Present(value["tolerance"]) && (!Finite(value["tolerance"]) || (double)value["tolerance"] < 0)) return "Accuracy Tolerance must be finite and nonnegative; use 0 to force the replicate limit.";
            return null;
        }

        /// <summary>Check row/channel dimensions and distinguish unknown values from sampled zeros.</summary>
        private static string Result(JObject value)
        {
            var senders = value["sender_ids"] as JArray;
            var channels = value["channels"] as JArray;
            var estimates = value["values"] as JArray;
            var errors = value["errors"] as JArray;
            var coverage = value["coverage"] as JArray;
            if (senders == null || channels == null || estimates == null || errors == null || coverage == null) return "Result needs senders, channels, values, errors and coverage; reconnect RS Solve or RS Load.";
            string idError = IDs(senders, "Result Senders"); if (idError != null) return idError;
            if (estimates.Count != senders.Count || errors.Count != senders.Count || coverage.Count != senders.Count) return "Result values, errors and coverage must match its sender row count.";
            foreach (var item in channels)
            {
                var channel = item as JObject;
                if (channel == null) return "Each Result channel must be a structured channel object.";
                string channelError = Channel(channel, "kind"); if (channelError != null) return channelError;
            }
            for (int r = 0; r < senders.Count; r++)
            {
                var row = estimates[r] as JArray; var errorRow = errors[r] as JArray;
                if (row == null || errorRow == null || row.Count != channels.Count || errorRow.Count != channels.Count) return "Result row " + r + " must have one value and error per channel.";
                if (coverage[r].Type != JTokenType.Integer || !new[] { -1L, 0L, 1L }.Contains((long)coverage[r])) return "Result Coverage must be -1 (unknown), 0 (uncomputed) or 1 (sampled).";
                for (int c = 0; c < channels.Count; c++)
                {
                    if (Present(row[c]) && (!Finite(row[c]) || (double)row[c] < 0)) return "Result estimates must be nonnegative finite numbers or null for unknown.";
                    if (Present(errorRow[c]) && (!Finite(errorRow[c]) || (double)errorRow[c] < 0)) return "Result errors must be nonnegative finite numbers or null for unknown.";
                    if (!Present(row[c]) && (Present(errorRow[c]) || (long)coverage[r] == 1)) return "An unknown Result estimate cannot have a known error or sampled coverage.";
                }
            }
            return null;
        }

        /// <summary>Validate a surface side, sky patch, escape or unrequested-hit channel.</summary>
        private static string Channel(JObject value, string kindField)
        {
            string kind = (string)value[kindField];
            if (!new[] { "surface", "sky", "rest", "unrequested" }.Contains(kind)) return "Channel kind must be surface, sky, rest or unrequested.";
            if (kind == "surface")
            {
                if (value["surface_id"] == null || value["surface_id"].Type != JTokenType.String || string.IsNullOrWhiteSpace((string)value["surface_id"])) return "Surface channel needs a receiver surface ID.";
                if (Present(value["side"]) && (string)value["side"] != "front" && (string)value["side"] != "back") return "Surface channel Side must be front, back or unknown.";
                if (Present(value["patch"])) return "Only sky channels can have a Patch.";
            }
            else if (Present(value["surface_id"]) || Present(value["side"])) return "Only surface channels can have a receiver ID and Side.";
            if (Present(value["patch"]) && (kind != "sky" || value["patch"].Type != JTokenType.Integer || (long)value["patch"] < 0 || (long)value["patch"] >= 145)) return "Sky Patch must be a zero-based index from 0 through 144, or null for merged sky.";
            return null;
        }

        /// <summary>Check an optional integer against a minimum without accepting numeric text.</summary>
        private static string Integer(JObject value, string field, long minimum)
        {
            return !Present(value[field]) || value[field].Type == JTokenType.Integer && (long)value[field] >= minimum ? null : field + " must be an integer >= " + minimum + ".";
        }

        /// <summary>Validate an optional text choice and list the supported values in the error.</summary>
        private static string Choice(JObject value, string field, string[] allowed)
        {
            return !Present(value[field]) || value[field].Type == JTokenType.String && allowed.Contains((string)value[field]) ? null : field + " must be " + string.Join(", ", allowed) + ".";
        }

        /// <summary>Identify a known JSON field rather than a missing or explicit null value.</summary>
        private static bool Present(JToken value) { return value != null && value.Type != JTokenType.Null; }

        /// <summary>Accept finite JSON numeric values and reject strings and booleans.</summary>
        private static bool Finite(JToken value)
        {
            return Present(value) && (value.Type == JTokenType.Integer || value.Type == JTokenType.Float) && !double.IsNaN((double)value) && !double.IsInfinity((double)value);
        }
    }

    /// <summary>Human-readable, bounded representations for Grasshopper panels and tooltips.</summary>
    internal static class SnapshotSummary
    {
        /// <summary>Explain the object type and how its output is used in a definition.</summary>
        internal static string Description(JObject value)
        {
            string kind = value == null || value["kind"] == null || value["kind"].Type != JTokenType.String ? "" : (string)value["kind"];
            switch (kind)
            {
                case "surface": return "Triangle geometry with a stable ID, label and rigid transform. Front follows face winding. Connect to RS Scene or RS Instance.";
                case "scene": return "Complete set of surfaces and blockers. Query selection keeps every scene occluder. Connect to RS Solve or RS Save.";
                case "query": return "Sender IDs, receiver sides and scene/sky channels requested from a complete Scene. Connect to RS Solve.";
                case "sampling": return "Estimator, ray density, seed and emitting-side settings. Connect to RS Options.";
                case "accuracy": return "Convergence test, tolerance and replicate limits. Connect to RS Options.";
                case "options": return "Unified sampling, accuracy, batch and reciprocity controls. Omitted values use Python defaults. Connect to RS Solve.";
                case "result": return "Immutable partial or final estimates with explicit sampled coverage, errors and ray counts. Connect to RS Result Table/Value, Inspect or Save.";
                case "channel": return "Structured receiver surface/side, sky patch, escape or unrequested-hit identifier in Result Table column order.";
                default: return "Portable Raystrack data. Connect a Raystrack component output.";
            }
        }

        /// <summary>Show useful counts/settings, handling incomplete imported data without throwing.</summary>
        internal static string Format(JObject value)
        {
            string kind = value == null || value["kind"] == null || value["kind"].Type != JTokenType.String ? "data" : (string)value["kind"];
            string title = "RS " + CultureInfo.InvariantCulture.TextInfo.ToTitleCase(kind);
            try
            {
                string error = SnapshotValidation.Error(value);
                if (error != null) return title + ": invalid data (" + error + ")";
                switch (kind)
                {
                    case "surface":
                        var mesh = (JObject)value["mesh"];
                        string label = (string)value["label"];
                        return title + " '" + Short((string)value["id"]) + "'" + (string.IsNullOrEmpty(label) ? "" : " (" + Short(label) + ")") + ": " +
                            Count(((JArray)mesh["vertices"]).Count, "vertex", "vertices") + ", " + Count(((JArray)mesh["faces"]).Count, "triangle") + "; transform: " + TransformText(value["transform"] as JArray);
                    case "scene":
                        var surfaces = value["surfaces"].Children<JObject>().ToArray();
                        int unique = surfaces.Select(s => s["mesh"]).Distinct(new JTokenEqualityComparer()).Count();
                        return title + ": " + Count(surfaces.Length, "surface") + ", " + Count(unique, "shared mesh", "shared meshes") + ", " + Count(surfaces.Sum(s => ((JArray)s["mesh"]["vertices"]).Count), "vertex", "vertices") + ", " +
                            Count(surfaces.Sum(s => ((JArray)s["mesh"]["faces"]).Count), "triangle") + "; IDs: " + Short(string.Join(", ", surfaces.Select(s => (string)s["id"])));
                    case "query":
                        return title + ": senders " + Selection(value["senders"]) + "; receivers " + ((bool?)value["scene"] == false ? "none (sky only)" : Selection(value["receivers"])) +
                            "; sides " + (value["receiver_sides"] == null ? "front + back" : string.Join(" + ", value["receiver_sides"].Values<string>())) + "; sky " + Text(value, "sky_mode", "none");
                    case "sampling": return title + ": " + Sampling(value);
                    case "accuracy": return title + ": " + Accuracy(value);
                    case "options": return title + ": " + Sampling(value["sampling"] as JObject ?? new JObject()) + "; " + Accuracy(value["accuracy"] as JObject ?? new JObject()) +
                        "; batch " + Text(value, "batch_size", "65536") + "; reciprocity " + Text(value["postprocessing"] as JObject ?? new JObject(), "reciprocity", "none");
                    case "result":
                        var coverage = value["coverage"].Values<int>().ToArray();
                        return title + ": " + Text(value, "status", "imported") + "; " + Count(((JArray)value["sender_ids"]).Count, "sender row") + " x " + Count(((JArray)value["channels"]).Count, "channel") +
                            "; " + coverage.Count(c => c == 1) + " sampled, " + coverage.Count(c => c == 0) + " uncomputed, " + coverage.Count(c => c == -1) + " unknown; " +
                            Text(value, "cumulative_rays", "unknown") + " cumulative rays (" + Text(value, "rays_used", "unknown") + " additional); convergence " + Text(value, "converged", "unknown");
                    case "channel": return "RS Channel: " + SnapshotGoo.ChannelLabel(value);
                }
                return title;
            }
            catch (Exception) { return title + ": invalid data (reconnect its component output)."; }
        }

        /// <summary>Describe default or explicit estimator settings without dumping their JSON.</summary>
        private static string Sampling(JObject value)
        {
            string strategy = Text(value, "strategy", "cosine");
            return strategy + (strategy == "area_pair" ? "; pair samples " + Text(value, "pair_samples", "8192") : "; density " + Text(value, "density", "16") + "; rays/cell " + Text(value, "rays_per_cell", "128")) +
                "; seed " + Text(value, "seed", "1") + "; " + Text(value, "mode", "fair") + "; flip " + Text(value, "flip_faces", "false");
        }

        /// <summary>Describe convergence limits using the same defaults as Python Accuracy.</summary>
        private static string Accuracy(JObject value)
        {
            return Text(value, "mode", "stderr") + " tolerance " + Text(value, "tolerance", "0.0001") + "; replicates min " + Text(value, "min_replicates", "5") +
                ", max " + Text(value, "max_replicates", "100") + "; min rays " + Text(value, "min_rays", "0");
        }

        /// <summary>Render selected IDs with a count and a bounded preview, or all surfaces.</summary>
        private static string Selection(JToken token)
        {
            return token == null || token.Type == JTokenType.Null ? "all" : token.Count() + " [" + Short(string.Join(", ", token.Values<string>())) + "]";
        }

        /// <summary>Show identity or rotation/translation in Rhino document units.</summary>
        private static string TransformText(JArray rows)
        {
            if (rows == null) return "identity";
            bool identity = true, rotation = false;
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
            {
                bool differs = Math.Abs((double)rows[r][c] - (r == c ? 1 : 0)) > 1e-8;
                identity &= !differs; if (r < 3 && c < 3) rotation |= differs;
            }
            if (identity) return "identity";
            return (rotation ? "rotation; " : "") + "translation (" + string.Join(", ", Enumerable.Range(0, 3).Select(r => ((double)rows[r][3]).ToString("G5", CultureInfo.InvariantCulture))) + ")";
        }

        /// <summary>Read a compact scalar or display its documented default when omitted.</summary>
        private static string Text(JObject value, string key, string fallback)
        {
            var token = value[key]; return token == null || token.Type == JTokenType.Null ? fallback : token.Type == JTokenType.String ? (string)token : token.ToString(Formatting.None);
        }

        /// <summary>Keep IDs and labels readable in panels even for large scene selections.</summary>
        private static string Short(string value) { value = (value ?? "").Replace("\r", " ").Replace("\n", " "); return value.Length <= 96 ? value : value.Substring(0, 93) + "..."; }

        /// <summary>Use readable singular/plural units in geometry and result summaries.</summary>
        private static string Count(int count, string singular, string plural = null) { return count.ToString(CultureInfo.InvariantCulture) + " " + (count == 1 ? singular : plural ?? singular + "s"); }
    }
}
