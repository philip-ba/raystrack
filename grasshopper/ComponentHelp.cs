namespace Raystrack.Grasshopper
{
    /// <summary>Practical, offline instructions shown by Grasshopper's component Help command.</summary>
    internal static class ComponentHelp
    {
        /// <summary>Return workflow examples and input conventions for a concrete component type.</summary>
        internal static string For(string type)
        {
            string text;
            switch (type)
            {
                case "SurfaceComponent":
                    text = "Connect a Mesh or Brep to Geometry and give it an ID such as roof or wall_A. Labels are optional display names; queries use IDs. Front is defined by mesh face winding.<br/><br/>" +
                        "Transform accepts a native Grasshopper Transform, not a vector or a list of numbers. Connect the X (Transform) output of Move, Rotate or another rigid transform component. Feed the original Rhino mesh/Brep into Move's Geometry (G) and a motion vector such as Unit X multiplied by 5 into Motion (T): the surface moves 5 Rhino document units. Leave Transform empty for identity.<br/><br/>" +
                        "Connect the ORIGINAL geometry with its Transform, or connect the already moved geometry and leave Transform empty. Applying both moves it twice. Only rotation/translation is allowed here; scale or mirror geometry upstream before creating the surface. Connect Surface to RS Scene or RS Instance.";
                    break;
                case "InstanceComponent":
                    text = "Connect an RS Surface as the prototype, use a different ID, then connect a native Grasshopper Transform (X) from Move or Rotate. Build that transform using the original Rhino geometry at Move's Geometry input and a vector at Motion. Leave Transform empty to copy the prototype placement. The new transform acts AFTER its existing placement: new = T * prototype, with column-vector convention. Reuse instances for moving repeated geometry. Scale/mirror the source mesh upstream instead of its instance transform.";
                    break;
                case "SceneComponent":
                    text = "Merge all RS Surface and Instance outputs into Surfaces, including geometry used only as blockers. Every ID must be unique. Connect Scene to RS Solve. Use the IDs output to build queries; selecting receivers keeps the other surfaces as occluders. A Panel shows surface, vertex, triangle and shared-mesh counts.";
                    break;
                case "QueryComponent":
                    text = "Example pair: Mode=pair, Senders=A, Receivers=B. Example row: Mode=row, Senders=A, leave Receivers empty for all. Matrix defaults to all senders/receivers. Sky mode requests escaped sky contributions; choose merged or tregenza145. Front/back refer to receiver face winding. IDs are case-sensitive; use RS Scene's IDs output, not labels. Mode/Sky/Sides choices ignore capitalization and surrounding spaces.";
                    break;
                case "SamplingComponent":
                    text = "Defaults: density 16, 128 rays/cell, seed 1, fair allocation, cosine estimator. Increase density/rays for additional spatial/angular sampling, or replicate limits in RS Accuracy for refinement. Flip reverses emitting normals while receiver labels keep their meaning. area_pair supports CPU pair queries only. Connect Sampling to RS Options. A Panel displays all relevant sampling settings.";
                    break;
                case "AccuracyComponent":
                    text = "Defaults: stderr, tolerance 0.0001, min 5 and max 100 replicates. Tolerance=0 runs to the maximum replicate count. A partial replicate can have unknown uncertainty; this is reported as NaN rather than zero. Minimum replicates can exceed maximum when deliberately forcing a fixed workload. Connect Accuracy to RS Options.";
                    break;
                case "OptionsComponent":
                    text = "Connect RS Sampling and RS Accuracy, or leave either empty for its documented defaults. Batch size is the maximum work per kernel chunk; smaller batches usually permit more frequent cancellation checks. Reciprocity is explicit postprocessing on completed full-scene results; rowsum additionally assumes a closed scene. Partial previews are raw estimates. Connect Options to RS Solve; a Panel also shows defaults used by empty inputs.";
                    break;
                case "SolveComponent":
                    text = "Connect Scene; Query and Options are optional. Press a Button at Run, or change a Boolean from false to true. Input changes alone do not start work. The worker runs outside Rhino; Status, Progress, Rays and partial Result refresh automatically. Initial Python import/JIT preparation may take time before rays appear.<br/><br/>" +
                        "Ray budget=0 runs until convergence or the replicate limit. A positive budget means ADDITIONAL rays. A paused compatible run continues on the next false-to-true Run pulse. If scene/query/options/device changes, the next launch starts a new run. Cancel keeps the partial result and is checked between kernel chunks. Saved true triggers are disarmed: set Run false then true after reopening.<br/><br/>" +
                        "Use RS Runtime for device availability; cpu is a useful first test. An explicit unsupported GPU/strategy combination reports an error. Result Table/Value read live snapshots; succeeded describes execution completion, so also inspect Result convergence and coverage.";
                    break;
                case "ResultTableComponent":
                    text = "Values/Errors are trees: branch {i} is sender i; each item matches Channels/Labels column order. Channels are structured identifiers, not strings with parsed suffixes. Coverage: 1=sampled, 0=uncomputed, -1=unknown (legacy). NaN means unavailable/uncomputed; a measured zero stays zero. Result readers wait quietly until a background solve produces its first snapshot.";
                    break;
                case "ResultValueComponent":
                    text = "For A to B's front: Sender=A, Receiver=B, Kind=surface, Side=front. IDs are case-sensitive. For merged sky: Kind=sky, Patch=-1. Tregenza patches use 0..144. Kind=rest reads escape; Kind=unrequested reads hits on surfaces omitted from the requested receiver outputs. These non-surface kinds ignore Receiver/Side. Consult RS Result Table's Labels when a requested channel is unavailable.";
                    break;
                case "InspectComponent":
                    text = "Connect any RS output to Data. Text gives a compact, readable summary and usage description. Set Details=true only when you want full JSON, geometry arrays, channel data and execution statistics. You can also connect any RS object directly to a Panel for its concise representation.";
                    break;
                case "SaveComponent":
                    text = "Use a new folder path such as C:\\Results\\case.raystrack. Connect the complete Scene and optionally its matching Result. Pulse Run once to write in the background. Existing folders are never overwritten: choose a new name. A result computed for different geometry cannot be saved with this scene. Saved true triggers are disarmed on reopen.";
                    break;
                case "LoadComponent":
                    text = "Connect an existing .raystrack folder path, or a legacy v1 JSON path, then pulse Run. Loading runs in the worker and returns portable Scene/Result objects. Imported legacy errors, coverage and statistics may be unknown. Missing legacy geometry must be supplied before a new v2 solve/save. A saved true trigger is disarmed after reopening.";
                    break;
                case "RuntimeComponent":
                    text = "Press Refresh once to check the adjacent bundled Python and available devices without a solve. Devices lists availability and failure reasons; copy cpu or an available backend name to RS Solve. Install the complete Yak/ZIP with runtime beside the GHA. Driver probing stays in the background. Restart Rhino after replacing/installing the plug-in.";
                    break;
                default: text = "Hover each input for its type, supported values and defaults. Connect RS data directly to a Panel for a readable summary."; break;
            }
            return "<br/><br/>" + text;
        }
    }
}
