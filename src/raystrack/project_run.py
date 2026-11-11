"""Run a file-driven moving shortwave project with bounded completed blocks.

Usage: python -m raystrack.project_run run project.json --output results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .project import load_project
from .project_store import ResultReader, ResultWriter
from .shortwave import ShortwaveSolver


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _event_time(event):
    return np.datetime64(event["timestamp_utc"], "us")


def _boundaries(project):
    timeline = project.timeline
    half = np.rint(timeline.durations_hours * 1_800_000_000).astype("timedelta64[us]")
    starts = timeline.timestamps.astype("datetime64[us]") - half
    indices = {}
    for event in project.states:
        instant = _event_time(event)
        index = int(np.searchsorted(starts, instant))
        if index >= len(starts) or starts[index] != instant:
            raise ValueError("state timestamps must coincide with weather interval start boundaries")
        indices.setdefault(index, []).append(event)
    return indices


def apply_states(project, solver, events, assignments):
    """Commit one simulation boundary while no solve is active.

    Transform batches are atomic; all other validated state edits execute before
    the next solve under the scene lock. Optical no-ops retain their revision.
    """
    transforms = {}
    with solver._lock, project.scene.lock:
        for event in events:
            for key, vertices in event.get("vertices", {}).items():
                project.scene.update_vertices(key, vertices)
            for key, mesh in event.get("meshes", {}).items():
                project.scene.update_mesh(key, mesh)
            transforms.update(event.get("transforms", {}))
            for key, material_id in event.get("materials", {}).items():
                if assignments[key] != material_id:
                    solver.set_material(key, project.material_library[material_id])
                    assignments[key] = material_id
        project.scene.update_transforms(transforms)


def run_project(path, output, *, backend=None, resume=False, max_blocks=None,
                block_steps=None, samples=None, replicates=None, retention=None,
                storage=None, fresh_probes=True):
    """Run/resume completed intervals without loading previous hourly arrays.

    Time partitioning retains native sample streams. Sensors are deliberately
    bounded by a configured admission limit: changing sensor partitions changes
    the native sensor seed, so independent sensor batching is not claimed.
    """
    project = load_project(path)
    settings = dict(project.settings)
    overrides = dict(backend=backend, block_steps=block_steps, samples=samples,
                     replicates=replicates, retention=retention, storage=storage)
    settings.update({key: value for key, value in overrides.items() if value is not None})
    defaults = dict(backend="embree", block_steps=168, samples=64, replicates=2,
                    bounces=3, seed=17, retention="hourly", storage="npy", max_sensors=4096,
                    max_result_bytes=32*1024*1024)
    unknown = set(settings).difference(defaults)
    if unknown:
        raise ValueError(f"unknown project settings: {sorted(unknown)}")
    settings = {**defaults, **settings}
    for name in ("block_steps", "replicates", "max_sensors", "max_result_bytes"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, maximum in (("samples", 2**32-1), ("seed", 2**64-1), ("bounces", 32)):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or not 0 <= settings[name] <= maximum:
            raise ValueError(f"{name} must be an integer from 0 to {maximum}")
    if settings["backend"] not in ("native-builtin", "embree", "vulkan-rt"):
        raise ValueError("unknown native backend")
    if max_blocks is not None and (isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or max_blocks < 1):
        raise ValueError("max_blocks must be a positive integer")
    if len(project.sensor_ids) > settings["max_sensors"]:
        raise ValueError("sensor count exceeds max_sensors memory admission limit")
    if settings["replicates"] < 2:
        raise ValueError("stored uncertainty requires at least two independent replicates")
    bytes_per_step = len(project.sensor_ids)*4*3*8
    bounded_steps = min(settings["block_steps"], settings["max_result_bytes"]//bytes_per_step)
    if bounded_steps < 1:
        raise ValueError("one sensor frame exceeds max_result_bytes")
    events = _boundaries(project)
    count = len(project.timeline.timestamps)
    cuts = sorted({0, count, *range(0, count, bounded_steps), *events})
    library = os.environ.get("RAYSTRACK_NATIVE_LIBRARY")
    forcing_digest = hashlib.sha256()
    for array in (project.timeline.timestamps, project.timeline.durations_hours,
                  project.timeline.directions, project.dni, project.dhi):
        forcing_digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
        forcing_digest.update(np.ascontiguousarray(array).tobytes())
    configuration = dict(project_sources=project.source_hashes, settings=settings,
                         forcing_sha256=forcing_digest.hexdigest(),
                         native_library_sha256=None if not library else _digest(library),
                         implementation_sha256={name: _digest(Path(__file__).parent / name) for name in
                             ("project_run.py", "project.py", "project_store.py", "shortwave/solver.py",
                              "shortwave/klems.py", "shortwave/materials.py", "shortwave/solar.py",
                              "model/scene.py", "model/geometry.py", "backends/native.py")},
                         state_semantics="absolute poses held from interval start; no interpolation",
                         forcing="UTC midpoint DNI/DHI W/m2; isotropic upper sky, black lower sky",
                         channel_basis="neutral RGB: three copies of broadband solar power")
    writer = ResultWriter(output, configuration=configuration, sensor_ids=project.sensor_ids,
                          sensor_groups=project.sensor_groups, components=("direct", "sky", "reflected", "transmitted"),
                          bands=("neutral_R", "neutral_G", "neutral_B"),
                          storage=settings["storage"], retention=settings["retention"], resume=resume)
    completed = writer.completed_steps
    if completed not in cuts:
        raise ValueError("checkpoint does not end at a configured block boundary")
    assignments = {entry["id"]: entry["material"] for entry in project.manifest["instances"]}
    timings, probes = [], []
    started = time.perf_counter()
    with ShortwaveSolver(project.scene, materials=project.materials, backend=settings["backend"]) as solver:
        blocks_now = 0
        for block_index, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
            tick = time.perf_counter()
            apply_states(project, solver, events.get(start, ()), assignments)
            edit_s = time.perf_counter() - tick
            if stop <= completed:
                saved = writer.manifest["blocks"][block_index]
                if (saved["scene_revision"] != project.scene.revision
                        or saved["material_revision"] != solver._material_revision):
                    raise ValueError("checkpoint revisions disagree with deterministic state replay")
                continue
            if max_blocks is not None and blocks_now >= max_blocks:
                break
            options = dict(dni=project.dni[start:stop], dhi=project.dhi[start:stop],
                           samples=settings["samples"], replicates=settings["replicates"],
                           bounces=settings["bounces"], seed=settings["seed"],
                           durations_hours=project.timeline.durations_hours[start:stop],
                           forcing_id="file project source hashes in result manifest")
            tick = time.perf_counter()
            result = solver.solve(project.points, project.normals,
                                  project.timeline.directions[start:stop], **options)
            solve_s = time.perf_counter() - tick
            if fresh_probes and (start == 0 or start in events):
                relative = int(np.argmax(project.dni[start:stop]+project.dhi[start:stop]))
                probe_step = start+relative
                probe_options = {**options, "dni": project.dni[probe_step:probe_step+1],
                                 "dhi": project.dhi[probe_step:probe_step+1],
                                 "durations_hours": project.timeline.durations_hours[probe_step:probe_step+1]}
                materials = {key: project.material_library[value] for key, value in assignments.items()}
                with ShortwaveSolver(project.scene, materials=materials, backend=settings["backend"]) as fresh:
                    reference = fresh.solve(project.points, project.normals,
                                            project.timeline.directions[probe_step:probe_step+1], **probe_options)
                actual = result.irradiance[relative:relative+1]
                difference = float(np.max(np.abs(reference.irradiance - actual)))
                if not np.allclose(reference.irradiance, actual, rtol=1e-10, atol=1e-7):
                    raise AssertionError(f"updated/fresh mismatch at interval {start}: {difference}")
                probes.append(dict(start=start, probe_step=probe_step,
                                   forcing_w_m2=float(project.dni[probe_step]+project.dhi[probe_step]),
                                   max_abs_w_m2=difference))
            tick = time.perf_counter()
            writer.append(project.timeline.timestamps[start:stop], project.timeline.durations_hours[start:stop],
                          result.irradiance, result.standard_error,
                          scene_revision=result.geometry_revision, material_revision=result.material_revision)
            timings.append(dict(start=start, stop=stop, edit_s=edit_s, solve_s=solve_s,
                                write_s=time.perf_counter()-tick,
                                geometry_mode=solver._native.geometry_mode,
                                instance_statistics=solver._native.instance_statistics,
                                geometry_revision=result.geometry_revision,
                                material_revision=result.material_revision,
                                state_changes={name: sum(len(event.get(name, {})) for event in events.get(start, ()))
                                               for name in ("transforms", "materials", "vertices", "meshes")}))
            blocks_now += 1
        if writer.completed_steps == count:
            writer.finish()
    report = dict(complete=writer.completed_steps == count, intervals=writer.completed_steps,
                  sensors=len(project.sensor_ids), instances=len(project.scene),
                  prototypes=len({id(surface.mesh) for surface in project.scene.surfaces}),
                  triangles=sum(len(surface.mesh.faces) for surface in project.scene.surfaces),
                  settings=settings, elapsed_s=time.perf_counter()-started,
                  maximum_result_array_bytes=bounded_steps*bytes_per_step,
                  memory_scope="output array cap; multiple native/NumPy arrays and geometry coexist; native ray paths use bounded chunks",
                  timings=timings, fresh_probes=probes, sources=project.source_hashes,
                  execution_only="district annual execution does not establish independent material accuracy")
    report_path = Path(output) / "run-report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume a project")
    run.add_argument("project")
    run.add_argument("--output", required=True)
    run.add_argument("--backend", choices=("native-builtin", "embree", "vulkan-rt"))
    run.add_argument("--resume", action="store_true")
    for name in ("block-steps", "samples", "replicates", "max-blocks"):
        run.add_argument("--"+name, type=int)
    run.add_argument("--retention", choices=("hourly", "exposure"))
    run.add_argument("--storage", choices=("npy", "npz"))
    export = commands.add_parser("export", help="export a selected half-open interval/sensor window")
    export.add_argument("results")
    export.add_argument("--output", required=True)
    export.add_argument("--start", type=int, default=0)
    export.add_argument("--stop", type=int, required=True)
    export.add_argument("--sensors", nargs="+")
    inspect = commands.add_parser("inspect", help="show axes, provenance and accumulated exposure")
    inspect.add_argument("results")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "run":
        project, output = args.pop("project"), args.pop("output")
        print(json.dumps(run_project(project, output, **args), indent=2))
    elif command == "export":
        reader = ResultReader(args["results"])
        reader.export_csv(args["output"], start=args["start"], stop=args["stop"], sensor_ids=args["sensors"])
        print(args["output"])
    else:
        print(json.dumps(ResultReader(args["results"]).manifest, indent=2))


if __name__ == "__main__":
    main()
