"""Bounded, restartable synthetic annual district experiment (not measured weather)."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from raystrack import Mesh, Scene
from raystrack.shortwave import ClearGlazing, DiffuseSheet, KlemsBSDF, Opaque, ShortwaveSolver
from raystrack.shortwave.klems import KLEMS_AZIMUTH_COUNTS, KLEMS_POLAR_BOUNDS_DEG
from validation.annual_radiance import synthetic_forcing


EVENTS = ((2190, "transform"), (4380, "material"), (6570, "deformation"), (7300, "topology"))
CGDB_SHA256 = "5f398e08a81fecd57f9e566fa2a16f08bb0f2ca2a22fdba04d5d949c818b3336"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bsdf_provenance(xml_sha256):
    """A file's presence is not evidence that its optical data were measured."""
    if xml_sha256 is None:
        return "controlled synthetic finite reciprocal Full Klems"
    if xml_sha256 == CGDB_SHA256:
        return "measurement-based LBNL CGDB product6006, original CC BY-ND4.0 fixture"
    return "user supplied XML, measurement provenance unverified"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def quad(corners, subdivisions=1):
    """Subdivide a planar parallelogram without changing its optical extent."""
    a, b, _, d = np.asarray(corners, dtype=np.float64)
    uv = np.array([(i / subdivisions, j / subdivisions)
                   for j in range(subdivisions + 1) for i in range(subdivisions + 1)])
    vertices = a + uv[:, :1] * (b-a) + uv[:, 1:] * (d-a)
    faces = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            k = j*(subdivisions+1)+i
            faces.extend(((k, k+1, k+subdivisions+2), (k, k+subdivisions+2, k+subdivisions+1)))
    return Mesh(vertices, faces)


def district_fixture(tiles=1, subdivisions=1, sensors_per_tile=5, material_types=4, bsdf_xml=None):
    """Repeated 12 m cells: courtyard walls, three canopies and six shade slats.

    Klems data are a controlled reciprocal finite lobe, not measured optics.
    Cells share Python optical objects, but native prototype instancing is absent.
    """
    if min(tiles, subdivisions, sensors_per_tile) < 1 or material_types not in (1, 2, 3, 4):
        raise ValueError("positive fixture dimensions and material_types in 1..4 required")
    matrices = np.empty((4, 145, 145))
    matrices[:2] = .12 / np.pi
    # Symmetric half-turn-invariant angular modulation preserves reciprocity.
    features = np.concatenate([
        np.sin(np.radians((low+high)/2))**2 * np.cos(4*np.pi*np.arange(count)/count)
        for low, high, count in zip(KLEMS_POLAR_BOUNDS_DEG[:-1],
                                   KLEMS_POLAR_BOUNDS_DEG[1:], KLEMS_AZIMUTH_COUNTS)
    ])
    matrices[2:] = .38/np.pi*(1 + .25*features[:, None]*features[None, :])
    finite = KlemsBSDF(matrices) if bsdf_xml is None else KlemsBSDF.from_xml(bsdf_xml, band="solar")
    choices = [Opaque(.25), ClearGlazing(1.52), DiffuseSheet(.15, .25, .45), finite]
    meshes, materials, points, labels = {}, {}, [], []
    for tile in range(tiles):
        x, y = 12*(tile % int(np.ceil(np.sqrt(tiles)))), 12*(tile // int(np.ceil(np.sqrt(tiles))))
        def add(name, corners, material, subdivide=True):
            key = f"t{tile}/{name}"
            meshes[key] = quad(np.asarray(corners) + [x, y, 0], subdivisions if subdivide else 1)
            materials[key] = material
        add("ground", [[0, 0, 0], [11, 0, 0], [11, 11, 0], [0, 11, 0]], choices[0])
        add("north", [[0, 10, 0], [11, 10, 0], [11, 10, 5], [0, 10, 5]], Opaque(.5))
        add("west", [[0, 10, 0], [0, 0, 0], [0, 0, 4], [0, 10, 4]], Opaque(.4))
        for j, name in enumerate(("glass", "sheet", "klems")):
            center = 2 + 3*j
            add(name, [[center-1.2, 2, 2.5], [center+1.2, 2, 2.5],
                       [center+1.2, 5, 2.5], [center-1.2, 5, 2.5]], choices[min(j+1, material_types-1)])
        for j in range(6):
            z = 3.4 + .08*(j % 2)
            add(f"slat{j}", [[1+j*1.5, 6, z], [1.55+j*1.5, 6, z],
                             [1.55+j*1.5, 9, z], [1+j*1.5, 9, z]], Opaque(.3), False)
        base = [[2.13, 3.37, .01], [5.13, 3.37, .01], [8.13, 3.37, .01],
                [4.2, 7.4, .01], [9.8, .8, .01]]
        for sensor in range(sensors_per_tile):
            point = np.array(base[sensor % 5]) + [x, y, 0]
            point[1] += .013*(sensor//5)
            points.append(point)
            labels.append(f"t{tile}/sensor{sensor}")
    scene = Scene.from_meshes(meshes)
    return scene, materials, np.asarray(points), np.tile([0, 0, 1], (len(points), 1)), labels


def apply_event(scene, materials, solver, event):
    """Absolute deterministic edits, replayed in order when restarting."""
    for key in scene.surface_ids:
        if event == "transform" and key.endswith("/slat2"):
            transform = np.eye(4)
            transform[1, 3] = -1.4
            scene.update_transform(key, transform)
        elif event == "material" and key.endswith("/sheet"):
            materials[key] = DiffuseSheet(.2, .1, .6)
            solver.set_material(key, materials[key])
        elif event == "deformation" and key.endswith("/north"):
            vertices = scene[key].world_vertices().copy()
            vertices[:, 2] *= .7
            scene.update_vertices(key, vertices)
        elif event == "topology" and key.endswith("/slat4"):
            mesh = scene[key].mesh
            scene.update_mesh(key, Mesh(mesh.vertices, mesh.faces[:1]))


def process_peak_bytes():
    """OS process-lifetime resident high-water mark, including native allocations."""
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def block_ranges(steps, block_steps):
    cuts = sorted({0, steps, *range(0, steps, block_steps), *(i for i, _ in EVENTS if i < steps)})
    return list(zip(cuts[:-1], cuts[1:]))


def fixture_hash(scene, points, normals, labels):
    h = hashlib.sha256()
    for key in scene.surface_ids:
        h.update(key.encode())
        h.update(scene[key].mesh.vertices.tobytes())
        h.update(scene[key].mesh.faces.tobytes())
    h.update(points.tobytes())
    h.update(normals.tobytes())
    h.update(json.dumps(labels).encode())
    return h.hexdigest()


def run_case(output, *, backend="embree", tiles=1, subdivisions=1, sensors_per_tile=5,
             material_types=4, steps=8760, block_steps=168, samples=64, replicates=2,
             bounces=3, seed=17, resume=False, max_blocks=None, bsdf_xml=None):
    """Write one bounded output block before atomically advancing the manifest.

    Checkpoints contain completed blocks, not in-flight native path state. A crash
    between block and manifest publication recomputes that uncommitted block.
    """
    if not 1 <= steps <= 8760 or min(block_steps, samples, replicates) < 1:
        raise ValueError("steps in 1..8760 and positive block/sampling sizes required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene, materials, points, normals, labels = district_fixture(tiles, subdivisions, sensors_per_tile, material_types, bsdf_xml)
    timeline, dni, dhi = synthetic_forcing()
    forcing_hash = hashlib.sha256(timeline.directions.tobytes()+dni.tobytes()+dhi.tobytes()
                                  +timeline.timestamps.tobytes()+timeline.durations_hours.tobytes()).hexdigest()
    library = os.environ.get("RAYSTRACK_NATIVE_LIBRARY")
    config = dict(backend=backend, tiles=tiles, subdivisions=subdivisions,
                  sensors_per_tile=sensors_per_tile, material_types=material_types, steps=steps,
                  block_steps=block_steps, samples=samples, replicates=replicates, bounces=bounces,
                  seed=seed, fixture_sha256=fixture_hash(scene, points, normals, labels),
                  forcing_sha256=forcing_hash, workflow_sha256=digest(__file__),
                  solver_sha256=digest(Path(__file__).parents[1]/"src/raystrack/shortwave/solver.py"),
                  native_library_sha256=digest(library) if library else None)
    config["bsdf_xml_sha256"] = None if bsdf_xml is None else digest(bsdf_xml)
    config["optics_sha256"] = {
        name: digest(Path(__file__).parents[1]/"src/raystrack/shortwave"/name)
        for name in ("materials.py", "klems.py")}
    manifest_path = output / "report.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("existing report requires --resume or a new output directory")
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report["configuration"] != config:
            raise ValueError("checkpoint configuration/provenance mismatch")
    else:
        report = dict(schema=1, configuration=config, intervals=0, complete=False, blocks=[],
                      probes=[], labels=labels, triangles=sum(len(scene[k].mesh.faces) for k in scene.surface_ids),
                      surfaces=len(scene.surface_ids), sensors=len(points), python=platform.python_version(),
                      platform=platform.platform(), numpy=np.__version__, units={"irradiance":"W/m2", "exposure":"Wh/m2"},
                      forcing="deterministic synthetic Berlin2025, fixed UTC+01:00 hourly midpoints, black lower sky",
                      site={"latitude_deg":52.52, "longitude_deg":13.405,
                            "utc_offset_minutes":60, "year":2025, "north":"+Y", "up":"+Z"},
                      events=[{"hour": hour, "edit": event} for hour, event in EVENTS],
                      channel_basis="neutral: three identical representations of broadband solar power",
                      limits={"bands":1, "spectral_transport":False, "measured_bsdf":False,
                              "transfer_cache":False, "gpu_resident_transport":False,
                              "native_shared_prototypes":False},
                      memory_kind="OS process-lifetime peak resident working set including native allocations")
        report["bsdf_source"] = "controlled synthetic finite reciprocal Full Klems" if bsdf_xml is None else str(Path(bsdf_xml).resolve())
        report["bsdf_provenance"] = bsdf_provenance(config["bsdf_xml_sha256"])
        report["limits"]["measured_bsdf"] = config["bsdf_xml_sha256"] == CGDB_SHA256 and material_types == 4
    ranges = block_ranges(steps, block_steps)
    if len(report["blocks"]) > len(ranges):
        raise ValueError("checkpoint contains excess blocks")
    exposure = np.zeros((len(points), 4, 3))
    exposure_se_bound = np.zeros_like(exposure)
    for index, block in enumerate(report["blocks"]):
        if [block["start"], block["stop"]] != list(ranges[index]):
            raise ValueError("checkpoint block sequence mismatch")
        path = output / block["file"]
        if block["file"] != f"block-{block['start']:05d}-{block['stop']:05d}.npz":
            raise ValueError("checkpoint block filename mismatch")
        if digest(path) != block["sha256"]:
            raise ValueError("checkpoint block checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            exposure += np.einsum("t,tscr->scr", arrays["durations_hours"], arrays["irradiance"])
            if arrays["standard_error"].size:
                exposure_se_bound += np.einsum("t,tscr->scr", arrays["durations_hours"], arrays["standard_error"])
    begin = time.perf_counter()
    with ShortwaveSolver(scene, materials=materials, backend=backend) as solver:
        completed_now = 0
        for index, (start, stop) in enumerate(ranges):
            edit_tick = time.perf_counter()
            edited = False
            for hour, event in EVENTS:
                if start == hour:
                    apply_event(scene, materials, solver, event)
                    edited = True
            edit_commit_s = time.perf_counter()-edit_tick if edited else 0.0
            if index < len(report["blocks"]):
                continue
            probing = index == 0 or start in dict(EVENTS)
            if probing:
                selection = np.linspace(0, stop-start-1, min(stop-start, 3), dtype=int)
                directions = timeline.directions[start:stop][selection]
                probe_options = dict(dni=dni[start:stop][selection], dhi=dhi[start:stop][selection],
                                     samples=samples, replicates=replicates, bounces=bounces, seed=seed)
                tick = time.perf_counter()
                first_probe = solver.solve(points, normals, directions, **probe_options)
                first_probe_s = time.perf_counter()-tick
            options = dict(samples=samples, replicates=replicates, bounces=bounces, seed=seed,
                           forcing_id="synthetic-Berlin2025-urban", durations_hours=timeline.durations_hours[start:stop])
            tick = time.perf_counter()
            result = solver.solve(points, normals, timeline.directions[start:stop],
                                  dni=dni[start:stop], dhi=dhi[start:stop], **options)
            solve_s = time.perf_counter()-tick
            if index == 0:
                tick = time.perf_counter()
                repeated_block = solver.solve(points, normals, timeline.directions[start:stop],
                                               dni=dni[start:stop], dhi=dhi[start:stop], **options)
                report["initial_block_repeat_s"] = time.perf_counter()-tick
                report["initial_block_repeat_steps"] = stop-start
                np.testing.assert_array_equal(result.irradiance, repeated_block.irradiance)
                del repeated_block
            if not np.isfinite(result.irradiance).all() or np.any(result.irradiance < 0):
                raise AssertionError("nonfinite or negative annual irradiance")
            night = ~timeline.above_horizon[start:stop]
            if night.any() and np.any(result.irradiance[night] != 0):
                raise AssertionError("nonzero synthetic night irradiance")
            # Initial and post-edit calls expose setup, repeat and fresh-build phases.
            if probing:
                tick = time.perf_counter()
                repeat = solver.solve(points, normals, directions, **probe_options)
                repeat_s = time.perf_counter()-tick
                tick = time.perf_counter()
                with ShortwaveSolver(scene, materials=materials, backend=backend) as fresh:
                    rebuilt = fresh.solve(points, normals, directions, **probe_options)
                fresh_s = time.perf_counter()-tick
                np.testing.assert_allclose(repeat.irradiance, result.irradiance[selection], rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(first_probe.irradiance, repeat.irradiance, rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(rebuilt.irradiance, repeat.irradiance, rtol=1e-12, atol=1e-10)
                report["probes"].append(dict(hour=start, steps=len(selection), repeat_s=repeat_s,
                    first_or_edited_solve_s=first_probe_s, edit_commit_s=edit_commit_s,
                    edit_plus_solve_s=edit_commit_s+first_probe_s,
                    fresh_setup_and_solve_s=fresh_s, max_abs_fresh_w_m2=float(np.max(np.abs(rebuilt.irradiance-repeat.irradiance))),
                    geometry_revision=result.geometry_revision, material_revision=result.material_revision))
            tick = time.perf_counter()
            name = f"block-{start:05d}-{stop:05d}.npz"
            path = output / name
            with path.with_suffix(".npz.tmp").open("wb") as stream:
                np.savez_compressed(stream, irradiance=result.irradiance, standard_error=result.standard_error
                    if result.standard_error is not None else np.empty(0),
                    timestamps=timeline.timestamps[start:stop], durations_hours=timeline.durations_hours[start:stop])
            os.replace(path.with_suffix(".npz.tmp"), path)
            exposure += result.exposure()
            if result.standard_error is not None:
                exposure_se_bound += np.einsum("t,tscr->scr", result.durations_hours, result.standard_error)
            report["blocks"].append(dict(start=start, stop=stop, file=name, sha256=digest(path), solve_s=solve_s,
                write_s=time.perf_counter()-tick, geometry_revision=result.geometry_revision,
                material_revision=result.material_revision,
                max_standard_error_w_m2=float(np.max(result.standard_error)) if result.standard_error is not None else None))
            report.update(intervals=stop, complete=stop == steps,
                          exposure_wh_m2=exposure.tolist(), peak_resident_bytes=process_peak_bytes(),
                          output_bytes=sum((output / b["file"]).stat().st_size for b in report["blocks"]))
            report["exposure_one_se_conservative_bound_wh_m2"] = exposure_se_bound.tolist() if replicates > 1 else None
            report["uncertainty_assumptions"] = (
                "Independent randomized replicate means; linear standard-error bound allows arbitrary "
                "covariance across reused hourly paths. Sampling error only, no model/reference accuracy claim.")
            if replicates == 2:
                report["exposure_95_half_width_conservative_wh_m2"] = (12.706204736*exposure_se_bound).tolist()
                report["uncertainty_assumptions"] += " Student t(1) 95% interval, approximate normal replicate means; two replicates give weak precision."
            report["solve_s"] = sum(b["solve_s"] for b in report["blocks"])
            report["measured_phases_s"] = dict(block_solves=report["solve_s"],
                block_writes=sum(b["write_s"] for b in report["blocks"]),
                probe_first_or_edit=sum(p["edit_plus_solve_s"] for p in report["probes"]),
                probe_repeats=sum(p["repeat_s"] for p in report["probes"]),
                probe_fresh_builds=sum(p["fresh_setup_and_solve_s"] for p in report["probes"]))
            report["last_invocation_s"] = time.perf_counter()-begin
            report["largest_result_array_bytes"] = min(block_steps, steps)*len(points)*4*3*8
            atomic_json(manifest_path, report)
            completed_now += 1
            if max_blocks is not None and completed_now >= max_blocks:
                break
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/annual-urban"))
    parser.add_argument("--backend", choices=("embree", "vulkan-rt", "native-builtin"), default="embree")
    parser.add_argument("--tiles", type=int, default=1)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument("--sensors-per-tile", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8760)
    parser.add_argument("--block-steps", type=int, default=168)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--bsdf-xml", type=Path, help="original Solar Full Klems WINDOW XML, retained unmodified")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="also measure geometry/sensor/time/material cases at three sizes")
    args = parser.parse_args()
    options = vars(args).copy()
    sweep = options.pop("sweep")
    output = options.pop("output")
    report = run_case(output, **options)
    print(json.dumps({key: report[key] for key in ("intervals", "complete", "triangles", "sensors", "solve_s", "peak_resident_bytes")}, indent=2))
    if sweep:
        records = []
        for axis, values in (("tiles", (1, 4, 16, 64)), ("subdivisions", (1, 4, 16, 64)),
                             ("sensors_per_tile", (5, 20, 80)), ("steps", (24, 72, 168)),
                             ("material_types", (1, 2, 4))):
            for value in values:
                case = dict(backend=args.backend, steps=24, samples=args.samples,
                            replicates=args.replicates, bounces=args.bounces, resume=args.resume, bsdf_xml=args.bsdf_xml)
                case[axis] = value
                evidence = run_case(output/f"sweep-{axis}-{value}", **case)
                records.append(dict(axis=axis, value=value, triangles=evidence["triangles"],
                    sensors=evidence["sensors"], intervals=evidence["intervals"], solve_s=evidence["solve_s"],
                    peak_resident_bytes=evidence["peak_resident_bytes"], probes=evidence["probes"]))
        atomic_json(output/"scale-sweep.json", dict(measured=records,
            bands={"measured":1, "additional_bands":"unsupported here; no spectral capacity claim"},
            caveat="sequential process; resident peaks are cumulative; January hours, not annual runtime projections"))


if __name__ == "__main__":
    main()
