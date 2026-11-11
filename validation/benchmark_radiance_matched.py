"""Matched cube wall timings using the existing validated annual workflow.

Only benchmark orchestration: the renderer and reference algorithms are unchanged.
All phases exclude common forcing loading and Python import/process startup.
Radiance reference construction includes the existing primary-sun diagnostic.
Warm Radiance application reuses coefficients; warm native solves retrace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from raystrack.backends.native import NativeGeometry
from raystrack.shortwave import Opaque, PatchSkySeries, ReinhartSkyBasis, ShortwaveSolver, ThinDiffuse

try:
    from .annual_radiance import Radiance, aggregate, cube_fixture, metrics, reference, polygon_text, rcontrib
    from .benchmark_native_shortwave import canopy
except ImportError:
    from annual_radiance import Radiance, aggregate, cube_fixture, metrics, reference, polygon_text, rcontrib
    from benchmark_native_shortwave import canopy


def timed(operation):
    begin = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - begin


def canopy_source(vertices):
    material = "void trans neutral\n0\n0\n7 .45 .45 .45 0 0 .222222222222 0\n"
    return material+"".join(polygon_text(f"tile{i}",quad,"neutral")
                              for i,quad in enumerate(vertices.reshape(-1,4,3)))


def canopy_reference(rad, source, points, normals, directions, sky, samples):
    """Same coefficient algorithm/settings as annual.reference, diffuse canopy."""
    out = rad.output
    (out/"scene.rad").write_text(source(),encoding="ascii")
    sensors = "".join(" ".join(f"{v:.10g}" for v in row)+"\n" for row in np.c_[points,normals]).encode()
    (out/"sensors.pts").write_bytes(sensors)
    (out/"sky.rad").write_text("void glow sky_glow\n0\n0\n4 1 1 1 0\nsky_glow source sky\n0\n0\n4 0 0 1 180\n")
    rad.run("oconv",["scene.rad","sky.rad"],target="sky.oct")
    coefficients = rcontrib(rad,"sky.oct",sensors,None,146,bounces=2,samples=samples,sky=True)[:,1:,:]
    np.save(out/"radiance_sky_coefficients.npy",coefficients)
    diffuse, apply_s = timed(lambda: np.einsum("spr,tpr->tsr",coefficients,sky.radiance))
    omega = np.pi*np.sin(np.radians(.01/2))**2
    sources = []
    for i,direction in enumerate(directions):
        sources.append(f"void light sun{i}\n0\n0\n3 {1/omega:.12g} {1/omega:.12g} {1/omega:.12g}\n"
            f"sun{i} source s{i}\n0\n0\n4 "+" ".join(f"{v:.12g}" for v in direction)+" .01\n")
    (out/"suns.rad").write_text("".join(sources))
    (out/"suns.mod").write_text("\n".join(f"sun{i}" for i in range(len(directions)))+"\n")
    rad.run("oconv",["scene.rad","suns.rad"],target="sun.oct")
    direct = rcontrib(rad,"sun.oct",sensors,"suns.mod",len(directions),bounces=1,samples=samples)
    np.save(out/"radiance_sun_coefficients.npy",direct)
    primary = rcontrib(rad,"sun.oct",sensors,"suns.mod",len(directions),bounces=0,samples=1)
    return direct.transpose(1,0,2),diffuse,primary.transpose(1,0,2),dict(
        diffuse_matrix_application_s=apply_s,sun_source_count=len(directions))


def reference_build(directory, polygons, points, normals, sun, sky, dni, samples, scene_source=None):
    directory.mkdir(parents=True)
    rad = Radiance("C:/Radiance/bin", directory)
    version = rad.run("rtrace", ["-version"]).decode().strip()
    data, build_s = timed(lambda: reference(rad, polygons, points, normals, sun, sky, samples, 1)
        if scene_source is None else canopy_reference(rad,scene_source,points,normals,sun,sky,samples))
    unit_sun, _, _, info = data
    coefficients = np.load(directory / "radiance_sky_coefficients.npy")

    def apply():
        direct = unit_sun * dni[:, None, None]
        diffuse = np.einsum("spr,tpr->tsr", coefficients, sky.radiance)
        return direct + diffuse

    first, first_apply_s = timed(apply)
    compilation = sum(c["seconds"] for c in rad.commands if Path(c["argv"][0]).stem == "oconv")
    traces = [c["seconds"] for c in rad.commands if Path(c["argv"][0]).stem == "rcontrib"]
    evidence = dict(version=version, coefficient_build_s=build_s,
        geometry_octree_s=compilation, sky_trace_s=traces[0], sun_trace_s=traces[1],
        primary_diagnostic_trace_s=traces[2], internal_sky_application_s=info["diffuse_matrix_application_s"],
        final_combined_application_s=first_apply_s, first_result_s=build_s+first_apply_s,
        sun_sources=info["sun_source_count"], commands=str(directory / "radiance_commands.json"),
        scope="reference build + final application; includes saved coefficient IO and primary diagnostic")
    return first, apply, evidence


def worker(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    source = args.forcing.resolve()
    with np.load(source) as saved:
        forcing = {name: saved[name].copy() for name in saved.files}
    scene, polygons, points, normals, labels = cube_fixture()
    np.testing.assert_array_equal(forcing["points"], points)
    np.testing.assert_array_equal(forcing["normals"], normals)
    np.testing.assert_array_equal(forcing["labels"], labels)
    selection = np.arange(8760) if args.scope == "annual" else np.array(
        [24*d+h for d in (15,79,171,264) for h in (7,11,16)])
    sun, dni = forcing["sun"][selection], forcing["dni"][selection]
    sky = PatchSkySeries(ReinhartSkyBasis(), forcing["sky"][selection])
    durations = forcing["durations_hours"][selection]
    scene_source = None
    materials = {name:Opaque(.2) for name in polygons}
    if args.scope=="canopy":
        scene = canopy(256)
        vertices = scene["canopy"].mesh.vertices
        polygons = {"canopy": None}
        rng = np.random.default_rng(11)
        points = np.column_stack((rng.uniform(-64,64,(64,2)),np.repeat([0.,2.],32)))
        normals = np.column_stack((np.zeros((64,2)),np.repeat([1.,-1.],32)))
        angles = np.linspace(-np.pi/3,np.pi/3,24)
        sun = np.column_stack((np.sin(angles),np.zeros(24),np.cos(angles)))
        dni = 700*sun[:,2]
        basis = ReinhartSkyBasis()
        sky = PatchSkySeries(basis,np.full((24,len(basis),3),100/np.pi))
        durations = np.ones(24)
        labels = ["lower"]*32+["upper"]*32
        materials = {"canopy":ThinDiffuse(.35,.1)}
        scene_source = lambda: canopy_source(vertices)
        np.savez_compressed(out/"canopy-inputs.npz",vertices=vertices,points=points,normals=normals,
                            sun=sun,dni=dni,sky=sky.radiance,durations_hours=durations,labels=labels)
    stored_forcing = (dict(sun=sun,dni=dni,dhi=np.full(24,100.),sky=sky.radiance,points=points,
        normals=normals,labels=np.asarray(labels),durations_hours=durations,step_indices=np.arange(24))
        if args.scope=="canopy" else {
        name: values[selection] if name in ("timestamps", "durations_hours", "sun", "dni", "dhi", "sky")
        else values for name, values in forcing.items()})
    np.savez_compressed(out / "forcing.npz", **stored_forcing)
    report = dict(scope=args.scope, intervals=len(sun), sensors=len(points), triangles=131072 if args.scope=="canopy" else 14,
        geometry=("256x256 diffuse transmitting canopy, 64 fixed sensors, no ground" if args.scope=="canopy" else
                  "1 m neutral cube, 20x20 m finite ground; 4 sensors/exposed face + 2 ground + open"),
        platform=platform.platform(), processor=platform.processor(), cpu_logical_count=os.cpu_count(),
        python=platform.python_version(), numpy=np.__version__, argv=sys.argv,
        native_library=os.environ["RAYSTRACK_NATIVE_LIBRARY"],
        native_sha256=hashlib.sha256(Path(os.environ["RAYSTRACK_NATIVE_LIBRARY"]).read_bytes()).hexdigest(),
        forcing_source=str(source) if args.scope!="canopy" else "fixed synthetic canopy benchmark",
        forcing_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if args.scope!="canopy" else None,
        comparison_groups=list(dict.fromkeys(labels)),
        common_input_load_and_save_s=time.perf_counter()-begin,
        samples=args.samples, replicates=args.replicates, seed=17, radiance_samples=args.radiance_samples,
        interactions=1, sky_patches=145, radiance_sky_ab=2, radiance_sun_ab=1,
        threads=dict(radiance=1, native_host_transport=1, embree_build="SDK default, not pinned",
                     numpy_environment={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS")}),
        timing_policy="sequential backends; process cold, driver/OS caches not flushed; inputs resident; no validation layers",
        repeats=args.repeats, backends={})
    expected, apply, report["radiance"] = reference_build(
        out/"reference", polygons, points, normals, sun, sky, dni, args.radiance_samples,scene_source)
    report["radiance"]["warm_application_s"] = [timed(apply)[1] for _ in range(args.repeats)]
    report["radiance"]["warm_application_median_s"] = statistics.median(report["radiance"]["warm_application_s"])
    edited_polygons, edited_points, edited_expected = None, None, None
    if args.edited:
        shift = np.array([.375 if args.scope=="canopy" else .2, 0, 0])
        edited_polygons = ({"canopy":None} if args.scope=="canopy" else
            {name: np.asarray(poly)+(shift if name!="ground" else 0) for name,poly in polygons.items()})
        edited_points = points.copy()
        if args.scope!="canopy": edited_points[:20] += shift
        edited_expected, _, report["radiance"]["edited"] = reference_build(
            out/"reference-edited", edited_polygons, edited_points, normals, sun, sky, dni, args.radiance_samples,
            (lambda: canopy_source(vertices+shift)) if args.scope=="canopy" else None)
    # Native geometry initialization/update durations instrument existing calls,
    # without replacing the traversal or changing solver behavior.
    init, update = NativeGeometry.__init__, NativeGeometry.update
    geometry_times, update_times = [], []

    def observed_init(instance, *positional, **keywords):
        _, seconds = timed(lambda: init(instance, *positional, **keywords))
        geometry_times.append(seconds)

    def observed_update(instance, *positional, **keywords):
        value, seconds = timed(lambda: update(instance, *positional, **keywords))
        update_times.append(seconds)
        return value

    scope = os.add_dll_directory(str(Path(args.embree_dll_dir).resolve())) if os.name=="nt" else None
    try:
        with patch.object(NativeGeometry, "__init__", observed_init), patch.object(NativeGeometry, "update", observed_update):
            for backend in ("embree", "vulkan-rt"):
                current_scene = canopy(256) if args.scope=="canopy" else cube_fixture()[0]
                options = dict(dni=dni, sky=sky, samples=args.samples, replicates=args.replicates,
                               seed=17, bounces=1, durations_hours=durations)
                geometry_times.clear()
                update_times.clear()
                begin = time.perf_counter()
                solver = ShortwaveSolver(current_scene, materials=materials, backend=backend)
                try:
                    first = solver.solve(points, normals, sun, **options)
                    cold_s = time.perf_counter()-begin
                    record = dict(cold_first_combined_s=cold_s, native_geometry_init_s=sum(geometry_times), warm_s=[])
                    for _ in range(args.repeats):
                        result, seconds = timed(lambda: solver.solve(points, normals, sun, **options))
                        record["warm_s"].append(seconds)
                        np.testing.assert_array_equal(first.irradiance, result.irradiance)
                    record["warm_median_s"] = statistics.median(record["warm_s"])
                    record["accuracy"] = metrics(aggregate(first.total,labels)[...,0],
                        aggregate(expected,labels)[...,0], sun[:,2]>0, durations)
                    difference = first.total-expected
                    record["accuracy"]["max_individual_sensor_error_w_m2"] = float(abs(difference).max())
                    record["accuracy"]["individual_sensor_rmse_w_m2"] = float(np.sqrt(np.mean(difference**2)))
                    if args.edited:
                        transform = np.eye(4)
                        transform[0,3] = shift[0]
                        begin = time.perf_counter()
                        for name in polygons:
                            if name!="ground": current_scene.update_transform(name, transform)
                        record["scene_edit_commit_s"] = time.perf_counter()-begin
                        edited, seconds = timed(lambda: solver.solve(edited_points,normals,sun,**options))
                        record["edited_solve_s"] = seconds
                        record["edited_update_plus_solve_s"] = seconds+record["scene_edit_commit_s"]
                        record["native_geometry_update_s"] = sum(update_times)
                        record["edited_accuracy"] = metrics(aggregate(edited.total,labels)[...,0],
                            aggregate(edited_expected,labels)[...,0],sun[:,2]>0,durations)
                        difference = edited.total-edited_expected
                        record["edited_accuracy"]["max_individual_sensor_error_w_m2"] = float(abs(difference).max())
                        record["edited_accuracy"]["individual_sensor_rmse_w_m2"] = float(np.sqrt(np.mean(difference**2)))
                        with ShortwaveSolver(current_scene,materials=materials,backend=backend) as fresh:
                            fresh_result = fresh.solve(edited_points,normals,sun,**options)
                        record["edited_fresh_max_abs_w_m2"] = float(abs(edited.total-fresh_result.total).max())
                        np.testing.assert_allclose(edited.total,fresh_result.total,atol=1e-5,rtol=1e-8)
                    report["backends"][backend] = record
                finally:
                    solver.close()
                (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    finally:
        if scope is not None: scope.close()
    failures = []
    for backend,record in report["backends"].items():
        for case in ("accuracy", "edited_accuracy"):
            if case not in record: continue
            passed = (np.max(np.abs(record[case]["annual_relative_error"][:5]))<=.02 and
                      np.max(record[case]["daylight_nrmse"][:5])<=.05)
            record[case]["exposed_faces_passed"] = bool(passed)
            if not passed: failures.append([backend,case])
    report["accuracy_failures"] = failures
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if failures: raise SystemExit(2)


def write_summary(output):
    """Summarize successful independent trials; never pool unlike workloads."""
    output = Path(output)
    trials = json.loads((output/"trials.json").read_text())
    if not trials or any(t["returncode"] for t in trials):
        raise ValueError("all requested trials must complete successfully")
    reports = [json.loads(Path(t["report"]).read_text()) for t in trials]
    first = reports[0]
    if any((r["scope"],r["samples"],r["replicates"],r["radiance_samples"],r["repeats"],r["native_sha256"])
           != (first["scope"],first["samples"],first["replicates"],first["radiance_samples"],first["repeats"],first["native_sha256"])
           for r in reports):
        raise ValueError("trial settings or native binaries differ")

    def stats(values):
        return dict(median_s=statistics.median(values),min_s=min(values),max_s=max(values),samples_s=values)

    summary = dict(scope=first["scope"],intervals=first["intervals"],sensors=first["sensors"],
        triangles=first["triangles"],independent_process_trials=len(reports),warm_repeats_per_trial=first["repeats"],
        samples=first["samples"],replicates=first["replicates"],radiance_samples=first["radiance_samples"],
        radiance={},backends={},ratio_definition="median Radiance wall time / median native wall time; >1 favors native",
        accuracy_failures=[x for r in reports for x in r["accuracy_failures"]],report_files=[t["report"] for t in trials])
    for name in ("geometry_octree_s","coefficient_build_s","first_result_s","warm_application_median_s"):
        summary["radiance"][name] = stats([r["radiance"][name] for r in reports])
    edited = "edited" in first["radiance"]
    if edited:
        summary["radiance"]["edited_first_result_s"] = stats([r["radiance"]["edited"]["first_result_s"] for r in reports])
    for backend in first["backends"]:
        record = {}
        for name in ("native_geometry_init_s","cold_first_combined_s","warm_median_s"):
            record[name] = stats([r["backends"][backend][name] for r in reports])
        if edited:
            for name in ("edited_update_plus_solve_s","native_geometry_update_s"):
                record[name] = stats([r["backends"][backend][name] for r in reports])
            record["edited_fresh_max_abs_w_m2"] = max(r["backends"][backend]["edited_fresh_max_abs_w_m2"] for r in reports)
        record["radiance_over_native"] = dict(
            cold_first_result=summary["radiance"]["first_result_s"]["median_s"]/record["cold_first_combined_s"]["median_s"],
            warm_evaluation=summary["radiance"]["warm_application_median_s"]["median_s"]/record["warm_median_s"]["median_s"])
        if edited:
            record["radiance_over_native"]["edited_result"] = summary["radiance"]["edited_first_result_s"]["median_s"]/record["edited_update_plus_solve_s"]["median_s"]
        record["max_group_exposure_relative_error"] = max(abs(v) for r in reports for v in r["backends"][backend]["accuracy"]["annual_relative_error"][:5])
        record["max_group_daylight_nrmse"] = max(v for r in reports for v in r["backends"][backend]["accuracy"]["daylight_nrmse"][:5])
        summary["backends"][backend] = record
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope",choices=("quick","annual","canopy"),default="quick")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--forcing",type=Path,default=Path("build/annual-validation/candidate-annual/forcing.npz"))
    parser.add_argument("--samples",type=int,default=2048)
    parser.add_argument("--replicates",type=int,default=2)
    parser.add_argument("--radiance-samples",type=int,default=4096)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--trials",type=int,default=1)
    parser.add_argument("--edited",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--embree-dll-dir",default=".local/embree-4.4.1/embree-4.4.1.x64.windows/bin")
    args = parser.parse_args()
    if min(args.samples,args.replicates,args.radiance_samples,args.repeats,args.trials)<1:
        parser.error("sample and repeat counts must be positive")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    reports = []
    for trial in range(args.trials):
        command = [sys.executable,str(Path(__file__).resolve()),"--worker","--scope",args.scope,
            "--output",str(args.output/f"trial-{trial}"),"--forcing",str(args.forcing),
            "--samples",str(args.samples),"--replicates",str(args.replicates),
            "--radiance-samples",str(args.radiance_samples),"--repeats",str(args.repeats),
            "--embree-dll-dir",args.embree_dll_dir]+(["--edited"] if args.edited else [])
        begin = time.perf_counter()
        result = subprocess.run(command,capture_output=True)
        (args.output/f"trial-{trial}.stdout").write_bytes(result.stdout)
        (args.output/f"trial-{trial}.stderr").write_bytes(result.stderr)
        reports.append(dict(command=command,process_elapsed_s=time.perf_counter()-begin,returncode=result.returncode,
                            report=str(args.output/f"trial-{trial}/report.json")))
        (args.output/"trials.json").write_text(json.dumps(reports,indent=2)+"\n")
        if result.returncode:
            sys.stderr.buffer.write(result.stderr)
            raise SystemExit(result.returncode)
    write_summary(args.output)
    print(json.dumps(reports,indent=2))


if __name__=="__main__":
    main()
