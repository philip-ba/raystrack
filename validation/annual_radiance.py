"""Annual radiometric cube comparison using explicit Radiance coefficients.

Outputs must go under ignored build/. The default evaluates all 8760 hours.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from raystrack import Mesh, Scene
from raystrack.shortwave import (
    Opaque,
    PatchSkySeries,
    ReinhartSkyBasis,
    ShortwaveSolver,
    SolarSite,
    SolarTimeline,
    load_radiance_diffuse_sky,
)


def cube_fixture(samples_per_edge=2):
    """Metres, +Y north, outward faces, 20x20 m finite ground, black below."""
    polygons = {
        "top": [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]],
        "north": [[-0.5, 0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 1], [0.5, 0.5, 0]],
        "east": [[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, -0.5, 1]],
        "south": [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, 1], [-0.5, -0.5, 1]],
        "west": [[-0.5, -0.5, 0], [-0.5, -0.5, 1], [-0.5, 0.5, 1], [-0.5, 0.5, 0]],
        "base": [[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0]],
        "ground": [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
    }
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    meshes = {name: Mesh(np.asarray(poly), faces) for name, poly in polygons.items()}
    points, normals, labels = [], [], []
    q = (np.arange(samples_per_edge) + 0.5) / samples_per_edge
    for name, poly in list(polygons.items())[:5]:
        p = np.asarray(poly, dtype=float)
        n = np.cross(p[1] - p[0], p[3] - p[0])
        n /= np.linalg.norm(n)
        for a in q:
            for b in q:
                points.append(p[0] + a * (p[1] - p[0]) + b * (p[3] - p[0]) + 1e-4 * n)
                normals.append(n)
                labels.append(name)
    for name, point in (
        ("ground_south", [0, -1, 1e-4]),
        ("ground_north", [0, 1, 1e-4]),
        ("open", [0, 0, 3]),
    ):
        points.append(point)
        normals.append([0, 0, 1])
        labels.append(name)
    return (
        Scene.from_meshes(meshes),
        polygons,
        np.asarray(points),
        np.asarray(normals),
        labels,
    )


def synthetic_forcing():
    timeline = SolarTimeline.for_year(
        2025, SolarSite(52.52, 13.405), utc_offset_minutes=60
    )
    z = np.maximum(timeline.directions[:, 2], 0)
    day = np.arange(8760) / 24
    clarity = 0.65 + 0.2 * np.sin(2 * np.pi * day / 9.7) * np.sin(2 * np.pi * day / 31)
    season = 0.85 + 0.15 * np.cos(2 * np.pi * (day - 172) / 365)
    dni = np.where(
        z > 0, 950 * np.exp(-0.16 / np.maximum(z, 0.01)) * clarity * season, 0
    )
    dhi = np.where(z > 0, (45 + 100 * (1 - clarity)) * z**0.45 * season, 0)
    return timeline, dni, dhi


def polygon_text(name, vertices, material):
    return (
        f"{material} polygon {name}\n0\n0\n{3*len(vertices)}\n"
        + "\n".join(" ".join(f"{v:.10g}" for v in p) for p in vertices)
        + "\n"
    )


class Radiance:
    def __init__(self, directory, output):
        self.directory = Path(directory).resolve()
        self.output = Path(output).resolve()
        self.commands = []
        self.environment = os.environ.copy()
        self.environment["PATH"] = (
            str(self.directory) + os.pathsep + self.environment.get("PATH", "")
        )
        self.environment["RAYPATH"] = (
            "." + os.pathsep + str(self.directory.parent / "lib")
        )

    def run(self, executable, arguments, *, data=None, target=None):
        command = [
            str(self.directory / (executable + (".exe" if os.name == "nt" else "")))
        ] + list(map(str, arguments))
        start = time.perf_counter()
        result = subprocess.run(
            command,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.output,
            env=self.environment,
            check=False,
        )
        self.commands.append(
            {
                "argv": command,
                "seconds": time.perf_counter() - start,
                "returncode": result.returncode,
                "stderr": result.stderr.decode(errors="replace"),
                "stdout_bytes": len(result.stdout),
                "stdout_file": target,
                "stdin_sha256": (
                    hashlib.sha256(data).hexdigest() if data is not None else None
                ),
            }
        )
        (self.output / "radiance_commands.json").write_text(
            json.dumps(self.commands, indent=2)
        )
        if result.returncode:
            raise RuntimeError(f"{command!r}\n{self.commands[-1]['stderr']}")
        if target:
            (self.output / target).write_bytes(result.stdout)
        return result.stdout


def write_weather(output, timeline, dni, dhi):
    local = timeline.timestamps + np.timedelta64(60, "m")
    lines = [
        "place Berlin_synthetic_2025",
        "latitude 52.52",
        "longitude -13.405",
        "time_zone -15",
        "site_elevation 34",
        "weather_data_file_units 1",
    ]
    for stamp, direct, diffuse in zip(local, dni, dhi):
        s = str(stamp)
        lines.append(
            f"{int(s[5:7])} {int(s[8:10])} {int(s[11:13])+.5:.1f} {direct:.9g} {diffuse:.9g}"
        )
    (output / "synthetic.wea").write_text("\n".join(lines) + "\n", encoding="ascii")


def rcontrib(rad, octree, sensors, modifiers, bins, *, bounces, samples, sky=False):
    args = [
        "-I+",
        "-V+",
        "-h",
        "-faa",
        "-n",
        1,
        "-u-",
        "-ab",
        bounces,
        "-ad",
        samples,
        "-as",
        0,
        "-aa",
        0,
        "-lw",
        "1e-8",
        "-lr",
        6,
        "-dt",
        0,
        "-dc",
        1,
        "-dj",
        0,
        "-ds",
        0,
    ]
    if sky:
        args += [
            "-e",
            "MF:1",
            "-f",
            "reinhart.cal",
            "-b",
            "rbin",
            "-bn",
            "Nrbins",
            "-m",
            "sky_glow",
        ]
    else:
        args += ["-M", modifiers]
    data = rad.run("rcontrib", args + [octree], data=sensors)
    values = np.fromstring(data.decode("ascii"), sep=" ")
    count = len(sensors.splitlines())
    if values.size != count * bins * 3:
        raise RuntimeError(f"Coefficient size {values.size} != {count*bins*3}")
    return values.reshape(count, bins, 3)


def reference(rad, polygons, points, normals, directions, sky, samples, bounces):
    output = rad.output
    scene = "void plastic neutral\n0\n0\n5 .2 .2 .2 0 0\n"
    scene += "".join(
        polygon_text(name, poly, "neutral") for name, poly in polygons.items()
    )
    (output / "scene.rad").write_text(scene)
    sensors = "".join(
        " ".join(f"{v:.10g}" for v in row) + "\n" for row in np.c_[points, normals]
    ).encode()
    (output / "sensors.pts").write_bytes(sensors)
    (output / "sky.rad").write_text(
        "void glow sky_glow\n0\n0\n4 1 1 1 0\nsky_glow source sky\n0\n0\n4 0 0 1 180\n"
    )
    rad.run("oconv", ["scene.rad", "sky.rad"], target="sky.oct")
    coeff = rcontrib(
        rad,
        "sky.oct",
        sensors,
        None,
        146,
        bounces=bounces + 1,
        samples=samples,
        sky=True,
    )[:, 1:, :]
    np.save(output / "radiance_sky_coefficients.npy", coeff)
    start = time.perf_counter()
    diffuse = np.einsum("spr,tpr->tsr", coeff, sky.radiance)
    diffuse_apply = time.perf_counter() - start
    active = np.flatnonzero(directions[:, 2] > 0)
    diameter = 0.01
    omega = np.pi * np.sin(np.radians(diameter / 2)) ** 2
    lines = []
    modifiers = []
    for j, step in enumerate(active):
        name = f"sun{j}"
        modifiers.append(name)
        lines += [
            f"void light {name}\n0\n0\n3 {1/omega:.12g} {1/omega:.12g} {1/omega:.12g}\n",
            f"{name} source s{j}\n0\n0\n4 "
            + " ".join(f"{v:.12g}" for v in directions[step])
            + f" {diameter}\n",
        ]
    (output / "suns.rad").write_text("".join(lines))
    (output / "suns.mod").write_text("\n".join(modifiers) + "\n")
    rad.run("oconv", ["scene.rad", "suns.rad"], target="sun.oct")
    dc = rcontrib(
        rad,
        "sun.oct",
        sensors,
        "suns.mod",
        len(active),
        bounces=bounces,
        samples=samples,
    )
    np.save(output / "radiance_sun_coefficients.npy", dc)
    direct = np.zeros((len(directions), len(points), 3))
    direct[active] = dc.transpose(1, 0, 2)
    dc0 = rcontrib(
        rad, "sun.oct", sensors, "suns.mod", len(active), bounces=0, samples=1
    )
    primary = np.zeros_like(direct)
    primary[active] = dc0.transpose(1, 0, 2)
    return (
        direct,
        diffuse,
        primary,
        {
            "diffuse_matrix_application_s": diffuse_apply,
            "sun_diameter_deg": diameter,
            "sun_source_count": len(active),
        },
    )


def reuse_reference(source, output, polygons, samples, bounces, *, rad=None):
    """Reuse expensive immutable coefficients only after checking the full input."""
    source = Path(source).resolve()
    if source == output:
        raise ValueError("reference input and output directories must differ")
    report = json.loads((source / "report.json").read_text())
    if report["radiance_samples"] != samples or report["bounces"] != bounces:
        raise ValueError("cached reference samples/bounces differ")
    with (
        np.load(source / "forcing.npz") as old,
        np.load(output / "forcing.npz") as current,
    ):
        if set(old.files) != set(current.files):
            raise ValueError("cached forcing fields differ")
        for name in old.files:
            np.testing.assert_array_equal(
                old[name], current[name], err_msg=f"cached reference {name}"
            )
    scene = "void plastic neutral\n0\n0\n5 .2 .2 .2 0 0\n"
    scene += "".join(
        polygon_text(name, poly, "neutral") for name, poly in polygons.items()
    )
    if (source / "scene.rad").read_text() != scene:
        raise ValueError("cached reference geometry differs")
    hashes = report.get("files_sha256", {})
    required = ("scene.rad", "sensors.pts", "suns.rad", "radiance_sky_coefficients.npy",
                "radiance_sun_coefficients.npy")
    if any(name not in hashes for name in required):
        raise ValueError("cached reference lacks required input/coefficient hashes")
    for name, digest in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"cached reference file hash differs: {name}")
    for name in (
        "scene.rad",
        "sensors.pts",
        "suns.rad",
        "radiance_sky_coefficients.npy",
        "radiance_sun_coefficients.npy",
    ):
        shutil.copy2(source / name, output / name)
    with np.load(output / "forcing.npz", allow_pickle=False) as forcing:
        directions, sky = forcing["sun"], forcing["sky"]
    active = np.flatnonzero(directions[:, 2] > 0)
    sky_coeff = np.load(output / "radiance_sky_coefficients.npy", allow_pickle=False)
    sun_coeff = np.load(output / "radiance_sun_coefficients.npy", allow_pickle=False)
    if (sky_coeff.ndim != 3 or sky_coeff.shape[1:] != sky.shape[1:]
            or sun_coeff.shape != (len(sky_coeff), len(active), 3)
            or not np.isfinite(sky_coeff).all() or not np.isfinite(sun_coeff).all()):
        raise ValueError("cached reference coefficient shape/values differ from forcing")
    direct = np.zeros((len(directions), len(sky_coeff), 3))
    direct[active] = sun_coeff.transpose(1, 0, 2)
    diffuse = np.einsum("spr,tpr->tsr", sky_coeff, sky)
    # Old reports did not hash primary coefficients. Recompute this inexpensive
    # diagnostic from verified geometry, sources and sensors, never hourly data.
    (output / "suns.mod").write_text("\n".join(f"sun{i}" for i in range(len(active))) + "\n")
    primary = np.zeros_like(direct)
    if len(active):
        rad = Radiance(report["radiance_bin"], output) if rad is None else rad
        if rad.output != output:
            raise ValueError("reference runner output directory differs")
        rad.run("oconv", ["scene.rad", "suns.rad"], target="sun.oct")
        dc0 = rcontrib(rad, "sun.oct", (output / "sensors.pts").read_bytes(),
                      "suns.mod", len(active), bounces=0, samples=1)
        primary[active] = dc0.transpose(1, 0, 2)
    np.save(output / "radiance_primary_coefficients.npy", primary)
    return (
        direct,
        diffuse,
        primary,
        {
            **report["reference"],
            "reused_from": str(source),
            "original_radiance_commands": str(source / "radiance_commands.json"),
            "original_radiance_version": report["radiance_version"],
            "original_missing_fields": [
                key
                for key in ("native_library_sha256", "acceptance", "spot_checks")
                if key not in report
            ],
            "reuse_validation": "exact forcing/sensors/scene; required source/coefficient hashes; "
                "reconstructed sun/sky from verified coefficients; recomputed zero-bounce primary",
        },
    )


def aggregate(values, labels):
    return np.stack(
        [
            values[:, np.array(labels) == name].mean(axis=1)
            for name in dict.fromkeys(labels)
        ],
        axis=1,
    )


def metrics(actual, expected, daylight, durations):
    d = actual - expected
    ref = np.einsum("t,tf->f", durations, expected)
    exposure = np.einsum("t,tf->f", durations, actual)
    rmse_day = np.sqrt(np.mean(d[daylight] ** 2, axis=0))
    return {
        "mae_w_m2": np.mean(np.abs(d), axis=0).tolist(),
        "rmse_w_m2": np.sqrt(np.mean(d**2, axis=0)).tolist(),
        "bias_w_m2": d.mean(axis=0).tolist(),
        "daylight_rmse_w_m2": rmse_day.tolist(),
        "daylight_nrmse": (
            rmse_day / np.maximum(expected[daylight].mean(axis=0), 1)
        ).tolist(),
        "annual_wh_m2": exposure.tolist(),
        "reference_annual_wh_m2": ref.tolist(),
        "annual_relative_error": ((exposure - ref) / np.maximum(ref, 1)).tolist(),
        "near_zero_mae_w_m2": (
            float(np.mean(np.abs(d[expected < 1]))) if np.any(expected < 1) else 0.0
        ),
    }


def acceptance_report(report):
    """Apply predeclared gates to each exposed cube face, keeping modes separate."""
    gates = []
    incomplete = []
    faces = ("top", "north", "east", "south", "west")
    groups = report.get("groups", [])
    if not report.get("backends"):
        incomplete.append("no completed backends")
    requested = report.get("requested_backends", list(report.get("backends", {})))
    for backend in requested:
        if backend not in report.get("backends", {}):
            incomplete.append(f"{backend}: requested backend not completed")
    if any(face not in groups for face in faces):
        incomplete.append("missing exposed cube face")
    for backend, evidence in report.get("backends", {}).items():
        for mode in ("direct", "diffuse", "combined"):
            result = evidence.get("metrics", {}).get(mode)
            if result is None:
                incomplete.append(f"{backend}: missing {mode}")
                continue
            for name in faces:
                if name not in groups:
                    continue
                i = groups.index(name)
                if any(i >= len(result.get(key, [])) for key in
                       ("annual_relative_error", "daylight_nrmse")):
                    incomplete.append(f"{backend}/{mode}: missing {name} metrics")
                    continue
                annual = abs(result["annual_relative_error"][i])
                nrmse = result["daylight_nrmse"][i]
                gates.append(
                    {
                        "backend": backend,
                        "mode": mode,
                        "face": name,
                        "annual_relative_error": annual,
                        "daylight_nrmse": nrmse,
                        "passed": bool(annual <= 0.02 and nrmse <= 0.05),
                    }
                )
    return {"all_passed": not incomplete and all(g["passed"] for g in gates),
            "face_mode_gates": gates, "incomplete": incomplete}


def supplement_saved_run(directory):
    """Add reporting evidence to an earlier completed run without retracing."""
    output = Path(directory).resolve()
    report = json.loads((output / "report.json").read_text())
    supplement = {"acceptance": acceptance_report(report), "confidence_intervals": {}}
    for backend, evidence in report["backends"].items():
        convergence = evidence.get("convergence", {})
        bound = convergence.get("annual_one_se_conservative_bound_wh_m2")
        if bound is not None and report["native_replicates"] == 2:
            mean = np.asarray(evidence["metrics"]["combined"]["annual_wh_m2"])
            half_width = 12.706204736 * np.asarray(bound)
            supplement["confidence_intervals"][backend] = {
                "groups": report["groups"],
                "annual_95_interval_conservative_wh_m2": np.column_stack(
                    (mean - half_width, mean + half_width)
                ).tolist(),
                "assumptions": "Student t(1), approximately normal independent replicate means; "
                    "linear covariance bound across components, sensor points and hours; "
                    "two replicates give weak interval precision.",
            }
    if "embree" in report["backends"] and "vulkan-rt" in report["backends"]:
        with np.load(output / "embree_hourly.npz") as cpu, np.load(output / "vulkan-rt_hourly.npz") as gpu:
            supplement["backend_parity"] = {
                mode + "_max_abs_w_m2": float(np.max(np.abs(cpu[mode] - gpu[mode])))
                for mode in ("direct", "diffuse", "combined")
            }
    supplement["files_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("forcing.npz", "radiance_sky_coefficients.npy", "radiance_sun_coefficients.npy")
    }
    (output / "validation-supplement.json").write_text(json.dumps(supplement, indent=2))
    return supplement


def diagnostic_indices(directions, dni, primary, labels):
    """Low sun, high sun and an actual ground shadow transition."""
    daylight = np.flatnonzero(dni > 50)
    low = daylight[np.argmin(np.abs(directions[daylight, 2] - np.sin(np.radians(5))))]
    high = int(np.argmax(directions[:, 2]))
    ground = labels.index("ground_north")
    illuminated = primary[:, ground, 0] > 1
    changes = np.flatnonzero(illuminated[1:] != illuminated[:-1]) + 1
    changes = changes[(dni[changes] > 100) & (dni[np.maximum(changes - 1, 0)] > 100)]
    transition = (
        int(changes[len(changes) // 2])
        if len(changes)
        else int(daylight[len(daylight) // 2])
    )
    return list(dict.fromkeys([int(low), transition, high]))


def spot_checks(rad, directions, dni, sky, expected, labels, samples, bounces):
    """Independent rtrace reconstruction of separate sky and direct sources."""
    checks = []
    sensor_data = (rad.output / "sensors.pts").read_bytes()
    cal = (rad.directory.parent / "lib" / "reinhart.cal").read_text()
    indices = diagnostic_indices(directions, dni, expected["primary"], labels)
    options = [
        "-I+",
        "-h",
        "-ov",
        "-n",
        1,
        "-u-",
        "-ad",
        samples,
        "-as",
        0,
        "-aa",
        0,
        "-lw",
        "1e-8",
        "-lr",
        6,
        "-dt",
        0,
        "-dc",
        1,
        "-dj",
        0,
        "-ds",
        0,
    ]
    for i in indices:
        values = ",".join(f"{v:.12g}" for v in sky.radiance[i, :, 0])
        (rad.output / "spot.cal").write_text(
            "MF:1;\n" + cal + f"\nskybright=select(rbin+1,0,{values});\n"
        )
        (rad.output / "spot-sky.rad").write_text(
            "void brightfunc skyfunc\n2 skybright spot.cal\n0\n0\n"
            "skyfunc glow spotglow\n0\n0\n4 1 1 1 0\nspotglow source spotsky\n0\n0\n4 0 0 1 180\n"
        )
        rad.run("oconv", ["scene.rad", "spot-sky.rad"], target="spot-sky.oct")
        sky_out = rad.run(
            "rtrace",
            options + ["-ab", bounces + 1, "spot-sky.oct"],
            data=sensor_data,
            target=f"spot-{i}-diffuse.txt",
        )
        omega = np.pi * np.sin(np.radians(0.01 / 2)) ** 2
        radiance = dni[i] / omega
        direction = " ".join(f"{v:.12g}" for v in directions[i])
        (rad.output / "spot-sun.rad").write_text(
            f"void light spotsun\n0\n0\n3 {radiance} {radiance} {radiance}\n"
            f"spotsun source spotsource\n0\n0\n4 {direction} .01\n"
        )
        rad.run("oconv", ["scene.rad", "spot-sun.rad"], target="spot-sun.oct")
        sun_out = rad.run(
            "rtrace",
            options + ["-ab", bounces, "spot-sun.oct"],
            data=sensor_data,
            target=f"spot-{i}-direct.txt",
        )
        actual_diffuse = np.fromstring(sky_out.decode(), sep=" ").reshape(-1, 3)
        actual_direct = np.fromstring(sun_out.decode(), sep=" ").reshape(-1, 3)
        checks.append(
            {
                "index": i,
                "sun_elevation_deg": float(np.degrees(np.arcsin(directions[i, 2]))),
                "direct_max_abs_w_m2": float(
                    np.max(np.abs(actual_direct - expected["direct"][i]))
                ),
                "diffuse_max_abs_w_m2": float(
                    np.max(np.abs(actual_diffuse - expected["diffuse"][i]))
                ),
                "rtrace_direct_group_w_m2": aggregate(actual_direct[None], labels)[
                    0, :, 0
                ].tolist(),
                "rtrace_diffuse_group_w_m2": aggregate(actual_diffuse[None], labels)[
                    0, :, 0
                ].tolist(),
            }
        )
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Path("build/annual-validation/annual")
    )
    p.add_argument("--radiance-bin", default="C:/Radiance/bin")
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--radiance-samples", type=int, default=4096)
    p.add_argument("--bounces", type=int, default=1)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument(
        "--quick", action="store_true", help="12 diagnostic steps; full year is default"
    )
    p.add_argument("--spot-samples", type=int, default=32768)
    p.add_argument("--skip-spots", action="store_true")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="retain failures in report without nonzero exit",
    )
    p.add_argument(
        "--reuse-reference",
        type=Path,
        help="reuse existing Radiance result after exact input checks",
    )
    p.add_argument("--backends", nargs="+", default=["embree", "vulkan-rt"])
    p.add_argument(
        "--embree-dll-dir", default=".local/embree-4.4.1/embree-4.4.1.x64.windows/bin"
    )
    a = p.parse_args()
    if min(a.samples, a.radiance_samples, a.replicates) < 1 or not 0 <= a.bounces <= 4:
        p.error("positive samples/replicates and bounces 0..4 required")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rad = Radiance(a.radiance_bin, out)
    version = rad.run("rtrace", ["-version"]).decode().strip()
    scene, polygons, points, normals, labels = cube_fixture()
    timeline, dni, dhi = synthetic_forcing()
    write_weather(out, timeline, dni, dhi)
    rad.run(
        "gendaymtx",
        ["-s", "-O1", "-m", 1, "-c", 1, 1, 1, "-g", 0, 0, 0, "synthetic.wea"],
        target="diffuse.mtx",
    )
    full_sky = load_radiance_diffuse_sky(out / "diffuse.mtx", ReinhartSkyBasis())
    selection = np.arange(8760)
    if a.quick:
        selection = np.array(
            [24 * d + h for d in (15, 79, 171, 264) for h in (7, 11, 16)]
        )
    directions = timeline.directions[selection]
    dni = dni[selection]
    dhi = dhi[selection]
    sky = PatchSkySeries(full_sky.basis, full_sky.radiance[selection])
    timestamps = timeline.timestamps[selection]
    durations = timeline.durations_hours[selection]
    np.savez_compressed(
        out / "forcing.npz",
        timestamps=timestamps,
        durations_hours=durations,
        sun=directions,
        dni=dni,
        dhi=dhi,
        sky=sky.radiance,
        points=points,
        normals=normals,
        labels=np.asarray(labels),
    )
    setup_s = time.perf_counter() - start
    if a.reuse_reference:
        dc, diffuse, primary_coeff, info = reuse_reference(
            a.reuse_reference, out, polygons, a.radiance_samples, a.bounces, rad=rad
        )
    else:
        dc, diffuse, primary_coeff, info = reference(
            rad,
            polygons,
            points,
            normals,
            directions,
            sky,
            a.radiance_samples,
            a.bounces,
        )
    direct = dc * dni[:, None, None]
    primary = primary_coeff * dni[:, None, None]
    np.save(out / "radiance_primary_coefficients.npy", primary_coeff)
    expected = {"direct": direct, "diffuse": diffuse, "combined": direct + diffuse}
    names = list(dict.fromkeys(labels))
    report = {
        "schema": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "numpy": np.__version__,
        "argv": sys.argv,
        "native_library": os.environ.get("RAYSTRACK_NATIVE_LIBRARY"),
        "native_library_sha256": hashlib.sha256(
            Path(os.environ["RAYSTRACK_NATIVE_LIBRARY"]).read_bytes()
        ).hexdigest(),
        "threads": {
            "radiance": 1,
            "native_host_transport": 1,
            "embree_build": "SDK default",
        },
        "radiance_version": version,
        "radiance_bin": str(rad.directory),
        "intervals": len(selection),
        "forcing": "synthetic deterministic Berlin2025; not measured weather",
        "year": 2025,
        "latitude_deg": 52.52,
        "longitude_deg": 13.405,
        "utc_offset_minutes": 60,
        "midpoints": True,
        "units": {"irradiance": "W/m2", "exposure": "Wh/m2"},
        "north": "+Y",
        "ground_extent_m": [-10, 10],
        "albedo": 0.2,
        "lower_environment": "black",
        "buried_face": "excluded; coincident sealed ground contact",
        "face_sensor_count": 4,
        "sensor_offset_m": 1e-4,
        "ray_offset_m": 1e-5,
        "native_samples": a.samples,
        "native_replicates": a.replicates,
        "radiance_samples": a.radiance_samples,
        "bounces": a.bounces,
        "radiance_sky_ab": a.bounces + 1,
        "radiance_sun_ab": a.bounces,
        "radiance_ambient": {"aa": 0, "as": 0, "lw": 1e-8, "u": False, "n": 1},
        "groups": names,
        "targets": {
            "annual_relative": 0.02,
            "daylight_nrmse": 0.05,
            "near_zero_floor_w_m2": 1,
            "deterministic_relative": 1e-5,
        },
        "setup_s": setup_s,
        "reference": info,
        "requested_backends": list(a.backends),
        "backends": {},
    }
    np.savez_compressed(out / "radiance_hourly.npz", **expected, primary=primary)
    if not a.skip_spots:
        report["spot_checks"] = spot_checks(
            rad,
            directions,
            dni,
            sky,
            {**expected, "primary": primary},
            labels,
            a.spot_samples,
            a.bounces,
        )
        for check in report["spot_checks"]:
            check["timestamp_utc"] = str(timestamps[check["index"]])
    scope = (
        os.add_dll_directory(str(Path(a.embree_dll_dir).resolve()))
        if os.name == "nt"
        else nullcontext()
    )
    with scope:
        for backend in a.backends:
            timing = {}
            actual = {}
            with ShortwaveSolver(
                scene,
                materials={name: Opaque(0.2) for name in polygons},
                backend=backend,
            ) as solver:
                for mode in ("direct", "diffuse", "combined"):
                    begin = time.perf_counter()
                    result = solver.solve(
                        points,
                        normals,
                        directions,
                        dni=0 if mode == "diffuse" else dni,
                        sky=(
                            sky
                            if mode != "direct"
                            else PatchSkySeries(sky.basis, np.zeros_like(sky.radiance))
                        ),
                        samples=a.samples,
                        bounces=a.bounces,
                        seed=17,
                        replicates=a.replicates,
                        durations_hours=durations,
                        forcing_id="Berlin2025-synthetic-annual-radiance",
                    )
                    timing[mode + "_s"] = time.perf_counter() - begin
                    actual[mode] = result.total
                begin = time.perf_counter()
                warm = solver.solve(
                    points,
                    normals,
                    directions,
                    dni=dni,
                    sky=sky,
                    samples=a.samples,
                    bounces=a.bounces,
                    seed=17,
                    replicates=a.replicates,
                )
                timing["combined_warm_s"] = time.perf_counter() - begin
                np.testing.assert_array_equal(warm.total, actual["combined"])
                se = result.standard_error
                diagnostic = diagnostic_indices(directions, dni, primary, labels)
                converged = solver.solve(
                    points,
                    normals,
                    directions[diagnostic],
                    dni=dni[diagnostic],
                    sky=PatchSkySeries(sky.basis, sky.radiance[diagnostic]),
                    samples=a.samples * 4,
                    bounces=a.bounces,
                    seed=1009,
                    replicates=a.replicates,
                )
                refinement = aggregate(converged.total, labels)[..., 0]
                base = aggregate(actual["combined"][diagnostic], labels)[..., 0]
            np.savez_compressed(
                out / f"{backend}_hourly.npz", **actual, standard_error=se
            )
            evidence = {"timing": timing, "metrics": {}, "worst": {}}
            evidence["convergence"] = {
                "indices": diagnostic,
                "seed": 1009,
                "samples": a.samples * 4,
                "refined_combined_w_m2": refinement.tolist(),
                "base_to_refined_difference_w_m2": (refinement - base).tolist(),
                "max_component_standard_error_w_m2": (
                    float(np.max(se)) if se is not None else None
                ),
            }
            if se is not None:
                # Linear bound allows arbitrary covariance between reused hourly
                # transfer samples; quadrature over hours would be unjustified.
                bound = aggregate(se.sum(axis=2), labels)[..., 0]
                evidence["convergence"]["annual_one_se_conservative_bound_wh_m2"] = (
                    durations @ bound
                ).tolist()
            if se is not None and a.replicates == 2:
                # Student t(1), two-sided 95%. The linear covariance bound
                # avoids pretending reused hourly paths are independent.
                mean_exposure = durations @ aggregate(actual["combined"], labels)[..., 0]
                half_width = 12.706204736 * (durations @ bound)
                evidence["convergence"]["annual_95_interval_conservative_wh_m2"] = (
                    np.column_stack((mean_exposure - half_width, mean_exposure + half_width)).tolist()
                )
                evidence["convergence"]["confidence_interval_assumptions"] = (
                    "Student t with one degree of freedom; approximately normal independent "
                    "replicate means; conservative linear covariance bound across components, "
                    "sensor points and hours. Two replicates give weak interval precision."
                )
            for mode in actual:
                av = aggregate(actual[mode], labels)[..., 0]
                ev = aggregate(expected[mode], labels)[..., 0]
                evidence["metrics"][mode] = metrics(
                    av, ev, directions[:, 2] > 0, durations
                )
                error = av - ev
                evidence.setdefault("aggregate_metrics", {})[mode] = {
                    "exposed_faces_mae_w_m2": float(np.mean(np.abs(error[:, :5]))),
                    "exposed_faces_rmse_w_m2": float(
                        np.sqrt(np.mean(error[:, :5] ** 2))
                    ),
                    "exposed_faces_bias_w_m2": float(np.mean(error[:, :5])),
                    "all_sensor_groups_mae_w_m2": float(np.mean(np.abs(error))),
                    "all_sensor_groups_rmse_w_m2": float(np.sqrt(np.mean(error**2))),
                    "all_sensor_groups_bias_w_m2": float(np.mean(error)),
                }
                err = np.abs(av - ev)
                ranked = np.argsort(err.ravel())[-20:][::-1]
                evidence["worst"][mode] = [
                    {
                        "timestamp_utc": str(timestamps[i]),
                        "group": names[j],
                        "actual": float(av[i, j]),
                        "reference": float(ev[i, j]),
                        "error_w_m2": float(err[i, j]),
                    }
                    for i, j in (np.unravel_index(k, err.shape) for k in ranked)
                ]
            evidence["linearity_max_abs_w_m2"] = float(
                np.max(
                    np.abs(actual["combined"] - actual["direct"] - actual["diffuse"])
                )
            )
            evidence["shadow_diagnostics"] = {}
            for name in ("ground_south", "ground_north"):
                j = labels.index(name)
                lit = (dni > 1) & (primary[:, j, 0] > 1)
                shadow = (dni > 1) & ~lit
                d = actual["combined"][:, j, 0] - expected["combined"][:, j, 0]
                evidence["shadow_diagnostics"][name] = {
                    "sunlit_hours": int(np.count_nonzero(lit)),
                    "shadowed_daylight_hours": int(np.count_nonzero(shadow)),
                    "sunlit_mae_w_m2": (
                        float(np.mean(np.abs(d[lit]))) if lit.any() else None
                    ),
                    "shadowed_mae_w_m2": (
                        float(np.mean(np.abs(d[shadow]))) if shadow.any() else None
                    ),
                }
            night = directions[:, 2] <= 0
            evidence["night_max_abs_w_m2"] = (
                float(np.max(np.abs(actual["combined"][night]))) if night.any() else 0.0
            )
            report["backends"][backend] = evidence
            (out / "report.json").write_text(json.dumps(report, indent=2))
    oi = labels.index("open")
    analytic = dni * np.maximum(directions[:, 2], 0)
    report["analytic"] = {
        "open_direct_max_abs_w_m2": float(np.max(np.abs(primary[:, oi, 0] - analytic))),
        "sky_horizontal_dhi_max_abs_w_m2": float(
            np.max(
                np.abs(sky.radiance[:, :, 0] @ sky.basis.projected_solid_angles - dhi)
            )
        ),
    }
    report["total_elapsed_s"] = time.perf_counter() - start
    report["files_sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in (
            "scene.rad",
            "sensors.pts",
            "synthetic.wea",
            "diffuse.mtx",
            "suns.rad",
            "forcing.npz",
            "radiance_sky_coefficients.npy",
            "radiance_sun_coefficients.npy",
            "radiance_primary_coefficients.npy",
            "radiance_hourly.npz",
        )
    }
    if "embree" in a.backends and "vulkan-rt" in a.backends:
        with (
            np.load(out / "embree_hourly.npz") as cpu,
            np.load(out / "vulkan-rt_hourly.npz") as gpu,
        ):
            report["backend_parity"] = {
                mode + "_max_abs_w_m2": float(np.max(np.abs(cpu[mode] - gpu[mode])))
                for mode in ("direct", "diffuse", "combined")
            }
    report["acceptance"] = acceptance_report(report)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["acceptance"]["all_passed"] and not a.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
