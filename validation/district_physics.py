"""Controlled selected-hour references for the moving district's materials.

These infinite-panel quadratures and shadow probes validate constituent paths,
not independent accuracy of the full annual district. No Radiance coefficients
or measured material values are altered to obtain equivalence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from raystrack import Mesh, Scene
from raystrack.shortwave import ClearGlazing, DiffuseSheet, KlemsBSDF, Opaque, ShortwaveSolver
from validation.validate_clear_glazing import plate_fresnel
from validation.validate_measured_bsdf import FIXTURE, inspect_fixture, quadrature


def panel(z, width=10000):
    return Mesh([[-width, -width, z], [width, -width, z],
                 [width, width, z], [-width, width, z]], [[0, 1, 2], [0, 2, 3]])


def _plate_patch_weights(order=64):
    """Integrate T_plate cos(theta) dOmega within each incident Klems cell."""
    bounds = [0, 5, 15, 25, 35, 45, 55, 65, 75, 90]
    counts = [1, 8, 16, 20, 24, 24, 24, 16, 12]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = []
    for low, high, count in zip(bounds[:-1], bounds[1:], counts):
        a, b = np.cos(np.radians([high, low]))
        cosine = a+(nodes+1)*(b-a)/2
        transmission = plate_fresnel(cosine)[1]
        integral = math.tau/count*(b-a)/2*np.dot(weights, cosine*transmission)
        result.extend([integral]*count)
    return np.array(result)


def selected_references(backend, *, samples=16384, replicates=4):
    centers, weights, _ = quadrature()
    weighted_plate = _plate_patch_weights()
    assert np.max(abs(weighted_plate-_plate_patch_weights(128))) < 1e-12
    # High sun, oblique sun, very low sun: all strictly inside named cells.
    directions = centers[[0, 69, 144]]*[-1, -1, 1]
    points = [[0, .17, 2.5], [0, .17, 0]]
    normals = [[0, 0, -1], [0, 0, 1]]
    sheet = DiffuseSheet(.2, .1, .4)
    materials = {
        "diffuse_sheet": sheet,
        "synthetic_rough": KlemsBSDF.rough_transmission(.55, 8, reflectance=.15),
        "original_measured_solar": KlemsBSDF.from_xml(FIXTURE),
    }
    records = []
    for label, material in materials.items():
        if isinstance(material, KlemsBSDF):
            matrices = material.matrices
        else:
            matrices = np.broadcast_to(np.array([.2, .1, .4, .4])[:, None, None]/math.pi,
                                       (4, 145, 145))
        hemisphere = matrices @ weights
        plate_t = plate_fresnel(directions[:, 2])[1]
        beam = 800*directions[:, 2, None]*plate_t[:, None]*hemisphere[[0, 2]][:, [0, 69, 144]].T
        diffuse = 100/math.pi*np.array([weighted_plate @ matrices[i] @ weights for i in (0, 2)])
        scene = Scene.from_meshes({"sheet": panel(2), "clear": panel(3)})
        assigned = {"sheet": material, "clear": ClearGlazing()}
        options = dict(samples=samples, replicates=replicates, bounces=2, seed=734)
        with ShortwaveSolver(scene, backend=backend, materials=assigned) as solver:
            isolated = {}
            for mode, dni, dhi, expected in (("sun", 800, 0, beam),
                    ("sky", 0, 100, np.broadcast_to(diffuse, beam.shape)),
                    ("combined", 800, 100, beam+diffuse)):
                result = solver.solve(points, normals, directions, dni=dni, dhi=dhi, **options)
                actual = result.total[:, :, 0]
                error = result.standard_error[:, :, :, 0].sum(axis=2)
                residual = abs(actual-expected)
                records.append({"material": label, "mode": mode,
                    "reference": "independent projected-angle quadrature, large separated finite panels",
                    "expected_W_m2": expected.tolist(), "actual_W_m2": actual.tolist(),
                    "absolute_error_W_m2": residual.tolist(), "standard_error_bound_W_m2": error.tolist(),
                    "confidence_95_half_width_W_m2": (3.182446305*error).tolist() if replicates == 4 else None,
                    "passed": bool(np.all(residual <= 5*error+.15))})
                isolated[mode] = result
            linearity = float(np.max(abs(isolated["combined"].irradiance
                                         -isolated["sun"].irradiance-isolated["sky"].irradiance)))
            records.append({"material": label, "check": "isolated forcing linearity",
                            "absolute_error_W_m2": linearity, "passed": linearity < 1e-10})
            # Rotate the complete fixture around +Z: sky and normals remain
            # unchanged, directional matrix tangent and sun must rotate together.
            rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            scene.update_transforms({"sheet": rotation, "clear": rotation})
            rotated_points = np.asarray(points) @ rotation[:3, :3].T
            rotated_sun = directions @ rotation[:3, :3].T
            moved = solver.solve(rotated_points, normals, rotated_sun, dni=800, dhi=100, **options)
            error = (moved.standard_error+isolated["combined"].standard_error).sum(axis=2)[:, :, 0]
            residual = abs(moved.total[:, :, 0]-isolated["combined"].total[:, :, 0])
            with ShortwaveSolver(scene, backend=backend, materials=assigned) as fresh:
                rebuilt = fresh.solve(rotated_points, normals, rotated_sun, dni=800, dhi=100, **options)
            records.append({"material": label, "check": "rotated BSDF frame and update/fresh",
                "absolute_error_W_m2": residual.tolist(),
                "passed": bool(np.all(residual <= 5*error+.15)
                               and np.array_equal(moved.irradiance, rebuilt.irradiance)
                               and not isolated["combined"].irradiance.flags.writeable)})
    return {"backend": backend, "samples": samples, "replicates": replicates,
            "bounces": 2, "seed": 734, "records": records,
            "shadow": shadow_reference(backend),
            "passed": all(item["passed"] for item in records)}


def shadow_reference(backend):
    """Deterministic low-sun visibility with an exactly positioned moving proxy."""
    scene = Scene.from_meshes({"person": panel(1, .2)})
    direction = np.array([np.sqrt(.99), 0, .1])
    transform = np.eye(4)
    transform[0, 3] = direction[0]/direction[2]
    options = dict(dni=800, dhi=0, samples=0, bounces=0)
    with ShortwaveSolver(scene, backend=backend, materials={"person": Opaque(0)}) as solver:
        open_result = solver.solve([0, 0, 0], [0, 0, 1], direction, **options)
        saved = open_result.irradiance.copy()
        scene.update_transform("person", transform)
        blocked = solver.solve([0, 0, 0], [0, 0, 1], direction, **options)
        with ShortwaveSolver(scene, backend=backend, materials={"person": Opaque(0)}) as fresh:
            rebuilt = fresh.solve([0, 0, 0], [0, 0, 1], direction, **options)
    passed = (np.max(abs(open_result.total-80)) < .002
              and np.max(abs(blocked.total)) < .002
              and np.array_equal(saved, open_result.irradiance)
              and np.array_equal(blocked.irradiance, rebuilt.irradiance))
    return {"expected_open_W_m2": 80, "expected_blocked_W_m2": 0,
            "open_W_m2": open_result.total.tolist(), "blocked_W_m2": blocked.total.tolist(),
            "absolute_error_at_zero_W_m2": float(np.max(abs(blocked.total))), "passed": bool(passed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [selected_references(backend, samples=args.samples, replicates=args.replicates)
               for backend in args.backend or ["embree"]]
    report = {"scope": "controlled constituent paths; not annual district accuracy",
              "measured_source": inspect_fixture(), "backends": reports,
              "passed": all(item["passed"] and item["shadow"]["passed"] for item in reports)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
