"""Independent projected-angle checks for an unchanged CGDB Solar fixture.

Run with PYTHONPATH=src;. No network, extra dependency, or native library is
needed for the import checks. --backend adds the large-sheet transport gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from raystrack.shortwave import KlemsBSDF


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/bsdf/2011-SA1.XML"
SHA256 = "5f398e08a81fecd57f9e566fa2a16f08bb0f2ca2a22fdba04d5d949c818b3336"
SOURCE_URL = "https://windows.lbl.gov/sites/default/files/2011-SA1.XML"


def quadrature():
    """Patch centers and exact integral of cos(theta) dOmega, independent of loader."""
    bounds = [0, 5, 15, 25, 35, 45, 55, 65, 75, 90]
    counts = [1, 8, 16, 20, 24, 24, 24, 16, 12]
    directions, weights, half_turn = [], [], []
    offset = 0
    for low, high, count in zip(bounds[:-1], bounds[1:], counts):
        theta = math.radians((low + high) / 2)
        weight = math.pi * (math.sin(math.radians(high))**2
                            - math.sin(math.radians(low))**2) / count
        for sector in range(count):
            phi = math.tau * sector / count
            directions.append([math.sin(theta)*math.cos(phi),
                               math.sin(theta)*math.sin(phi), math.cos(theta)])
            weights.append(weight)
            half_turn.append(offset + (sector + count//2) % count)
        offset += count
    return np.array(directions), np.array(weights), np.array(half_turn)


def inspect_fixture(path=FIXTURE, *, expected_sha256=SHA256):
    """Fail on changed bytes; report data residuals without repairing measurements."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("fixture SHA256 does not match pinned original bytes")
    bsdf = KlemsBSDF.from_xml(path, band="solar", window_front="front")
    centers, weights, half = quadrature()
    hemispherical = bsdf.matrices @ weights
    pdf_residual, value_residual, weight_residual = 0.0, 0.0, 0.0
    rng = np.random.default_rng(671)
    for side in (1, -1):
        for patch in (0, 1, 25, 69, 133, 144):
            fixed = centers[patch] * [-1, -1, side]
            for reverse in (False, True):
                integral = 0.0
                for outgoing_side in (1, -1):
                    for direction, weight in zip(centers, weights):
                        sampled = direction * ([-1, -1, outgoing_side] if reverse
                                               else [1, 1, outgoing_side])
                        wi, wo = (sampled, fixed) if reverse else (fixed, sampled)
                        integral += bsdf.pdf(wi, wo, reverse=reverse) / abs(sampled[2]) * weight
                pdf_residual = max(pdf_residual, abs(integral - 1))
                for draw in rng.random((8, 3)):
                    sample = bsdf.sample(fixed, draw, reverse=reverse)
                    wi, wo = (sample.direction, fixed) if reverse else (fixed, sample.direction)
                    value_residual = max(value_residual, abs(sample.value-bsdf.evaluate(wi, wo)))
                    weight_residual = max(weight_residual, abs(sample.pdf-bsdf.pdf(wi, wo, reverse=reverse)))
    # Swap incident/outgoing away-pointing vectors: transpose plus XY half turn.
    def reverse(matrix):
        return matrix.T[half][:, half]

    reciprocity = [float(np.max(np.abs(bsdf.matrices[i]-reverse(bsdf.matrices[j]))))
                   for i, j in ((0, 0), (1, 1), (2, 3))]
    return {
        "source": asdict(bsdf.source), "original_url": SOURCE_URL if digest == SHA256 else None,
        "data_status": ("measurement-based CGDB product; not raw goniometer data"
                        if digest == SHA256 else "user-supplied; measurement provenance unverified"),
        "units": {"bsdf": "sr^-1", "hemispherical": "dimensionless"},
        "normal_rho_tau_front": hemispherical[[0, 2], 0].tolist(),
        "normal_rho_tau_back": hemispherical[[1, 3], 0].tolist(),
        "maximum_energy": float(np.max(hemispherical[:2]+hemispherical[2:])),
        "reciprocity_max_absolute_sr_inverse": reciprocity,
        "pdf_integral_max_residual": pdf_residual,
        "sample_evaluate_max_residual": value_residual,
        "sample_pdf_max_residual": weight_residual,
        "directional_transmission_range": [float(np.min(hemispherical[2])),
                                             float(np.max(hemispherical[2]))],
    }


def native_reference(backend, path=FIXTURE, *, samples=32768, replicates=4,
                     reverse_sheet=False):
    """One-event sun/sky integrals for a large sheet; no reciprocity assumption."""
    from raystrack import Mesh, Scene
    from raystrack.shortwave import ShortwaveSolver

    bsdf = KlemsBSDF.from_xml(path)
    centers, weights, _ = quadrature()
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    scene = Scene.from_meshes({"sheet": Mesh(
        [[-10000, -10000, 1], [10000, -10000, 1],
         [10000, 10000, 1], [-10000, 10000, 1]],
        faces[:, ::-1] if reverse_sheet else faces)})
    # Reflection above sheet and transmission below it, both looking at sheet.
    points, normals = [[0, .17, 2], [0, .17, 0]], [[0, 0, -1], [0, 0, 1]]
    sun = centers[[0, 25, 69]] * [-1, -1, 1]
    hemi = bsdf.matrices @ weights
    lobes = (1, 3) if reverse_sheet else (0, 2)
    beam = 800 * sun[:, 2, None] * hemi[list(lobes)][:, [0, 25, 69]].T
    sky = 100 / math.pi * np.array([weights @ bsdf.matrices[i] @ weights for i in lobes])
    records = []
    with ShortwaveSolver(scene, backend=backend, materials={"sheet": bsdf}) as solver:
        for name, dni, dhi, expected in (
            ("direct", 800, 0, beam), ("diffuse", 0, 100, np.broadcast_to(sky, beam.shape)),
            ("combined", 800, 100, beam+sky),
        ):
            result = solver.solve(points, normals, sun, dni=dni, dhi=dhi,
                                  samples=samples, replicates=replicates, bounces=1, seed=671)
            actual = result.total[:, :, 0]
            # Independent randomized replicate SE, fixed five-SE plus finite-size floor.
            error = np.sqrt(np.sum(result.standard_error[:, :, :, 0]**2, axis=2))
            tolerance = 5*error + .1
            # Student-t(3) interval is descriptive for the fixed four-replicate run.
            interval = 3.182446305 * error if replicates == 4 else None
            records.append({"mode": name, "expected_W_m2": expected.tolist(),
                            "actual_W_m2": actual.tolist(), "standard_error_W_m2": error.tolist(),
                            "confidence_95_half_width_W_m2": (interval.tolist()
                                                               if interval is not None else None),
                            "tolerance_W_m2": tolerance.tolist(),
                            "passed": bool(np.all(np.abs(actual-expected) <= tolerance))})
    return {"backend": backend, "illuminated_side": "back" if reverse_sheet else "front",
            "samples": samples, "replicates": replicates,
            "seed": 671, "bounces": 1, "records": records,
            "passed": all(record["passed"] for record in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=FIXTURE)
    parser.add_argument("--sha256", default=SHA256)
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_fixture(args.xml, expected_sha256=args.sha256)
    report["native"] = [native_reference(backend, args.xml, reverse_sheet=reverse)
                        for backend in args.backend for reverse in (False, True)]
    report["passed"] = (report["maximum_energy"] <= 1+1e-6
                        and report["pdf_integral_max_residual"] < 1e-12
                        and report["sample_evaluate_max_residual"] < 1e-12
                        and report["sample_pdf_max_residual"] < 1e-12
                        and all(item["passed"] for item in report["native"]))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
