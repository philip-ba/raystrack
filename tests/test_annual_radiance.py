"""Independent fixture, time and error-accounting checks for annual validation."""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pytest
from validation import annual_radiance

from validation.annual_radiance import (
    Radiance,
    acceptance_report,
    aggregate,
    cube_fixture,
    metrics,
    rcontrib,
    reuse_reference,
    synthetic_forcing,
)


def test_cube_uses_outward_normals_and_same_area_points():
    scene, polygons, points, normals, labels = cube_fixture()
    assert len(points) == 23
    assert len(scene.surface_ids) == 7
    expected = {
        "top": [0, 0, 1],
        "north": [0, 1, 0],
        "east": [1, 0, 0],
        "south": [0, -1, 0],
        "west": [-1, 0, 0],
    }
    for name, normal in expected.items():
        mask = np.asarray(labels) == name
        np.testing.assert_array_equal(normals[mask], np.broadcast_to(normal, (4, 3)))
        center = points[mask].mean(axis=0)
        assert np.dot(center - np.array([0, 0, 0.5]), normal) == pytest.approx(0.5001)
    assert polygons["ground"][0] == [-10, -10, 0]


def test_forcing_is_full_standard_time_year_and_night_zero():
    timeline, dni, dhi = synthetic_forcing()
    assert len(dni) == len(dhi) == 8760
    assert timeline.timestamps[0] == np.datetime64("2024-12-31T23:30")
    assert timeline.timestamps[-1] == np.datetime64("2025-12-31T22:30")
    np.testing.assert_array_equal(timeline.durations_hours, np.ones(8760))
    np.testing.assert_array_equal(dni[~timeline.above_horizon], 0)
    np.testing.assert_array_equal(dhi[~timeline.above_horizon], 0)
    assert 4000 < np.count_nonzero(dni) < 5000
    np.testing.assert_array_equal(synthetic_forcing()[1], dni)


def test_metrics_do_not_hide_hourly_compensation_or_amplify_zero():
    expected = np.array([[0.0, 0.0], [10.0, 100.0], [10.0, 100.0]])
    actual = np.array([[0.0, 0.0], [12.0, 90.0], [8.0, 110.0]])
    report = metrics(actual, expected, np.array([False, True, True]), np.ones(3))
    np.testing.assert_array_equal(report["annual_relative_error"], 0)
    np.testing.assert_allclose(report["daylight_nrmse"], [0.2, 0.1])
    assert report["near_zero_mae_w_m2"] == 0
    values = np.array([[[1.0], [3.0], [8.0]]])
    np.testing.assert_array_equal(
        aggregate(values, ["face", "face", "ground"]), [[[2.0], [8.0]]]
    )


def test_acceptance_requires_every_exposed_face_and_every_mode():
    values = {"annual_relative_error": [0, 0.03, 0, 0, 0, 1], "daylight_nrmse": [0] * 6}
    report = {
        "groups": ["top", "north", "east", "south", "west", "ground"],
        "backends": {"embree": {"metrics": {"diffuse": values}}},
    }
    result = acceptance_report(report)
    assert not result["all_passed"]
    assert [
        gate["face"] for gate in result["face_mode_gates"] if not gate["passed"]
    ] == ["north"]
    values["annual_relative_error"][1] = 0.01
    assert not acceptance_report(report)["all_passed"]
    report["backends"]["embree"]["metrics"].update(direct=values, combined=values)
    assert acceptance_report(report)["all_passed"]
    report["groups"] = report["groups"][:4]
    assert not acceptance_report(report)["all_passed"]


@pytest.mark.parametrize("backends", [{}, {"embree": {"metrics": {}}}])
def test_acceptance_cannot_pass_without_completed_backend_modes(backends):
    assert not acceptance_report({"groups": ["top", "north", "east", "south", "west"],
                                  "backends": backends})["all_passed"]


def test_acceptance_rejects_missing_requested_backend():
    values = {"annual_relative_error": [0]*5, "daylight_nrmse": [0]*5}
    report = {"groups": ["top", "north", "east", "south", "west"],
              "requested_backends": ["embree", "vulkan-rt"],
              "backends": {"embree": {"metrics": {mode: values for mode in ("direct", "diffuse", "combined")}}}}
    assert not acceptance_report(report)["all_passed"]
    report["backends"]["vulkan-rt"] = report["backends"]["embree"]
    assert acceptance_report(report)["all_passed"]


def cached_reference_fixture(tmp_path):
    source, output = tmp_path/"source", tmp_path/"output"
    source.mkdir()
    output.mkdir()
    polygons = {"wall": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]}
    scene = "void plastic neutral\n0\n0\n5 .2 .2 .2 0 0\n"
    scene += annual_radiance.polygon_text("wall", polygons["wall"], "neutral")
    (source/"scene.rad").write_text(scene)
    (source/"sensors.pts").write_text("0 0 0 0 0 1\n")
    (source/"suns.rad").write_text("verified source placeholder\n")
    (source/"suns.mod").write_text("sun0\n")
    np.save(source/"radiance_sun_coefficients.npy", np.full((1, 1, 3), .6))
    np.save(source/"radiance_sky_coefficients.npy", np.broadcast_to(np.array([.1, .2])[None, :, None], (1, 2, 3)))
    np.savez(source/"forcing.npz", sun=[[0, 0, 1], [0, 0, -1]], dni=[700, 0],
             sky=np.broadcast_to(np.array([[3., 4.], [0, 0]])[:, :, None], (2, 2, 3)))
    shutil.copy2(source/"forcing.npz", output/"forcing.npz")
    # Stale hourly results must not override verified coefficients and forcing.
    wrong = np.full((2, 1, 3), 999.)
    np.savez(source/"radiance_hourly.npz", direct=wrong, diffuse=wrong, primary=wrong)
    report = dict(radiance_samples=4096, bounces=1, reference={}, radiance_version="pinned",
                  radiance_bin="unused", files_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                      for name in ("scene.rad", "sensors.pts", "suns.rad", "radiance_sky_coefficients.npy",
                                   "radiance_sun_coefficients.npy")})
    (source/"report.json").write_text(json.dumps(report))
    return source, output, polygons


def test_reference_reuse_reconstructs_from_verified_coefficients_not_stale_hourly(tmp_path, monkeypatch):
    source, output, polygons = cached_reference_fixture(tmp_path)
    class ReferenceRunner:
        def __init__(self, directory, destination):
            self.output = destination
        def run(self, executable, arguments, **kwargs):
            assert executable == "oconv"
    monkeypatch.setattr(annual_radiance, "Radiance", ReferenceRunner)
    calls = []
    def primary(rad, octree, sensors, modifiers, bins, *, bounces, samples, sky=False):
        calls.append((octree, sensors, modifiers, bins, bounces, samples, sky))
        return np.full((1, 1, 3), .75)
    monkeypatch.setattr(annual_radiance, "rcontrib", primary)
    direct, diffuse, primary_coefficients, _ = reuse_reference(source, output, polygons, 4096, 1)
    np.testing.assert_allclose(direct[:, 0, 0], [.6, 0])
    np.testing.assert_allclose(diffuse[:, 0, 0], [1.1, 0])
    np.testing.assert_allclose(primary_coefficients[:, 0, 0], [.75, 0])
    assert len(calls) == 1 and calls[0][3:] == (1, 0, 1, False)


def test_reference_reuse_rejects_missing_coefficient_hash(tmp_path):
    source, output, polygons = cached_reference_fixture(tmp_path)
    report = json.loads((source/"report.json").read_text())
    del report["files_sha256"]["radiance_sun_coefficients.npy"]
    (source/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="required.*hash"):
        reuse_reference(source, output, polygons, 4096, 1)


def test_radiance_unit_sky_and_dni_establish_radiometric_conversion(tmp_path):
    if not Path("C:/Radiance/bin/rcontrib.exe").exists():
        pytest.skip("Pinned Radiance installation unavailable")
    rad = Radiance("C:/Radiance/bin", tmp_path)
    (tmp_path / "unit.rad").write_text(
        "void glow sky_glow\n0\n0\n4 1 1 1 0\n"
        "sky_glow source sky\n0\n0\n4 0 0 1 180\n"
    )
    rad.run("oconv", ["unit.rad"], target="unit.oct")
    result = rcontrib(
        rad, "unit.oct", b"0 0 0 0 0 1\n", None, 146, bounces=1, samples=4096, sky=True
    )
    np.testing.assert_allclose(result.sum(axis=1), np.pi, rtol=1e-5)
    diameter = 0.01
    omega = np.pi * np.sin(np.radians(diameter / 2)) ** 2
    (tmp_path / "sun.rad").write_text(
        f"void light sun\n0\n0\n3 {1/omega} {1/omega} {1/omega}\n"
        "sun source sun_disk\n0\n0\n4 .6 0 .8 .01\n"
    )
    (tmp_path / "sun.mod").write_text("sun\n")
    rad.run("oconv", ["sun.rad"], target="sun.oct")
    result = rcontrib(
        rad, "sun.oct", b"0 0 0 0 0 1\n", "sun.mod", 1, bounces=0, samples=1
    )
    np.testing.assert_allclose(result, 0.8, rtol=1e-5)
