"""Analytic and mutation regressions through the native shortwave entry point."""

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from raystrack import CapabilityError, Mesh, Scene, Surface
from raystrack.backends.native import native_library
from raystrack.shortwave import (
    KlemsBSDF, Opaque, PatchSky, PatchSkySeries, ReinhartSkyBasis, ShortwaveSolver,
    SolarSite, SolarTimeline, ThinDiffuse,
)
from raystrack.shortwave.klems import KLEMS_PROJECTED_SOLID_ANGLES


@pytest.fixture(params=("native-builtin", "embree", "vulkan-rt"))
def backend(request):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("set RAYSTRACK_NATIVE_LIBRARY to exercise compiled transport")
    try:
        native_library(request.param)
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    return request.param


def panel(z=1, width=1):
    return Mesh([[-width, -width, z], [width, -width, z], [width, width, z],
                 [-width, width, z]], [[0, 1, 2], [0, 2, 3]])


def test_material_validation():
    assert Opaque([0.3, 0.4, 0.5], 0.1).back == (0.1,) * 3
    with pytest.raises(ValueError, match="at most one"):
        ThinDiffuse([0.1, 0.2, 0.3], 0.8)
    with pytest.raises(ValueError, match="finite"):
        Opaque(np.nan)


def test_direct_sun_geometry_updates_and_immutable_snapshots(backend):
    scene = Scene.from_meshes({"shade": panel()})
    points, normals = [[0.1, 0, 0], [3.1, 0, 0]], [[0, 0, 1], [0, 3, 4]]
    sun = [[0, 0, 1], [0, 0, -1]]
    options = dict(dni=800, samples=0, bounces=0, durations_hours=[0.5, 1])
    with ShortwaveSolver(scene, backend=backend) as solver:
        first = solver.solve(points, normals, sun, **options)
        shift = np.eye(4)
        shift[0, 3] = 3
        scene.update_transform("shade", shift)
        second = solver.solve(points, normals, sun, **options)
        scene.update_vertices("shade", panel(width=0.01).vertices)
        third = solver.solve(points, normals, sun, **options)
        scene.update_mesh("shade", Mesh(panel().vertices, [[0, 1, 2]]))
        fourth = solver.solve(points, normals, sun, **options)
        with ShortwaveSolver(scene, backend=backend) as fresh:
            np.testing.assert_array_equal(fourth.irradiance, fresh.solve(points, normals, sun, **options).irradiance)
    np.testing.assert_allclose(first.total[:, :, 0], [[0, 640], [0, 0]])
    np.testing.assert_allclose(second.total[:, :, 0], [[800, 0], [0, 0]])
    np.testing.assert_allclose(third.total[:, :, 0], [[800, 640], [0, 0]])
    np.testing.assert_allclose(second.exposure()[:, 0, 0], [400, 0])
    assert [first.geometry_revision, second.geometry_revision, third.geometry_revision, fourth.geometry_revision] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        first.irradiance.setflags(write=True)


def test_diffuse_transmission_and_material_reuse(backend):
    scene = Scene.from_meshes({"sheet": panel(width=100)})
    points, normals = [[0.12, 0, 2], [0.12, 0, 0]], [[0, 0, -1], [0, 0, 1]]
    args = (points, normals, [0, 0, 1])
    with ShortwaveSolver(scene, backend=backend, materials={"sheet": ThinDiffuse(0.3, 0.4)}) as solver:
        first = solver.solve(*args, dni=800, samples=4096, replicates=3, seed=42)
        handle = solver._native
        solver.set_material("sheet", Opaque(0.5, 0.1))
        scene.set_label("sheet", "Changed label")
        second = solver.solve(*args, dni=800, samples=4096, seed=42)
        assert solver._native is handle
        scene.remove("sheet")
        scene.add(Surface("sheet", panel(width=100)))
        third = solver.solve(*args, dni=800, samples=4096, seed=42)
    np.testing.assert_allclose(first.irradiance[0, :, :, 0], [[0, 0, 240, 0], [0, 0, 0, 320]], atol=1)
    np.testing.assert_allclose(second.irradiance[0, :, :, 0], [[0, 0, 400, 0], [0, 0, 0, 0]], atol=1)
    np.testing.assert_array_equal(third.total, 0)
    assert first.material_revision < second.material_revision
    assert first.geometry_revision == second.geometry_revision == 0
    assert first.standard_error is not None


def test_uniform_sky_and_patch_series(backend):
    basis = ReinhartSkyBasis(1, north_deg=35)
    patches = PatchSky.isotropic(basis, 100)
    series = PatchSkySeries(basis, np.stack([patches.radiance, 2 * patches.radiance]))
    args = ([[0, 0, 0], [0, 0, 0]], [[0, 0, 1], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]])
    with ShortwaveSolver(Scene(), backend=backend) as solver:
        patch = solver.solve(*args, sky=series, dni=0, bounces=0, samples=16384)
        isotropic = solver.solve(*args, dhi=[100, 200], dni=0, bounces=0, samples=16384)
    np.testing.assert_allclose(patch.irradiance, isotropic.irradiance, atol=1e-10)
    np.testing.assert_allclose(patch.total[:, :, 0], [[100, 50], [200, 100]], atol=2)


def test_annual_direct_energy_and_night(backend):
    timeline = SolarTimeline.for_year(2024, SolarSite(52.5, 13.4), utc_offset_minutes=60)
    with ShortwaveSolver(Scene(), backend=backend) as solver:
        result = solver.solve([0, 0, 0], [0, 0, 1], timeline, dni=700, samples=0, bounces=0)
    assert len(result.timestamps) == 8784
    reference = np.maximum(timeline.directions[:, 2], 0) * 700
    np.testing.assert_allclose(result.total[:, 0, 0], reference, atol=5e-5)
    np.testing.assert_allclose(result.exposure()[0, 0], reference.sum(), rtol=1e-7)


def test_two_transmission_interactions_and_large_triangle_hit_position(backend):
    scene = Scene.from_meshes({"lower": panel(1, 10000), "upper": panel(2, 10000)})
    material = ThinDiffuse(0, 0.5)
    args = ([0, 0.1, 0], [0, 0, 1], [0, 0, 1])
    with ShortwaveSolver(scene, backend=backend, materials=dict.fromkeys(scene.surface_ids, material)) as solver:
        once = solver.solve(*args, dni=1000, dhi=100, samples=4096)
        twice = solver.solve(*args, dni=1000, dhi=100, samples=4096, bounces=2)
        scene.remove("upper")
        single_sheet = solver.solve(*args, dni=1000, dhi=100, samples=4096)
    np.testing.assert_allclose(once.total, 0, atol=1e-10)
    np.testing.assert_allclose(twice.irradiance[0, 0, 3], 275, atol=1e-7)
    np.testing.assert_allclose(single_sheet.irradiance[0, 0, 3], 550, atol=1e-7)


def test_klems_solar_lobes_through_native_transport(backend):
    matrix = np.empty((4, 145, 145))
    for slot, coefficient in enumerate((0.3, 0.1, 0.4, 0.2)):
        matrix[slot] = coefficient / np.pi
    scene = Scene.from_meshes({"sheet": panel(width=1000)})
    points, normals = [[0, 0.1, 2], [0, 0.1, 0]], [[0, 0, -1], [0, 0, 1]]
    with ShortwaveSolver(scene, backend=backend, materials={"sheet": KlemsBSDF(matrix)}) as solver:
        result = solver.solve(points, normals, [0, 0, 1], dni=800, dhi=100, samples=4096)
        with pytest.raises(ValueError, match="Solar"):
            solver.set_material("sheet", KlemsBSDF(matrix, band="visible"))
    np.testing.assert_allclose(result.irradiance[0, :, :, 0], [[0, 0, 270, 0], [0, 0, 0, 360]], atol=0.1)


def test_anisotropic_klems_incident_and_outgoing_patches(backend):
    # The +X sun selects the -X incident sector under WINDOW's incident
    # convention. Only directions in outgoing patch 69 reach the sensor.
    incident, outgoing = 5, 69
    matrix = np.zeros((4, 145, 145))
    matrix[2, incident, outgoing] = 0.8
    tilt = np.radians(10)
    sun = [np.sin(tilt), 0, np.cos(tilt)]
    scene = Scene.from_meshes({"sheet": panel(width=10000)})
    options = dict(dhi=100, samples=32768, replicates=4, seed=83)
    with ShortwaveSolver(scene, backend=backend, materials={"sheet": KlemsBSDF(matrix)}) as solver:
        lit = solver.solve([0, 0.1, 0], [0, 0, 1], sun, dni=800, **options)
        other = solver.solve([0, 0.1, 0], [0, 0, 1], [-sun[0], 0, sun[2]], dni=800, **options)
    incident_sky = 100 / np.pi * KLEMS_PROJECTED_SOLID_ANGLES[incident]
    outgoing_weight = 0.8 * KLEMS_PROJECTED_SOLID_ANGLES[outgoing]
    for result, expected in ((lit, outgoing_weight * (800 * np.cos(tilt) + incident_sky)),
                             (other, outgoing_weight * incident_sky)):
        value = result.irradiance[0, 0, 3, 0]
        error = result.standard_error[0, 0, 3, 0]
        assert abs(value - expected) <= 4 * error + 0.01 * expected


def test_radiance_thin_sheet_reference(backend, tmp_path):
    executable_dir = os.environ.get("RAYSTRACK_RADIANCE_BIN")
    commands = [shutil.which(name, path=executable_dir) for name in ("oconv", "rtrace")]
    if None in commands:
        pytest.skip("Radiance oconv and rtrace are unavailable")
    oconv, rtrace = commands
    size = 0.533
    brightness = 800 / (np.pi * np.sin(np.radians(size / 2))**2)
    source = tmp_path / "sheet.rad"
    source.write_text(f"""void light solar
0
0
3 {brightness} {brightness} {brightness}
solar source sun
0
0
4 0 0 1 {size}
void trans sheet_mat
0
0
7 .7 .7 .7 0 0 {4/7} 0
sheet_mat polygon sheet
0
0
12 -100 -100 1  100 -100 1  100 100 1  -100 100 1
""", encoding="ascii")
    octree = tmp_path / "sheet.oct"
    with octree.open("wb") as stream:
        subprocess.run([oconv, "-f", source.name], cwd=tmp_path, stdout=stream, check=True)
    reference = subprocess.run(
        [rtrace, "-I+", "-h", "-ab", "1", "-aa", "0", "-ad", "8192", "-as", "0",
         "-av", "0", "0", "0", "-ov", Path(octree).name],
        input="0 .1 2 0 0 -1\n0 .1 0 0 0 1\n", text=True, capture_output=True, check=True, cwd=tmp_path,
    )
    target = np.fromstring(reference.stdout, sep=" ").reshape(2, 3)
    with ShortwaveSolver(Scene.from_meshes({"sheet": panel(width=100)}), backend=backend,
                         materials={"sheet": ThinDiffuse(0.3, 0.4)}) as solver:
        result = solver.solve([[0, 0.1, 2], [0, 0.1, 0]], [[0, 0, -1], [0, 0, 1]],
                              [0, 0, 1], dni=800, samples=8192, replicates=4, seed=91)
    np.testing.assert_allclose(result.total[0], target, atol=0.5)
