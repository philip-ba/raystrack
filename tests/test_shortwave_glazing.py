"""Delta-light connections, depth accounting and mutable clear plates."""
import os

import numpy as np
import pytest

from raystrack import CapabilityError, Mesh, Scene
from raystrack.backends.native import native_library
from raystrack.shortwave import ClearGlazing, KlemsBSDF, Opaque, ShortwaveSolver
from raystrack.shortwave.klems import KLEMS_AZIMUTH_COUNTS, KLEMS_POLAR_BOUNDS_DEG
from validation.validate_clear_glazing import plate_fresnel


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


def panel(z=1, width=100, tilt=0):
    vertices = np.array([[-width, -width, 0], [width, -width, 0],
                         [width, width, 0], [-width, width, 0]], dtype=float)
    angle = np.radians(tilt)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    return Mesh(vertices @ rotation.T + [0, 0, z], [[0, 1, 2], [0, 2, 3]])


def test_oblique_sun_transmission_reflection_rgb_and_depth(backend):
    angles = np.radians([0, 20, 45, 60, 75, 85])
    sun = np.column_stack((np.sin(angles), np.zeros(len(angles)), np.cos(angles)))
    dni = np.broadcast_to([800, 600, 400], (len(angles), 3))
    scene = Scene.from_meshes({"glass": panel(width=1000)})
    with ShortwaveSolver(scene, backend=backend, materials={"glass": ClearGlazing()}) as solver:
        args = ([[0, .1, 0], [0, .1, 2]], [[0, 0, 1], [0, 0, -1]], sun)
        dark = solver.solve(*args, dni=dni, samples=8, bounces=0, channel_basis="RGB test")
        result = solver.solve(*args, dni=dni, samples=8, bounces=1, channel_basis="RGB test")
        repeated = solver.solve(*args, dni=dni, samples=31, bounces=1, seed=17, channel_basis="RGB test")
    r, t = plate_fresnel(np.cos(angles))
    expected = np.zeros_like(result.irradiance)
    expected[:, 0, 3] = dni * (np.cos(angles) * t)[:, None]
    expected[:, 1, 2] = dni * (np.cos(angles) * r)[:, None]
    np.testing.assert_array_equal(dark.total, 0)
    np.testing.assert_allclose(result.irradiance, expected, rtol=2e-6, atol=2e-5)
    np.testing.assert_array_equal(result.irradiance, repeated.irradiance)


def test_two_plates_and_material_geometry_snapshot_updates(backend):
    scene = Scene.from_meshes({"lower": panel(1), "upper": panel(2)})
    materials = dict.fromkeys(scene.surface_ids, ClearGlazing())
    args = ([0, .1, 0], [0, 0, 1], [0, 0, 1])
    options = dict(dni=800, samples=8)
    with ShortwaveSolver(scene, backend=backend, materials=materials) as solver:
        once = solver.solve(*args, bounces=1, **options)
        twice = solver.solve(*args, bounces=2, **options)
        solver.set_material("upper", ClearGlazing(1))
        changed = solver.solve(*args, bounces=2, **options)
        scene.update_mesh("upper", panel(2, tilt=20))
        tilted = solver.solve(*args, bounces=2, **options)
        scene.update_mesh("upper", panel(2, width=.001))
        removed_from_beam = solver.solve(*args, bounces=1, **options)
    _, t = plate_fresnel(1)
    np.testing.assert_array_equal(once.total, 0)
    np.testing.assert_allclose(twice.irradiance[0, 0, 3], 800*t*t, rtol=1e-6)
    for result in (changed, tilted, removed_from_beam):
        np.testing.assert_allclose(result.irradiance[0, 0, 3], 800*t, rtol=1e-6)
    assert twice.material_revision < changed.material_revision
    assert twice.geometry_revision == changed.geometry_revision < tilted.geometry_revision
    with pytest.raises(ValueError):
        twice.irradiance.setflags(write=True)


def test_finite_plate_off_axis_direct_and_reflected_beam_visibility(backend):
    points = [[0, .1, 0], [2, .1, 0], [0, .1, 2], [2, .1, 2]]
    normals = [[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1]]
    with ShortwaveSolver(Scene.from_meshes({"glass": panel(width=1)}), backend=backend,
                         materials={"glass": ClearGlazing()}) as solver:
        result = solver.solve(points, normals, [.6, 0, .8], dni=800, samples=8, bounces=1)
    r, t = plate_fresnel(.8)
    expected = np.zeros_like(result.irradiance)
    expected[0, 0, 3] = 640*t
    expected[0, 1, 0] = 640
    expected[0, 2, 2] = 640*r
    np.testing.assert_allclose(result.irradiance, expected, rtol=2e-6, atol=1e-5)


def test_finite_event_connection_through_glass_has_one_sun_estimator(backend):
    scene = Scene.from_meshes({"receiver": panel(0, 1000), "glass": panel(1, 1000)})
    rho = np.array([.2, .4, .6])
    with ShortwaveSolver(scene, backend=backend,
                         materials={"receiver": Opaque(rho), "glass": ClearGlazing()}) as solver:
        args = ([0, .1, .5], [0, 0, -1], [0, 0, 1])
        shallow = solver.solve(*args, dni=800, samples=1024, bounces=1, channel_basis="RGB test")
        result = solver.solve(*args, dni=800, samples=1024, bounces=2, channel_basis="RGB test")
    _, t = plate_fresnel(1)
    np.testing.assert_array_equal(shallow.total, 0)
    expected = np.zeros_like(result.irradiance)
    expected[0, 0, 2] = 800*rho*t
    np.testing.assert_allclose(result.irradiance, expected, rtol=1e-6)


def test_nonparallel_plate_reflection_with_two_transmissions(backend):
    scene = Scene.from_meshes({"tilted": panel(0, 100, tilt=20), "horizontal": panel(2, 100)})
    with ShortwaveSolver(scene, backend=backend,
                         materials=dict.fromkeys(scene.surface_ids, ClearGlazing())) as solver:
        result = solver.solve([0, .1, 3], [0, 0, -1], [0, 0, 1],
                              dni=800, samples=8, bounces=3)
    r0, t0 = plate_fresnel(1)
    r20, _ = plate_fresnel(np.cos(np.radians(20)))
    _, t40 = plate_fresnel(np.cos(np.radians(40)))
    expected = np.zeros_like(result.irradiance)
    expected[0, 0, 2] = 800*r0
    expected[0, 0, 3] = 800*np.cos(np.radians(40))*t40*r20*t0
    np.testing.assert_allclose(result.irradiance, expected, rtol=3e-6, atol=2e-5)


def test_sky_through_plate_matches_hemisphere_integral(backend):
    scene = Scene.from_meshes({"glass": panel(width=10000)})
    with ShortwaveSolver(scene, backend=backend, materials={"glass": ClearGlazing()}) as solver:
        result = solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], dni=0,
                              dhi=100, samples=16384, replicates=4, bounces=1, seed=23)
    cosine = (np.arange(100000) + .5) / 100000
    _, t = plate_fresnel(cosine)
    expected = 100 * np.mean(2*cosine*t)
    value = result.irradiance[0, 0, 3, 0]
    error = result.standard_error[0, 0, 3, 0]
    assert abs(value - expected) < 4*error + .1
    np.testing.assert_array_equal(result.irradiance[0, 0, :3], 0)


def test_parallel_plates_internal_reflection_series_is_counted_once(backend):
    scene = Scene.from_meshes({"lower": panel(1, 1000), "upper": panel(2, 1000)})
    with ShortwaveSolver(scene, backend=backend,
                         materials=dict.fromkeys(scene.surface_ids, ClearGlazing())) as solver:
        result = solver.solve([[0, .1, 0], [0, .1, 3]], [[0, 0, 1], [0, 0, -1]],
                              [0, 0, 1], dni=800, samples=8, bounces=32)
    r, t = plate_fresnel(1)
    np.testing.assert_allclose(result.irradiance[0, 0, 3], 800*t*t/(1-r*r), rtol=1e-6)
    np.testing.assert_allclose(result.irradiance[0, 1, 2], 800*r, rtol=1e-6)
    np.testing.assert_allclose(result.irradiance[0, 1, 3], 800*t*t*r/(1-r*r), rtol=1e-6)
    np.testing.assert_allclose(result.total[0].sum(axis=0), 800, rtol=1e-6)


@pytest.mark.parametrize("tabulated", [False, True])
def test_glass_before_and_after_finite_scatter(backend, tabulated):
    scene = Scene.from_meshes({"receiver": panel(0, 1000), "glass": panel(1, 1000)})
    if tabulated:
        bounds = KLEMS_POLAR_BOUNDS_DEG
        ring_rho = .2 + .3*np.cos(np.radians((bounds[:-1] + bounds[1:])/2))
        outgoing_rho = np.repeat(ring_rho, KLEMS_AZIMUTH_COUNTS)
        matrix = np.zeros((4, 145, 145))
        matrix[:2] = outgoing_rho[None, :]/np.pi
        receiver = KlemsBSDF(matrix)
    else:
        receiver = Opaque(.4)
    with ShortwaveSolver(scene, backend=backend,
                         materials={"receiver": receiver, "glass": ClearGlazing()}) as solver:
        result = solver.solve([0, .1, 2], [0, 0, -1], [0, 0, 1], dni=800,
                              samples=8192, replicates=4, bounces=3, seed=37)
    cosine = (np.arange(100000) + .5)/100000
    _, transmittance = plate_fresnel(cosine)
    _, normal_transmittance = plate_fresnel(1)
    if tabulated:
        rings = np.searchsorted(bounds[1:], np.degrees(np.arccos(cosine)), side="right")
        reflectance = ring_rho[rings]
    else:
        reflectance = .4
    expected = 800*normal_transmittance*np.mean(2*cosine*transmittance*reflectance)
    value = result.irradiance[0, 0, 3, 0]
    error = result.standard_error[0, 0, 3, 0]
    assert abs(value-expected) < 4*error + .2


def test_unresolved_distinct_image_directions_raise_capability_error(backend):
    scene = Scene.from_meshes({"lower": panel(1), "near_parallel": panel(2, tilt=.00001)})
    with ShortwaveSolver(scene, backend=backend,
                         materials=dict.fromkeys(scene.surface_ids, ClearGlazing())) as solver:
        with pytest.raises(CapabilityError, match="unresolved at float ray precision"):
            solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], dni=800, samples=8, bounces=1)


def test_ambiguous_facet_normals_raise_capability_error(backend):
    lower, upper = panel(1), panel(2, tilt=.00008)
    mesh = Mesh(np.vstack((lower.vertices, upper.vertices)),
                np.vstack((lower.faces, upper.faces+4)))
    with ShortwaveSolver(Scene.from_meshes({"facets": mesh}), backend=backend,
                         materials={"facets": ClearGlazing()}) as solver:
        with pytest.raises(CapabilityError, match="hit normal is ambiguous"):
            solver.solve([0, .1, 3], [0, 0, -1], [0, 0, 1], dni=800, samples=8, bounces=1)


def test_extreme_finite_index_produces_no_nan(backend):
    with ShortwaveSolver(Scene.from_meshes({"glass": panel()}), backend=backend,
                         materials={"glass": ClearGlazing(1e308)}) as solver:
        result = solver.solve([[0, .1, 0], [0, .1, 2]], [[0, 0, 1], [0, 0, -1]],
                              [0, 0, 1], dni=800, samples=8, bounces=1)
    expected = np.zeros_like(result.irradiance)
    expected[0, 1, 2] = 800
    np.testing.assert_array_equal(result.irradiance, expected)
