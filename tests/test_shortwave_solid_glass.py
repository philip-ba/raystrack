"""Solid-boundary Fresnel/Snell, Beer transport and explicit capability limits."""
import os

import numpy as np
import pytest

from raystrack import CapabilityError, Mesh, Scene
from raystrack.backends.native import native_library
from raystrack.shortwave import ClearGlazing, Opaque, ShortwaveSolver, SolidDielectric


@pytest.fixture(params=("native-builtin", "embree", "vulkan-rt"))
def backend(request):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("set RAYSTRACK_NATIVE_LIBRARY for compiled transport")
    try:
        native_library(request.param)
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    return request.param


def slab(width=100, low=1, high=2):
    vertices = [[-width, -width, low], [width, -width, low],
                [width, width, low], [-width, width, low],
                [-width, -width, high], [width, -width, high],
                [width, width, high], [-width, width, high]]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return Mesh(vertices, faces)


def panel(z, width=100):
    return Mesh([[-width, -width, z], [width, -width, z], [width, width, z], [-width, width, z]],
                [[0, 1, 2], [0, 2, 3]])


def test_interface_fresnel_snell_critical_angle_and_adjoint_weights():
    glass = SolidDielectric(1.5)
    r, t, cosine = glass.coefficients(1.)
    assert r == pytest.approx(.04)
    assert r+t == 1
    assert cosine == 1
    incidence = np.array([.6, 0, .8])
    entering = glass.sample(incidence, .9)
    np.testing.assert_allclose(entering.direction, [-.4, 0, -np.sqrt(.84)])
    assert entering.weight == pytest.approx(1/1.5**2)
    exiting = glass.sample(entering.direction, .9)
    np.testing.assert_allclose(exiting.direction, incidence)
    assert entering.weight*exiting.weight == pytest.approx(1)
    critical = np.arcsin(1/1.5)
    for angle in (critical+1e-6, np.radians(80)):
        direction = [np.sin(angle), 0, -np.cos(angle)]
        event = glass.sample(direction, .999999)
        assert not event.transmitted
        assert event.coefficient == event.probability_mass == 1
        np.testing.assert_allclose(event.direction, [-direction[0], 0, direction[2]])
    assert glass.coefficients(np.cos(critical-1e-6), entering=False)[1] > 0
    assert SolidDielectric(1).coefficients(0)[0] == 0
    # At a prism's second face, 30-degree internal incidence exits at 48.5904
    # degrees relative to that face normal.
    event = glass.sample([.5, 0, -np.sqrt(.75)], .999)
    assert event.transmitted
    assert np.degrees(np.arcsin(abs(event.direction[0]))) == pytest.approx(48.59037789072914)


def test_absorption_units_rgb_and_input_ownership():
    values = np.array([0., .2, 2.])
    glass = SolidDielectric(absorption=values)
    values[:] = 100
    np.testing.assert_allclose(glass.attenuation([0, 2]), np.exp(-np.array([[0, 0, 0], [0, .4, 4]])))
    for bad in (-1, np.inf, np.nan, True, [1, 2], "0.1"):
        with pytest.raises(ValueError, match="absorption"):
            SolidDielectric(absorption=bad)
    for bad in (.9, 10.01, 1e7, np.inf, True, [1.5]):
        with pytest.raises(ValueError, match="refractive_index"):
            SolidDielectric(bad)


def _slab_quadrature(index, absorption, bounces):
    # Independent interface equations and finite internal-reflection series.
    cosine = (np.arange(200000)+.5)/200000
    internal = np.sqrt(1-(1-cosine*cosine)/(index*index))
    rs = ((cosine-index*internal)/(cosine+index*internal))**2
    rp = ((index*cosine-internal)/(index*cosine+internal))**2
    r = .5*(rs+rp)
    attenuation = np.exp(-np.asarray(absorption)[None, :]/internal[:, None])
    first = (1-r[:, None])**2*attenuation
    ratio = r[:, None]**2*attenuation**2
    series = sum(ratio**j for j in range(bounces//2))
    return 100*np.mean(2*cosine[:, None]*first*series, axis=0)


@pytest.mark.parametrize("index,absorption,bounces", [(1., [0, .2, .8], 2),
                                                    (1.52, [0, .2, .8], 2),
                                                    (1.52, [0, 0, 0], 12)])
def test_native_slab_sky_matches_finite_series(backend, index, absorption, bounces):
    scene = Scene.from_meshes({"glass": slab(width=2000)})
    with ShortwaveSolver(scene, backend=backend,
                         materials={"glass": SolidDielectric(index, absorption)}) as solver:
        result = solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], dni=0, dhi=100,
                              samples=16384, replicates=4, bounces=bounces, seed=41,
                              channel_basis="RGB absorption test")
    expected = _slab_quadrature(index, absorption, bounces)
    value = result.irradiance[0, 0, 3]
    error = result.standard_error[0, 0, 3]
    assert np.all(abs(value-expected) < 4*error+.15), (value, expected, error)
    np.testing.assert_allclose(result.total[0, 0], value, atol=.02)


def test_solid_sun_closed_winding_sensor_overlap_and_precision_limits(backend):
    mesh = slab(width=2)
    glass = SolidDielectric()
    options = dict(dni=0, dhi=100, samples=8, bounces=2)
    with ShortwaveSolver(Scene.from_meshes({"glass": mesh}), backend=backend,
                         materials={"glass": glass}) as solver:
        with pytest.raises(CapabilityError, match="sky forcing only"):
            solver.solve([0, 0, 0], [0, 0, 1], [0, 0, -1], dni=1, samples=8)
        for point in ([0, 0, 1.5], [0, 0, 1]):
            with pytest.raises(CapabilityError, match="strictly outside"):
                solver.solve(point, [0, 0, 1], [0, 0, 1], **options)
    duplicated_shell = Mesh(np.concatenate((mesh.vertices, mesh.vertices)),
                            np.concatenate((mesh.faces, mesh.faces+len(mesh.vertices))))
    invalid = [Mesh(mesh.vertices, mesh.faces[:-1]), Mesh(mesh.vertices, mesh.faces[:, ::-1]),
               duplicated_shell,
               slab(width=2, high=1.000001)]
    for bad in invalid:
        with ShortwaveSolver(Scene.from_meshes({"glass": bad}), backend=backend,
                             materials={"glass": glass}) as solver:
            with pytest.raises(CapabilityError, match="closed|convex|thickness|connected"):
                solver.solve([0, 0, 0], [0, 0, 1], [0, 0, 1], **options)
    with ShortwaveSolver(Scene.from_meshes({"glass": mesh, "inside": panel(1.5, 1)}),
                         backend=backend, materials={"glass": glass}) as solver:
        with pytest.raises(CapabilityError, match="disjoint bounding boxes"):
            solver.solve([0, 0, 0], [0, 0, 1], [0, 0, 1], **options)


def test_beer_distance_restores_skipped_normal_offset(backend):
    with ShortwaveSolver(Scene.from_meshes({"glass": slab(width=2000)}), backend=backend,
                         materials={"glass": SolidDielectric(1, [0, .2, 2])}) as solver:
        options = dict(dni=0, dhi=100, samples=4096, bounces=2, seed=37,
                       channel_basis="RGB absorption test")
        near = solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], ray_offset=1e-5, **options)
        far = solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], ray_offset=1e-3, **options)
    np.testing.assert_allclose(near.total, far.total, rtol=0, atol=5e-5)


def test_solid_finite_plate_mixture_and_revision_snapshots(backend):
    scene = Scene.from_meshes({"glass": slab(width=200), "receiver": panel(0, 1000),
                               "plate": panel(3, 1000)})
    assignments = {"glass": SolidDielectric(1.), "receiver": Opaque(.4),
                   "plate": ClearGlazing(1.)}
    args = ([0, .1, .5], [0, 0, -1], [0, 0, 1])
    options = dict(dni=0, dhi=100, samples=8192, replicates=4, bounces=4, seed=19)
    with ShortwaveSolver(scene, backend=backend, materials=assignments) as solver:
        first = solver.solve(*args, **options)
        handle = solver._native
        solver.set_material("glass", SolidDielectric(1., .5))
        changed = solver.solve(*args, **options)
        assert solver._native is handle
        shift = np.eye(4)
        shift[2, 3] = .25
        scene.update_transform("glass", shift)
        moved = solver.solve(*args, **options)
    assignments["glass"] = SolidDielectric(1., .5)
    with ShortwaveSolver(scene, backend=backend, materials=assignments) as solver:
        fresh = solver.solve(*args, **options)
    np.testing.assert_allclose(first.total, 40., atol=.2)
    expected = .4*_slab_quadrature(1., [.5]*3, 2)
    assert np.all(abs(changed.total[0, 0]-expected) < 4*changed.standard_error[0, 0, 2]+.2)
    np.testing.assert_array_equal(moved.irradiance, fresh.irradiance)
    assert first.material_revision < changed.material_revision
    assert first.geometry_revision == changed.geometry_revision < moved.geometry_revision
    with pytest.raises(ValueError):
        first.irradiance.setflags(write=True)


def _wedge_reference(resolution, bounces=4, half_y=2.):
    """Independent float64 plane-clipping quadrature, splitting every event.

    No native geometry, model coefficients or sampling functions are used.
    Uniform quadrature over the cosine measure has irradiance weight DHI/N.
    """
    u = (np.arange(resolution)+.5)/resolution
    phi = 2*np.pi*(np.arange(2*resolution)+.5)/(2*resolution)
    cosine, azimuth = np.meshgrid(np.sqrt(u), phi, indexing="ij")
    cosine, azimuth = cosine.ravel(), azimuth.ravel()
    direction = np.column_stack((np.sqrt(1-cosine**2)*np.cos(azimuth),
                                 np.sqrt(1-cosine**2)*np.sin(azimuth), cosine))
    origin = np.broadcast_to([0., .1, 0.], direction.shape).copy()
    normal = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                       [0, 0, -1], [-.2, 0, 1]], dtype=float)
    bound = np.array([2., 2., half_y, half_y, -1., 1.5])
    lengths = np.linalg.norm(normal, axis=1)
    normal, bound = normal/lengths[:, None], bound/lengths
    inside = np.zeros(len(origin), dtype=bool)
    weight = np.ones(len(origin))/len(origin)
    component = np.ones(len(origin), dtype=int)  # native sky component
    result = np.zeros(4)
    tir_count = 0
    for depth in range(bounces+1):
        denominator = direction @ normal.T
        with np.errstate(divide="ignore", invalid="ignore"):
            times = (bound-origin @ normal.T)/denominator
        entering = np.where(denominator < -1e-14, times, -np.inf)
        exiting = np.where(denominator > 1e-14, times, np.inf)
        near_face, far_face = entering.argmax(axis=1), exiting.argmin(axis=1)
        slots = np.arange(len(origin))
        near, far = entering[slots, near_face], exiting[slots, far_face]
        hit = np.where(inside, far > 1e-8, (near > 1e-8) & (far >= near))
        for output in range(4):
            escape = ~hit & (direction[:, 2] > 0) & (component == output)
            result[output] += 100*weight[escape].sum()
        if depth == bounces:
            break
        face = np.where(inside, far_face, near_face)[hit]
        distance = np.where(inside, far, near)[hit]
        origin, direction, weight = origin[hit], direction[hit], weight[hit]
        inside, component = inside[hit], component[hit]
        origin = origin+distance[:, None]*direction
        # Scalar absorption .15 m^-1 acts on actual in-medium path distance.
        weight *= np.where(inside, np.exp(-.15*distance), 1.)
        facing = normal[face]*np.where(inside, -1, 1)[:, None]
        cos_i = -np.einsum("ij,ij->i", direction, facing)
        eta = np.where(inside, 1.5, 1/1.5)
        sine_squared = eta**2*(1-cos_i**2)
        tir_count += np.count_nonzero(sine_squared >= 1)
        cos_t = np.sqrt(np.maximum(0, 1-sine_squared))
        rs = ((eta*cos_i-cos_t)/(eta*cos_i+cos_t))**2
        rp = ((cos_i-eta*cos_t)/(cos_i+eta*cos_t))**2
        rho = np.where(sine_squared >= 1, 1., .5*(rs+rp))
        reflected_direction = direction+2*cos_i[:, None]*facing
        transmitted_direction = eta[:, None]*direction+(eta*cos_i-cos_t)[:, None]*facing
        reflected_component = np.where(component == 1, 2, component)
        transmitted_component = np.where(component == 1, 3, component)
        reflect_weight, transmit_weight = weight*rho, weight*(1-rho)*eta**2
        origin = np.concatenate((origin, origin))
        direction = np.concatenate((reflected_direction, transmitted_direction))
        inside = np.concatenate((inside, ~inside))
        component = np.concatenate((reflected_component, transmitted_component))
        weight = np.concatenate((reflect_weight, transmit_weight))
        active = weight > 0
        origin, direction = origin[active], direction[active]
        weight, inside, component = weight[active], inside[active], component[active]
    return result, tir_count


def test_native_wedge_refraction_and_tir_match_plane_quadrature(backend):
    mesh = slab(width=2)
    vertices = mesh.vertices.astype(float).copy()
    vertices[4:, 2] = 1.5+.2*vertices[4:, 0]
    wedge = Mesh(vertices, mesh.faces)
    reference, tir_count = _wedge_reference(192)
    coarse, _ = _wedge_reference(96)
    assert tir_count > 0
    # Declare a deterministic quadrature floor independently of native results.
    assert np.max(abs(reference-coarse)) < .12
    with ShortwaveSolver(Scene.from_meshes({"wedge": wedge}), backend=backend,
                         materials={"wedge": SolidDielectric(1.5, .15)}) as solver:
        result = solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], dni=0, dhi=100,
                              samples=32768, replicates=4, bounces=4, seed=71)
    value = result.irradiance[0, 0, :, 0]
    error = result.standard_error[0, 0, :, 0]
    assert np.all(abs(value-reference) < 4*error+.15), (value, reference, error)


def test_slender_vulkan_prism_never_hides_unresolved_medium(backend):
    if backend != "vulkan-rt":
        pytest.skip("regression for hardware traversal of slender boundary facets")
    mesh = slab(width=2)
    vertices = mesh.vertices.astype(float).copy()
    vertices[:, 1] *= 250
    vertices[4:, 2] = 1.5+.2*vertices[4:, 0]
    with ShortwaveSolver(Scene.from_meshes({"wedge": Mesh(vertices, mesh.faces)}),
                         backend=backend, materials={"wedge": SolidDielectric(1.5, .15)}) as solver:
        try:
            result = solver.solve([0, .1, 0], [0, 0, 1], [0, 0, 1], dni=0, dhi=100,
                                  samples=32768, replicates=4, bounces=4, seed=71)
        except CapabilityError as error:
            # Radeon860M missed the entrance on this very slender boundary
            # fixture. Refuse the inconsistent medium rather than fabricate a
            # transmitted path. Other hardware may resolve the same geometry.
            assert "current medium is inconsistent" in str(error)
            return
    reference, _ = _wedge_reference(192, half_y=500)
    assert np.all(abs(result.irradiance[0, 0, :, 0]-reference)
                  < 4*result.standard_error[0, 0, :, 0]+.15)
