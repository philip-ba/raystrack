"""Independent-band energy and contract checks, without broadband inference."""
import os

import numpy as np
import pytest

from raystrack import CapabilityError, Mesh, Scene
from raystrack.backends.native import native_library
from raystrack.shortwave import ShortwaveSolver, ThinDiffuse
from raystrack.shortwave.spectral import BandInput, WavelengthBand, solve_independent_bands


def band(name="visible", low=400, high=700, materials=None, dni=0, dhi=0):
    return BandInput(WavelengthBand(name, low, high), materials or {}, dni, dhi,
                     "synthetic constant optics; independent validation fixture",
                     "synthetic band-integrated W/m²; unit top-hat response")


@pytest.fixture(params=("native-builtin", "embree", "vulkan-rt"))
def backend(request):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("native library not configured")
    try:
        native_library(request.param)
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    return request.param


@pytest.mark.parametrize("low,high", [(700, 400), (0, 700), (np.nan, 700), (400, np.inf), (True, 700)])
def test_invalid_band_edges(low, high):
    with pytest.raises(ValueError):
        WavelengthBand("invalid", low, high)


def test_band_inputs_owned_and_no_implicit_broadband_or_rgb():
    dni = np.array([10., 20.])
    item = band(dni=dni)
    dni[:] = 0
    np.testing.assert_array_equal(item.dni, [10, 20])
    with pytest.raises(ValueError):
        item.dni.setflags(write=True)
    with pytest.raises(TypeError):
        item.materials["x"] = ThinDiffuse(0, .5)
    for forcing in ([[1, 2, 3]], True, -1, np.inf):
        with pytest.raises(ValueError):
            band(dni=forcing)
    with pytest.raises(ValueError, match="optical_provenance"):
        BandInput(WavelengthBand("v", 400, 700), {}, 1, 0, "", "weather")


def test_overlap_missing_optics_and_overrides_rejected():
    with pytest.raises(ValueError, match="overlap"):
        solve_independent_bands(Scene(), [0, 0, 0], [0, 0, 1], [0, 0, 1],
                                [band(), band("nir", 600, 900)])
    scene = Scene.from_meshes({"panel": Mesh([[0, 0, 1], [1, 0, 1], [0, 1, 1]], [[0, 1, 2]])})
    with pytest.raises(ValueError, match="assignments"):
        solve_independent_bands(scene, [0, 0, 0], [0, 0, 1], [0, 0, 1], [band()])
    with pytest.raises(ValueError, match="overridden"):
        solve_independent_bands(Scene(), [0, 0, 0], [0, 0, 1], [0, 0, 1], [band()], dni=100)


def test_band_forcing_sum_and_duration_integration(backend):
    items = [band(dni=[100, 200]), band("nir", 700, 2500, dni=[300, 400])]
    result = solve_independent_bands(Scene(), [0, 0, 0], [0, 0, 1], [[0, 0, 1], [0, 0, -1]],
                                     items, backend=backend, bounces=0, samples=0,
                                     durations_hours=[.5, 2])
    np.testing.assert_allclose(result.total[:, 0], [400, 0])
    np.testing.assert_allclose(result.exposure()[0], [200, 0, 0, 0])
    with pytest.raises(ValueError):
        result.total.setflags(write=True)


def test_different_band_optics_and_direct_diffuse_linearity(backend):
    panel = Mesh([[-10000, -10000, 1], [10000, -10000, 1], [10000, 10000, 1],
                  [-10000, 10000, 1]], [[0, 1, 2], [0, 2, 3]])
    scene = Scene.from_meshes({"panel": panel})
    optics = [ThinDiffuse(0, .2), ThinDiffuse(0, .8)]
    items = [band(materials={"panel": optics[0]}, dni=100, dhi=20),
             band("nir", 700, 2500, {"panel": optics[1]}, dni=300, dhi=40)]
    args = ([.123, .234, 0], [0, 0, 1], [0, 0, 1])
    options = dict(samples=4096, bounces=1, seed=37, durations_hours=[2])
    result = solve_independent_bands(scene, *args, items, backend=backend, **options)
    np.testing.assert_allclose(result.irradiance[:, 0, 0, 3], [24, 272], atol=.01)
    np.testing.assert_allclose(result.total, 296, atol=.02)
    np.testing.assert_allclose(result.exposure()[0, 3], 592, atol=.04)
    for item, actual in zip(items, result.results):
        with ShortwaveSolver(scene, backend=backend, materials=item.materials) as solver:
            direct = solver.solve(*args, dni=item.dni, dhi=0, **options)
            diffuse = solver.solve(*args, dni=0, dhi=item.dhi, **options)
        np.testing.assert_allclose(actual.irradiance, direct.irradiance + diffuse.irradiance, atol=1e-10)


def test_rgb_optics_are_not_spectral_data(backend):
    scene = Scene.from_meshes({"panel": Mesh([[0, 0, 1], [1, 0, 1], [0, 1, 1]], [[0, 1, 2]])})
    item = band(materials={"panel": ThinDiffuse(0, [.1, .2, .3])}, dni=1)
    with pytest.raises(ValueError, match="non-neutral RGB"):
        solve_independent_bands(scene, [0, 0, 0], [0, 0, 1], [0, 0, 1], [item], backend=backend)


def test_caller_array_edits_between_bands_do_not_change_snapshot(backend, monkeypatch):
    from raystrack.shortwave import spectral
    points, normals = np.array([[0., 0, 0]]), np.array([[0., 0, 1]])
    sun, hours = np.array([[0., 0, 1]]), np.array([2.])
    class EditingSolver(ShortwaveSolver):
        def solve(self, *args, **kwargs):
            result = super().solve(*args, **kwargs)
            normals[:] = [0, 0, -1]
            sun[:] = [0, 0, -1]
            hours[:] = 9
            return result
    monkeypatch.setattr(spectral, "ShortwaveSolver", EditingSolver)
    result = spectral.solve_independent_bands(Scene(), points, normals, sun,
        [band(dni=100), band("nir", 700, 2500, dni=300)], backend=backend,
        samples=0, bounces=0, durations_hours=hours)
    np.testing.assert_allclose(result.total, 400)
    np.testing.assert_allclose(result.exposure()[0, 0], 800)
