"""Diffuse sidedness and discrete lossless parallel-plate material contracts."""
import math

import numpy as np
import pytest

from raystrack.shortwave.materials import ClearGlazing, DiffuseSheet, ThinDiffuse
from validation.validate_clear_glazing import plate_fresnel


def test_asymmetric_diffuse_sheet_energy_rgb_and_ownership():
    front = np.array([.1, .2, .3])
    sheet = DiffuseSheet(front, .4, [.5, .4, .3])
    front[:] = 1
    assert sheet.front_reflectance == (.1, .2, .3)
    assert sheet.back_reflectance == (.4,) * 3
    assert sheet.transmittance == (.5, .4, .3)
    assert ThinDiffuse(.2, .3).reflectance == (.2,) * 3
    for args in ((.8, .1, .3), (.1, .8, .3), (.1, .1, np.nan), (.1, .1, -.1)):
        with pytest.raises(ValueError):
            DiffuseSheet(*args)


@pytest.mark.parametrize("index", [1, 1.01, 1.52, 2.5])
def test_clear_plate_matches_independent_reference_and_energy(index):
    material = ClearGlazing(index)
    cosine = np.r_[0, np.geomspace(1e-8, 1, 32)]
    reflectance, transmittance = material.coefficients(cosine)
    expected = plate_fresnel(cosine, index)
    np.testing.assert_allclose([reflectance, transmittance], expected, atol=4e-15)
    np.testing.assert_array_equal(reflectance + transmittance, 1)
    assert np.all(reflectance >= 0) and np.all(transmittance >= 0)
    interface = ((index - 1) / (index + 1))**2
    assert material.coefficients(1)[0] == pytest.approx(2*interface/(1+interface))


@pytest.mark.parametrize("front", [False, True])
def test_clear_plate_delta_events_direction_mass_and_immutable_direction(front):
    material = ClearGlazing()
    cosine = math.cos(math.radians(55))
    wi = np.array([math.sqrt(1-cosine**2), 0, cosine * (1 if front else -1)])
    reflectance, transmittance = material.coefficients(cosine)
    for draw, transmitted, coefficient in ((0, False, reflectance),
                                            (reflectance, True, transmittance)):
        sample = material.sample(wi*9, draw)
        expected = -wi
        if not transmitted:
            expected[2] = wi[2]
        np.testing.assert_allclose(sample.direction, expected)
        assert sample.transmitted is transmitted
        assert sample.coefficient == pytest.approx(coefficient)
        assert sample.probability_mass == pytest.approx(coefficient)
        assert sample.weight == pytest.approx(1)
        assert not hasattr(sample, "pdf") and not hasattr(sample, "value")
        with pytest.raises(ValueError):
            sample.direction.setflags(write=True)
        # Both the reflection and transmission mapping are involutions.
        reverse = material.sample(sample.direction, draw)
        np.testing.assert_allclose(reverse.direction, wi)


def test_clear_plate_event_frequencies_are_discrete_masses():
    material = ClearGlazing()
    cosine = .4
    reflectance, _ = material.coefficients(cosine)
    count = 1000
    events = [material.sample([math.sqrt(1-cosine**2), 0, cosine], (i+.5)/count)
              for i in range(count)]
    assert sum(not event.transmitted for event in events)/count == pytest.approx(reflectance, abs=1/count)
    assert ClearGlazing(1).sample([0, 0, 1], 0).transmitted
    assert material.sample([1, 0, 0], .5) is None


@pytest.mark.parametrize("index", [True, "1.52", [1.52], 1+0j, .9, np.inf, np.nan])
def test_clear_plate_rejects_unsupported_index(index):
    with pytest.raises(ValueError, match="refractive_index"):
        ClearGlazing(index)


@pytest.mark.parametrize("cosine", [True, "0.5", 1+0j, -.1, 1.1, np.inf, np.nan])
def test_clear_plate_rejects_invalid_incidence(cosine):
    with pytest.raises(ValueError, match="cosine"):
        ClearGlazing().coefficients(cosine)


@pytest.mark.parametrize("direction,draw", [([0, 0, 0], .5), ([0, np.inf, 1], .5),
    ([0, np.nan, 1], .5), ([0, 1], .5), ([0, 0, 1], -.1), ([0, 0, 1], 1),
    ([0, 0, 1], np.nan), ([0, 0, 1], [.5]), ([0, 0, 1], True)])
def test_clear_plate_rejects_invalid_sample_inputs(direction, draw):
    with pytest.raises(ValueError):
        ClearGlazing().sample(direction, draw)
