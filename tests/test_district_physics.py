"""Synthetic finite rough transmission and moving-district optical gates."""
import math
import os

import numpy as np
import pytest

from raystrack.shortwave import KlemsBSDF
from validation.validate_measured_bsdf import quadrature


@pytest.mark.parametrize("exponent", [0., 1., 8., 64.])
def test_rough_transmission_finite_reciprocal_energy_and_pdf(exponent):
    material = KlemsBSDF.rough_transmission(.65, exponent, reflectance=.2)
    centers, weights, half = quadrature()
    energy = material.matrices @ weights
    np.testing.assert_allclose(material.matrices[2,0,1:9], material.matrices[2,0,1])
    assert np.max(energy[:2]+energy[2:]) == pytest.approx(.85)
    assert np.min(energy) >= 0
    np.testing.assert_allclose(material.matrices[2], material.matrices[3].T[half][:, half], atol=1e-14)
    assert material.source is None  # Synthetic, never measurement provenance.
    rng = np.random.default_rng(734)
    for side in (1, -1):
        for patch in (0, 25, 69, 144):
            incident = centers[patch]*[-1, -1, side]
            for reverse in (False, True):
                for draw in rng.random((8, 3)):
                    sample = material.sample(incident, draw, reverse=reverse)
                    wi, wo = (sample.direction, incident) if reverse else (incident, sample.direction)
                    assert sample.pdf == pytest.approx(material.pdf(wi, wo, reverse=reverse))
                    assert sample.value == pytest.approx(material.evaluate(wi, wo))
                    assert 0 < sample.weight <= .85+1e-14
    if exponent == 0:
        np.testing.assert_allclose(material.matrices[2:], .65/math.pi)
    else:
        assert material.matrices[2, 25, 25] > material.matrices[2, 25, 100]


@pytest.mark.parametrize("tau,exponent,rho", [(-.1, 1, 0), (1.1, 1, 0),
    (.5, -1, 0), (.5, np.inf, 0), (.5, True, 0), (.5, 1, .6),
    (.5, 1, np.nan), ([.5], 1, 0), (.5, [1], 0), (.5, 257, 0)])
def test_rough_transmission_rejects_invalid_parameters(tau, exponent, rho):
    with pytest.raises(ValueError):
        KlemsBSDF.rough_transmission(tau, exponent, reflectance=rho)


def test_snell_angular_jacobian_is_not_fresnel_or_radiance_weight():
    # Differential solid-angle map at one smooth interface, independently
    # differentiating Snell. A point-source connection additionally needs a
    # spatial boundary-value solution; this check alone cannot enable DNI.
    from raystrack.shortwave import SolidDielectric

    glass = SolidDielectric(1.5)
    for entering, eta, angles in ((True, 1/1.5, [10, 50, 85]),
                                  (False, 1.5, [10, 30, 40])):
        for degrees in angles:
            theta = np.radians(degrees)
            sign = 1 if entering else -1
            direction = [np.sin(theta), 0, sign*np.cos(theta)]
            event = glass.sample(direction, .999999)
            assert event.transmitted
            transmitted = np.arccos(abs(event.direction[2]))
            h = 1e-6
            lo = np.arcsin(eta*np.sin(theta-h))
            hi = np.arcsin(eta*np.sin(theta+h))
            numeric = np.sin(transmitted)/np.sin(theta)*(hi-lo)/(2*h)
            expected = eta**2*np.cos(theta)/np.cos(transmitted)
            assert numeric == pytest.approx(expected, rel=1e-8)
            assert event.weight == pytest.approx(eta**2)
    # Parallel entrance/exit maps cancel in angle; wedges generally do not.
    cosine = .6
    inside = glass.coefficients(cosine)[2]
    assert (cosine/(1.5**2*inside))*(1.5**2*inside/cosine) == pytest.approx(1)


@pytest.mark.parametrize("backend", ["native-builtin", "embree", "vulkan-rt"])
def test_district_selected_constituent_references(backend):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("requires explicitly selected native library")
    from validation.district_physics import selected_references

    report = selected_references(backend, samples=4096, replicates=4)
    assert report["passed"], report
    assert report["shadow"]["passed"], report["shadow"]
