"""Unmodified, licensed measurement-based Solar Full Klems evidence."""
import hashlib
import math
import os

import numpy as np
import pytest

from raystrack.shortwave import KlemsBSDF
from validation.validate_measured_bsdf import (
    FIXTURE, SHA256, inspect_fixture, native_reference, quadrature,
)


def test_pinned_cgdb_source_and_optical_conventions():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    bsdf = KlemsBSDF.from_xml(FIXTURE)
    assert bsdf.source.incident_order == "columns"
    assert bsdf.source.band == "solar"
    assert bsdf.source.derived_lobes == ()
    assert len(bsdf.source.supplied_lobes) == 4
    assert set(bsdf.source.spectra) == {("ASTM 891 direct 1.5 atmosphere", "Flat")}
    assert "Satine 5500" in bsdf.source.material_xml
    # Independent CGDB V8 release-note normal values rounded to three digits.
    assert bsdf.hemispherical([0, 0, 1]) == pytest.approx((.465, .118), abs=.0005)
    assert bsdf.hemispherical([0, 0, -1]) == pytest.approx((.526, .116), abs=.0005)
    assert not np.array_equal(bsdf.matrices, KlemsBSDF.from_xml(FIXTURE, band="visible").matrices)


def test_import_gate_energy_pdf_and_measured_reciprocity_residuals():
    report = inspect_fixture()
    assert report["maximum_energy"] < 1
    assert report["pdf_integral_max_residual"] < 1e-12
    assert report["sample_evaluate_max_residual"] == 0
    assert report["sample_pdf_max_residual"] < 1e-12
    # Report measurement inconsistency, never symmetrize the original to pass.
    np.testing.assert_allclose(report["reciprocity_max_absolute_sr_inverse"],
                               [.050214, .0483, .019976], atol=1e-12)


def test_directional_finite_transmission_integrates_original_density():
    bsdf = KlemsBSDF.from_xml(FIXTURE)
    centers, weights, _ = quadrature()
    assert weights.sum() == pytest.approx(math.pi)
    for patch in (0, 25, 69, 144):
        wi = centers[patch] * [-1, -1, 1]
        explicit_integral = sum(bsdf.evaluate(wi, center*[1, 1, -1])*weight
                                for center, weight in zip(centers, weights))
        assert explicit_integral == pytest.approx(bsdf.hemispherical(wi)[1], abs=1e-14)
    # Actual finite-width outgoing scatter exists away from straight-through.
    assert bsdf.evaluate([0, 0, 1], [1, 0, -1]) > 0
    assert bsdf.hemispherical(centers[144]*[-1, -1, 1])[1] < .1
    assert bsdf.hemispherical([0, 0, 1])[1] > .11


def test_changed_source_bytes_rejected_before_import(tmp_path):
    changed = tmp_path / "changed.xml"
    changed.write_bytes(FIXTURE.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        inspect_fixture(changed)


@pytest.mark.parametrize("backend", ["native-builtin", "embree", "vulkan-rt"])
@pytest.mark.parametrize("reverse_sheet", [False, True])
def test_measured_solar_native_direct_diffuse_combined(backend, reverse_sheet):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("requires explicitly selected native library")
    # Explicitly requested backend availability failures must remain failures.
    report = native_reference(backend, reverse_sheet=reverse_sheet)
    assert report["passed"], report
