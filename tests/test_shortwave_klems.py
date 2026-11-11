"""Full Klems orientation, WINDOW import, and BSDF sampling contracts."""
import math
import hashlib
from dataclasses import FrozenInstanceError
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from raystrack.shortwave.klems import (
    KLEMS_AZIMUTH_COUNTS,
    KLEMS_POLAR_BOUNDS_DEG,
    KLEMS_PROJECTED_SOLID_ANGLES,
    KlemsBSDF,
)


def _direction(theta, phi, *, front=True, incident=False):
    theta, phi = np.radians([theta, phi])
    xy_sign = -1 if incident else 1
    return np.array([
        xy_sign * np.sin(theta) * np.cos(phi),
        xy_sign * np.sin(theta) * np.sin(phi),
        (1 if front else -1) * np.cos(theta),
    ])


def _write_window(path, order, layers):
    root = ET.Element("WindowElement", xmlns="http://windows.lbl.gov")
    layer = ET.SubElement(ET.SubElement(root, "Optical"), "Layer")
    definition = ET.SubElement(layer, "DataDefinition")
    ET.SubElement(definition, "IncidentDataStructure").text = order
    basis = ET.SubElement(definition, "AngleBasis")
    ET.SubElement(basis, "AngleBasisName").text = "LBNL/Klems Full"
    for index, count in enumerate(KLEMS_AZIMUTH_COUNTS):
        ring = ET.SubElement(basis, "AngleBasisBlock")
        ET.SubElement(ring, "nPhis").text = str(count)
        limits = ET.SubElement(ring, "ThetaBounds")
        ET.SubElement(limits, "LowerTheta").text = str(KLEMS_POLAR_BOUNDS_DEG[index])
        ET.SubElement(limits, "UpperTheta").text = str(KLEMS_POLAR_BOUNDS_DEG[index + 1])
    for band, lobes in layers.items():
        data = ET.SubElement(layer, "WavelengthData")
        ET.SubElement(data, "Wavelength", unit="Integral").text = band
        for label, values in lobes.items():
            block = ET.SubElement(data, "WavelengthDataBlock")
            ET.SubElement(block, "WavelengthDataDirection").text = label
            for axis in ("ColumnAngleBasis", "RowAngleBasis"):
                ET.SubElement(block, axis).text = "LBNL/Klems Full"
            matrix = values if order == "Rows" else values.T
            ET.SubElement(block, "ScatteringData").text = ",\n".join(map(str, matrix.ravel()))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_full_klems_projected_energy_and_immutable_lobes():
    np.testing.assert_allclose(KLEMS_PROJECTED_SOLID_ANGLES.sum(), math.pi)
    values = np.broadcast_to(
        np.array([0.2, 0.3, 0.6, 0.4])[:, None, None] / math.pi, (4, 145, 145),
    ).copy()
    bsdf = KlemsBSDF(values)
    values[:] = 0
    assert bsdf.hemispherical([0, 0, 1]) == pytest.approx((0.2, 0.6))
    assert bsdf.hemispherical([0, 0, -1]) == pytest.approx((0.3, 0.4))
    assert bsdf.evaluate([0, 0, -1], [0, 0, 1]) == pytest.approx(0.4 / math.pi)
    with pytest.raises(ValueError):
        bsdf.matrices.setflags(write=True)
    values[[0, 2]] = 0.6 / math.pi
    with pytest.raises(ValueError, match="exceeds unit energy"):
        KlemsBSDF(values)
    values[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        KlemsBSDF(values)


@pytest.mark.parametrize("reverse", [False, True])
def test_sampling_matches_evaluation_pdf_and_directional_energy(reverse):
    values = np.zeros((4, 145, 145))
    values[0, 1, 3] = 0.2 / KLEMS_PROJECTED_SOLID_ANGLES[1 if reverse else 3]
    if reverse:
        values[3, 2, 3] = 0.4 / KLEMS_PROJECTED_SOLID_ANGLES[2]
        fixed = _direction(10, 90)
    else:
        values[2, 1, 2] = 0.4 / KLEMS_PROJECTED_SOLID_ANGLES[2]
        fixed = _direction(10, 0, incident=True)
    bsdf = KlemsBSDF(values)
    rng = np.random.default_rng(4)
    sample_count = 120
    draws = rng.random((sample_count, 3))
    draws[:, 0] = (np.arange(sample_count) + 0.5) / sample_count
    draws[0, 1:] = 0.0
    transmitted = 0
    for draw in draws:
        sample = bsdf.sample(fixed, draw, reverse=reverse)
        incident, outgoing = (sample.direction, fixed) if reverse else (fixed, sample.direction)
        assert sample.value == pytest.approx(bsdf.evaluate(incident, outgoing))
        assert sample.pdf == pytest.approx(bsdf.pdf(incident, outgoing, reverse=reverse))
        assert sample.weight == pytest.approx(0.6)
        assert np.linalg.norm(sample.direction) == pytest.approx(1)
        assert sample.transmitted == (sample.direction[2] < 0.0)
        transmitted += sample.transmitted
    assert transmitted == 80
    assert bsdf.sample([0, 0, 1], [0, 0, 0], reverse=reverse) is None
    assert bsdf.sample([1, 0, 0], [0, 0, 0], reverse=reverse) is None


@pytest.mark.parametrize("order", ["Rows", "Columns"])
def test_window_import_selects_band_order_and_front_mapping(tmp_path, order):
    path = tmp_path / "bands.xml"
    shape = (145, 145)
    front = np.zeros(shape)
    front[1, 3] = 0.5
    back = np.zeros(shape)
    back[2, 4] = 0.75
    _write_window(path, order, {
        "Solar": {
            "Reflection Front": front,
            "Reflection Back": back,
            "Transmission Front": front * 2,
            "Transmission Back": back * 2,
        },
        "Visible": {"Reflection Front": front * 3},
    })
    solar = KlemsBSDF.from_xml(path)
    np.testing.assert_array_equal(solar.matrices, [front, back, front*2, back*2])
    reversed_sides = KlemsBSDF.from_xml(path, window_front="back")
    np.testing.assert_array_equal(reversed_sides.matrices, solar.matrices[[1, 0, 3, 2]])
    visible = KlemsBSDF.from_xml(path, band="visible")
    assert visible.band == "visible"
    np.testing.assert_array_equal(visible.matrices[0], front * 3)
    np.testing.assert_array_equal(visible.matrices[1:], 0.0)
    assert solar.evaluate(_direction(10, 0, incident=True), _direction(10, 90)) == 0.5
    assert solar.evaluate(_direction(10, 0, incident=True), _direction(10, 90, front=False)) == 1.0


def test_single_window_transmission_can_be_reversed_by_reciprocity(tmp_path):
    path = tmp_path / "visible.xml"
    values = np.zeros((145, 145))
    values[1, 3] = 0.8
    _write_window(path, "Columns", {"Visible": {"Transmission Front": values}})
    with pytest.raises(ValueError, match="no solar"):
        KlemsBSDF.from_xml(path)
    bsdf = KlemsBSDF.from_xml(path, band="visible")
    wi = _direction(10, 0, incident=True)
    wo = _direction(10, 90, front=False)
    assert bsdf.evaluate(wi, wo) == pytest.approx(0.8)
    assert bsdf.evaluate(wo, wi) == pytest.approx(0.8)


def test_directional_reciprocal_sheet_energy_and_reverse_pdf():
    # A symmetric, half-turn invariant anisotropic BTDF is reciprocal when
    # directions point away from the sheet on both sides (same refractive index).
    features = np.concatenate([
        np.sin(np.radians((low+high)/2))**2 * np.cos(4*np.pi*np.arange(count)/count)
        for low, high, count in zip(KLEMS_POLAR_BOUNDS_DEG[:-1],
                                   KLEMS_POLAR_BOUNDS_DEG[1:], KLEMS_AZIMUTH_COUNTS)
    ])
    values = np.empty((4,145,145))
    values[:2] = .1/math.pi
    values[2:] = .35/math.pi*(1+.6*features[:,None]*features[None,:])
    bsdf = KlemsBSDF(values)
    rng = np.random.default_rng(317)
    for front in (True,False):
        for theta, phi in ((0,0),(30,20),(60,110),(80,230)):
            fixed = _direction(theta,phi,front=front)
            rho,tau = bsdf.hemispherical(fixed)
            assert rho+tau <= 1
            for reverse in (False,True):
                for draw in rng.random((12,3)):
                    sampled = bsdf.sample(fixed,draw,reverse=reverse)
                    wi,wo = (sampled.direction,fixed) if reverse else (fixed,sampled.direction)
                    assert bsdf.evaluate(wi,wo) == pytest.approx(bsdf.evaluate(wo,wi))
                    assert sampled.pdf == pytest.approx(bsdf.pdf(wi,wo,reverse=reverse))
                    assert 0 < sampled.weight <= 1


def test_window_import_preserves_source_metadata_and_hash(tmp_path):
    path = tmp_path / "controlled-synthetic.xml"
    _write_window(path, "Columns", {"Solar": {"Transmission Front": np.full((145, 145), .2/math.pi)}})
    tree = ET.parse(path)
    # The fixture declares metadata, not genuine measurement or a data license.
    layer = tree.getroot().find("{*}Optical/{*}Layer")
    material = ET.SubElement(layer, "Material")
    ET.SubElement(material, "Name").text = "Controlled synthetic specimen"
    ET.SubElement(material, "Thickness", unit="Millimeter").text = "1.5"
    data = layer.find("{*}WavelengthData")
    ET.SubElement(data, "SourceSpectrum").text = "Controlled Solar input"
    ET.SubElement(data, "DetectorSpectrum").text = "Neutral radiometric"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    bsdf = KlemsBSDF.from_xml(path, window_front="back")
    source = bsdf.source
    assert source.filename == str(path)
    assert source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.incident_order == "columns" and source.window_front == "back"
    assert source.supplied_lobes == ("transmission back",)
    assert source.derived_lobes == ("transmission front",)
    assert source.band == "solar" and source.basis == "LBNL/Klems Full"
    assert source.units == "sr^-1"
    assert 'unit="Millimeter"' in source.material_xml
    assert "Controlled synthetic specimen" in source.material_xml
    assert source.spectra == (("Controlled Solar input", "Neutral radiometric"),)
    with pytest.raises(FrozenInstanceError):
        source.sha256 = "changed"
    assert KlemsBSDF(np.zeros((4, 145, 145))).source is None
    with path.open("rb") as stream:
        assert KlemsBSDF.from_xml(stream).source.sha256 == source.sha256


@pytest.mark.parametrize("target,value,attribute,match", [
    ("Wavelength", "nm", "unit", "Integral"),
    ("ScatteringData", "W/m2", "unit", "sr\\^-1"),
    ("ScatteringDataType", "Radiance", None, "BSDF densities"),
    ("RowAngleBasis", "LBNL/Klems Half", None, "both matrix axes"),
    ("IncidentDataStructure", "TensorTree4", None, "Rows or Columns"),
])
def test_window_rejects_unsupported_units_and_structures(tmp_path, target, value, attribute, match):
    path = tmp_path / "unsupported.xml"
    _write_window(path, "Rows", {"Solar": {"Reflection Front": np.zeros((145, 145))}})
    tree = ET.parse(path)
    element = tree.getroot().find(f".//{{*}}{target}")
    if element is None:
        element = ET.SubElement(tree.getroot().find(".//{*}WavelengthDataBlock"), target)
    if attribute:
        element.set(attribute, value)
    else:
        element.text = value
    tree.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match=match):
        KlemsBSDF.from_xml(path)
