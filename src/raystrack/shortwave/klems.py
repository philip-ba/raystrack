"""Full Klems scattering matrices and WINDOW XML import."""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model.geometry import _immutable_array

# The full Klems basis and direction conventions follow Radiance bsdf_m.c.
# https://github.com/LBNL-ETA/Radiance/blob/master/src/common/bsdf_m.c
KLEMS_POLAR_BOUNDS_DEG = _immutable_array(np.array([0, 5, 15, 25, 35, 45, 55, 65, 75, 90]))
KLEMS_AZIMUTH_COUNTS = _immutable_array(np.array([1, 8, 16, 20, 24, 24, 24, 16, 12]))
KLEMS_RING_OFFSETS = _immutable_array(np.r_[0, np.cumsum(KLEMS_AZIMUTH_COUNTS)])
_cosines = np.cos(np.radians(KLEMS_POLAR_BOUNDS_DEG))
_cosines[-1] = 0.0
KLEMS_COSINE_BOUNDS = _immutable_array(_cosines)
KLEMS_PROJECTED_SOLID_ANGLES = _immutable_array(np.repeat(
    -np.pi * np.diff(KLEMS_COSINE_BOUNDS**2) / KLEMS_AZIMUTH_COUNTS, KLEMS_AZIMUTH_COUNTS,
))
_PATCHES = int(KLEMS_RING_OFFSETS[-1])
_PATCH_RINGS = np.repeat(np.arange(len(KLEMS_AZIMUTH_COUNTS)), KLEMS_AZIMUTH_COUNTS)
_HALF_TURN = np.concatenate([
    start + (np.arange(count) + count // 2) % count
    for start, count in zip(KLEMS_RING_OFFSETS, KLEMS_AZIMUTH_COUNTS)
])
_LOBE_NAMES = ("reflection front", "reflection back", "transmission front", "transmission back")


def _unit_vector(vector):
    values = np.asarray(vector)
    if values.shape != (3,) or values.dtype.kind not in "fiu":
        raise ValueError("direction must be a real 3-vector")
    values = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("direction must be finite and nonzero")
    values = values / scale
    return values / np.linalg.norm(values)


def _index(direction, *, incident):
    x, y, z = direction
    theta = math.degrees(math.atan2(math.hypot(x, y), abs(z)))
    ring = min(np.searchsorted(KLEMS_POLAR_BOUNDS_DEG, theta, side="right") - 1,
               len(KLEMS_AZIMUTH_COUNTS) - 1)
    count = int(KLEMS_AZIMUTH_COUNTS[ring])
    if incident:
        x, y = -x, -y
    width = math.tau / count
    column = math.floor(math.atan2(y, x) / width + 0.5) % count
    return int(KLEMS_RING_OFFSETS[ring]) + column


def _band_name(band):
    if not isinstance(band, str) or band.lower() not in ("solar", "visible"):
        raise ValueError("band must be 'solar' or 'visible'")
    return band.lower()


@dataclass(frozen=True, eq=False)
class KlemsSample:
    """Direction and BSDF/PDF, both densities per unit solid angle."""

    direction: np.ndarray = field(repr=False)
    value: float
    pdf: float
    transmitted: bool

    @property
    def weight(self) -> float:
        """Path throughput factor for this sample."""
        return self.value * abs(self.direction[2]) / self.pdf


@dataclass(frozen=True)
class KlemsSource:
    """Immutable provenance of a WINDOW XML import, without a license inference.

    File and binary-stream bytes are identified by SHA256; text streams hash
    their UTF-8 encoding. Material XML and spectral
    labels retain source declarations; their presence does not establish that
    the data were measured or grant permission to redistribute them.
    """

    filename: str | None
    sha256: str
    incident_order: str
    window_front: str
    band: str
    supplied_lobes: tuple
    derived_lobes: tuple
    material_xml: str
    spectra: tuple
    basis: str = "LBNL/Klems Full"
    units: str = "sr^-1"


@dataclass(frozen=True, eq=False)
class KlemsBSDF:
    """An energy-bounded, single-band thin-surface scattering material.

    Supply four full Klems matrices in sr⁻¹, ordered front reflection, back
    reflection, front transmission, and back transmission. Each is indexed
    [incident, outgoing]. Front/back refer to the geometric normal.

    Directions point away from the surface and are in a right-handed material
    frame. Local +Z is the front normal and +X is the zero-azimuth tangent.
    Incident directions use reversed XY for binning, so straight-through
    transmission falls on the matrix diagonal. Both sides share XY.
    """

    matrices: np.ndarray = field(repr=False)
    band: str = "solar"
    source: KlemsSource | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        values = np.asarray(self.matrices)
        if values.shape != (4, _PATCHES, _PATCHES) or values.dtype.kind not in "fiu":
            raise ValueError("Klems matrices must be real with shape (4, 145, 145)")
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Klems scattering values must be finite and nonnegative")
        hemispherical = values @ KLEMS_PROJECTED_SOLID_ANGLES
        if np.any(hemispherical[:2] + hemispherical[2:] > 1.0 + 1e-6):
            raise ValueError("Klems reflection plus transmission exceeds unit energy")
        object.__setattr__(self, "matrices", _immutable_array(values))
        object.__setattr__(self, "band", _band_name(self.band))

    @classmethod
    def rough_transmission(cls, transmittance, exponent, *, reflectance=0) -> KlemsBSDF:
        """Construct a synthetic, reciprocal finite directional sheet.

        The Full Klems cell density is proportional to
        ``max(dot(c_i, c_o), 0)**exponent``, using polar/azimuth cell centers
        on one hemisphere. Incident XY reversal makes its diagonal straight
        through. One global factor sets the largest directional-hemispherical
        transmission to ``transmittance``; absorption varies with direction.
        Reflection is Lambertian with the same coefficient on both sides.

        This defines a piecewise constant finite BTDF in sr^-1, not a
        microfacet dielectric, measured product, or delta event. Exponent 0
        is diffuse; 1..256 concentrates the lobe within this coarse basis.
        A global normalization preserves reciprocity, unlike row normalization.
        """
        parameters = []
        for name, value, limit in (("transmittance", transmittance, 1),
                                   ("exponent", exponent, 256),
                                   ("reflectance", reflectance, 1)):
            raw = np.asarray(value)
            if (raw.shape != () or raw.dtype.kind not in "fiu"
                    or not np.isfinite(raw) or not 0 <= raw <= limit):
                raise ValueError(f"{name} must be a finite real scalar in [0, {limit}]")
            parameters.append(float(raw))
        tau, power, rho = parameters
        if tau+rho > 1:
            raise ValueError("reflectance plus transmittance must be at most one")
        centers = []
        for low, high, count in zip(KLEMS_POLAR_BOUNDS_DEG[:-1],
                                    KLEMS_POLAR_BOUNDS_DEG[1:], KLEMS_AZIMUTH_COUNTS):
            theta = 0.0 if count == 1 else math.radians((low+high)/2)
            for sector in range(int(count)):
                phi = math.tau*sector/count
                centers.append([math.sin(theta)*math.cos(phi),
                                math.sin(theta)*math.sin(phi), math.cos(theta)])
        centers = np.asarray(centers)
        density = np.maximum(centers @ centers.T, 0)**power
        # Enforce the exact reciprocity permutation despite trigonometric
        # roundoff; this is synthetic construction, not repair of imported data.
        density = .5*(density+density.T[np.ix_(_HALF_TURN, _HALF_TURN)])
        density *= tau/np.max(density @ KLEMS_PROJECTED_SOLID_ANGLES)
        matrices = np.empty((4, _PATCHES, _PATCHES))
        matrices[:2] = rho/math.pi
        matrices[2:] = density
        return cls(matrices)

    @classmethod
    def from_xml(cls, path, *, band="solar", window_front="front") -> KlemsBSDF:
        """Load one named band of a WINDOW Full Klems XML file.

        By default WINDOW Front corresponds to mesh front. Pass
        `window_front="back"` to match Radiance's mapping of WINDOW labels to
        the same mesh winding. Missing reflection is zero. A single supplied
        transmission matrix is reversed by reciprocity to fill the other side.
        ``source`` records immutable import provenance. Missing XML units use
        the standard implicit sr^-1 density and Integral band conventions.
        """
        band = _band_name(band)
        if window_front not in ("front", "back"):
            raise ValueError("window_front must be 'front' or 'back'")
        if hasattr(path, "read"):
            raw = path.read()
            filename = getattr(path, "name", None)
        else:
            raw = Path(path).read_bytes()
            filename = str(path)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        root = ET.fromstring(raw)
        for element in root.iter():
            element.tag = element.tag.rsplit("}", 1)[-1]
        layers = root.findall("./Optical/Layer")
        if root.tag != "WindowElement" or len(layers) != 1:
            raise ValueError("expected one optical layer in WINDOW BSDF XML")
        layer = layers[0]
        definition = layer.find("DataDefinition")
        if definition is None:
            raise ValueError("WINDOW XML is missing DataDefinition")
        order = definition.findtext("IncidentDataStructure", "").strip().lower()
        if order not in ("rows", "columns"):
            raise ValueError("IncidentDataStructure must be Rows or Columns")
        _validate_angle_basis(definition)

        matrices = np.zeros((4, _PATCHES, _PATCHES))
        supplied = set()
        spectra = []
        for data in layer.findall("WavelengthData"):
            if data.findtext("Wavelength", "").strip().lower() != band:
                continue
            wavelength = data.find("Wavelength")
            if wavelength.get("unit", "Integral").strip().lower() != "integral":
                raise ValueError("WINDOW Solar/Visible data must use Integral wavelength units")
            spectra.append((data.findtext("SourceSpectrum", "").strip(),
                            data.findtext("DetectorSpectrum", "").strip()))
            for block in data.findall("WavelengthDataBlock"):
                label = " ".join(block.findtext("WavelengthDataDirection", "").lower().split())
                if label not in _LOBE_NAMES:
                    raise ValueError(f"unsupported WINDOW scattering direction {label!r}")
                lobe = _LOBE_NAMES.index(label)
                if window_front == "back":
                    lobe ^= 1
                if lobe in supplied:
                    raise ValueError(f"duplicate {band} {label} scattering matrix")
                supplied.add(lobe)
                matrices[lobe] = _read_matrix(block, order)
        if not supplied:
            raise ValueError(f"WINDOW XML has no {band} Full Klems scattering data")
        transmission = supplied & {2, 3}
        derived = set()
        if len(transmission) == 1:
            source = transmission.pop()
            matrices[source ^ 1] = matrices[source][np.ix_(_HALF_TURN, _HALF_TURN)].T
            derived.add(source ^ 1)
        result = cls(matrices, band)
        material = layer.find("Material")
        provenance = KlemsSource(
            None if filename is None else str(filename), hashlib.sha256(raw).hexdigest(),
            order, window_front, band, tuple(_LOBE_NAMES[i] for i in sorted(supplied)),
            tuple(_LOBE_NAMES[i] for i in sorted(derived)),
            "" if material is None else ET.tostring(material, encoding="unicode"), tuple(spectra),
        )
        object.__setattr__(result, "source", provenance)
        return result

    def evaluate(self, incident, outgoing) -> float:
        """Evaluate the selected BRDF or BTDF in sr⁻¹."""
        wi, wo = _unit_vector(incident), _unit_vector(outgoing)
        if wi[2] == 0.0 or wo[2] == 0.0:
            return 0.0
        front = wi[2] > 0.0
        reflected = front == (wo[2] > 0.0)
        lobe = int(not front) + 2 * int(not reflected)
        return float(self.matrices[lobe, _index(wi, incident=True), _index(wo, incident=False)])

    def _scattering(self, direction, *, reverse):
        side = int(direction[2] < 0.0)
        patch = _index(direction, incident=not reverse)
        if reverse:
            return self.matrices[side, :, patch], self.matrices[3 - side, :, patch]
        return self.matrices[side, patch], self.matrices[2 + side, patch]

    def hemispherical(self, incident) -> tuple[float, float]:
        """Return directional-hemispherical reflectance and transmittance."""
        wi = _unit_vector(incident)
        if wi[2] == 0.0:
            return 0.0, 0.0
        reflect, transmit = self._scattering(wi, reverse=False)
        return (float(reflect @ KLEMS_PROJECTED_SOLID_ANGLES),
                float(transmit @ KLEMS_PROJECTED_SOLID_ANGLES))

    def pdf(self, incident, outgoing, *, reverse=False) -> float:
        """Evaluate the sampling density per steradian, over both hemispheres."""
        wi, wo = _unit_vector(incident), _unit_vector(outgoing)
        if wi[2] == 0.0 or wo[2] == 0.0:
            return 0.0
        fixed, sampled = (wo, wi) if reverse else (wi, wo)
        reflect, transmit = self._scattering(fixed, reverse=reverse)
        normalization = (reflect + transmit) @ KLEMS_PROJECTED_SOLID_ANGLES
        if normalization == 0.0:
            return 0.0
        lobe = reflect if (fixed[2] > 0.0) == (sampled[2] > 0.0) else transmit
        value = lobe[_index(sampled, incident=reverse)]
        return float(value * abs(sampled[2]) / normalization)

    def sample(self, direction, random, *, reverse=False) -> KlemsSample | None:
        """Sample scattering from three uniform values in [0, 1).

        Fix an incident direction by default or an outgoing direction when
        reverse=True. The sampled distribution is conditional on scattering
        and returns None for an absorbing or exactly grazing direction.
        """
        fixed = _unit_vector(direction)
        draws = np.asarray(random)
        if (draws.shape != (3,) or draws.dtype.kind not in "fiu"
                or not np.isfinite(draws).all() or np.any((draws < 0.0) | (draws >= 1.0))):
            raise ValueError("random must contain three uniform values in [0, 1)")
        if fixed[2] == 0.0:
            return None
        reflect, transmit = self._scattering(fixed, reverse=reverse)
        cdf = np.cumsum(np.concatenate((reflect, transmit)) * np.tile(KLEMS_PROJECTED_SOLID_ANGLES, 2))
        total = cdf[-1]
        if total == 0.0:
            return None
        target = min(draws[0] * total, np.nextafter(total, 0.0))
        index = int(np.searchsorted(cdf, target, side="right"))
        transmitted = index >= _PATCHES
        patch = index % _PATCHES
        ring = _PATCH_RINGS[patch]
        sector = patch - KLEMS_RING_OFFSETS[ring]
        # Keep finite-precision endpoints inside the selected angular bin.
        polar, azimuth = np.clip(draws[1:], 1e-12, 1.0 - 1e-12)
        high, low = KLEMS_COSINE_BOUNDS[ring:ring + 2]
        z = math.sqrt((1.0 - polar) * high**2 + polar * low**2)
        phi = math.tau * (sector + azimuth - 0.5) / KLEMS_AZIMUTH_COUNTS[ring]
        radius = math.sqrt(max(0.0, 1.0 - z*z))
        xy_sign = -1.0 if reverse else 1.0
        z_sign = 1.0 if (fixed[2] > 0.0) != transmitted else -1.0
        sampled = np.array([xy_sign * radius * math.cos(phi), xy_sign * radius * math.sin(phi), z_sign*z])
        value = float((transmit if transmitted else reflect)[patch])
        return KlemsSample(_immutable_array(sampled), value, float(value * z / total), transmitted)


def _validate_angle_basis(definition):
    bases = [
        basis for basis in definition.findall("AngleBasis")
        if basis.findtext("AngleBasisName", "").strip().lower() == "lbnl/klems full"
    ]
    if len(bases) != 1:
        raise ValueError("WINDOW XML must define one LBNL/Klems Full angle basis")
    blocks = bases[0].findall("AngleBasisBlock")
    if len(blocks) != len(KLEMS_AZIMUTH_COUNTS):
        raise ValueError("invalid Full Klems angle basis ring count")
    for ring, block in enumerate(blocks):
        try:
            count = int(block.findtext("nPhis", ""))
            low = float(block.findtext("ThetaBounds/LowerTheta", ""))
            high = float(block.findtext("ThetaBounds/UpperTheta", ""))
        except ValueError as error:
            raise ValueError("invalid Full Klems angular ring") from error
        if count != KLEMS_AZIMUTH_COUNTS[ring] or (low, high) != tuple(KLEMS_POLAR_BOUNDS_DEG[ring:ring + 2]):
            raise ValueError("WINDOW XML angle basis differs from standard Full Klems")


def _read_matrix(block, order):
    # WINDOW uses BTDF as the data type even for its reflection blocks.
    # https://windows.lbl.gov/sites/default/files/Downloads/Solar%20BSDF%20documentation.pdf
    data_type = block.findtext("ScatteringDataType", "BTDF").strip().lower()
    if data_type not in ("btdf", "brdf", "bsdf"):
        raise ValueError("WINDOW scattering data must contain finite BSDF densities")
    data = block.find("ScatteringData")
    if data is not None and data.get("unit", "sr^-1").strip().lower() not in ("sr^-1", "1/sr", "sr-1"):
        raise ValueError("WINDOW scattering data units must be sr^-1")
    for axis in ("RowAngleBasis", "ColumnAngleBasis"):
        if block.findtext(axis, "").strip().lower() != "lbnl/klems full":
            raise ValueError("both matrix axes must use the LBNL/Klems Full basis")
    text = block.findtext("ScatteringData", "")
    entries = text.replace(",", " ").split()
    if len(entries) != _PATCHES**2:
        raise ValueError("Full Klems scattering data must contain 145 × 145 values")
    try:
        matrix = np.asarray(entries, dtype=np.float64).reshape(_PATCHES, _PATCHES)
    except ValueError as error:
        raise ValueError("invalid Full Klems scattering value") from error
    return matrix.T if order == "columns" else matrix
