"""Explicit independent top-hat wavelength bands using native scalar transport.

Forcing is already integrated over each wavelength interval, in W/m². Optical
coefficients must be supplied independently for each interval. No spectral
distribution is inferred from broadband weather, RGB, or WINDOW Visible data.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..model import Scene
from ..model.geometry import _immutable_array
from .solver import ShortwaveSolver
from .solar import SolarPosition, SolarTimeline


def _provenance(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify the source and integration convention")
    return value


@dataclass(frozen=True)
class WavelengthBand:
    """A nonoverlapping interval [lower_nm, upper_nm), with unit response."""

    name: str
    lower_nm: float
    upper_nm: float

    def __post_init__(self):
        _provenance(self.name, "band name")
        for name in ("lower_nm", "upper_nm"):
            raw = np.asarray(getattr(self, name))
            if raw.shape != () or raw.dtype.kind not in "fiu":
                raise ValueError("band edges must be finite positive real scalars in nm")
            value = float(raw)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("band edges must be finite positive real scalars in nm")
            object.__setattr__(self, name, value)
        if self.upper_nm <= self.lower_nm:
            raise ValueError("upper_nm must exceed lower_nm")


@dataclass(frozen=True, eq=False)
class BandInput:
    """Band-integrated DNI/DHI and explicit per-surface, scalar band optics.

    Coefficients represent constants within the band, or effective values whose
    weighting must be recorded in optical_provenance. DHI drives an isotropic
    upper sky. Missing materials are rejected, including intended black surfaces.
    """

    band: WavelengthBand
    materials: object
    dni: object
    dhi: object
    optical_provenance: str
    forcing_provenance: str

    def __post_init__(self):
        if not isinstance(self.band, WavelengthBand):
            raise TypeError("band must be a WavelengthBand")
        for name in ("optical_provenance", "forcing_provenance"):
            _provenance(getattr(self, name), name)
        object.__setattr__(self, "materials", MappingProxyType(dict(self.materials)))
        if any(value is None for value in self.materials.values()):
            raise ValueError("each band requires an explicit optical material per surface")
        for name in ("dni", "dhi"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "fiu" or raw.ndim > 1:
                raise ValueError("band forcing must be a scalar or one-dimensional array in W/m²")
            values = np.asarray(raw, dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("band forcing must be finite and nonnegative")
            object.__setattr__(self, name, _immutable_array(values))


@dataclass(frozen=True, eq=False)
class IndependentBandResult:
    """Immutable per-band snapshots; sums cover only the supplied intervals.

    Results remain correlated when they share sampling seeds. No combined
    standard error is inferred from the individual band standard errors.
    """

    inputs: tuple
    results: tuple

    @property
    def irradiance(self):
        """W/m² indexed by band, timestep, sensor, and component."""
        return _immutable_array(np.stack([result.irradiance[..., 0] for result in self.results]))

    @property
    def total(self):
        """W/m² summed over supplied bands and components, by timestep/sensor."""
        return _immutable_array(self.irradiance.sum(axis=(0, 3)))

    def exposure(self):
        """Wh/m² over supplied bands, indexed by sensor and component."""
        return _immutable_array(np.sum([result.exposure()[..., 0] for result in self.results], axis=0))


def solve_independent_bands(scene, points, normals, sun, inputs, *, backend="embree", **options):
    """Solve independent wavelength intervals with explicitly supplied optics.

    This intentionally creates one native scene per band and does not cache
    transfer coefficients. No cross-band scattering, fluorescence, interference,
    polarization state, or wavelength-dependent sky generation is implemented.
    Scalar optical models apply within each band. Transport capability errors
    (including solid-glass sun restrictions) propagate unchanged.
    """
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene")
    inputs = tuple(inputs)
    if not inputs or any(not isinstance(item, BandInput) for item in inputs):
        raise ValueError("inputs must contain at least one BandInput")
    if len({item.band.name for item in inputs}) != len(inputs):
        raise ValueError("band names must be unique")
    ordered = sorted((item.band for item in inputs), key=lambda band: band.lower_nm)
    if any(a.upper_nm > b.lower_nm for a, b in zip(ordered, ordered[1:])):
        raise ValueError("wavelength bands must not overlap")
    forbidden = {"dni", "dhi", "sky", "channel_basis", "forcing_id"}.intersection(options)
    if forbidden:
        raise ValueError(f"band forcing and basis cannot be overridden: {sorted(forbidden)}")
    # Capture caller-owned arrays once so all bands see the same forcing frame.
    points, normals = np.array(points, copy=True), np.array(normals, copy=True)
    if isinstance(sun, SolarPosition):
        fields = ("timestamps", "directions", "elevation_deg", "azimuth_deg", "above_horizon")
        captured = {name: _immutable_array(np.asarray(getattr(sun, name))) for name in fields}
        if isinstance(sun, SolarTimeline):
            captured["durations_hours"] = _immutable_array(np.asarray(sun.durations_hours))
        sun = replace(sun, **captured)
    else:
        sun = np.array(sun, copy=True)
    if options.get("durations_hours") is not None:
        options["durations_hours"] = np.array(options["durations_hours"], copy=True)
    results = []
    # Retain one geometry revision across all independent native solves.
    with scene.lock:
        surface_ids = set(scene.surface_ids)
        for item in inputs:
            if set(item.materials) != surface_ids:
                raise ValueError("each band requires exactly the scene's surface material assignments")
        revision = scene.revision
        for item in inputs:
            with ShortwaveSolver(scene, materials=item.materials, backend=backend) as solver:
                result = solver.solve(points, normals, sun, dni=item.dni, dhi=item.dhi,
                                      channel_basis="neutral", forcing_id=item.forcing_provenance,
                                      **options)
            if result.geometry_revision != revision or scene.revision != revision:
                raise RuntimeError("scene geometry changed during independent-band transport")
            results.append(result)
    return IndependentBandResult(inputs, tuple(results))
