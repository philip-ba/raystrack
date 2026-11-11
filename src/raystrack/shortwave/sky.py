"""Shortwave sky radiance on the Radiance Reinhart upper hemisphere."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path

import numpy as np

from ..model.geometry import _immutable_array


@dataclass(frozen=True)
class ReinhartSkyBasis:
    """Radiance Reinhart directions, with zero-based sky indices and no ground.

    The world is +Z up and +Y north at zero rotation. The north angle is
    degrees counterclockwise about +Z from +Y, as for SolarSite.
    """

    multiplier: int = 1
    north_deg: float = 0.0
    ring_height_rad: float = field(init=False)
    ring_offsets: np.ndarray = field(init=False, compare=False, repr=False)
    directions: np.ndarray = field(init=False, compare=False, repr=False)
    solid_angles: np.ndarray = field(init=False, compare=False, repr=False)
    projected_solid_angles: np.ndarray = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        m = self.multiplier
        if isinstance(m, bool) or not isinstance(m, Integral) or m < 1:
            raise ValueError("multiplier must be a positive integer")
        m = int(m)
        north = float(self.north_deg)
        if not np.isfinite(north):
            raise ValueError("north_deg must be finite")
        north %= 360.0

        ring_counts = np.append(m * np.repeat([30, 30, 24, 24, 18, 12, 6], m), 1)
        offsets = np.insert(np.cumsum(ring_counts), 0, 0)
        ring_height = (np.pi / 2.0) / (7 * m + 0.5)
        rings = np.repeat(np.arange(len(ring_counts)), ring_counts)
        width = 2.0 * np.pi / ring_counts[rings]
        azimuth = width * (np.arange(offsets[-1]) - offsets[rings])
        lower = ring_height * rings
        upper = np.minimum(lower + ring_height, np.pi / 2.0)
        altitude = (lower + upper) / 2.0
        altitude[-1] = np.pi / 2.0
        world_azimuth = azimuth - np.radians(north)
        directions = np.column_stack((
            np.cos(altitude) * np.sin(world_azimuth),
            np.cos(altitude) * np.cos(world_azimuth),
            np.sin(altitude),
        ))
        directions[-1] = (0.0, 0.0, 1.0)
        sin_lo = np.sin(lower)
        sin_hi = np.sin(upper)
        solid = width * (sin_hi - sin_lo)
        projected = width * (sin_hi**2 - sin_lo**2) / 2.0

        object.__setattr__(self, "multiplier", m)
        object.__setattr__(self, "north_deg", north)
        object.__setattr__(self, "ring_height_rad", ring_height)
        object.__setattr__(self, "ring_offsets", _immutable_array(offsets))
        object.__setattr__(self, "directions", _immutable_array(directions))
        object.__setattr__(self, "solid_angles", _immutable_array(solid))
        object.__setattr__(self, "projected_solid_angles", _immutable_array(projected))

    def __len__(self):
        return len(self.directions)

    def patch_indices(self, directions) -> np.ndarray:
        """Map finite nonzero world vectors of shape (..., 3) to sky indices.

        Vector magnitude is irrelevant. Below-horizon directions return -1.
        """
        vectors = np.asarray(directions)
        if vectors.ndim == 0 or vectors.shape[-1] != 3 or vectors.dtype.kind not in "fiu":
            raise ValueError("directions must be real vectors of shape (..., 3)")
        vectors = np.asarray(vectors, dtype=np.float64)
        scale = np.max(np.abs(vectors), axis=-1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(scale == 0.0):
            raise ValueError("directions must be finite and nonzero")
        x, y, z = np.moveaxis(vectors / scale, -1, 0)
        altitude = np.arctan2(z, np.hypot(x, y))
        row = np.floor(np.maximum(altitude, 0) / self.ring_height_rad).astype(np.intp)
        counts = self.ring_offsets[row + 1] - self.ring_offsets[row]
        azimuth = np.degrees(np.arctan2(x, y)) + self.north_deg
        width = 360.0 / counts
        column = np.floor((azimuth + width / 2.0) / width).astype(np.intp) % counts
        return np.where(z >= 0.0, self.ring_offsets[row] + column, -1)


def _radiance_array(radiance, shape: tuple) -> np.ndarray:
    values = np.asarray(radiance)
    if values.dtype.kind not in "fiu" or values.shape != shape:
        raise ValueError(f"radiance must be a real array with shape {shape}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("radiance must be finite and nonnegative")
    return _immutable_array(values)


@dataclass(frozen=True, eq=False)
class PatchSky:
    """RGB sky radiance in W/(m² sr) for one timestep, excluding sun and ground."""

    basis: ReinhartSkyBasis
    radiance: np.ndarray = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.basis, ReinhartSkyBasis):
            raise TypeError("basis must be a ReinhartSkyBasis")
        object.__setattr__(self, "radiance", _radiance_array(self.radiance, (len(self.basis), 3)))

    @classmethod
    def isotropic(cls, basis: ReinhartSkyBasis, dhi) -> PatchSky:
        """Make a uniform sky from scalar or RGB diffuse horizontal irradiance."""
        if not isinstance(basis, ReinhartSkyBasis):
            raise TypeError("basis must be a ReinhartSkyBasis")
        values = np.asarray(dhi)
        if values.shape not in ((), (3,)) or values.dtype.kind not in "fiu":
            raise ValueError("dhi must be a real scalar or RGB triplet")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("dhi must be finite and nonnegative")
        return cls(basis, np.broadcast_to(values / np.pi, (len(basis), 3)))

    def radiance_for(self, directions) -> np.ndarray:
        """Return RGB radiance from the direction's patch, or zero below the sky."""
        patch = self.basis.patch_indices(directions)
        return np.where(patch[..., None] >= 0, self.radiance[patch], 0.0)


@dataclass(frozen=True, eq=False)
class PatchSkySeries:
    """A timestep-major diffuse-sky matrix in W/(m² sr)."""

    basis: ReinhartSkyBasis
    radiance: np.ndarray = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.basis, ReinhartSkyBasis):
            raise TypeError("basis must be a ReinhartSkyBasis")
        values = np.asarray(self.radiance)
        if values.ndim != 3:
            raise ValueError("radiance must have shape (timesteps, patches, 3)")
        object.__setattr__(
            self, "radiance", _radiance_array(values, (len(values), len(self.basis), 3)),
        )

    def __len__(self):
        return len(self.radiance)

    def frame(self, index: int) -> PatchSky:
        """Select one sky by timestep index."""
        return PatchSky(self.basis, self.radiance[index])


def load_radiance_diffuse_sky(path, basis: ReinhartSkyBasis) -> PatchSkySeries:
    """Import an ASCII matrix from `gendaymtx -s -O1 -m N -c 1 1 1 weather.wea`.

    Use the same standard-time interval order for WEA and the simulation.
    Preserve all columns by leaving off the averaging and sun-up filters.
    The matrix's ground row is discarded. Rotate via the basis.
    """
    if not isinstance(basis, ReinhartSkyBasis):
        raise TypeError("basis must be a ReinhartSkyBasis")
    with Path(path).open(encoding="ascii") as stream:
        if stream.readline().strip() != "#?RADIANCE":
            raise ValueError("expected a Radiance matrix header")
        header = {}
        command = []
        for line in stream:
            line = line.strip()
            if not line:
                break
            if "gendaymtx" in line.split(maxsplit=1)[0]:
                command = line.split()
            key, separator, value = line.partition("=")
            if separator:
                header[key] = value.strip()
        if (
            "-s" not in command or "-O1" not in command or
            any(option in command for option in ("-d", "-A", "-u", "-r", "-rz"))
        ):
            raise ValueError("matrix must use gendaymtx -s -O1 with all timesteps and default north")
        try:
            rows, columns, components = (int(header[key]) for key in ("NROWS", "NCOLS", "NCOMP"))
        except (KeyError, ValueError) as error:
            raise ValueError("invalid Radiance matrix dimensions") from error
        if rows != len(basis) + 1 or columns < 1 or components != 3 or header.get("FORMAT") != "ascii":
            raise ValueError("matrix must be ASCII RGB with ground and the basis's sky patches")
        values = np.loadtxt(stream, comments=None, ndmin=2)
    if values.shape != (rows * columns, components):
        raise ValueError("matrix data size does not match the header")
    return PatchSkySeries(basis, values.reshape(rows, columns, components)[1:].transpose(1, 0, 2))
