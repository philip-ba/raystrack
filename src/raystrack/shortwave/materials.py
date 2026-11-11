"""Energy-bounded diffuse materials and discrete clear-plate scattering."""
from dataclasses import dataclass, field

import numpy as np

from ..model.geometry import _immutable_array


def _rgb(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape not in ((), (3,)):
        raise ValueError("optical coefficients must be a real scalar or RGB triple")
    rgb = np.broadcast_to(np.asarray(raw, dtype=np.float64), (3,))
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("optical coefficients must be finite and in [0, 1]")
    return tuple(float(channel) for channel in rgb)


@dataclass(frozen=True, init=False)
class Opaque:
    """Lambertian RGB reflectance independently assigned to each winding side."""

    front: tuple
    back: tuple

    def __init__(self, front, back=None):
        front = _rgb(front)
        object.__setattr__(self, "front", front)
        object.__setattr__(self, "back", front if back is None else _rgb(back))


@dataclass(frozen=True, init=False)
class ThinDiffuse:
    """One symmetric diffuse reflection and transmission model for a thin sheet."""

    reflectance: tuple
    transmittance: tuple

    def __init__(self, reflectance, transmittance):
        rho, tau = _rgb(reflectance), _rgb(transmittance)
        if any(r + t > 1 for r, t in zip(rho, tau)):
            raise ValueError("reflectance plus transmittance must be at most one per channel")
        object.__setattr__(self, "reflectance", rho)
        object.__setattr__(self, "transmittance", tau)


@dataclass(frozen=True, init=False)
class DiffuseSheet:
    """Two-sided Lambertian sheet with reciprocal RGB transmission.

    Reflection may differ on the mesh winding sides. Transmission is shared
    by both sides, for a sheet surrounded by the same optical medium.
    All coefficients are dimensionless; absorption is the remaining energy.
    """

    front_reflectance: tuple
    back_reflectance: tuple
    transmittance: tuple

    def __init__(self, front_reflectance, back_reflectance, transmittance):
        front, back, tau = map(_rgb, (front_reflectance, back_reflectance, transmittance))
        if any(r + t > 1 for side in (front, back) for r, t in zip(side, tau)):
            raise ValueError("reflectance plus transmittance must be at most one per side and channel")
        object.__setattr__(self, "front_reflectance", front)
        object.__setattr__(self, "back_reflectance", back)
        object.__setattr__(self, "transmittance", tau)


@dataclass(frozen=True, eq=False)
class DeltaSample:
    """A discrete scattering event, with dimensionless coefficient and mass.

    These are discrete measures, without a finite solid-angle BSDF or PDF.
    Directions point away from the surface in the material frame.
    """

    direction: np.ndarray = field(repr=False)
    coefficient: float
    probability_mass: float
    transmitted: bool

    @property
    def weight(self):
        """Dimensionless path throughput factor."""
        return self.coefficient / self.probability_mass


@dataclass(frozen=True)
class ClearGlazing:
    """Lossless, smooth, zero-thickness parallel plate in air.

    The scalar index is achromatic. Incoherent internal reflections are summed
    separately for both polarizations before averaging unpolarized light.
    Net transmission is straight through; this model has no ray displacement,
    absorption, dispersion, interference, or volumetric refraction.
    """

    refractive_index: float = 1.52

    def __post_init__(self):
        raw = np.asarray(self.refractive_index)
        if raw.shape != () or raw.dtype.kind not in "fiu":
            raise ValueError("refractive_index must be a finite real scalar >= 1")
        index = float(raw)
        if not np.isfinite(index) or index < 1:
            raise ValueError("refractive_index must be a finite real scalar >= 1")
        object.__setattr__(self, "refractive_index", index)

    def coefficients(self, cosine):
        """Return dimensionless unpolarized plate (R, T) for cosine in [0, 1]."""
        raw = np.asarray(cosine)
        if raw.dtype.kind not in "fiu":
            raise ValueError("incidence cosine must be finite and in [0, 1]")
        cosine = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(cosine).all() or np.any((cosine < 0) | (cosine > 1)):
            raise ValueError("incidence cosine must be finite and in [0, 1]")
        if self.refractive_index == 1:
            return np.zeros_like(cosine), np.ones_like(cosine)
        eta = 1 / self.refractive_index
        inside = np.sqrt(1 - (1 - cosine**2) * eta**2)
        rs = ((eta*cosine - inside) / (eta*cosine + inside))**2
        rp = ((cosine - eta*inside) / (cosine + eta*inside))**2
        transmission = .5 * ((1-rs)/(1+rs) + (1-rp)/(1+rp))
        return 1-transmission, transmission

    def sample(self, direction, random):
        """Sample reflection/transmission using one uniform scalar in [0, 1).

        Both directions point away; local +Z is the front normal. The same
        events and masses apply in reverse because both external media are air.
        Exactly grazing directions do not define a surface event and return None.
        """
        raw = np.asarray(direction)
        if raw.shape != (3,) or raw.dtype.kind not in "fiu":
            raise ValueError("direction must be a finite nonzero real 3-vector")
        direction = np.asarray(raw, dtype=np.float64)
        scale = np.max(np.abs(direction))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("direction must be a finite nonzero real 3-vector")
        direction = direction / scale
        direction = direction / np.linalg.norm(direction)
        draw = np.asarray(random)
        if (draw.shape != () or draw.dtype.kind not in "fiu"
                or not np.isfinite(draw) or not 0 <= draw < 1):
            raise ValueError("random must be one uniform scalar in [0, 1)")
        if direction[2] == 0:
            return None
        reflectance, transmittance = self.coefficients(abs(direction[2]))
        transmitted = bool(draw >= reflectance)
        sampled = -direction
        if not transmitted:
            sampled[2] = direction[2]
        coefficient = float(transmittance if transmitted else reflectance)
        return DeltaSample(_immutable_array(sampled), coefficient, coefficient, transmitted)


@dataclass(frozen=True)
class SolidDielectric:
    """Smooth air/glass boundary of a closed outward convex solid, in metres.

    ``absorption`` is a nonnegative scalar or RGB Beer coefficient in m^-1.
    The achromatic index is limited to [1, 10] for float ray angular precision.
    Only sky forcing is supported. Solids require disjoint bounding boxes from
    every other surface and exterior sensors. Each boundary uses unpolarized
    Fresnel coefficients; polarization is not tracked between events.
    """

    refractive_index: float = 1.52
    absorption: tuple = (0., 0., 0.)

    def __post_init__(self):
        raw = np.asarray(self.refractive_index)
        if (raw.shape != () or raw.dtype.kind not in "fiu" or not np.isfinite(raw)
                or not 1 <= raw <= 10):
            raise ValueError("refractive_index must be a finite real scalar in [1, 10]")
        raw_absorption = np.asarray(self.absorption)
        if (raw_absorption.dtype.kind not in "fiu" or raw_absorption.shape not in ((), (3,))
                or not np.isfinite(raw_absorption).all() or np.any(raw_absorption < 0)):
            raise ValueError("absorption must be a finite nonnegative scalar or RGB triple in m^-1")
        object.__setattr__(self, "refractive_index", float(raw))
        object.__setattr__(self, "absorption", tuple(float(x) for x in np.broadcast_to(raw_absorption, (3,))))

    def coefficients(self, cosine, *, entering=True):
        """Interface power (R,T) and transmitted cosine; TIR gives T=0.

        Reverse radiance transmission additionally uses
        (eta_current / eta_next)^2.
        """
        raw = np.asarray(cosine)
        if (raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()
                or np.any((raw < 0) | (raw > 1))):
            raise ValueError("incidence cosine must be finite and in [0, 1]")
        if not isinstance(entering, (bool, np.bool_)):
            raise ValueError("entering must be boolean")
        cosine = np.asarray(raw, dtype=np.float64)
        eta = 1 / self.refractive_index if entering else self.refractive_index
        sine_squared = eta**2 * np.maximum(0, 1 - cosine**2)
        transmitted_cosine = np.sqrt(np.maximum(0, 1 - sine_squared))
        if self.refractive_index == 1:
            return np.zeros_like(cosine), np.ones_like(cosine), cosine.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = ((eta*cosine - transmitted_cosine) / (eta*cosine + transmitted_cosine))**2
            rp = ((cosine - eta*transmitted_cosine) / (cosine + eta*transmitted_cosine))**2
        reflectance = np.where(sine_squared >= 1, 1., .5*(rs+rp))
        return reflectance, 1-reflectance, transmitted_cosine

    def attenuation(self, distance):
        """Return exp(-absorption * distance) per channel, distance in metres."""
        raw = np.asarray(distance)
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("distance must be finite and nonnegative in metres")
        with np.errstate(over="ignore"):
            return np.exp(-np.asarray(raw, dtype=np.float64)[..., None] * self.absorption)

    def sample(self, direction, random):
        """Sample a reverse radiance event; +Z is the outward winding normal.

        ``direction`` points away into the current medium. The returned mass
        is Fresnel power, while transmission coefficient includes adjoint eta².
        """
        raw = np.asarray(direction)
        if raw.shape != (3,) or raw.dtype.kind not in "fiu":
            raise ValueError("direction must be a finite nonzero real 3-vector")
        direction = np.asarray(raw, dtype=np.float64)
        scale = np.max(np.abs(direction))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("direction must be a finite nonzero real 3-vector")
        direction = direction / scale
        direction /= np.linalg.norm(direction)
        draw = np.asarray(random)
        if (draw.shape != () or draw.dtype.kind not in "fiu" or not np.isfinite(draw)
                or not 0 <= draw < 1):
            raise ValueError("random must be one uniform scalar in [0, 1)")
        if direction[2] == 0:
            return None
        entering = bool(direction[2] > 0)
        reflectance, transmittance, cosine = self.coefficients(abs(direction[2]), entering=entering)
        transmitted = bool(draw >= reflectance)
        sampled = -direction
        if transmitted:
            eta = 1/self.refractive_index if entering else self.refractive_index
            sampled[:2] *= eta
            sampled[2] = -np.copysign(cosine, direction[2])
            mass = float(transmittance)
            coefficient = mass * eta**2
        else:
            sampled[2] = direction[2]
            mass = coefficient = float(reflectance)
        return DeltaSample(_immutable_array(sampled), coefficient, mass, transmitted)
