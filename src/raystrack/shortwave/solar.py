"""Geometric sun directions and fixed-standard-time simulation intervals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral

import numpy as np

from ..model.geometry import _immutable_array


@dataclass(frozen=True)
class SolarSite:
    """Latitude north, longitude east, and CCW world north rotation in degrees.

    At zero rotation the world axes are east, north, and up. At 90 degrees
    geographic north is world -X and east is world +Y.
    """

    latitude_deg: float
    longitude_deg: float
    north_deg: float = 0.0

    def __post_init__(self):
        for name, limit in (("latitude_deg", 90), ("longitude_deg", 180)):
            value = float(getattr(self, name))
            if not np.isfinite(value) or abs(value) > limit:
                raise ValueError(f"{name} must be between {-limit} and {limit}")
            object.__setattr__(self, name, value)
        north = float(self.north_deg)
        if not np.isfinite(north):
            raise ValueError("north_deg must be finite")
        object.__setattr__(self, "north_deg", north % 360.0)


@dataclass(frozen=True, eq=False)
class SolarPosition:
    """Read-only UTC timestamps, unit vectors, and geographic sun angles."""

    timestamps: np.ndarray
    directions: np.ndarray
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    above_horizon: np.ndarray


def solar_position(site: SolarSite, timestamps) -> SolarPosition:
    """Compute solar center directions without atmospheric refraction.

    Supply a one-dimensional sequence of timezone-aware datetimes or an
    array of datetime64 UTC instants in years 1900 through 2100. Azimuth is
    clockwise from geographic north. The NOAA/Meeus approximation uses UTC
    as UT1 and ignores solar parallax.
    """
    if not isinstance(site, SolarSite):
        raise TypeError("site must be a SolarSite")
    utc = np.asarray(timestamps)
    if utc.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if utc.dtype.kind != "M":
        instants = []
        for value in utc:
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError("timestamps must be datetime64 UTC or timezone-aware datetimes")
            instants.append(value.astimezone(timezone.utc).replace(tzinfo=None))
        utc = np.asarray(instants, dtype="datetime64[us]")
    if np.isnat(utc).any():
        raise ValueError("timestamps must be finite")
    years = utc.astype("datetime64[Y]")
    if np.any((years < np.datetime64("1900")) | (years > np.datetime64("2100"))):
        raise ValueError("timestamps must be in years 1900 through 2100")
    utc = utc.astype("datetime64[us]")

    # NOAA Solar Calculator (Meeus), using the Unix epoch's exact Julian day:
    # https://gml.noaa.gov/grad/solcalc/calcdetails.html
    seconds = utc.astype(np.int64) / 1_000_000.0
    centuries = (seconds / 86400.0 + 2440587.5 - 2451545.0) / 36525.0
    mean_longitude = np.radians(
        (280.46646 + centuries * (36000.76983 + 0.0003032 * centuries)) % 360.0
    )
    mean_anomaly = np.radians(357.52911 + centuries * (35999.05029 - 0.0001537 * centuries))
    eccentricity = 0.016708634 - centuries * (0.000042037 + 0.0000001267 * centuries)
    equation_center = (
        np.sin(mean_anomaly) * (1.914602 - centuries * (0.004817 + 0.000014 * centuries))
        + np.sin(2.0 * mean_anomaly) * (0.019993 - 0.000101 * centuries)
        + 0.000289 * np.sin(3.0 * mean_anomaly)
    )
    omega = np.radians(125.04 - 1934.136 * centuries)
    apparent_longitude = mean_longitude + np.radians(
        equation_center - 0.00569 - 0.00478 * np.sin(omega)
    )
    mean_obliquity = 23.0 + (
        26.0 + (
            21.448 - centuries * (46.815 + centuries * (0.00059 - 0.001813 * centuries))
        ) / 60.0
    ) / 60.0
    obliquity = np.radians(mean_obliquity + 0.00256 * np.cos(omega))
    declination = np.arcsin(np.sin(obliquity) * np.sin(apparent_longitude))

    obliquity_factor = np.tan(obliquity / 2.0) ** 2
    equation_of_time = 4.0 * np.degrees(
        obliquity_factor * np.sin(2.0 * mean_longitude) - 2.0 * eccentricity * np.sin(mean_anomaly)
        + 4.0 * eccentricity * obliquity_factor * np.sin(mean_anomaly) * np.cos(2.0 * mean_longitude)
        - 0.5 * obliquity_factor**2 * np.sin(4.0 * mean_longitude)
        - 1.25 * eccentricity**2 * np.sin(2.0 * mean_anomaly)
    )
    utc_minutes = np.remainder(seconds, 86400.0) / 60.0
    solar_minutes = utc_minutes + equation_of_time + 4.0 * site.longitude_deg
    hour_angle = np.radians(solar_minutes / 4.0 - 180.0)
    latitude = np.radians(site.latitude_deg)
    east = -np.cos(declination) * np.sin(hour_angle)
    north = np.cos(latitude) * np.sin(declination) - (
        np.sin(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    up = np.sin(latitude) * np.sin(declination) + (
        np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    rotation = np.radians(site.north_deg)
    world = np.column_stack((
        east * np.cos(rotation) - north * np.sin(rotation),
        east * np.sin(rotation) + north * np.cos(rotation),
        up,
    ))
    elevation = np.degrees(np.arctan2(up, np.hypot(east, north)))
    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
    return SolarPosition(
        _immutable_array(utc), _immutable_array(world), _immutable_array(elevation),
        _immutable_array(azimuth), _immutable_array(up > 0),
    )


@dataclass(frozen=True, eq=False)
class SolarTimeline(SolarPosition):
    """Annual midpoint samples with interval durations in hours."""

    durations_hours: np.ndarray
    site: SolarSite
    year: int
    step_minutes: int
    utc_offset_minutes: int

    @classmethod
    def for_year(cls, year: int, site: SolarSite, *, step_minutes: int = 60,
                 utc_offset_minutes: int = 0) -> SolarTimeline:
        """Cover a local calendar year at a fixed UTC offset, including leap day.

        Use minutes east of UTC for the offset. A final shorter interval
        covers any remainder. Timestamps in the result are UTC instants.
        """
        for name, value, minimum, maximum in (
            ("year", year, 1901, 2099),
            ("step_minutes", step_minutes, 1, 366 * 1440),
            ("utc_offset_minutes", utc_offset_minutes, -1439, 1439),
        ):
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not minimum <= value <= maximum):
                raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        start = np.datetime64(f"{year:04d}-01-01", "m")
        end = np.datetime64(f"{year + 1:04d}-01-01", "m")
        total_minutes = int((end - start) / np.timedelta64(1, "m"))
        offsets = np.arange(0, total_minutes, step_minutes, dtype=np.int64)
        minutes = np.minimum(step_minutes, total_minutes - offsets)
        start_utc = start - np.timedelta64(int(utc_offset_minutes), "m")
        timestamps = start_utc + (offsets * 60 + minutes * 30).astype("timedelta64[s]")
        position = solar_position(site, timestamps)
        return cls(position.timestamps, position.directions, position.elevation_deg,
                   position.azimuth_deg, position.above_horizon,
                   _immutable_array(minutes / 60.0), site, int(year),
                   int(step_minutes), int(utc_offset_minutes))
