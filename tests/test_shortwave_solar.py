"""Geometric solar position, world orientation, and annual interval semantics."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from raystrack.shortwave.solar import SolarSite, SolarTimeline, solar_position


def test_solar_position_matches_published_spa_example():
    # NREL/TP-560-34302 Appendix A.5. Geometric elevation from the listed
    # topocentric declination and hour angle, before refraction correction.
    latitude, longitude = 39.742476, -105.1786
    date = datetime(2003, 10, 17, 12, 30, 30, tzinfo=timezone(timedelta(hours=-7)))
    result = solar_position(SolarSite(latitude, longitude), [date])
    lat, dec, hour = np.radians([latitude, -9.316179, 11.10629])
    elevation = np.degrees(np.arcsin(
        np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour)
    ))
    assert result.elevation_deg[0] == pytest.approx(elevation, abs=0.01)
    assert result.azimuth_deg[0] == pytest.approx(194.34024, abs=0.01)
    assert result.timestamps[0] == np.datetime64("2003-10-17T19:30:30")
    assert result.above_horizon[0]


def test_geographic_angles_are_preserved_when_north_rotates():
    times = np.array(["2026-03-20T09:00", "2026-03-20T15:00", "2026-03-20T23:00"],
                     dtype="datetime64[m]")
    geographic = solar_position(SolarSite(52, 0), times)
    rotated = solar_position(SolarSite(52, 0, north_deg=90), times)
    east, north, up = geographic.directions.T
    assert east[0] > 0 and east[1] < 0
    np.testing.assert_array_equal(geographic.above_horizon, [True, True, False])
    np.testing.assert_allclose(rotated.directions, np.column_stack((-north, east, up)))
    np.testing.assert_allclose(np.linalg.norm(rotated.directions, axis=1), 1)
    np.testing.assert_array_equal(rotated.azimuth_deg, geographic.azimuth_deg)
    with pytest.raises(ValueError):
        rotated.directions.setflags(write=True)


@pytest.mark.parametrize(("year", "hours"), [(2024, 8784), (2026, 8760)])
def test_annual_timeline_uses_standard_time_and_leap_day(year, hours):
    result = SolarTimeline.for_year(year, SolarSite(52, 13), utc_offset_minutes=60)
    assert len(result.timestamps) == hours
    assert result.timestamps[0] == np.datetime64(f"{year - 1}-12-31T23:30")
    assert result.timestamps[-1] == np.datetime64(f"{year}-12-31T22:30")
    np.testing.assert_array_equal(np.diff(result.timestamps), np.timedelta64(1, "h"))
    np.testing.assert_array_equal(result.durations_hours, 1.0)


def test_timeline_final_interval_can_be_shorter_than_regular_step():
    result = SolarTimeline.for_year(2026, SolarSite(52, 13), step_minutes=51)
    assert result.durations_hours.sum() == pytest.approx(8760)
    assert result.durations_hours[-1] == pytest.approx(45 / 60)
    assert result.timestamps[-1] == np.datetime64("2026-12-31T23:37:30")


def test_solar_inputs_reject_ambiguous_or_invalid_values():
    with pytest.raises(ValueError, match="latitude"):
        SolarSite(91, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        solar_position(SolarSite(0, 0), [datetime(2026, 1, 1)])
    with pytest.raises(ValueError, match="finite"):
        solar_position(SolarSite(0, 0), np.array(["NaT"], dtype="datetime64[s]"))
    with pytest.raises(ValueError, match="step_minutes"):
        SolarTimeline.for_year(2026, SolarSite(0, 0), step_minutes=0)
