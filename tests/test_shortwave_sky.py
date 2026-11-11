"""Sky geometry, radiometric normalization, and Radiance Perez import."""
import shutil
import subprocess

import numpy as np
import pytest

from raystrack.shortwave.sky import PatchSky, ReinhartSkyBasis, load_radiance_diffuse_sky


@pytest.mark.parametrize(("multiplier", "count"), [(1, 145), (np.uint64(2), 577), (4, 2305)])
def test_reinhart_geometry_covers_exactly_the_upper_hemisphere(multiplier, count):
    basis = ReinhartSkyBasis(multiplier, north_deg=23)
    assert len(basis) == count
    assert len(basis.ring_offsets) == 7 * multiplier + 2
    np.testing.assert_array_equal(basis.patch_indices(basis.directions), np.arange(count))
    np.testing.assert_allclose(np.linalg.norm(basis.directions, axis=1), 1.0)
    assert np.all(basis.solid_angles > 0)
    assert basis.solid_angles.sum() == pytest.approx(2 * np.pi)
    assert basis.projected_solid_angles.sum() == pytest.approx(np.pi)


def test_radiance_reinhart_binning_and_rotation():
    basis = ReinhartSkyBasis()
    # First bin centered north, clockwise ordering, one dedicated zenith cap.
    np.testing.assert_array_equal(
        basis.patch_indices([[0, 2, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0],
                             [0, 0, 2], [0, 0, -1]]),
        [0, 8, 15, 23, 144, -1],
    )
    # A quarter turn puts geographic north at world -X.
    assert ReinhartSkyBasis(north_deg=90).patch_indices([-1, 0, 0]) == 0
    with pytest.raises(ValueError, match="nonzero"):
        basis.patch_indices([0, 0, 0])


def test_patch_sky_copies_radiance_and_isotropic_sky_matches_dhi():
    basis = ReinhartSkyBasis()
    dhi = [100, 110, 120]
    sky = PatchSky.isotropic(basis, dhi)
    np.testing.assert_allclose(basis.projected_solid_angles @ sky.radiance, dhi)
    np.testing.assert_allclose(sky.radiance_for([0, 0, 1]), np.array(dhi) / np.pi)
    np.testing.assert_array_equal(sky.radiance_for([0, 0, -1]), [0, 0, 0])
    with pytest.raises(ValueError):
        sky.radiance.setflags(write=True)
    values = np.ones((len(basis), 3))
    sky = PatchSky(basis, values)
    values[:] = 0
    np.testing.assert_array_equal(sky.radiance, 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        PatchSky.isotropic(basis, -1)


def test_radiance_loader_preserves_timestep_order_and_drops_ground(tmp_path):
    basis = ReinhartSkyBasis()
    matrix = tmp_path / "sky.mtx"
    rgb = np.arange((len(basis) + 1) * 2 * 3).reshape(len(basis) + 1, 2, 3)
    header = "#?RADIANCE\ngendaymtx -s -O1 -m 1\nNROWS=146\nNCOLS=2\nNCOMP=3\nFORMAT=ascii\n\n"
    with matrix.open("w", encoding="ascii") as stream:
        stream.write(header)
        np.savetxt(stream, rgb.reshape(-1, 3), fmt="%d")
    series = load_radiance_diffuse_sky(matrix, basis)
    assert len(series) == 2
    np.testing.assert_array_equal(series.frame(0).radiance, rgb[1:, 0])
    np.testing.assert_array_equal(series.frame(1).radiance, rgb[1:, 1])
    with pytest.raises(ValueError, match="sky patches"):
        load_radiance_diffuse_sky(matrix, ReinhartSkyBasis(2))
    matrix.write_text(header.replace("-s -O1", "-d -O1"), encoding="ascii")
    with pytest.raises(ValueError, match="gendaymtx -s"):
        load_radiance_diffuse_sky(matrix, basis)


def test_installed_radiance_perez_diffuse_preserves_night_and_daylight_columns(tmp_path):
    gendaymtx = shutil.which("gendaymtx")
    if gendaymtx is None:
        pytest.skip("Radiance gendaymtx is unavailable")
    wea = (
        "place Test\nlatitude 52\nlongitude -13\ntime_zone -15\nsite_elevation 0\n"
        "weather_data_file_units 1\n3 21 0.5 0 0\n3 21 12.5 800 100\n3 21 13.5 600 90\n"
    )
    matrix = tmp_path / "sky.mtx"
    with matrix.open("wb") as output:
        subprocess.run([gendaymtx, "-s", "-O1", "-m", "1", "-c", "1", "1", "1"],
                       input=wea.encode("ascii"), stdout=output, check=True)
    basis = ReinhartSkyBasis()
    series = load_radiance_diffuse_sky(matrix, basis)
    integrated = np.einsum("p,tpc->tc", basis.projected_solid_angles, series.radiance)
    np.testing.assert_allclose(integrated, [[0]*3, [100]*3, [90]*3], rtol=0.01)
