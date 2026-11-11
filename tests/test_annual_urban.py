"""Annual workflow contracts, restart safety and real native block equivalence."""
import os

import numpy as np
import pytest

from raystrack.shortwave import ShortwaveSolver
from validation.annual_radiance import synthetic_forcing
from validation.annual_urban import (
    CGDB_SHA256, EVENTS, apply_event, block_ranges, bsdf_provenance, district_fixture, fixture_hash, run_case,
)


def test_block_ranges_cover_full_year_once_and_never_cross_edits():
    ranges = block_ranges(8760, 168)
    assert ranges[0][0] == 0 and ranges[-1][1] == 8760
    assert sum(stop-start for start, stop in ranges) == 8760
    assert all(stop-start <= 168 for start, stop in ranges)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert all(hour in [start for start, _ in ranges] for hour, _ in EVENTS)


def test_only_pinned_measurement_fixture_is_labeled_measurement_based():
    assert bsdf_provenance(CGDB_SHA256).startswith("measurement-based LBNL CGDB")
    assert "unverified" in bsdf_provenance("0"*64)
    assert "synthetic" in bsdf_provenance(None)


def test_fixture_scales_geometry_sensors_and_preserves_finite_reciprocity():
    scene, materials, points, normals, labels = district_fixture(4, 3, 7)
    assert len(scene.surface_ids) == 48
    assert points.shape == normals.shape == (28, 3)
    assert len(set(labels)) == 28
    assert sum(len(scene[key].mesh.faces) for key in scene.surface_ids) == 4*(6*18+6*2)
    assert len({type(value) for value in materials.values()}) == 4
    finite = materials["t0/klems"]
    for wi, wo in (([.3, .2, 1], [.4, .5, -1]), ([.2, .6, -1], [.7, .2, 1])):
        assert finite.evaluate(wi, wo) == pytest.approx(finite.evaluate(wo, wi))
    other = district_fixture(4, 3, 7)
    assert fixture_hash(scene, points, normals, labels) == fixture_hash(other[0], *other[2:])


def test_scheduled_edits_have_distinct_revision_semantics_and_keep_old_mesh():
    scene, materials, *_ = district_fixture()
    class Assignments:
        def __init__(self):
            self.edits = []
        def set_material(self, key, value):
            self.edits.append(key)
    assignments = Assignments()
    original = scene["t0/north"].mesh.vertices.copy()
    for index, (_, event) in enumerate(EVENTS):
        before = scene.revision
        apply_event(scene, materials, assignments, event)
        assert scene.revision == before + (event != "material")
    np.testing.assert_array_equal(original[:, 2], [0, 0, 5, 5])
    assert assignments.edits == ["t0/sheet"]
    assert len(scene["t0/slat4"].mesh.faces) == 1


@pytest.mark.skipif(not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"), reason="compiled native library required")
def test_native_blocks_resume_hashes_and_unblocked_parity(tmp_path):
    options = dict(backend="native-builtin", steps=24, block_steps=7, samples=16, replicates=2)
    first = run_case(tmp_path/"resume", max_blocks=1, **options)
    assert first["intervals"] == 7 and not first["complete"]
    final = run_case(tmp_path/"resume", resume=True, **options)
    whole = run_case(tmp_path/"whole", **{**options, "block_steps":24})
    assert final["intervals"] == 24 and final["complete"]
    np.testing.assert_allclose(final["exposure_wh_m2"], whole["exposure_wh_m2"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(final["exposure_one_se_conservative_bound_wh_m2"],
                               whole["exposure_one_se_conservative_bound_wh_m2"], rtol=1e-12, atol=1e-9)
    with pytest.raises(ValueError, match="configuration"):
        run_case(tmp_path/"resume", resume=True, **{**options, "samples":17})
    block = tmp_path/"resume"/final["blocks"][0]["file"]
    block.write_bytes(block.read_bytes()+b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        run_case(tmp_path/"resume", resume=True, **options)


@pytest.mark.skipif(not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"), reason="compiled native library required")
def test_each_edit_matches_fresh_scene_and_old_result_is_immutable():
    scene, materials, points, normals, _ = district_fixture()
    timeline, dni, dhi = synthetic_forcing()
    selected = [12, 80*24+12, 172*24+12]
    options = dict(dni=dni[selected], dhi=dhi[selected], samples=32, replicates=2, bounces=3, seed=17)
    with ShortwaveSolver(scene, materials=materials, backend="native-builtin") as solver:
        original = solver.solve(points, normals, timeline.directions[selected], **options)
        saved = original.irradiance.copy()
        for _, event in EVENTS:
            apply_event(scene, materials, solver, event)
            edited = solver.solve(points, normals, timeline.directions[selected], **options)
            with ShortwaveSolver(scene, materials=materials, backend="native-builtin") as fresh:
                rebuilt = fresh.solve(points, normals, timeline.directions[selected], **options)
            np.testing.assert_allclose(edited.irradiance, rebuilt.irradiance, rtol=1e-12, atol=1e-10)
        np.testing.assert_array_equal(original.irradiance, saved)
        assert not original.irradiance.flags.writeable


@pytest.mark.skipif(not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"), reason="compiled native library required")
def test_restart_replays_transform_then_material_deformation_and_topology(tmp_path):
    options = dict(backend="native-builtin", steps=7400, block_steps=2048,
                   samples=8, replicates=2, bounces=3)
    partial = run_case(tmp_path/"resume-edits", max_blocks=3, **options)
    assert 2190 < partial["intervals"] < 4380
    resumed = run_case(tmp_path/"resume-edits", resume=True, **options)
    uninterrupted = run_case(tmp_path/"uninterrupted", **options)
    np.testing.assert_array_equal(resumed["exposure_wh_m2"], uninterrupted["exposure_wh_m2"])
    assert [p["hour"] for p in resumed["probes"]] == [0, *[hour for hour, _ in EVENTS]]
    assert [(b["geometry_revision"], b["material_revision"]) for b in resumed["blocks"]] == [
        (b["geometry_revision"], b["material_revision"]) for b in uninterrupted["blocks"]]
