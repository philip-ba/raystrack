"""Resumable estimates preserve the seeded ray stream across solve boundaries."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from tests.v2_cases import MatrixCase, SkyCase
from raystrack.utils.prepared import PreparedSolver
from tests.v2_cases import CaseSession


def scene():
    vertices = np.asarray([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return [("lower", vertices, faces), ("upper", vertices + [0, 0, 1], faces[:, ::-1].copy())]


def parameters(device="cpu", **overrides):
    values = dict(samples=2, rays=8, seed=19, bvh="builtin", device=device,
                  min_iters=2, max_iters=10, tol=0.0, reciprocity=False,
                  emitter_names=["lower"], ray_batch_size=17, auto_tune=False)
    values.update(overrides)
    return MatrixCase(**values)


def sky_parameters(matrix, **overrides):
    values = {key: value for key, value in matrix.as_dict().items()
              if key in SkyCase.__dataclass_fields__}
    values.update(overrides)
    return SkyCase(**values)


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.logging = patch("tests.v2_cases._log")
        self.logging.start()
        self.addCleanup(self.logging.stop)

    def reference(self, mp, sp, budget):
        with CaseSession(PreparedSolver(scene()), mp, sp) as session:
            return session.solve(max_total_rays=budget, max_time_ms=None)

    def assert_factors_equal(self, actual, expected):
        self.assertEqual(actual.scene, expected.scene)
        self.assertEqual(actual.sky, expected.sky)
        self.assertEqual(actual.rest, expected.rest)
        self.assertEqual(actual.cumulative_rays, actual.statistics["cumulative_rays"])
        self.assertEqual(expected.cumulative_rays, expected.statistics["cumulative_rays"])

    def test_partial_replicate_resumes_without_retracing(self):
        mp = parameters()
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            counts = []
            refined = session.refine(max_total_rays=91, progress=counts.append)
        self.assertEqual(first.rays_used, 37)
        self.assertEqual(first.cumulative_rays, 37)
        self.assertEqual(first.statistics["emitters"]["lower"]["ray_offset"], 37)
        self.assertIsNone(first.statistics["emitters"]["lower"]["matrix"]["stderr"])
        self.assertEqual(refined.rays_used, 91)
        self.assertEqual(sum(counts), 91)
        self.assertEqual(refined.cumulative_rays, 128)
        self.assertEqual(refined.statistics["emitters"]["lower"]["replicates"], 1)
        self.assertEqual(refined.statistics["emitters"]["lower"]["ray_offset"], 0)
        self.assert_factors_equal(refined, self.reference(mp, None, 128))

    def test_split_shared_scene_and_sky_equal_uninterrupted_with_new_chunk_size(self):
        mp = parameters()
        sp = sky_parameters(mp)
        with CaseSession(PreparedSolver(scene()), mp, sp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            total = 37
            for budget in (23, 91, 104):
                counts = []
                refined = session.refine(max_total_rays=budget, ray_batch_size=7, progress=counts.append)
                total += budget
                self.assertEqual(refined.rays_used, budget)
                self.assertEqual(sum(counts), budget)
                self.assertLessEqual(max(counts), 7)
                self.assertEqual(refined.cumulative_rays, total)
        self.assert_factors_equal(refined, self.reference(replace(mp, ray_batch_size=7),
                                                         replace(sp, ray_batch_size=7), total))
        # Earlier result metadata is a snapshot, independent of the live accumulator.
        self.assertEqual(first.statistics["emitters"]["lower"]["ray_offset"], 37)
        with self.assertRaises(TypeError):
            first.statistics["emitters"]["lower"]["rays"] = 0
        exported = first.as_dict()
        exported["statistics"]["emitters"]["lower"]["rays"] = 0
        self.assertEqual(first.statistics["emitters"]["lower"]["rays"], 37)

    def test_fair_multi_emitter_schedule_survives_partial_chunk_boundary(self):
        mp = parameters(emitter_names=["lower", "upper"])
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.preview(max_total_rays=31, max_time_ms=None)
            refined = session.refine(max_total_rays=33)
        self.assertTrue(all(row["rays"] > 0 for row in first.statistics["emitters"].values()))
        self.assertEqual(refined.rays_used, 33)
        self.assertEqual(refined.cumulative_rays, 64)
        expected = self.reference(mp, None, 64)
        self.assert_factors_equal(refined, expected)
        self.assertEqual({name: row["rays"] for name, row in refined.statistics["emitters"].items()},
                         {name: row["rays"] for name, row in expected.statistics["emitters"].items()})

    def test_default_refinement_doubles_total_with_only_missing_rays(self):
        mp = parameters()
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            second = session.refine()
            third = session.refine()
        self.assertEqual((first.rays_used, second.rays_used, third.rays_used), (37, 37, 74))
        self.assertEqual((first.cumulative_rays, second.cumulative_rays, third.cumulative_rays), (37, 74, 148))
        self.assert_factors_equal(third, self.reference(mp, None, 148))

    def test_geometry_change_requires_new_preview_and_resets_counts(self):
        mp = parameters()
        prepared = PreparedSolver(scene())
        with CaseSession(prepared, mp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            transform = np.eye(4)
            transform[0, 3] = 4
            session.update_transform("upper", transform)
            with self.assertRaisesRegex(RuntimeError, "Scene geometry changed"):
                session.refine()
            current = session.preview(max_total_rays=37, max_time_ms=None)
            with CaseSession(PreparedSolver(prepared.meshes), mp) as fresh:
                expected = fresh.solve(max_total_rays=37)
        self.assertEqual(current.cumulative_rays, 37)
        self.assertEqual(current.scene_version, first.scene_version + 1)
        self.assert_factors_equal(current, expected)

    def test_incompatible_refinement_fails_without_corrupting_retained_state(self):
        mp = parameters()
        with CaseSession(PreparedSolver(scene()), mp) as session:
            session.preview(max_total_rays=37, max_time_ms=None)
            for settings in ({"matrix_params": replace(mp, seed=20)},
                             {"matrix_params": replace(mp, rays=16)},
                             {"emitter_names": ["upper"]},
                             {"sky_params": sky_parameters(mp)}):
                with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "start a new run"):
                    session.refine(**settings)
            with self.assertRaisesRegex(ValueError, "rays"):
                session.refine(max_total_rays=-1)
            refined = session.refine(max_total_rays=91)
            restarted = session.preview(matrix_params=replace(mp, seed=20),
                                        max_total_rays=37, max_time_ms=None)
        self.assert_factors_equal(refined, self.reference(mp, None, 128))
        self.assertEqual(restarted.cumulative_rays, 37)
        self.assert_factors_equal(restarted, self.reference(replace(mp, seed=20), None, 37))

    def test_cancelled_refinement_retains_snapshot_and_stops_new_counts(self):
        mp = parameters()
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            counts = []
            cancelled = session.refine(max_total_rays=91, cancel=lambda: bool(counts), progress=counts.append)
            self.assertTrue(cancelled.cancelled)
            self.assertGreater(cancelled.rays_used, 0)
            self.assertEqual(cancelled.cumulative_rays, 37 + sum(counts))
            self.assertEqual(first.cumulative_rays, 37)
            stopped = session.refine(max_total_rays=91)
            self.assertEqual(stopped.rays_used, 0)
            self.assertEqual(stopped.cumulative_rays, cancelled.cumulative_rays)
            restarted = session.preview(max_total_rays=37, max_time_ms=None)
        self.assertEqual(restarted.cumulative_rays, 37)
        self.assert_factors_equal(restarted, self.reference(mp, None, 37))

    def test_zero_budget_preview_can_resume_later(self):
        mp = parameters()
        with CaseSession(PreparedSolver(scene()), mp) as session:
            empty = session.preview(max_total_rays=0, max_time_ms=None)
            refined = session.refine(max_total_rays=37)
        self.assertEqual(empty.cumulative_rays, 0)
        self.assertEqual(empty.rays_used, 0)
        self.assertEqual(refined.cumulative_rays, 37)
        self.assert_factors_equal(refined, self.reference(mp, None, 37))

    def test_tighter_tolerance_reuses_converged_replicates(self):
        mp = parameters(tol=1.0)
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.preview(max_total_rays=512, max_time_ms=None)
            self.assertTrue(first.converged)
            self.assertEqual(first.rays_used, 256)
            refined = session.refine(matrix_params=replace(mp, tol=0.0), max_total_rays=128)
        self.assertEqual(refined.rays_used, 128)
        self.assertEqual(refined.cumulative_rays, 384)
        self.assertEqual(refined.statistics["emitters"]["lower"]["replicates"], 3)
        self.assert_factors_equal(refined, self.reference(replace(mp, tol=0.0), None, 384))

    def test_iteration_limit_can_expand_without_replaying_completed_replicates(self):
        mp = parameters(min_iters=1, max_iters=1)
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.solve()
            refined = session.refine()
        self.assertEqual(first.rays_used, 128)
        self.assertEqual(first.status, "max_iters")
        self.assertFalse(first.converged)
        self.assertEqual(refined.rays_used, 128)
        self.assertEqual(refined.cumulative_rays, 256)
        self.assert_factors_equal(refined, self.reference(replace(mp, max_iters=2), None, 256))

    def test_distinct_matrix_and_sky_queries_keep_independent_partial_states(self):
        from raystrack import Solver, Query, Budget
        from tests.v2_cases import scene_for, options_for
        mp = parameters(tol=1)
        sp = sky_parameters(mp, seed=20, tol=0)
        with Solver(scene_for(scene()), device="cpu", auto_tune=False) as solver:
            matrix = solver.start(Query.row("lower"), options_for(mp))
            sky = solver.start(Query.sky(["lower"]), options_for(sp, matrix=False))
            first = matrix.advance(Budget(rays=256))
            partial = sky.advance(Budget(rays=37))
            refined = sky.advance(Budget(rays=100))
            expected = solver.solve(Query.sky(["lower"]), options_for(sp, matrix=False), Budget(rays=137))
            self.assertEqual(first.cumulative_rays, 256)
            self.assertEqual(partial.cumulative_rays, 37)
            self.assertEqual(refined.rays_used, 100)
            self.assertEqual(refined.cumulative_rays, 137)
            self.assertEqual(matrix.cumulative_rays + sky.cumulative_rays, 393)
            np.testing.assert_array_equal(refined.dense(), expected.dense())
            self.assertEqual(matrix.result.cumulative_rays, 256)

    def test_warmup_probes_do_not_change_retained_counts(self):
        mp = parameters()
        with CaseSession(PreparedSolver(scene()), mp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            plan = session.warmup(repeats=1, max_probe_rays=16)
            self.assertIs(session.latest, first)
            self.assertGreater(plan.warmup_rays, 0)
            refined = session.refine(max_total_rays=91)
        self.assertEqual(refined.cumulative_rays, 128)
        self.assertEqual(refined.rays_used, 91)
        self.assert_factors_equal(refined, self.reference(mp, None, 128))


@unittest.skipUnless(importlib.util.find_spec("taichi") is not None, "optional Taichi dependency is absent")
class VulkanRefinementTests(unittest.TestCase):
    """Exercise the same continuation guarantees on real Vulkan hardware."""

    @classmethod
    def setUpClass(cls):
        from raystrack.utils.taichi_trace import get_taichi_backend, TaichiUnavailableError
        try:
            get_taichi_backend(arch="vulkan")
        except TaichiUnavailableError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        self.logging = patch("tests.v2_cases._log")
        self.logging.start()
        self.addCleanup(self.logging.stop)

    def reference(self, mp, sp, budget):
        with CaseSession(PreparedSolver(scene()), mp, sp) as session:
            return session.solve(max_total_rays=budget, max_time_ms=None)

    def assert_factors_equal(self, actual, expected):
        self.assertEqual(actual.scene, expected.scene)
        self.assertEqual(actual.sky, expected.sky)
        self.assertEqual(actual.rest, expected.rest)
        self.assertEqual(actual.cumulative_rays, actual.statistics["cumulative_rays"])
        self.assertEqual(expected.cumulative_rays, expected.statistics["cumulative_rays"])

    def test_vulkan_split_shared_preview_matches_uninterrupted(self):
        mp = parameters(device="vulkan")
        sp = sky_parameters(mp)
        with CaseSession(PreparedSolver(scene()), mp, sp) as session:
            first = session.preview(max_total_rays=37, max_time_ms=None)
            refined = session.refine(max_total_rays=219)
        self.assertEqual(first.cumulative_rays, 37)
        self.assertEqual(refined.rays_used, 219)
        self.assertEqual(refined.cumulative_rays, 256)
        self.assert_factors_equal(refined, self.reference(mp, sp, 256))

    def test_vulkan_fair_multi_emitter_partial_chunk_resume(self):
        mp = parameters(device="vulkan", emitter_names=["lower", "upper"])
        with CaseSession(PreparedSolver(scene()), mp) as session:
            session.preview(max_total_rays=31, max_time_ms=None)
            refined = session.refine(max_total_rays=225)
        self.assertEqual(refined.rays_used, 225)
        self.assertEqual(refined.cumulative_rays, 256)
        self.assert_factors_equal(refined, self.reference(mp, None, 256))


if __name__ == "__main__":
    unittest.main()
