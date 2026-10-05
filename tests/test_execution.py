"""Cross-path correctness and global budgets for dynamic execution."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from raystrack import (MatrixParams, SkyParams, PreparedSolver, PreviewSession,
    view_factor_matrix, view_factor_to_tregenza_sky, view_factor_outside_workflow,
    view_factor_targeted)


def scene():
    v = np.asarray([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
    f = np.asarray([[0,1,2], [0,2,3]], np.int32)
    return [("lower", v, f), ("upper", v + [0,0,1], f[:,::-1].copy())]


class ExecutionTests(unittest.TestCase):
    def params(self, **kwargs):
        common = dict(samples=2, rays=8, min_iters=3, max_iters=3,
                      device="cpu", reciprocity=False)
        common.update(kwargs)
        return MatrixParams(**common)

    def test_chunked_replicates_match_full_cpu_rays(self):
        meshes = scene()
        for bvh in ("off", "builtin"):
            p = self.params(bvh=bvh)
            with patch("raystrack.main._log"):
                expected = view_factor_matrix(meshes, p)
                counted = []
                actual = view_factor_matrix(meshes, replace(p, ray_batch_size=17), progress=counted.append)
            self.assertEqual(actual, expected)
            self.assertEqual(sum(counted), 2*16*8*3)
            self.assertLessEqual(max(counted), 17)

    def test_shared_budget_is_global_and_rest_is_sampled(self):
        p = self.params(emitter_names=["lower"], max_total_rays=67, ray_batch_size=17)
        sp = SkyParams(**{k:v for k,v in p.as_dict().items() if k in SkyParams.__dataclass_fields__})
        counts = []
        matrix, sky, rest = view_factor_outside_workflow(scene(), matrix_params=p, sky_params=sp,
                                                        progress=counts.append)
        self.assertEqual(sum(counts), 67)
        self.assertEqual(set(matrix), {"lower"})
        self.assertEqual(set(sky), {"lower"})
        self.assertAlmostEqual(sum(matrix["lower"].values()) + sky["lower"]["Sky"] + rest["lower"]["Rest"], 1)

    def test_incompatible_workflow_has_one_budget(self):
        p = self.params(max_total_rays=29, ray_batch_size=7, emitter_names=["lower"])
        sp = SkyParams(samples=3, rays=8, min_iters=3, max_iters=3, device="cpu",
                       max_total_rays=29, ray_batch_size=7, emitter_names=["lower"])
        counts = []
        matrix, sky, rest = view_factor_outside_workflow(scene(), matrix_params=p, sky_params=sp,
                                                        progress=counts.append)
        self.assertEqual(sum(counts), 29)
        self.assertEqual(set(matrix), {"lower"})
        self.assertEqual(sky, {})
        self.assertEqual(rest, {})

    def test_zero_budget_and_deadline_do_not_claim_results(self):
        p = self.params(max_total_rays=0)
        self.assertEqual(view_factor_matrix(scene(), p), {})
        self.assertEqual(view_factor_matrix(scene(), replace(p, max_total_rays=None, max_time_ms=0)), {})
        counts = []
        self.assertEqual(view_factor_to_tregenza_sky(scene(), SkyParams(max_total_rays=0), progress=counts.append), {})
        self.assertEqual(counts, [])

    def test_lone_upward_emitter_sees_the_sky(self):
        p = SkyParams(samples=1, rays=4, min_iters=2, max_iters=2, device="cpu")
        with patch("raystrack.main._log"):
            sky = view_factor_to_tregenza_sky(scene()[:1], p)
        self.assertEqual(sky, {"lower": {"Sky": 1.0}})

    def test_targeted_accepts_matching_float64_prepared_inputs(self):
        meshes = [(name, vertices.astype(np.float64) + [.1,.1,0], faces)
                  for name, vertices, faces in scene()]
        prepared = PreparedSolver(meshes)
        result = view_factor_targeted(meshes, "lower", "upper", samples=32, prepared=prepared)
        self.assertGreater(result["upper_front"], 0)
        transform = np.eye(4)
        transform[0,3] = 10
        prepared.update_transform("upper", transform)
        with self.assertRaisesRegex(ValueError, "prepared"):
            view_factor_targeted(meshes, "lower", "upper", samples=32, prepared=prepared)

    def test_cancellation_stops_between_chunks(self):
        counts = []
        view_factor_matrix(scene(), self.params(ray_batch_size=11),
                           cancel=lambda: bool(counts), progress=counts.append)
        self.assertEqual(len(counts), 1)
        self.assertGreater(counts[0], 0)
        self.assertLessEqual(counts[0], 11)

    def test_selected_emitter_still_checks_all_occluders(self):
        meshes = scene()
        blocker = ("blocker", meshes[0][1] + [0,0,.5], meshes[0][2])
        result = view_factor_matrix([blocker] + meshes,
            self.params(emitter_names=["lower"], max_total_rays=128))
        self.assertEqual(result["lower"].get("upper_front", 0), 0)
        self.assertGreater(result["lower"].get("blocker_back", 0), 0)

    def test_invalid_budgets_and_selection(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_total_rays"):
                view_factor_matrix(scene(), self.params(max_total_rays=value))
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "max_time_ms"):
                view_factor_matrix(scene(), self.params(max_time_ms=value))
        with self.assertRaisesRegex(ValueError, "Unknown emitter"):
            view_factor_matrix(scene(), self.params(emitter_names=["missing"]))

    def test_session_moving_scene_outside_workflow(self):
        prepared = PreparedSolver(scene())
        p = self.params()
        sp = SkyParams(samples=2, rays=8, min_iters=3, max_iters=3, device="cpu")
        with PreviewSession(prepared, p, sp) as session:
            first = session.preview(emitter_names=["lower"], max_time_ms=None, max_total_rays=96)
            transform = np.eye(4)
            transform[0,3] = 5
            session.update_transform("upper", transform)
            moved = session.preview(emitter_names=["lower"], max_time_ms=None, max_total_rays=96)
            refined = session.refine()
        self.assertEqual(first.scene_version, 0)
        self.assertEqual(moved.scene_version, 1)
        self.assertEqual(refined.scene_version, 1)
        self.assertGreater(first.scene["lower"].get("upper_front", 0), moved.scene["lower"].get("upper_front", 0))
        self.assertGreaterEqual(refined.rays_used, moved.rays_used)


if __name__ == "__main__":
    unittest.main()
