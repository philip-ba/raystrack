"""Cross-path correctness and global budgets for dynamic execution."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from tests.v2_cases import MatrixCase, SkyCase, CaseSession, matrix_case, sky_case, outside_case, pair_case
from raystrack.utils.prepared import PreparedSolver


def scene():
    v = np.asarray([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
    f = np.asarray([[0,1,2], [0,2,3]], np.int32)
    return [("lower", v, f), ("upper", v + [0,0,1], f[:,::-1].copy())]


class ExecutionTests(unittest.TestCase):
    def params(self, **kwargs):
        common = dict(samples=2, rays=8, min_iters=3, max_iters=3,
                      device="cpu", reciprocity=False)
        common.update(kwargs)
        return MatrixCase(**common)

    def test_chunked_replicates_match_full_cpu_rays(self):
        meshes = scene()
        for bvh in ("off", "builtin"):
            p = self.params(bvh=bvh)
            with patch("tests.v2_cases._log"):
                expected = matrix_case(meshes, p)
                counted = []
                actual = matrix_case(meshes, replace(p, ray_batch_size=17), progress=counted.append)
            self.assertEqual(actual, expected)
            self.assertEqual(sum(counted), 2*16*8*3)
            self.assertLessEqual(max(counted), 17)

    def test_shared_budget_is_global_and_rest_is_sampled(self):
        p = self.params(emitter_names=["lower"], max_total_rays=67, ray_batch_size=17)
        sp = SkyCase(**{k:v for k,v in p.as_dict().items() if k in SkyCase.__dataclass_fields__})
        counts = []
        matrix, sky, rest = outside_case(scene(), matrix_params=p, sky_params=sp,
                                                        progress=counts.append)
        self.assertEqual(sum(counts), 67)
        self.assertEqual(set(matrix), {"lower"})
        self.assertEqual(set(sky), {"lower"})
        self.assertAlmostEqual(sum(matrix["lower"].values()) + sky["lower"]["Sky"] + rest["lower"]["Rest"], 1)

    def test_combined_outputs_use_one_sampling_configuration_and_budget(self):
        from raystrack import Solver, Query, Budget
        from tests.v2_cases import scene_for, options_for
        p = self.params(max_total_rays=29, ray_batch_size=7, emitter_names=["lower"])
        counts = []
        with Solver(scene_for(scene()), device="cpu", auto_tune=False) as solver:
            result = solver.solve(Query.row("lower", sky="merged"), options_for(p),
                                  Budget(rays=29), progress=counts.append)
        self.assertEqual(sum(counts), 29)
        self.assertEqual(result.sender_ids, ("lower",))
        self.assertEqual(result.statistics["emitters"]["lower"]["rays"], 29)
        self.assertAlmostEqual(sum(result.row("lower").values()), 1)

    def test_zero_budget_and_deadline_do_not_claim_results(self):
        p = self.params(max_total_rays=0)
        self.assertEqual(matrix_case(scene(), p), {})
        self.assertEqual(matrix_case(scene(), replace(p, max_total_rays=None, max_time_ms=0)), {})
        counts = []
        self.assertEqual(sky_case(scene(), SkyCase(max_total_rays=0), progress=counts.append), {})
        self.assertEqual(counts, [])

    def test_lone_upward_emitter_sees_the_sky(self):
        p = SkyCase(samples=1, rays=4, min_iters=2, max_iters=2, device="cpu")
        with patch("tests.v2_cases._log"):
            sky = sky_case(scene()[:1], p)
        self.assertEqual(sky, {"lower": {"Sky": 1.0}})

    def test_targeted_accepts_matching_float64_prepared_inputs(self):
        meshes = [(name, vertices.astype(np.float64) + [.1,.1,0], faces)
                  for name, vertices, faces in scene()]
        prepared = PreparedSolver(meshes)
        result = pair_case(meshes, "lower", "upper", samples=32, prepared=prepared)
        self.assertGreater(result["upper_front"], 0)
        transform = np.eye(4)
        transform[0,3] = 10
        prepared.update_transform("upper", transform)
        with self.assertRaisesRegex(ValueError, "prepared"):
            pair_case(meshes, "lower", "upper", samples=32, prepared=prepared)

    def test_cancellation_stops_between_chunks(self):
        counts = []
        matrix_case(scene(), self.params(ray_batch_size=11),
                           cancel=lambda: bool(counts), progress=counts.append)
        self.assertEqual(len(counts), 1)
        self.assertGreater(counts[0], 0)
        self.assertLessEqual(counts[0], 11)

    def test_selected_emitter_still_checks_all_occluders(self):
        meshes = scene()
        blocker = ("blocker", meshes[0][1] + [0,0,.5], meshes[0][2])
        result = matrix_case([blocker] + meshes,
            self.params(emitter_names=["lower"], max_total_rays=128))
        self.assertEqual(result["lower"].get("upper_front", 0), 0)
        self.assertGreater(result["lower"].get("blocker_back", 0), 0)

    def test_invalid_budgets_and_selection(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "rays"):
                matrix_case(scene(), self.params(max_total_rays=value))
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "time_ms"):
                matrix_case(scene(), self.params(max_time_ms=value))
        with self.assertRaisesRegex(ValueError, "Unknown surface"):
            matrix_case(scene(), self.params(emitter_names=["missing"]))

    def test_session_moving_scene_outside_workflow(self):
        prepared = PreparedSolver(scene())
        p = self.params()
        sp = SkyCase(samples=2, rays=8, min_iters=3, max_iters=3, device="cpu")
        with CaseSession(prepared, p, sp) as session:
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
