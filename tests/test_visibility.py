from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from raystrack import (MatrixParams, SkyParams, view_factor_matrix, view_factor_targeted,
                       view_factor_outside_workflow, view_factor_to_tregenza_sky)


def square(name: str, height: float, radius: float):
    vertices = np.asarray(
        [[-radius, -radius, height], [radius, -radius, height],
         [radius, radius, height], [-radius, radius, height]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return name, vertices, faces


class VisibilityTests(unittest.TestCase):
    def test_targeted_pair_finds_rare_receiver_with_correct_area_weight(self):
        emitter = square("emitter", 0, 1)
        tiny = square("tiny", 1, 0.005)
        tiny = (tiny[0], tiny[1], tiny[2][:, ::-1].copy())
        row = view_factor_targeted([emitter, tiny], "emitter", "tiny",
                                   samples=8192, seed=2)
        self.assertEqual(set(row), {"tiny_front"})

        # Receiver is small enough for its center-point limit to be accurate.
        xy = (np.arange(400) + 0.5) / 200.0 - 1.0
        x, y = np.meshgrid(xy, xy)
        reference = (0.01 ** 2 / np.pi
                     * np.mean(1.0 / (1.0 + x*x + y*y) ** 2))
        self.assertLess(abs(row["tiny_front"] - reference) / reference, 0.04)

        blocker = square("blocker", 0.5, 2)
        blocked = view_factor_targeted([emitter, blocker, tiny], "emitter", "tiny",
                                       samples=1024, seed=2)
        self.assertEqual(blocked, {})

        upward = square("tiny", 1, 0.005)
        back = view_factor_targeted([emitter, upward], "emitter", "tiny",
                                    samples=1024, seed=2)
        self.assertEqual(set(back), {"tiny_back"})

        with self.assertRaisesRegex(ValueError, "sequence"):
            view_factor_targeted([emitter, tiny], "emitter", "tiny",
                                 samples=16, sequence="unknown")
        with self.assertRaisesRegex(ValueError, "unique"):
            view_factor_targeted([emitter, emitter], "emitter", "tiny",
                                 samples=16)

    def test_minimum_ray_budget_prevents_zero_hit_early_stop(self):
        meshes = [square("emitter", 0, 1), square("tiny", 1, 1e-6)]
        base = dict(samples=4, rays=8, seed=5, bvh="off", device="cpu",
                    min_iters=2, max_iters=6, tol=1.0, reciprocity=False)
        with patch("raystrack.main._log") as log:
            view_factor_matrix(meshes, MatrixParams(**base))
        self.assertIn("2 iter", log.call_args_list[0].args[0])

        with patch("raystrack.main._log") as log:
            scene = view_factor_matrix(
                meshes, MatrixParams(**base, min_total_rays=640))
        self.assertIn("5 iter", log.call_args_list[0].args[0])
        self.assertEqual(scene["emitter"], {})

        sky_base = {key: value for key, value in base.items()
                    if key != "reciprocity"}
        with patch("raystrack.main._log") as log:
            view_factor_to_tregenza_sky(
                meshes, SkyParams(**sky_base, min_total_rays=640))
        self.assertIn("5 iter", log.call_args_list[0].args[0])

        with self.assertRaisesRegex(ValueError, "min_total_rays"):
            view_factor_matrix(meshes, MatrixParams(min_total_rays=-1))

    def test_bidirectional_front_exchange_is_reciprocal(self):
        lower = square("lower", 0, 1)
        upper = square("upper", 1, 2)
        upper = (upper[0], upper[1], upper[2][:, ::-1].copy())
        meshes = [lower, upper]
        params = MatrixParams(samples=4, rays=32, seed=8, bvh="off",
                              device="cpu", reciprocity_mode="bidirectional",
                              min_iters=4, max_iters=4)
        scene = view_factor_matrix(meshes, params)
        self.assertGreater(scene["lower"]["upper_front"], 0.0)
        self.assertAlmostEqual(
            4.0 * scene["lower"]["upper_front"],
            16.0 * scene["upper"]["lower_front"], places=12,
        )
        self.assertFalse(any(key.endswith("_back") for row in scene.values()
                             for key in row))

        sky_params = SkyParams(samples=4, rays=32, seed=8, bvh="off",
                               device="cpu", min_iters=4, max_iters=4)
        shared, _, _ = view_factor_outside_workflow(
            meshes, matrix_params=params, sky_params=sky_params
        )
        self.assertAlmostEqual(
            4.0 * shared["lower"]["upper_front"],
            16.0 * shared["upper"]["lower_front"], places=12,
        )

    def test_invalid_bidirectional_settings(self):
        meshes = [square("lower", 0, 1), square("upper", 1, 1)]
        with self.assertRaisesRegex(ValueError, "reciprocity_mode"):
            view_factor_matrix(meshes, MatrixParams(reciprocity_mode="unknown"))
        with self.assertRaisesRegex(ValueError, "requires reciprocity"):
            view_factor_matrix(meshes, MatrixParams(
                reciprocity=False, reciprocity_mode="bidirectional"))

    def test_reciprocity_does_not_see_through_lower_index_occluder(self):
        # Every ray from emitter to far must cross the middle square first.
        meshes = [square("blocker", 1, 2), square("emitter", 0, 1),
                  square("far", 2, 1)]
        for bvh in ("off", "builtin"):
            with self.subTest(bvh=bvh):
                matrix_params = MatrixParams(samples=4, rays=16, seed=3,
                                             bvh=bvh, device="cpu", reciprocity=True,
                                             min_iters=2, max_iters=2)
                scene = view_factor_matrix(meshes, params=matrix_params)
                self.assertEqual(scene["emitter"].get("far_front", 0.0), 0.0)
                self.assertEqual(scene["emitter"].get("far_back", 0.0), 0.0)

                sky_params = SkyParams(samples=4, rays=16, seed=3, bvh=bvh,
                                       device="cpu", min_iters=2, max_iters=2)
                combined, _, _ = view_factor_outside_workflow(
                    meshes, matrix_params=matrix_params, sky_params=sky_params
                )
                self.assertEqual(combined["emitter"].get("far_front", 0.0), 0.0)
                self.assertEqual(combined["emitter"].get("far_back", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
