from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tests.v2_cases import MatrixCase, SkyCase, matrix_case, pair_case, outside_case, sky_case


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
        row = pair_case([emitter, tiny], "emitter", "tiny",
                                   samples=8192, seed=2)
        self.assertEqual(set(row), {"tiny_front"})

        # Receiver is small enough for its center-point limit to be accurate.
        xy = (np.arange(400) + 0.5) / 200.0 - 1.0
        x, y = np.meshgrid(xy, xy)
        reference = (0.01 ** 2 / np.pi
                     * np.mean(1.0 / (1.0 + x*x + y*y) ** 2))
        self.assertLess(abs(row["tiny_front"] - reference) / reference, 0.04)

        blocker = square("blocker", 0.5, 2)
        blocked = pair_case([emitter, blocker, tiny], "emitter", "tiny",
                                       samples=1024, seed=2)
        self.assertEqual(blocked, {})

        upward = square("tiny", 1, 0.005)
        back = pair_case([emitter, upward], "emitter", "tiny",
                                    samples=1024, seed=2)
        self.assertEqual(set(back), {"tiny_back"})

        with self.assertRaisesRegex(ValueError, "sequence"):
            pair_case([emitter, tiny], "emitter", "tiny",
                                 samples=16, sequence="unknown")
        with self.assertRaisesRegex(ValueError, "unique"):
            pair_case([emitter, emitter], "emitter", "tiny",
                                 samples=16)

    def test_minimum_ray_budget_prevents_zero_hit_early_stop(self):
        from raystrack import Solver, Query
        from tests.v2_cases import scene_for, options_for
        meshes = [square("emitter", 0, 1), square("tiny", 1, 1e-6)]
        base = MatrixCase(samples=4, rays=8, seed=5, bvh="off", device="cpu",
                          min_iters=2, max_iters=6, tol=1, reciprocity=False)
        with Solver(scene_for(meshes), device="cpu", bvh="off", auto_tune=False) as solver:
            early = solver.solve(Query.row("emitter"), options_for(base))
            self.assertEqual(early.statistics["emitters"]["emitter"]["replicates"], 2)
            base.min_total_rays = 640
            floor = solver.solve(Query.row("emitter"), options_for(base))
            sky = solver.solve(Query.sky(["emitter"]), options_for(base))
            self.assertEqual(floor.statistics["emitters"]["emitter"]["replicates"], 5)
            self.assertEqual(sky.statistics["emitters"]["emitter"]["replicates"], 5)
            self.assertFalse(any(value for channel, value in floor.row("emitter").items() if channel.kind == "surface"))
        with self.assertRaisesRegex(ValueError, "min_rays"):
            options_for(MatrixCase(min_total_rays=-1))

    def test_bidirectional_front_exchange_is_reciprocal(self):
        lower = square("lower", 0, 1)
        upper = square("upper", 1, 2)
        upper = (upper[0], upper[1], upper[2][:, ::-1].copy())
        meshes = [lower, upper]
        params = MatrixCase(samples=4, rays=32, seed=8, bvh="off",
                              device="cpu", reciprocity_mode="bidirectional",
                              min_iters=4, max_iters=4)
        scene = matrix_case(meshes, params)
        self.assertGreater(scene["lower"]["upper_front"], 0.0)
        self.assertAlmostEqual(
            4.0 * scene["lower"]["upper_front"],
            16.0 * scene["upper"]["lower_front"], places=12,
        )
        self.assertFalse(any(key.endswith("_back") for row in scene.values()
                             for key in row))

        sky_params = SkyCase(samples=4, rays=32, seed=8, bvh="off",
                               device="cpu", min_iters=4, max_iters=4)
        shared, _, _ = outside_case(
            meshes, matrix_params=params, sky_params=sky_params
        )
        self.assertAlmostEqual(
            4.0 * shared["lower"]["upper_front"],
            16.0 * shared["upper"]["lower_front"], places=12,
        )

    def test_invalid_bidirectional_settings(self):
        from raystrack import Solver, Query, Postprocessing, SolveOptions
        from tests.v2_cases import scene_for
        meshes = [square("lower", 0, 1), square("upper", 1, 1)]
        with self.assertRaisesRegex(ValueError, "reciprocity"):
            Postprocessing(reciprocity="unknown")
        with Solver(scene_for(meshes), device="cpu") as solver:
            with self.assertRaisesRegex(ValueError, "all sender"):
                solver.start(Query.row("lower"), SolveOptions(postprocessing=Postprocessing("bidirectional")))
            with self.assertRaisesRegex(ValueError, "receiver sides"):
                solver.start(Query.matrix(receiver_sides=("back",)),
                             SolveOptions(postprocessing=Postprocessing("bidirectional")))

    def test_reciprocity_does_not_see_through_lower_index_occluder(self):
        # Every ray from emitter to far must cross the middle square first.
        meshes = [square("blocker", 1, 2), square("emitter", 0, 1),
                  square("far", 2, 1)]
        for bvh in ("off", "builtin"):
            with self.subTest(bvh=bvh):
                matrix_params = MatrixCase(samples=4, rays=16, seed=3,
                                             bvh=bvh, device="cpu", reciprocity=True,
                                             min_iters=2, max_iters=2)
                scene = matrix_case(meshes, params=matrix_params)
                self.assertEqual(scene["emitter"].get("far_front", 0.0), 0.0)
                self.assertEqual(scene["emitter"].get("far_back", 0.0), 0.0)

                sky_params = SkyCase(samples=4, rays=16, seed=3, bvh=bvh,
                                       device="cpu", min_iters=2, max_iters=2)
                combined, _, _ = outside_case(
                    meshes, matrix_params=matrix_params, sky_params=sky_params
                )
                self.assertEqual(combined["emitter"].get("far_front", 0.0), 0.0)
                self.assertEqual(combined["emitter"].get("far_back", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
