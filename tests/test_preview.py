"""Persistent public runs: immutable snapshots, cancellation and async work."""
from dataclasses import FrozenInstanceError
import threading
import unittest

import numpy as np

from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Budget, Channel


def scene():
    vertices = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], np.float32)
    faces = np.array([[0,1,2],[0,2,3]], np.int32)
    return Scene.from_meshes({"lower": Mesh(vertices, faces),
                             "upper": Mesh(vertices+[0,0,1], faces[:,::-1])})


def options(*, seed=99, batch_size=16):
    return SolveOptions(Sampling(density=2, rays_per_cell=8, seed=seed),
                        Accuracy(max_replicates=10, min_replicates=3, tolerance=0),
                        batch_size=batch_size)


class PreviewTests(unittest.TestCase):
    def test_result_is_deeply_immutable_and_dense_exports_are_independent(self):
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            result = solver.solve(Query.row("lower"), options=options(), budget=Budget(rays=80))
        channel = Channel("surface", "upper", "front")
        before = result.value("lower", channel)
        with self.assertRaises(TypeError):
            result.row("lower")[channel] = .5
        with self.assertRaises(TypeError):
            result.statistics["emitters"]["lower"]["rays"] = 0
        with self.assertRaises(FrozenInstanceError):
            result.scene_revision = 4
        with self.assertRaises(ValueError):
            result.coverage.setflags(write=True)
        exported = result.dense()
        exported[:] = 1
        self.assertEqual(result.value("lower", channel), before)

    def test_query_owns_sender_selection_and_run_reports_bounded_progress(self):
        names = ["lower"]
        query = Query.matrix(senders=names)
        names.clear()
        progress = []
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            result = solver.solve(query, options=options(), budget=Budget(rays=80), progress=progress.append)
        self.assertEqual(result.sender_ids, ("lower",))
        self.assertEqual(result.rays_used, 80)
        self.assertEqual(sum(progress), 80)
        self.assertLessEqual(max(progress), 16)
        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_combined_scene_and_sky_share_one_selection_and_budget(self):
        progress = []
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            result = solver.solve(Query.row("upper", sky="merged"), options=options(),
                                  budget=Budget(rays=12), progress=progress.append)
        self.assertEqual(result.sender_ids, ("upper",))
        self.assertEqual(result.rays_used, sum(progress))
        self.assertEqual(result.rays_used, 12)
        self.assertAlmostEqual(sum(result.row("upper").values()), 1)
        self.assertIn(Channel("sky"), result.channels)
        self.assertIn(Channel("rest"), result.channels)

    def test_cancelled_run_is_terminal_and_previous_snapshot_is_unchanged(self):
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            run = solver.start(Query.row("lower"), options=options())
            previous = run.advance(Budget(rays=7))
            previous_values = previous.dense().copy()
            result = run.advance(Budget(rays=80), progress=lambda count: run.cancel())
            self.assertEqual(run.status, "cancelled")
            self.assertEqual(result.status, "cancelled")
            self.assertLessEqual(result.rays_used, 16)
            np.testing.assert_array_equal(previous.dense(), previous_values)
            stopped = run.advance(Budget(rays=1))
            self.assertEqual(stopped.rays_used, 0)
            self.assertEqual(stopped.cumulative_rays, result.cumulative_rays)
            self.assertEqual(stopped.status, "cancelled")

    def test_pre_cancelled_run_does_not_trace(self):
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            run = solver.start(Query.row("lower"), options=options())
            run.cancel()
            progress = []
            result = run.advance(Budget(rays=80), progress=progress.append)
            self.assertEqual(result.rays_used, 0)
            self.assertEqual(result.status, "cancelled")
            self.assertEqual(progress, [])

    def test_repeated_submit_queues_additive_budgets_and_serializes_execution(self):
        entered, release = threading.Event(), threading.Event()
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            run = solver.start(Query.row("lower"), options=options())
            def hold(count):
                entered.set()
                self.assertTrue(release.wait(10))
            first = run.submit(Budget(rays=10), progress=hold)
            try:
                self.assertTrue(entered.wait(10))
                second = run.submit(Budget(rays=20))
                third = run.submit(Budget(rays=30))
                self.assertFalse(second.done())
                self.assertFalse(third.done())
            finally:
                release.set()
            self.assertEqual(first.result(timeout=10).cumulative_rays, 10)
            self.assertEqual(second.result(timeout=10).cumulative_rays, 30)
            self.assertEqual(third.result(timeout=10).cumulative_rays, 60)

    def test_update_invalidates_worker_before_waiting_for_scene_lock(self):
        current = scene()
        entered, release, notified = threading.Event(), threading.Event(), threading.Event()
        current._subscribe(notified.set)
        with Solver(current, device="cpu", auto_tune=False) as solver:
            run = solver.start(Query.row("lower"), options=options())
            def hold(count):
                entered.set()
                self.assertTrue(release.wait(10))
            pending = run.submit(Budget(rays=80), progress=hold)
            transform = np.eye(4); transform[0,3] = .5
            updater = threading.Thread(target=current.update_transform, args=("upper", transform))
            try:
                self.assertTrue(entered.wait(10))
                updater.start()
                self.assertTrue(notified.wait(10))
                self.assertEqual(run.status, "invalidated")
            finally:
                release.set()
                updater.join(10)
            stale = pending.result(timeout=10)
            self.assertEqual(stale.status, "invalidated")
            self.assertEqual(stale.scene_revision, 0)
            self.assertEqual(current.revision, 1)
            with self.assertRaisesRegex(RuntimeError, "Scene geometry changed"):
                run.advance(Budget(rays=1))

    def test_topology_update_invalidates_old_work_and_rebuilds_new_acceleration(self):
        current = scene()
        with Solver(current, device="cpu", acceleration="instanced", auto_tune=False) as solver:
            run = solver.start(Query.row("lower"), options=options())
            previous = run.advance(Budget(rays=48))
            current.update_mesh("upper", Mesh(current["upper"].world_vertices(), np.array([[0,2,1]], np.int32)))
            self.assertEqual(run.status, "invalidated")
            with self.assertRaisesRegex(RuntimeError, "Scene geometry changed"):
                run.advance(Budget(rays=1))
            fresh = solver.solve(Query.row("lower"), options=options(), budget=Budget(rays=48))
            self.assertEqual(previous.scene_revision, 0)
            self.assertEqual(fresh.scene_revision, 1)
            self.assertEqual(fresh.rays_used, 48)

    def test_additional_budget_preserves_seed_and_requires_stationary_scene(self):
        current = scene()
        with Solver(current, device="cpu", auto_tune=False) as solver:
            run = solver.start(Query.row("lower"), options=options())
            first = run.advance(Budget(rays=10))
            refined = run.advance(Budget(rays=30))
            reference = solver.solve(Query.row("lower"), options=options(), budget=Budget(rays=40))
            np.testing.assert_array_equal(refined.dense(), reference.dense())
            self.assertEqual(first.cumulative_rays, 10)
            self.assertEqual(refined.rays_used, 30)
            self.assertEqual(refined.cumulative_rays, 40)
            transform = np.eye(4); transform[1,3] = .5
            current.update_transform("upper", transform)
            with self.assertRaisesRegex(RuntimeError, "Scene geometry changed"):
                run.advance(Budget(rays=1))

    def test_closed_run_and_solver_reject_work(self):
        solver = Solver(scene(), device="cpu", auto_tune=False)
        run = solver.start(Query.row("lower"), options=options())
        run.close(); run.close()
        for action in (lambda: run.advance(Budget(rays=1)), lambda: run.submit(Budget(rays=1))):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                action()
        solver.close(); solver.close()
        for action in (lambda: solver.start(Query.row("lower")), lambda: solver.solve(Query.row("lower"))):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                action()

    def test_fresh_cpu_runs_obey_budget_and_repeat_seed(self):
        with Solver(scene(), device="cpu", auto_tune=False) as solver:
            first = solver.solve(Query.row("lower"), options=options(), budget=Budget(rays=48))
            second = solver.solve(Query.row("lower"), options=options(), budget=Budget(rays=48))
        self.assertEqual(first.rays_used, 48)
        np.testing.assert_array_equal(first.dense(), second.dense())
        self.assertEqual(first.sender_ids, ("lower",))


if __name__ == "__main__":
    unittest.main()
