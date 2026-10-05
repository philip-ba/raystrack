from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import FrozenInstanceError
import threading
import unittest
from unittest.mock import patch

import numpy as np

from raystrack.params import MatrixParams, SkyParams
from raystrack.preview import PreviewResult, PreviewSession
from raystrack.utils.prepared import PreparedSolver


def square(name, z):
    vertices = np.asarray([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return name, vertices, faces


def solver():
    return PreparedSolver([square("lower", 0), square("upper", 1)])


class PreviewTests(unittest.TestCase):
    def test_result_is_deeply_immutable_and_exports_independent_dicts(self):
        source = {"lower": {"upper_front": 0.25}}
        result = PreviewResult(3, source, {}, {}, True, False, 80, 12.0)
        source["lower"]["upper_front"] = 0.9
        self.assertEqual(result.scene["lower"]["upper_front"], 0.25)
        with self.assertRaises(TypeError):
            result.scene["lower"]["upper_front"] = 0.5
        with self.assertRaises(FrozenInstanceError):
            result.scene_version = 4
        exported = result.as_dict()
        exported["scene"]["lower"]["upper_front"] = 1.0
        self.assertEqual(result.scene["lower"]["upper_front"], 0.25)

    def test_preview_clones_params_selects_emitters_and_reports_rays(self):
        names = ["lower"]
        params = MatrixParams(device="cpu", emitter_names=names, max_iters=5)
        progress = []

        def trace(meshes, passed, *, prepared, cancel, progress, accumulator=None):
            self.assertEqual(passed.emitter_names, ["lower"])
            self.assertEqual(passed.max_total_rays, 80)
            self.assertIsNone(passed.max_time_ms)
            self.assertEqual(passed.ray_batch_size, 8192)
            self.assertFalse(cancel())
            progress(30)
            progress(50)
            return {"lower": {"upper_back": 0.25}}

        with PreviewSession(solver(), params) as session:
            names.clear()
            with patch("raystrack.preview.view_factor_matrix", side_effect=trace):
                result = session.preview(max_total_rays=80, max_time_ms=None, progress=progress.append)
            self.assertTrue(result.completed)
            self.assertFalse(result.cancelled)
            self.assertEqual(result.rays_used, 80)
            self.assertEqual(progress, [30, 50])
            self.assertIs(session.latest, result)
            self.assertGreaterEqual(result.elapsed_ms, 0)
        self.assertEqual(params.emitter_names, [])
        self.assertIsNone(params.max_total_rays)

    def test_outside_workflow_gets_matching_selection_and_budget(self):
        def trace(meshes, *, matrix_params, sky_params, prepared, cancel, progress, accumulator=None):
            self.assertEqual(matrix_params.emitter_names, ["upper"])
            self.assertEqual(sky_params.emitter_names, ["upper"])
            self.assertEqual(matrix_params.max_total_rays, 12)
            self.assertEqual(sky_params.max_total_rays, 12)
            progress(12)
            return {"upper": {}}, {"upper": {"Sky": 0.75}}, {"upper": {"Rest": 0.25}}

        with PreviewSession(solver(), MatrixParams(), SkyParams()) as session:
            with patch("raystrack.preview.view_factor_outside_workflow", side_effect=trace):
                result = session.preview(emitter_names=["upper"], max_total_rays=12)
        self.assertEqual(result.sky["upper"]["Sky"], 0.75)
        self.assertEqual(result.rest["upper"]["Rest"], 0.25)
        self.assertEqual(result.rays_used, 12)

    def test_cancelled_request_drops_partial_results(self):
        cancelled = threading.Event()

        def trace(meshes, params, *, prepared, cancel, progress, accumulator=None):
            progress(7)
            cancelled.set()
            self.assertTrue(cancel())
            return {"lower": {"upper_back": 0.5}}

        with PreviewSession(solver()) as session:
            with patch("raystrack.preview.view_factor_matrix", side_effect=trace):
                result = session.solve(cancel=cancelled.is_set)
            self.assertIsNone(session.latest)
        self.assertFalse(result.completed)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.rays_used, 7)
        self.assertEqual(dict(result.scene), {})

    def test_pre_cancelled_request_does_not_trace(self):
        with PreviewSession(solver()) as session:
            with patch("raystrack.preview.view_factor_matrix") as trace:
                result = session.preview(cancel=lambda: True)
            trace.assert_not_called()
            self.assertTrue(result.cancelled)
            self.assertEqual(result.rays_used, 0)

    def test_submit_coalesces_queued_frames_and_cancels_running_frame(self):
        started = threading.Event()
        release = threading.Event()
        active = 0
        peak_active = 0
        calls = []

        def trace(meshes, params, *, prepared, cancel, progress, accumulator=None):
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            calls.append(params.max_total_rays)
            try:
                if len(calls) == 1:
                    started.set()
                    self.assertTrue(release.wait(5))
                    self.assertTrue(cancel())
                progress(1)
                return {"lower": {"upper_back": 0.2}}
            finally:
                active -= 1

        with patch("raystrack.preview.view_factor_matrix", side_effect=trace):
            with PreviewSession(solver()) as session:
                first = session.submit(max_total_rays=10, max_time_ms=None)
                try:
                    self.assertTrue(started.wait(5))
                    queued = session.submit(max_total_rays=20, max_time_ms=None)
                    newest = session.submit(max_total_rays=30, max_time_ms=None)
                finally:
                    release.set()
                self.assertTrue(first.result(timeout=5).cancelled)
                with self.assertRaises(CancelledError):
                    queued.result(timeout=5)
                result = newest.result(timeout=5)
                self.assertTrue(result.completed)
                self.assertIs(session.latest, result)
        self.assertEqual(calls, [10, 30])
        self.assertEqual(peak_active, 1)

    def test_update_cancels_worker_before_mutating_scene(self):
        started = threading.Event()
        stopped = threading.Event()
        prepared = solver()
        original_version = prepared.version

        def trace(meshes, params, *, prepared, cancel, progress, accumulator=None):
            started.set()
            deadline = threading.Event()
            for _ in range(500):
                if cancel():
                    stopped.set()
                    progress(1)
                    return {"lower": {"upper_back": 0.3}}
                deadline.wait(0.01)
            self.fail("scene update failed to cancel the worker")

        translation = np.eye(4)
        translation[0, 3] = 0.5
        with patch("raystrack.preview.view_factor_matrix", side_effect=trace):
            with PreviewSession(prepared) as session:
                pending = session.submit(max_time_ms=None)
                self.assertTrue(started.wait(5))
                session.update_transform("upper", translation)
                self.assertTrue(stopped.is_set())
                self.assertTrue(pending.result(timeout=5).cancelled)
                self.assertEqual(prepared.version, original_version + 1)
                self.assertIsNone(session.latest)

    def test_rebuild_cancels_worker_and_replaces_acceleration_snapshot(self):
        prepared = solver()
        original_scene = prepared.get_scene(use_bvh=True)
        original_version = prepared.version
        started = threading.Event()

        def trace(meshes, params, *, prepared, cancel, progress, accumulator=None):
            started.set()
            pause = threading.Event()
            for _ in range(500):
                if cancel():
                    return {"lower": {"upper_back": 0.3}}
                pause.wait(0.01)
            self.fail("BVH rebuilding failed to cancel the worker")

        with patch("raystrack.preview.view_factor_matrix", side_effect=trace):
            with PreviewSession(prepared) as session:
                pending = session.submit(max_time_ms=None)
                self.assertTrue(started.wait(5))
                new_version = session.rebuild_bvh()
                self.assertTrue(pending.result(timeout=5).cancelled)
                self.assertEqual(new_version, original_version + 1)
                self.assertIsNot(prepared.get_scene(use_bvh=True), original_scene)
                self.assertIsNone(session.latest)

    def test_refinement_increases_budget_preserves_seed_and_requires_stationary_scene(self):
        captured = []

        def trace(meshes, params, *, prepared, cancel, progress, accumulator=None):
            captured.append(params)
            progress(params.max_total_rays)
            return {"lower": {"upper_back": 0.5}}

        with PreviewSession(solver(), MatrixParams(seed=99, max_iters=3)) as session:
            with self.assertRaisesRegex(RuntimeError, "before refining"):
                session.refine()
            with patch("raystrack.preview.view_factor_matrix", side_effect=trace):
                first = session.preview(max_total_rays=10)
                refined = session.refine(level=2)
                self.assertEqual(captured[1].max_total_rays, 30)
                self.assertEqual(captured[1].max_iters, 12)
                self.assertEqual(captured[1].seed, 99)
                self.assertIsNone(captured[1].max_time_ms)
                self.assertEqual(refined.scene_version, first.scene_version)
                self.assertEqual(refined.rays_used, 30)
                self.assertEqual(refined.cumulative_rays, 40)
                translation = np.eye(4)
                translation[1, 3] = 0.5
                session.update_transform("upper", translation)
                with self.assertRaisesRegex(RuntimeError, "scene changed"):
                    session.refine()

    def test_closed_session_rejects_work(self):
        session = PreviewSession(solver())
        session.close()
        session.close()
        for action in (session.preview, session.submit, session.solve, session.refine):
            with self.subTest(action=action.__name__):
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    action()

    def test_real_cpu_preview_obeys_budget_and_repeats_seed(self):
        params = MatrixParams(samples=2, rays=8, min_iters=2, max_iters=3,
                              reciprocity=False, bvh="builtin", device="cpu")
        with patch("raystrack.main._log"):
            with PreviewSession(solver(), params) as session:
                first = session.preview(emitter_names=["lower"], max_total_rays=48, max_time_ms=None)
                second = session.preview(emitter_names=["lower"], max_total_rays=48, max_time_ms=None)
        self.assertTrue(first.completed)
        self.assertEqual(first.rays_used, 48)
        self.assertEqual(first.scene, second.scene)
        self.assertEqual(set(first.scene), {"lower"})


if __name__ == "__main__":
    unittest.main()
