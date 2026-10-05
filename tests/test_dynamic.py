from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from tests.v2_cases import MatrixCase, matrix_case
from raystrack.utils.prepared import PreparedSolver
from raystrack.utils.bvh import build_bvh, refit_bvh


def square(name: str, z: float = 0, radius: float = 1, down: bool = False):
    vertices = np.asarray([[-radius, -radius, z], [radius, -radius, z],
                           [radius, radius, z], [-radius, radius, z]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return name, vertices, faces[:, ::-1].copy() if down else faces


def translation(x: float = 0, y: float = 0, z: float = 0):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


class DynamicSceneTests(unittest.TestCase):
    def assert_matches_fresh(self, prepared):
        fresh = PreparedSolver(prepared.meshes)
        for use_bvh in (False, True):
            actual = prepared.get_scene(use_bvh=use_bvh)
            expected = fresh.get_scene(use_bvh=use_bvh)
            actual_order = np.argsort(actual.permutation)
            expected_order = np.argsort(expected.permutation)
            for attr in ("v0", "e1", "e2", "normals", "sid"):
                np.testing.assert_array_equal(getattr(actual, attr)[actual_order],
                                              getattr(expected, attr)[expected_order])
            if actual.use_bvh:
                np.testing.assert_array_equal(actual.bb_min[0], expected.bb_min[0])
                np.testing.assert_array_equal(actual.bb_max[0], expected.bb_max[0])
        for flip in (False, True):
            emitters = prepared.get_emitters(samples=4, rays=2, flip_faces=flip)
            fresh_emitters = fresh.get_emitters(samples=4, rays=2, flip_faces=flip)
            for actual, expected in zip(emitters, fresh_emitters):
                for attr in ("tri_a", "tri_e1", "tri_e2", "tri_u", "tri_v", "tri_n",
                             "tri_origin_eps", "plane_origin", "plane_normal", "cdf"):
                    np.testing.assert_array_equal(getattr(actual, attr), getattr(expected, attr))
                self.assertEqual(actual.total_area, expected.total_area)
                self.assertEqual(actual.plane_is_planar, expected.plane_is_planar)
        for actual, expected in zip(prepared.get_mesh_bounds(), fresh.get_mesh_bounds()):
            np.testing.assert_array_equal(actual, expected)

    def test_owned_geometry_and_external_validation(self):
        source = [square("surface")]
        # Most mesh loaders supply float64 vertices; comparisons must use the
        # precision of the owned tracing geometry.
        source[0] = (source[0][0], source[0][1].astype(np.float64) + 0.1, source[0][2])
        prepared = PreparedSolver(source)
        prepared.validate_meshes(source)
        prepared.validate_meshes(prepared.meshes)
        exposed = prepared.meshes
        with self.assertRaises(ValueError):
            exposed[0][1][0] = 9
        with self.assertRaises(ValueError):
            exposed[0][1].setflags(write=True)
        exposed.clear()
        self.assertEqual(len(prepared.meshes), 1)
        expected = prepared.meshes[0][1].copy()
        source[0][1][:] += 1
        np.testing.assert_array_equal(prepared.meshes[0][1], expected)
        with self.assertRaisesRegex(ValueError, "prepared geometry"):
            prepared.validate_meshes(source)

    def test_absolute_rigid_transforms_refresh_frames_and_reuse_tables(self):
        prepared = PreparedSolver([square("moving"), square("fixed", 2)])
        old_scene = prepared.get_scene(use_bvh=True)
        old_v0 = old_scene.v0.copy()
        old_emitters = prepared.get_emitters(samples=4, rays=2, flip_faces=False)
        old_bounds = prepared.get_mesh_bounds()
        original = prepared.meshes[0][1].copy()
        self.assertEqual(prepared.version, 0)
        self.assertEqual(prepared.update_transform("moving", translation(x=2)), 1)
        self.assertEqual(prepared.update_transform(0, translation(x=3)), 2)
        np.testing.assert_array_equal(prepared.meshes[0][1], original + [3, 0, 0])
        current = prepared.get_scene(use_bvh=True)
        self.assertIs(current.left, old_scene.left)
        self.assertIs(current.permutation, old_scene.permutation)
        np.testing.assert_array_equal(old_scene.v0, old_v0)
        emitters = prepared.get_emitters(samples=4, rays=2, flip_faces=False)
        self.assertIs(emitters[1], old_emitters[1])
        self.assertIs(emitters[0].halton_u, old_emitters[0].halton_u)
        self.assertFalse(emitters[0].halton_u.flags.writeable)
        self.assertIsNot(prepared.get_mesh_bounds(), old_bounds)
        transform = translation(x=1, z=4)
        transform[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        prepared.update_transform("moving", transform)
        np.testing.assert_array_equal(
            prepared.meshes[0][1], original @ transform[:3, :3].T + transform[:3, 3])
        self.assert_matches_fresh(prepared)

    def test_refit_preserves_triangle_mapping_across_a_partitioned_tree(self):
        meshes = []
        for idx in range(12):
            name, vertices, faces = square(f"surface{idx}", z=(idx % 3) * 2)
            meshes.append((name, vertices + [((idx * 7) % 12) * 3, 0, 0], faces))
        prepared = PreparedSolver(meshes)
        before = prepared.get_scene(use_bvh=True)
        self.assertGreater(len(before.left), 1)
        self.assertFalse(np.array_equal(before.permutation, np.arange(len(before.v0))))
        prepared.update_transform("surface5", translation(x=-25, z=5))
        deformed = prepared.meshes[3][1].copy()
        deformed[2, 2] += 0.75
        prepared.update_vertices(3, deformed)
        after = prepared.get_scene(use_bvh=True)
        self.assertIs(after.left, before.left)
        self.assertIs(after.permutation, before.permutation)
        self.assert_matches_fresh(prepared)
        for idx, size in enumerate(after.count):
            if size:
                sl = slice(after.start[idx], after.start[idx] + size)
                vertices = np.concatenate([after.v0[sl], after.v0[sl] + after.e1[sl],
                                           after.v0[sl] + after.e2[sl]])
                np.testing.assert_array_equal(after.bb_min[idx], vertices.min(axis=0))
                np.testing.assert_array_equal(after.bb_max[idx], vertices.max(axis=0))

    def test_deformation_resets_transform_basis_and_topology_rebuilds(self):
        prepared = PreparedSolver([square("surface")])
        before = prepared.get_scene(use_bvh=True)
        deformed = prepared.meshes[0][1].copy()
        deformed[2, 2] = 1
        prepared.update_vertices("surface", deformed)
        prepared.update_transform("surface", translation(y=4))
        np.testing.assert_array_equal(prepared.meshes[0][1], deformed + [0, 4, 0])
        self.assertIs(prepared.get_scene(use_bvh=True).left, before.left)
        # A different index order with the same triangle count is a topology
        # change as well, not merely a deformation.
        vertices = prepared.meshes[0][1].copy()
        faces = np.asarray([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        prepared.update_mesh("surface", vertices, faces)
        self.assertIsNot(prepared.get_scene(use_bvh=True).left, before.left)
        self.assert_matches_fresh(prepared)
        prepared.update_mesh(0, vertices, faces[:1])
        self.assertEqual(prepared.total_faces, 1)
        self.assert_matches_fresh(prepared)

    def test_moving_blocker_changes_stationary_pair_and_matches_fresh_solve(self):
        for bvh in ("off", "builtin"):
            with self.subTest(bvh=bvh), patch("tests.v2_cases._log"):
                prepared = PreparedSolver([square("emitter"), square("blocker", 0.5, 3),
                                           square("receiver", 2, 1, True)])
                params = MatrixCase(samples=4, rays=16, min_iters=2, max_iters=2,
                                      seed=3, device="cpu", bvh=bvh, reciprocity=False)
                before = matrix_case(prepared.meshes, params, prepared=prepared)
                self.assertEqual(before["emitter"].get("receiver_front", 0), 0)
                prepared.update_transform("blocker", translation(x=10))
                after = matrix_case(prepared.meshes, params, prepared=prepared)
                fresh = matrix_case(prepared.meshes, params)
                self.assertEqual(after, fresh)
                self.assertGreater(after["emitter"].get("receiver_front", 0), 0)

    def test_invalid_updates_are_atomic(self):
        prepared = PreparedSolver([square("surface")])
        before = prepared.get_scene(use_bvh=True)
        invalid = [np.eye(3), np.eye(4) * np.nan]
        for kind in ("scale", "shear", "reflection", "projective"):
            transform = np.eye(4)
            if kind == "scale":
                transform[0, 0] = 2
            elif kind == "shear":
                transform[0, 1] = 0.1
            elif kind == "reflection":
                transform[0, 0] = -1
            else:
                transform[3, 0] = 1
            invalid.append(transform)
        for transform in invalid:
            with self.assertRaises(ValueError):
                prepared.update_transform("surface", transform)
        with self.assertRaises(KeyError):
            prepared.update_transform("missing", np.eye(4))
        with self.assertRaises(IndexError):
            prepared.update_vertices(-1, prepared.meshes[0][1])
        with self.assertRaisesRegex(ValueError, "finite"):
            prepared.update_vertices(0, np.full((4, 3), np.inf))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            prepared.update_mesh(0, prepared.meshes[0][1], np.asarray([[0, 1, 4]]))
        with self.assertRaisesRegex(ValueError, "integer"):
            prepared.update_mesh(0, prepared.meshes[0][1], np.asarray([[0., 1., 2.]]))
        self.assertEqual(prepared.version, 0)
        self.assertIs(prepared.get_scene(use_bvh=True), before)

    def test_explicit_rebuild_keeps_current_geometry_and_emitter_cache(self):
        prepared = PreparedSolver([square("moving"), square("fixed", 2)])
        original = prepared.get_scene(use_bvh=True)
        emitter = prepared.get_emitter(0, samples=4, rays=2, flip_faces=False)
        self.assertEqual(prepared.rebuild_bvh(), 1)
        rebuilt = prepared.get_scene(use_bvh=True)
        self.assertIsNot(rebuilt.left, original.left)
        np.testing.assert_array_equal(rebuilt.v0, original.v0)
        self.assertIs(prepared.get_emitter(0, samples=4, rays=2, flip_faces=False), emitter)
        with self.assertRaisesRegex(ValueError, "unique"):
            PreparedSolver([square("duplicate"), square("duplicate", 2)])

    def test_empty_bvh_and_mesh_updates(self):
        empty = np.empty((0, 3), dtype=np.float32)
        bmin, bmax, left, right, start, count, perm = build_bvh(empty, empty, empty)
        self.assertEqual(bmin.shape, (0, 3))
        self.assertEqual(refit_bvh(empty, empty, empty, left, right, start, count)[0].shape, (0, 3))
        prepared = PreparedSolver([("empty", empty, np.empty((0, 3), dtype=np.int32))])
        self.assertFalse(prepared.get_scene(use_bvh=True).use_bvh)
        prepared.update_vertices("empty", empty)
        prepared.update_mesh("empty", square("new")[1], square("new")[2])
        self.assertTrue(prepared.get_scene(use_bvh=True).use_bvh)
        self.assert_matches_fresh(prepared)

    def test_cuda_allocations_refresh_and_survive_topology_changes(self):
        code = textwrap.dedent("""
            from types import SimpleNamespace
            import numpy as np
            from numba import cuda
            from raystrack.utils.prepared import PreparedSolver
            cuda.get_current_device = lambda: SimpleNamespace(id=0)
            v = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], np.float32)
            f = np.array([[0,1,2], [1,3,2]], np.int32)
            prepared = PreparedSolver([('moving', v, f), ('fixed', v + [0,0,3], f)])
            for bvh in (False, True):
                initial = prepared.get_device_scene(use_bvh=bvh)
                old_emitter = prepared.get_device_emitter(0, samples=4, rays=2, flip_faces=False)
                fixed_emitter = prepared.get_device_emitter(1, samples=4, rays=2, flip_faces=False)
                transform = np.eye(4)
                transform[0,3] = 5 if bvh else 2
                prepared.update_transform('moving', transform)
                updated = prepared.get_device_scene(use_bvh=bvh)
                emitter = prepared.get_device_emitter(0, samples=4, rays=2, flip_faces=False)
                assert updated.v0 is initial.v0
                assert emitter.tri_a is old_emitter.tri_a
                assert emitter.halton_u is old_emitter.halton_u
                assert prepared.get_device_emitter(1, samples=4, rays=2, flip_faces=False) is fixed_emitter
                np.testing.assert_array_equal(updated.v0.copy_to_host(), prepared.get_scene(use_bvh=bvh).v0)
                np.testing.assert_array_equal(emitter.tri_a.copy_to_host(), prepared.get_emitter(0, samples=4, rays=2, flip_faces=False).tri_a)
                if bvh:
                    assert updated.left is initial.left
                    np.testing.assert_array_equal(updated.bb_min.copy_to_host(), prepared.get_scene(use_bvh=True).bb_min)
            previous = prepared.get_device_scene(use_bvh=True)
            vertices = prepared.meshes[0][1]
            prepared.update_mesh(0, vertices, np.array([[0,2,1], [1,2,3]], np.int32))
            same_size = prepared.get_device_scene(use_bvh=True)
            assert same_size.v0 is previous.v0
            np.testing.assert_array_equal(same_size.v0.copy_to_host(), prepared.get_scene(use_bvh=True).v0)
            prepared.update_mesh(0, vertices, np.array([[0,2,1]], np.int32))
            smaller = prepared.get_device_scene(use_bvh=True)
            assert smaller.v0 is not same_size.v0
            np.testing.assert_array_equal(smaller.sid.copy_to_host(), prepared.get_scene(use_bvh=True).sid)
            prepared._execution_workspace_cache = {'test': 1}
            prepared.clear_device_cache()
            assert not prepared._execution_workspace_cache
            assert prepared.get_device_scene(use_bvh=True).v0 is not smaller.v0
        """)
        env = os.environ.copy()
        env["NUMBA_ENABLE_CUDASIM"] = "1"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
