from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from tests.v2_cases import MatrixCase, SkyCase, matrix_case, outside_case
from raystrack.utils.prepared import PreparedSolver
from raystrack.utils.cpu_trace import trace_cpu_combined, trace_cpu_instanced_combined


def square(name="square", z=0.0, half=1.0, down=False):
    v = np.asarray([[-half, -half, z], [half, -half, z],
                    [half, half, z], [-half, half, z]], np.float32)
    f = np.asarray([[0, 1, 2], [0, 2, 3]], np.int32)
    return name, v, f[:, ::-1].copy() if down else f


def transform(x=0.0, y=0.0, z=0.0, angle=0.0):
    matrix = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def rays_for(prepared, n=2000):
    rng = np.random.default_rng(876)
    origin = rng.uniform(-7, 7, (n, 3)).astype(np.float32)
    directions = rng.normal(size=(n, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origin, directions


def trace_reference(prepared, origin, directions, active=None, emit=-1, minimum=0):
    scene = prepared.get_scene(use_bvh=False)
    active = np.ones(len(prepared.meshes), np.uint8) if active is None else active
    result = (np.empty(len(origin), np.int32), np.empty(len(origin), np.uint8), np.empty(len(origin), np.uint8))
    trace_cpu_combined(origin, directions, scene.v0, scene.e1, scene.e2,
                       scene.normals, scene.sid, active, emit, minimum, *result)
    return result


def trace_instanced(prepared, origin, directions, active=None, emit=-1, minimum=0):
    scene = prepared.get_instanced_scene()
    active = np.ones(len(scene.roots), np.uint8) if active is None else active
    result = (np.empty(len(origin), np.int32), np.empty(len(origin), np.uint8), np.empty(len(origin), np.uint8))
    trace_cpu_instanced_combined(origin, directions, scene, active, emit, minimum, *result)
    return result


class InstancingTests(unittest.TestCase):
    def test_vectorized_frames_exactly_match_scalar_axes_degenerate_and_tiny_cases(self):
        from raystrack.utils.prepared import _triangle_frames
        rng = np.random.default_rng(8)
        normals = rng.normal(size=(2048, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        boundary = np.float32(.9)
        extras = np.concatenate((np.eye(3, dtype=np.float32), -np.eye(3, dtype=np.float32),
                                 np.eye(3, dtype=np.float32)*1e-13, np.eye(3, dtype=np.float32)*1e-10,
                                 np.zeros((1,3), np.float32),
                                 np.array([[boundary,.2,.3], [-boundary,.2,.3],
                                           [np.nextafter(boundary,np.float32(1)),.2,.3]], np.float32)))
        normals = np.concatenate((normals, extras))
        expected_u, expected_v = np.empty_like(normals), np.empty_like(normals)
        axis_x, axis_y = np.array([1,0,0], np.float32), np.array([0,1,0], np.float32)
        for i, normal in enumerate(normals):
            ref = axis_x if abs(float(normal[0])) < .9 else axis_y
            u = np.cross(ref, normal).astype(np.float32)
            length = float(np.linalg.norm(u))
            if length <= 1e-12:
                ref = axis_y if ref is axis_x else axis_x
                u = np.cross(ref, normal).astype(np.float32)
                length = float(np.linalg.norm(u))
            if length <= 1e-12:
                expected_u[i], expected_v[i] = axis_x, axis_y
            else:
                u /= length
                expected_u[i], expected_v[i] = u, np.cross(normal,u).astype(np.float32)
        actual = _triangle_frames(normals)
        np.testing.assert_array_equal(actual[0], expected_u)
        np.testing.assert_array_equal(actual[1], expected_v)

    def test_transform_overflow_is_rejected_before_lazy_scene_build(self):
        vertices = np.full((3,3), 1e38, np.float32)
        prepared = PreparedSolver([("large", vertices, np.array([[0,1,2]], np.int32))],
                                  acceleration="instanced")
        invalid = transform(x=3e38)
        original = prepared._meshes[0][1]
        with self.assertRaisesRegex(ValueError, "float32"):
            prepared.update_transform(0, invalid)
        self.assertEqual(prepared.version, 0)
        self.assertIsNone(prepared._instancing)
        self.assertIs(prepared._meshes[0][1], original)
        prepared.get_instanced_scene()
        with self.assertRaisesRegex(ValueError, "float32"):
            prepared.update_transform(0, invalid)
        self.assertEqual(prepared.version, 0)

    def test_shared_prototypes_transform_without_rewriting_local_buffers(self):
        prototype = square()
        prepared = PreparedSolver.from_instances([prototype], [
            ("left", "square", transform(x=-3)), ("right", 0, transform(x=3))])
        before = prepared.get_instanced_scene()
        self.assertEqual(before.unique_geometries, 1)
        self.assertEqual(before.local_triangle_count, 2)
        self.assertEqual(prepared.total_faces, 4)
        self.assertIs(prepared._local_vertices[0], prepared._local_vertices[1])
        old_world = prepared._meshes[0][1]
        old_geom = before.blas.geom.copy()
        prepared.update_transform("left", transform(x=-4, z=2, angle=.4))
        after = prepared.get_instanced_scene()
        self.assertIs(before.blas, after.blas)
        self.assertIs(before.blas.geom, after.blas.geom)
        self.assertIs(before.blas.bb_min, after.blas.bb_min)
        self.assertIs(prepared._meshes[0][1], old_world)  # World mesh is lazy.
        np.testing.assert_array_equal(after.blas.geom, old_geom)
        world = prepared.meshes[0][1]
        expected = prototype[1] @ transform(angle=.4)[:3, :3].T + [-4, 0, 2]
        np.testing.assert_allclose(world, expected, rtol=1e-6, atol=1e-6)
        self.assertFalse(after.blas.geom.flags.writeable)
        self.assertNotEqual(before.revision, after.revision)

    def test_rotated_nonunit_rays_front_back_and_filters_match_flat_geometry(self):
        prepared = PreparedSolver.from_instances([square()], [
            (f"instance{i}", 0, transform(x=(i % 3 - 1)*3, y=(i // 3 - 1)*3,
                                         z=i % 2, angle=.21*i)) for i in range(9)])
        origin, directions = rays_for(prepared, 5000)
        directions *= 3  # Ray parameter must survive rigid inverse transforms.
        active = np.ones(9, np.uint8)
        active[3] = 0
        for emit, minimum in ((-1, 0), (2, 4)):
            actual = trace_instanced(prepared, origin, directions, active, emit, minimum)
            expected = trace_reference(prepared, origin, directions, active, emit, minimum)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_deformation_detaches_shared_geometry_and_topology_refreshes(self):
        prepared = PreparedSolver.from_instances([square()], [
            ("first", 0, transform()), ("second", 0, transform(z=3))])
        old = prepared.get_instanced_scene()
        original_second = prepared._instancing._instance_geometries[1]
        vertices = prepared.meshes[0][1].copy()
        vertices[2, 2] = .75
        prepared.update_vertices("first", vertices)
        scene = prepared.get_instanced_scene()
        self.assertEqual(scene.unique_geometries, 2)
        self.assertEqual(scene.local_triangle_count, 4)
        self.assertIs(prepared._instancing._instance_geometries[1], original_second)
        self.assertIsNot(scene.blas, old.blas)
        origin, directions = rays_for(prepared)
        for a, b in zip(trace_instanced(prepared, origin, directions), trace_reference(prepared, origin, directions)):
            np.testing.assert_array_equal(a, b)
        prepared.update_mesh("first", vertices, np.asarray([[0, 3, 1]], np.int32))
        self.assertEqual(prepared.total_faces, 3)
        for a, b in zip(trace_instanced(prepared, origin, directions), trace_reference(prepared, origin, directions)):
            np.testing.assert_array_equal(a, b)

    def test_tlas_quality_rebuild_and_explicit_rebuild_preserve_blas(self):
        prepared = PreparedSolver.from_instances([square()], [
            (f"i{i}", 0, transform(x=i*4)) for i in range(32)])
        original = prepared.get_instanced_scene()
        prepared.update_transform("i0", transform(x=10000))
        rebuilt = prepared.get_instanced_scene()
        self.assertGreater(rebuilt.tlas_rebuilds, original.tlas_rebuilds)
        self.assertIs(rebuilt.blas, original.blas)
        prepared.rebuild_bvh()
        self.assertGreater(prepared.get_instanced_scene().tlas_rebuilds, rebuilt.tlas_rebuilds)
        self.assertIs(prepared.get_instanced_scene().blas, original.blas)

    def test_empty_instances_and_invalid_rigid_updates(self):
        empty = ("empty", np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))
        prepared = PreparedSolver([empty], acceleration="instanced")
        result = trace_instanced(prepared, np.zeros((1, 3), np.float32), np.ones((1, 3), np.float32))
        self.assertEqual(result[0][0], -1)
        self.assertEqual(result[2][0], 0)
        prepared.update_mesh("empty", square()[1], square()[2])
        self.assertEqual(prepared.get_instanced_scene().local_triangle_count, 2)
        before = prepared.get_instanced_scene()
        version = prepared.version
        invalid = np.eye(4)
        invalid[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "rigid"):
            prepared.update_transform(0, invalid)
        self.assertEqual(prepared.version, version)
        self.assertIs(prepared.get_instanced_scene(), before)
        with self.assertRaisesRegex(ValueError, "acceleration"):
            PreparedSolver([square()], acceleration="unknown")

    def test_moving_blocker_stationary_rows_and_sky_match_fresh_reference(self):
        prepared = PreparedSolver([square("emitter"), square("blocker", .5, 3),
                                   square("receiver", 2, 1, True)], acceleration="instanced")
        params = MatrixCase(samples=2, rays=8, min_iters=2, max_iters=2,
                              seed=3, device="cpu", bvh="builtin", reciprocity=False)
        sky = SkyCase(samples=2, rays=8, min_iters=2, max_iters=2,
                        seed=3, device="cpu", bvh="builtin")
        with patch("tests.v2_cases._log"):
            blocked = matrix_case(prepared.meshes, params, prepared=prepared)
            self.assertEqual(blocked["emitter"].get("receiver_front", 0), 0)
            prepared.update_transform("blocker", transform(x=10))
            actual = outside_case(prepared.meshes, matrix_params=params,
                                                  sky_params=sky, prepared=prepared)
            expected = outside_case(prepared.meshes, matrix_params=params, sky_params=sky)
        self.assertEqual(actual, expected)
        self.assertGreater(actual[0]["emitter"].get("receiver_front", 0), 0)
        emitters = prepared.get_emitters(samples=2, rays=8, flip_faces=False)
        self.assertEqual([e.total_area for e in emitters], [4., 36., 4.])

    def test_cuda_sim_two_level_traversal_and_blas_allocation_reuse(self):
        code = textwrap.dedent("""
            from types import SimpleNamespace
            import numpy as np
            from numba import cuda
            from raystrack.utils.prepared import PreparedSolver
            from raystrack.utils.cuda_trace import kernel_trace_instanced_combined
            cuda.get_current_device = lambda: SimpleNamespace(id=0)
            v = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
            f = np.array([[0,1,2], [0,2,3]], np.int32)
            matrices = [np.eye(4), np.eye(4)]
            matrices[0][2,3], matrices[1][2,3] = 1, 2
            p = PreparedSolver.from_instances([('prototype',v,f)], [('near',0,matrices[0]),('far',0,matrices[1])])
            o = cuda.to_device(np.array([[0,0,0],[0,0,3],[4,0,0]], np.float32))
            d = cuda.to_device(np.array([[0,0,1],[0,0,-2],[0,0,1]], np.float32))
            active = cuda.to_device(np.ones(2,np.uint8))
            h, fr, mask = cuda.device_array(3,np.int32), cuda.device_array(3,np.uint8), cuda.device_array(3,np.uint8)
            before = p.get_device_instanced_scene()
            kernel_trace_instanced_combined[1,32](o,d,*before,active,-1,0,h,fr,mask)
            np.testing.assert_array_equal(h.copy_to_host(),[0,1,-1])
            np.testing.assert_array_equal(fr.copy_to_host(),[0,1,0])
            np.testing.assert_array_equal(mask.copy_to_host(),[1,1,0])
            moved = matrices[0].copy(); moved[0,3] = 10
            p.update_transform('near',moved)
            after = p.get_device_instanced_scene()
            assert after[0] is before[0] and after[1] is before[1]
            kernel_trace_instanced_combined[1,32](o,d,*after,active,-1,0,h,fr,mask)
            np.testing.assert_array_equal(h.copy_to_host(),[1,1,-1])
            p.update_mesh('near',v,np.array([[0,1,2]],np.int32))
            changed = p.get_device_instanced_scene()
            assert len(changed[0]) == 3
            p.clear_device_cache()
            assert p.get_device_instanced_scene()[0] is not changed[0]
        """)
        env = os.environ.copy()
        env["NUMBA_ENABLE_CUDASIM"] = "1"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("taichi"), "optional Taichi dependency absent")
    def test_portable_instancing_matches_cpu_and_reuses_geometry_upload(self):
        from raystrack.utils.taichi_trace import get_taichi_backend, TaichiUnavailableError
        from raystrack.utils.cpu_trace import reduce_first_hits, bin_tregenza_cpu
        try:
            gpu = get_taichi_backend()
        except TaichiUnavailableError as exc:
            self.skipTest(str(exc))
        prepared = PreparedSolver.from_instances([square()], [
            (f"i{i}", 0, transform(x=(i%3-1)*3, z=i//3, angle=.17*i)) for i in range(9)])
        origins, directions = rays_for(prepared, 1000)
        active = np.ones(9, np.uint8)
        active[2] = 0
        for step in range(2):
            hit, front, mask = trace_reference(prepared, origins, directions, active)
            hf, hb, sky = np.zeros(9, np.int64), np.zeros(9, np.int64), np.zeros(145, np.int64)
            reduce_first_hits(hit, front, hf, hb)
            bin_tregenza_cpu(directions, mask, sky)
            actual = gpu.trace_rays(prepared.get_instanced_scene(), origins, directions,
                                    surf_active=active, include_sky=True, discrete=True)
            for a, b in zip(actual, (hf, hb, sky)):
                np.testing.assert_array_equal(a, b)
            if step == 0:
                geometry_buffer = gpu._buffers["inst_geom"]
                prepared.update_transform("i0", transform(x=10, z=2, angle=.6))
            else:
                self.assertIs(gpu._buffers["inst_geom"], geometry_buffer)
        emitter = prepared.get_emitter(0, samples=2, rays=8, flip_faces=False)
        controls = dict(rays=8, surf_active=active, emit_sid=0, include_sky=True,
                        discrete=True, ray_count=31, ray_offset=5)
        grids = np.array([[.1,.2], [.4,.5]], np.float32)
        dims = np.array([[.11,.22,.33,.44,.55], [.16,.27,.38,.49,.61]], np.float32)
        singles = [gpu.trace_batch(prepared.get_instanced_scene(), emitter,
                                   cp_grid=g, cp_dims=d, **controls) for g, d in zip(grids,dims)]
        grouped = gpu.trace_iterations(prepared.get_instanced_scene(), emitter,
                                       cp_grids=grids, cp_dims=dims, **controls)
        for i, values in enumerate(grouped):
            np.testing.assert_array_equal(values, np.stack([x[i] for x in singles]))


if __name__ == "__main__":
    unittest.main()
