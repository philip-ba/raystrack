from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest


class CudaSimulatorTests(unittest.TestCase):
    def test_deferred_summaries_and_global_chunk_budget(self):
        code = textwrap.dedent("""
            from types import SimpleNamespace
            from dataclasses import replace
            import numpy as np
            from numba import cuda
            from raystrack import MatrixParams, SkyParams, PreparedSolver, view_factor_outside_workflow
            cuda.get_current_device = lambda: SimpleNamespace(id=0, MAX_THREADS_PER_BLOCK=256)
            v = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], np.float32)
            f = np.array([[0,1,2], [0,2,3]], np.int32)
            meshes = [('lower',v,f), ('upper',v+[0,0,1],f[:,::-1].copy())]
            def solve(device, budget=None, callback=None, prepared=None, async_=True, raygen=True):
                common = dict(samples=1, rays=1, min_iters=3, max_iters=3,
                              convergence_interval=3, bvh='builtin', device=device,
                              max_total_rays=budget, ray_batch_size=7,
                              cuda_async=async_, gpu_raygen=raygen)
                return view_factor_outside_workflow(meshes if prepared is None else prepared.meshes,
                    matrix_params=MatrixParams(**common, reciprocity_mode='bidirectional'),
                    sky_params=SkyParams(**common, discrete=True),
                    prepared=prepared, progress=callback)
            expected = solve('cpu')
            prepared = PreparedSolver(meshes)
            assert solve('cuda', prepared=prepared) == expected
            ws = list(prepared._execution_workspace_cache.values())[0]
            assert ws.summary_capacity == 3
            assert solve('cuda', prepared=prepared) == expected
            assert list(prepared._execution_workspace_cache.values())[0] is ws
            for async_, raygen in ((True, True), (False, False), (True, False)):
                counts=[]
                actual=solve('cuda', 43, counts.append, async_=async_, raygen=raygen)
                reference=solve('cpu', 43, lambda n: None)
                assert actual == reference, (actual, reference)
                assert sum(counts)==43 and max(counts)<=7
        """)
        env = os.environ.copy()
        env["NUMBA_ENABLE_CUDASIM"] = "1"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_buffer_reuse_with_unequal_emitter_sizes(self):
        # CUDASIM must be enabled before importing numba, so use a subprocess.
        code = textwrap.dedent("""
            from types import SimpleNamespace
            import numpy as np
            from numba import cuda
            from raystrack import MatrixParams, SkyParams, view_factor_outside_workflow

            cuda.get_current_device = lambda: SimpleNamespace(id=0, MAX_THREADS_PER_BLOCK=256)
            def square(name, z, h, down=False):
                v = np.array([[-h,-h,z], [h,-h,z], [h,h,z], [-h,h,z]], np.float32)
                f = np.array([[0,1,2], [0,2,3]], np.int32)
                return name, v, f[:,::-1].copy() if down else f

            meshes = [square('small', 0, 1), square('large', 1, 2, True)]
            def solve(device, async_, raygen):
                common = dict(samples=2, rays=1, seed=3, device=device,
                              bvh='off', cuda_async=async_, gpu_raygen=raygen,
                              min_iters=1, max_iters=1)
                return view_factor_outside_workflow(
                    meshes,
                    matrix_params=MatrixParams(**common, reciprocity_mode='bidirectional'),
                    sky_params=SkyParams(**common),
                )

            expected = solve('cpu', False, False)
            assert solve('gpu', True, True) == expected
            assert solve('gpu', False, False) == expected
        """)
        env = os.environ.copy()
        env["NUMBA_ENABLE_CUDASIM"] = "1"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
