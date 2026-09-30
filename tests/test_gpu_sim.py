from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest


class CudaSimulatorTests(unittest.TestCase):
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
