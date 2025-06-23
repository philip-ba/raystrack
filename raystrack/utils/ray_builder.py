from __future__ import annotations
import math
import numpy as np
import numba as nb

@nb.njit(parallel=True, cache=True, fastmath=True)
def build_rays(u_grid, v_grid,
               cdf, tri_a, tri_b, tri_c,
               rng_uni,
               samples, rays, orig, dire):
    """Create arrays of Monte Carlo rays for every emitter grid-cell."""
    g = samples
    n_cells = g * g
    two_pi = 6.283185307179586
    for cell in nb.prange(n_cells):
        u = u_grid[cell]
        v = v_grid[cell]
        tri = np.searchsorted(cdf, rng_uni[cell])
        a = tri_a[tri]; b = tri_b[tri]; c = tri_c[tri]

        s = math.sqrt(u)
        p = (1.0 - s) * a + s * (v * b + (1.0 - v) * c)

        n0 = np.cross(b - a, c - a)
        nlen = math.sqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2])
        n0 /= nlen

        for r in range(rays):
            idx = cell * rays + r
            r1 = rng_uni[n_cells + idx]
            r2 = rng_uni[n_cells + idx + orig.shape[0]]

            sin_t = math.sqrt(1.0 - r1)
            phi = two_pi * r2
            x = sin_t * math.cos(phi)
            y = sin_t * math.sin(phi)
            z = math.sqrt(r1)

            uvec = np.empty(3, np.float32)
            if abs(n0[0]) < 0.9:
                uvec[0], uvec[1], uvec[2] = 1.0, 0.0, 0.0
            else:
                uvec[0], uvec[1], uvec[2] = 0.0, 1.0, 0.0
            cx = uvec[1] * n0[2] - uvec[2] * n0[1]
            cy = uvec[2] * n0[0] - uvec[0] * n0[2]
            cz = uvec[0] * n0[1] - uvec[1] * n0[0]
            clen = math.sqrt(cx * cx + cy * cy + cz * cz)
            uvec[0], uvec[1], uvec[2] = cx / clen, cy / clen, cz / clen

            vvec = np.array([n0[1] * uvec[2] - n0[2] * uvec[1],
                             n0[2] * uvec[0] - n0[0] * uvec[2],
                             n0[0] * uvec[1] - n0[1] * uvec[0]],
                            np.float32)

            d = x * uvec + y * vvec + z * n0
            orig[idx] = p
            dire[idx] = d

__all__ = ["build_rays"]
