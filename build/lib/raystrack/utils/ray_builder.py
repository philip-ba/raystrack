from __future__ import annotations
import math
import numpy as np
import numba as nb


@nb.njit(inline="always", cache=True)
def _halton_1d(i: int, base: int) -> float:
    f = 1.0
    r = 0.0
    while i:
        f /= base
        r += f * (i % base)
        i //= base
    return r


@nb.njit(parallel=True, cache=True, fastmath=True)
def build_rays(u_grid, v_grid,
               cdf, tri_a, tri_b, tri_c,
               samples, rays, orig, dire,
               cp_grid, cp_dims):
    """Create arrays of quasi Monte Carlo rays with randomized low-discrepancy.

    Parameters
    ----------
    u_grid, v_grid : float32[...]
        2D Halton jitter grid (kept for tiling).
    cdf, tri_a, tri_b, tri_c : arrays
        Triangle selection data and vertices.
    samples : int
        Grid resolution per side (g). Total cells = g*g.
    rays : int
        Rays per cell.
    orig, dire : outputs
        Ray origins and directions.
    cp_grid : float32[2]
        Cranley-Patterson shifts for (u,v) grid; kept for decorrelation.
    cp_dims : float32[5]
        CP shifts for dims: [tri, u_r, v_r, r1, r2].
    """
    g = samples
    n_cells = g * g
    two_pi = 6.283185307179586

    # Bases for low-discrepancy per-ray dimensions
    b_tri = 5
    b_u = 2
    b_v = 3
    b_r1 = 7
    b_r2 = 11

    for cell in nb.prange(n_cells):
        ug = (u_grid[cell] + cp_grid[0]) % 1.0
        vg = (v_grid[cell] + cp_grid[1]) % 1.0
        for r in range(rays):
            idx = cell * rays + r
            qidx = idx + 1  # Halton indices are 1-based

            # Triangle selection (1D)
            q_tri = (_halton_1d(qidx, b_tri) + cp_dims[0]) % 1.0
            tri = np.searchsorted(cdf, q_tri)

            # Per-ray surface sample in triangle (2D)
            ur = (_halton_1d(qidx, b_u) + cp_dims[1] + ug) % 1.0
            vr = (_halton_1d(qidx, b_v) + cp_dims[2] + vg) % 1.0

            a = tri_a[tri]
            b = tri_b[tri]
            c = tri_c[tri]

            s = math.sqrt(ur)
            p = (1.0 - s) * a + s * (vr * b + (1.0 - vr) * c)

            # Local frame from triangle normal
            n0 = np.cross(b - a, c - a)
            nlen = math.sqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2])
            n0 /= nlen

            r1 = (_halton_1d(qidx, b_r1) + cp_dims[3]) % 1.0
            r2 = (_halton_1d(qidx, b_r2) + cp_dims[4]) % 1.0

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
