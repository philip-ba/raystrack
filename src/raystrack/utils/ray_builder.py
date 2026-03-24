from __future__ import annotations

import math

import numba as nb


@nb.njit(inline="always", cache=True)
def _binary_search_cdf(cdf, x):
    lo = 0
    hi = cdf.shape[0] - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if cdf[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo < 0:
        return 0
    if lo >= cdf.shape[0]:
        return cdf.shape[0] - 1
    return lo


@nb.njit(parallel=True, cache=True, fastmath=True)
def build_rays(
    u_grid,
    v_grid,
    halton_tri,
    halton_u,
    halton_v,
    halton_r1,
    halton_r2,
    cdf,
    tri_a,
    tri_e1,
    tri_e2,
    tri_u,
    tri_v,
    tri_n,
    tri_origin_eps,
    rays_per_cell,
    orig,
    dire,
    cp_grid,
    cp_dims,
):
    """Create ray origins and directions from precomputed emitter data."""
    n_rays = orig.shape[0]
    two_pi = 6.283185307179586

    for idx in nb.prange(n_rays):
        cell = idx // rays_per_cell
        ug = (u_grid[cell] + cp_grid[0]) % 1.0
        vg = (v_grid[cell] + cp_grid[1]) % 1.0

        q_tri = (halton_tri[idx] + cp_dims[0]) % 1.0
        tri = _binary_search_cdf(cdf, q_tri)

        ur = (halton_u[idx] + cp_dims[1] + ug) % 1.0
        vr = (halton_v[idx] + cp_dims[2] + vg) % 1.0

        s = math.sqrt(ur)
        mix_b = s * vr
        mix_c = s * (1.0 - vr)

        ax = tri_a[tri, 0]
        ay = tri_a[tri, 1]
        az = tri_a[tri, 2]

        px = ax + mix_b * tri_e1[tri, 0] + mix_c * tri_e2[tri, 0]
        py = ay + mix_b * tri_e1[tri, 1] + mix_c * tri_e2[tri, 1]
        pz = az + mix_b * tri_e1[tri, 2] + mix_c * tri_e2[tri, 2]

        r1 = (halton_r1[idx] + cp_dims[3]) % 1.0
        r2 = (halton_r2[idx] + cp_dims[4]) % 1.0

        sin_t = math.sqrt(1.0 - r1)
        phi = two_pi * r2
        x = sin_t * math.cos(phi)
        y = sin_t * math.sin(phi)
        z = math.sqrt(r1)

        dx = x * tri_u[tri, 0] + y * tri_v[tri, 0] + z * tri_n[tri, 0]
        dy = x * tri_u[tri, 1] + y * tri_v[tri, 1] + z * tri_n[tri, 1]
        dz = x * tri_u[tri, 2] + y * tri_v[tri, 2] + z * tri_n[tri, 2]
        eps = tri_origin_eps[tri]

        orig[idx, 0] = px + eps * tri_n[tri, 0]
        orig[idx, 1] = py + eps * tri_n[tri, 1]
        orig[idx, 2] = pz + eps * tri_n[tri, 2]
        dire[idx, 0] = dx
        dire[idx, 1] = dy
        dire[idx, 2] = dz


__all__ = ["build_rays"]
