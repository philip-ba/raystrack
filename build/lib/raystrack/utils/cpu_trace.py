from __future__ import annotations
import numpy as np
import numba as nb

@nb.njit(inline="always", cache=True, fastmath=True)
def _cross_cpu(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1],
                     a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]], np.float32)

@nb.njit(inline="always", cache=True, fastmath=True)
def _dot_cpu(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@nb.njit(fastmath=True, cache=True, parallel=True)
def trace_cpu(orig, dirs, v0, e1, e2, norm, sid,
               out_front, out_back):
    n_rays = orig.shape[0]
    for k in nb.prange(n_rays):
        o = orig[k]; d = dirs[k]
        best = 1e20; hit = -1; front = 1
        for i in range(v0.shape[0]):
            p = _cross_cpu(d, e2[i]); det = _dot_cpu(e1[i], p)
            if abs(det) < 1e-7:
                continue
            inv = 1.0 / det
            tvec = o - v0[i]
            u = _dot_cpu(tvec, p) * inv
            if u < 0 or u > 1:
                continue
            q = _cross_cpu(tvec, e1[i])
            v = _dot_cpu(d, q) * inv
            if v < 0 or u + v > 1:
                continue
            t = _dot_cpu(e2[i], q) * inv
            if 1e-6 < t < best:
                best = t; hit = sid[i]
                front = 1 if -_dot_cpu(d, norm[i]) > 0 else 0
        if hit >= 0:
            if front:
                out_front[hit] += 1
            else:
                out_back[hit] += 1

@nb.njit(cache=True, fastmath=True, inline="always")
def _aabb_hit(o: np.ndarray, invd: np.ndarray,
              bmin: np.ndarray, bmax: np.ndarray) -> bool:
    tmin = (bmin[0] - o[0]) * invd[0]
    tmax = (bmax[0] - o[0]) * invd[0]
    if tmin > tmax:
        tmin, tmax = tmax, tmin
    tymin = (bmin[1] - o[1]) * invd[1]
    tymax = (bmax[1] - o[1]) * invd[1]
    if tymin > tymax:
        tymin, tymax = tymax, tymin
    if (tmin > tymax) or (tymin > tmax):
        return False
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax
    tzmin = (bmin[2] - o[2]) * invd[2]
    tzmax = (bmax[2] - o[2]) * invd[2]
    if tzmin > tzmax:
        tzmin, tzmax = tzmax, tzmin
    if (tmin > tzmax) or (tzmin > tmax):
        return False
    return tzmax >= max(tmin, 0.0)

@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_bvh(orig, dirs,
                   v0, e1, e2, norm, sid,
                   bb_min, bb_max, left, right, start, count,
                   out_front, out_back):
    n_rays = orig.shape[0]
    for k in nb.prange(n_rays):
        o = orig[k]
        d = dirs[k]
        invd = np.empty(3, np.float32)
        for i in range(3):
            invd[i] = 1.0 / d[i] if abs(d[i]) > 1e-9 else 1e10
        stack = [0]
        best = 1e20
        hit = -1
        front = 1
        while stack:
            node = stack.pop()
            if not _aabb_hit(o, invd, bb_min[node], bb_max[node]):
                continue
            if count[node] > 0:
                for t in range(count[node]):
                    tri = start[node] + t
                    p = _cross_cpu(d, e2[tri])
                    det = _dot_cpu(e1[tri], p)
                    if abs(det) < 1e-7:
                        continue
                    inv_det = 1.0 / det
                    tvec = o - v0[tri]
                    u = _dot_cpu(tvec, p) * inv_det
                    if u < 0.0 or u > 1.0:
                        continue
                    qvec = _cross_cpu(tvec, e1[tri])
                    v = _dot_cpu(d, qvec) * inv_det
                    if v < 0.0 or u + v > 1.0:
                        continue
                    t_param = _dot_cpu(e2[tri], qvec) * inv_det
                    if 1e-6 < t_param < best:
                        best = t_param
                        hit = sid[tri]
                        front = 1 if -_dot_cpu(d, norm[tri]) > 0 else 0
            else:
                stack.append(left[node])
                stack.append(right[node])
        if hit >= 0:
            if front:
                out_front[hit] += 1
            else:
                out_back[hit] += 1

__all__ = ["trace_cpu", "trace_cpu_bvh", "_aabb_hit"]
