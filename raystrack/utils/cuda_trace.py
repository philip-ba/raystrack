from __future__ import annotations
import numpy as np
import numba as nb
from numba import cuda
import math

@cuda.jit(device=True, inline=True)
def _cross(a, b, out):
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]

@cuda.jit(device=True, inline=True)
def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cuda.jit
def kernel_trace(orig, dirs, v0, e1, e2, norm, sid, hits_f, hits_b):
    k = cuda.grid(1)
    if k >= orig.shape[0]:
        return
    o0,o1,o2 = orig[k,0], orig[k,1], orig[k,2]
    d0,d1,d2 = dirs[k,0], dirs[k,1], dirs[k,2]
    best=1e20; hit=-1; front=1
    p=cuda.local.array(3, np.float32)
    q=cuda.local.array(3, np.float32)
    for i in range(v0.shape[0]):
        _cross((d0,d1,d2), (e2[i,0],e2[i,1],e2[i,2]), p)
        det=_dot((e1[i,0],e1[i,1],e1[i,2]), p)
        if abs(det)<1e-7:
            continue
        inv=1.0/det
        t0=o0-v0[i,0]; t1=o1-v0[i,1]; t2=o2-v0[i,2]
        u=(t0*p[0]+t1*p[1]+t2*p[2])*inv
        if u<0 or u>1:
            continue
        _cross((t0,t1,t2), (e1[i,0],e1[i,1],e1[i,2]), q)
        v=(d0*q[0]+d1*q[1]+d2*q[2])*inv
        if v<0 or u+v>1:
            continue
        t=(e2[i,0]*q[0]+e2[i,1]*q[1]+e2[i,2]*q[2])*inv
        if 1e-6<t<best:
            best=t; hit=sid[i]
            front = 1 if -(d0*norm[i,0]+d1*norm[i,1]+d2*norm[i,2])>0 else 0
    if hit>=0:
        if front:
            cuda.atomic.add(hits_f, hit, 1)
        else:
            cuda.atomic.add(hits_b, hit, 1)

# BVH helpers
@cuda.jit(device=True, inline=True)
def _aabb_hit_dev(o0,o1,o2, inv0,inv1,inv2,
                  bmin0,bmin1,bmin2, bmax0,bmax1,bmax2):
    tmin=(bmin0-o0)*inv0; tmax=(bmax0-o0)*inv0
    if tmin>tmax:
        tmp=tmin; tmin=tmax; tmax=tmp
    tymin=(bmin1-o1)*inv1; tymax=(bmax1-o1)*inv1
    if tymin>tymax:
        tmp=tymin; tymin=tymax; tymax=tmp
    if (tmin>tymax) or (tymin>tmax):
        return False
    if tymin>tmin: tmin=tymin
    if tymax<tmax: tmax=tymax
    tzmin=(bmin2-o2)*inv2; tzmax=(bmax2-o2)*inv2
    if tzmin>tzmax:
        tmp=tzmin; tzmin=tzmax; tzmax=tmp
    if (tmin>tzmax) or (tzmin>tmax):
        return False
    return tzmax>=tmin

@cuda.jit
def kernel_trace_bvh(orig, dirs, v0, e1, e2, norm, sid,
                     bb_min, bb_max, left, right, start, cnt,
                     hits_f, hits_b):
    k = cuda.grid(1)
    if k >= orig.shape[0]:
        return
    o0,o1,o2 = orig[k,0], orig[k,1], orig[k,2]
    d0,d1,d2 = dirs[k,0], dirs[k,1], dirs[k,2]
    inv0 = 1.0/d0 if abs(d0)>1e-9 else 1e10
    inv1 = 1.0/d1 if abs(d1)>1e-9 else 1e10
    inv2 = 1.0/d2 if abs(d2)>1e-9 else 1e10

    stack = cuda.local.array(64, nb.int32)
    sp = 0
    stack[sp] = 0
    sp += 1

    best = 1e20
    hit  = -1
    front = 1

    while sp>0:
        sp -= 1
        node = stack[sp]
        if not _aabb_hit_dev(o0,o1,o2, inv0,inv1,inv2,
                              bb_min[node,0],bb_min[node,1],bb_min[node,2],
                              bb_max[node,0],bb_max[node,1],bb_max[node,2]):
            continue
        if cnt[node] > 0:
            for t in range(cnt[node]):
                tri = start[node] + t
                px = d1*e2[tri,2]-d2*e2[tri,1]
                py = d2*e2[tri,0]-d0*e2[tri,2]
                pz = d0*e2[tri,1]-d1*e2[tri,0]
                det = (e1[tri,0]*px + e1[tri,1]*py + e1[tri,2]*pz)
                if abs(det) < 1e-7:
                    continue
                inv_det = 1.0/det
                tx = o0 - v0[tri,0]
                ty = o1 - v0[tri,1]
                tz = o2 - v0[tri,2]
                u = (tx*px + ty*py + tz*pz)*inv_det
                if u<0 or u>1:
                    continue
                qx = ty*e1[tri,2]-tz*e1[tri,1]
                qy = tz*e1[tri,0]-tx*e1[tri,2]
                qz = tx*e1[tri,1]-ty*e1[tri,0]
                v = (d0*qx + d1*qy + d2*qz)*inv_det
                if v<0 or u+v>1:
                    continue
                tparam = (e2[tri,0]*qx + e2[tri,1]*qy + e2[tri,2]*qz)*inv_det
                if 1e-6 < tparam < best:
                    best = tparam
                    hit  = sid[tri]
                    front = 1 if -(d0*norm[tri,0]+d1*norm[tri,1]+d2*norm[tri,2]) > 0 else 0
        else:
            if sp+2 >= 64:
                continue
            stack[sp] = left[node]
            stack[sp+1] = right[node]
            sp += 2
    if hit >= 0:
        if front:
            cuda.atomic.add(hits_f, hit, 1)
        else:
            cuda.atomic.add(hits_b, hit, 1)

__all__ = ["kernel_trace", "kernel_trace_bvh"]

# Utility kernels
@cuda.jit
def kernel_zero_i64(a):
    i = cuda.grid(1)
    if i < a.size:
        a[i] = 0

@cuda.jit
def kernel_zero_i32(a):
    i = cuda.grid(1)
    if i < a.size:
        a[i] = 0

__all__ += ["kernel_zero_i64", "kernel_zero_i32"]

# ---------------------------------------------------------------
# On-GPU ray generation (optional)
# ---------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _halton_1d_dev(i, base):
    f = 1.0
    r = 0.0
    while i > 0:
        f = f / base
        r = r + f * (i % base)
        i = i // base
    return r

@cuda.jit(device=True, inline=True)
def _binary_search_cdf(cdf, x):
    lo = 0
    hi = cdf.shape[0] - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        v = cdf[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo < 0:
        lo = 0
    if lo >= cdf.shape[0]:
        lo = cdf.shape[0] - 1
    return lo

@cuda.jit
def kernel_build_rays(u_grid, v_grid,
                      cdf, tri_a, tri_b, tri_c,
                      g, rays_per_cell,
                      orig, dire,
                      cp_grid, cp_dims):
    k = cuda.grid(1)
    n_cells = g * g
    n_rays = n_cells * rays_per_cell
    if k >= n_rays:
        return

    cell = k // rays_per_cell
    ug = (u_grid[cell] + cp_grid[0]) % 1.0
    vg = (v_grid[cell] + cp_grid[1]) % 1.0
    qidx = k + 1  # 1-based

    # Bases for low-discrepancy dimensions
    b_tri = 5
    b_u = 2
    b_v = 3
    b_r1 = 7
    b_r2 = 11

    q_tri = (_halton_1d_dev(qidx, b_tri) + cp_dims[0]) % 1.0
    tri = _binary_search_cdf(cdf, q_tri)

    ur = (_halton_1d_dev(qidx, b_u) + cp_dims[1] + ug) % 1.0
    vr = (_halton_1d_dev(qidx, b_v) + cp_dims[2] + vg) % 1.0

    ax = tri_a[tri, 0]; ay = tri_a[tri, 1]; az = tri_a[tri, 2]
    bx = tri_b[tri, 0]; by = tri_b[tri, 1]; bz = tri_b[tri, 2]
    cx = tri_c[tri, 0]; cy = tri_c[tri, 1]; cz = tri_c[tri, 2]

    s = math.sqrt(ur)
    px = (1.0 - s) * ax + s * (vr * bx + (1.0 - vr) * cx)
    py = (1.0 - s) * ay + s * (vr * by + (1.0 - vr) * cy)
    pz = (1.0 - s) * az + s * (vr * bz + (1.0 - vr) * cz)

    # Normal and local frame
    n0x = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
    n0y = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
    n0z = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    nlen = math.sqrt(n0x * n0x + n0y * n0y + n0z * n0z) + 1e-12
    n0x /= nlen; n0y /= nlen; n0z /= nlen

    if abs(n0x) < 0.9:
        ux, uy, uz = 1.0, 0.0, 0.0
    else:
        ux, uy, uz = 0.0, 1.0, 0.0
    cx0 = uy * n0z - uz * n0y
    cy0 = uz * n0x - ux * n0z
    cz0 = ux * n0y - uy * n0x
    clen = math.sqrt(cx0 * cx0 + cy0 * cy0 + cz0 * cz0) + 1e-12
    ux = cx0 / clen; uy = cy0 / clen; uz = cz0 / clen

    vx = n0y * uz - n0z * uy
    vy = n0z * ux - n0x * uz
    vz = n0x * uy - n0y * ux

    r1 = (_halton_1d_dev(qidx, b_r1) + cp_dims[3]) % 1.0
    r2 = (_halton_1d_dev(qidx, b_r2) + cp_dims[4]) % 1.0
    sin_t = math.sqrt(1.0 - r1)
    phi = 6.283185307179586 * r2
    x = sin_t * math.cos(phi)
    y = sin_t * math.sin(phi)
    z = math.sqrt(r1)

    dx = x * ux + y * vx + z * n0x
    dy = x * uy + y * vy + z * n0y
    dz = x * uz + y * vz + z * n0z

    orig[k, 0] = px; orig[k, 1] = py; orig[k, 2] = pz
    dire[k, 0] = dx; dire[k, 1] = dy; dire[k, 2] = dz

__all__ += ["kernel_build_rays"]
