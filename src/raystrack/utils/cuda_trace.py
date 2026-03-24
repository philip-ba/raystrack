from __future__ import annotations

import math

import numba as nb
import numpy as np
from numba import cuda

INF = 1.0e20
STACK_SIZE = 64
MAX_SHARED_SURF = 1024
TREGENZA_BINS = 145


@cuda.jit(device=True, inline=True)
def _aabb_tmin_dev(o0, o1, o2, inv0, inv1, inv2, bmin0, bmin1, bmin2, bmax0, bmax1, bmax2):
    tmin = (bmin0 - o0) * inv0
    tmax = (bmax0 - o0) * inv0
    if tmin > tmax:
        tmp = tmin
        tmin = tmax
        tmax = tmp

    tymin = (bmin1 - o1) * inv1
    tymax = (bmax1 - o1) * inv1
    if tymin > tymax:
        tmp = tymin
        tymin = tymax
        tymax = tmp
    if (tmin > tymax) or (tymin > tmax):
        return INF
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (bmin2 - o2) * inv2
    tzmax = (bmax2 - o2) * inv2
    if tzmin > tzmax:
        tmp = tzmin
        tzmin = tzmax
        tzmax = tmp
    if (tmin > tzmax) or (tzmin > tmax):
        return INF
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax
    if tmax < 0.0:
        return INF
    return tmin if tmin > 0.0 else 0.0


@cuda.jit(device=True, inline=True)
def _skip_surface(surface_id, emit_sid, min_sid):
    if surface_id < min_sid:
        return True
    return surface_id == emit_sid


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
        return 0
    if lo >= cdf.shape[0]:
        return cdf.shape[0] - 1
    return lo


@cuda.jit
def kernel_trace_firsthit(orig, dirs, v0, e1, e2, norm, sid, emit_sid, min_sid, hit_sid, hit_front):
    k = cuda.grid(1)
    if k >= orig.shape[0]:
        return

    o0 = orig[k, 0]
    o1 = orig[k, 1]
    o2 = orig[k, 2]
    d0 = dirs[k, 0]
    d1 = dirs[k, 1]
    d2 = dirs[k, 2]

    best = INF
    hit = -1
    front = 0

    for i in range(v0.shape[0]):
        surf = sid[i]
        if _skip_surface(surf, emit_sid, min_sid):
            continue

        px = d1 * e2[i, 2] - d2 * e2[i, 1]
        py = d2 * e2[i, 0] - d0 * e2[i, 2]
        pz = d0 * e2[i, 1] - d1 * e2[i, 0]
        det = e1[i, 0] * px + e1[i, 1] * py + e1[i, 2] * pz
        if abs(det) < 1e-7:
            continue

        inv_det = 1.0 / det
        tx = o0 - v0[i, 0]
        ty = o1 - v0[i, 1]
        tz = o2 - v0[i, 2]
        u = (tx * px + ty * py + tz * pz) * inv_det
        if u < 0.0 or u > 1.0:
            continue

        qx = ty * e1[i, 2] - tz * e1[i, 1]
        qy = tz * e1[i, 0] - tx * e1[i, 2]
        qz = tx * e1[i, 1] - ty * e1[i, 0]
        v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
        if v < 0.0 or u + v > 1.0:
            continue

        tparam = (e2[i, 0] * qx + e2[i, 1] * qy + e2[i, 2] * qz) * inv_det
        if 1e-6 < tparam < best:
            best = tparam
            hit = surf
            front = 1 if -(d0 * norm[i, 0] + d1 * norm[i, 1] + d2 * norm[i, 2]) > 0.0 else 0

    hit_sid[k] = hit
    hit_front[k] = front if hit >= 0 else 0


@cuda.jit
def kernel_trace_bvh_firsthit(
    orig,
    dirs,
    v0,
    e1,
    e2,
    norm,
    sid,
    bb_min,
    bb_max,
    left,
    right,
    start,
    cnt,
    emit_sid,
    min_sid,
    hit_sid,
    hit_front,
):
    k = cuda.grid(1)
    if k >= orig.shape[0]:
        return

    o0 = orig[k, 0]
    o1 = orig[k, 1]
    o2 = orig[k, 2]
    d0 = dirs[k, 0]
    d1 = dirs[k, 1]
    d2 = dirs[k, 2]

    inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
    inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
    inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

    root_t = _aabb_tmin_dev(
        o0,
        o1,
        o2,
        inv0,
        inv1,
        inv2,
        bb_min[0, 0],
        bb_min[0, 1],
        bb_min[0, 2],
        bb_max[0, 0],
        bb_max[0, 1],
        bb_max[0, 2],
    )
    if root_t >= INF:
        hit_sid[k] = -1
        hit_front[k] = 0
        return

    stack = cuda.local.array(STACK_SIZE, nb.int32)
    tstack = cuda.local.array(STACK_SIZE, nb.float32)
    sp = 0
    stack[sp] = 0
    tstack[sp] = root_t
    sp += 1

    best = INF
    hit = -1
    front = 0

    while sp > 0:
        sp -= 1
        node = stack[sp]
        node_t = tstack[sp]
        if node_t >= best:
            continue

        if cnt[node] > 0:
            for t in range(cnt[node]):
                tri = start[node] + t
                surf = sid[tri]
                if _skip_surface(surf, emit_sid, min_sid):
                    continue

                px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
                py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
                pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
                det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
                if abs(det) < 1e-7:
                    continue

                inv_det = 1.0 / det
                tx = o0 - v0[tri, 0]
                ty = o1 - v0[tri, 1]
                tz = o2 - v0[tri, 2]
                u = (tx * px + ty * py + tz * pz) * inv_det
                if u < 0.0 or u > 1.0:
                    continue

                qx = ty * e1[tri, 2] - tz * e1[tri, 1]
                qy = tz * e1[tri, 0] - tx * e1[tri, 2]
                qz = tx * e1[tri, 1] - ty * e1[tri, 0]
                v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
                if v < 0.0 or u + v > 1.0:
                    continue

                tparam = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                if 1e-6 < tparam < best:
                    best = tparam
                    hit = surf
                    front = 1 if -(d0 * norm[tri, 0] + d1 * norm[tri, 1] + d2 * norm[tri, 2]) > 0.0 else 0
        else:
            ln = left[node]
            rn = right[node]
            tl = _aabb_tmin_dev(
                o0,
                o1,
                o2,
                inv0,
                inv1,
                inv2,
                bb_min[ln, 0],
                bb_min[ln, 1],
                bb_min[ln, 2],
                bb_max[ln, 0],
                bb_max[ln, 1],
                bb_max[ln, 2],
            )
            tr = _aabb_tmin_dev(
                o0,
                o1,
                o2,
                inv0,
                inv1,
                inv2,
                bb_min[rn, 0],
                bb_min[rn, 1],
                bb_min[rn, 2],
                bb_max[rn, 0],
                bb_max[rn, 1],
                bb_max[rn, 2],
            )

            if tl < tr:
                if tr < best and sp < STACK_SIZE:
                    stack[sp] = rn
                    tstack[sp] = tr
                    sp += 1
                if tl < best and sp < STACK_SIZE:
                    stack[sp] = ln
                    tstack[sp] = tl
                    sp += 1
            else:
                if tl < best and sp < STACK_SIZE:
                    stack[sp] = ln
                    tstack[sp] = tl
                    sp += 1
                if tr < best and sp < STACK_SIZE:
                    stack[sp] = rn
                    tstack[sp] = tr
                    sp += 1

    hit_sid[k] = hit
    hit_front[k] = front if hit >= 0 else 0


@cuda.jit
def kernel_trace_combined(orig, dirs, v0, e1, e2, norm, sid, emit_sid, matrix_min_sid, hit_sid, hit_front, any_hitmask):
    k = cuda.grid(1)
    if k >= orig.shape[0]:
        return

    o0 = orig[k, 0]
    o1 = orig[k, 1]
    o2 = orig[k, 2]
    d0 = dirs[k, 0]
    d1 = dirs[k, 1]
    d2 = dirs[k, 2]

    best_matrix = INF
    hit = -1
    front = 0
    any_hit = 0

    for i in range(v0.shape[0]):
        surf = sid[i]
        if surf == emit_sid:
            continue

        px = d1 * e2[i, 2] - d2 * e2[i, 1]
        py = d2 * e2[i, 0] - d0 * e2[i, 2]
        pz = d0 * e2[i, 1] - d1 * e2[i, 0]
        det = e1[i, 0] * px + e1[i, 1] * py + e1[i, 2] * pz
        if abs(det) < 1e-7:
            continue

        inv_det = 1.0 / det
        tx = o0 - v0[i, 0]
        ty = o1 - v0[i, 1]
        tz = o2 - v0[i, 2]
        u = (tx * px + ty * py + tz * pz) * inv_det
        if u < 0.0 or u > 1.0:
            continue

        qx = ty * e1[i, 2] - tz * e1[i, 1]
        qy = tz * e1[i, 0] - tx * e1[i, 2]
        qz = tx * e1[i, 1] - ty * e1[i, 0]
        v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
        if v < 0.0 or u + v > 1.0:
            continue

        tparam = (e2[i, 0] * qx + e2[i, 1] * qy + e2[i, 2] * qz) * inv_det
        if tparam <= 1e-6:
            continue

        any_hit = 1
        if surf < matrix_min_sid:
            continue
        if tparam < best_matrix:
            best_matrix = tparam
            hit = surf
            front = 1 if -(d0 * norm[i, 0] + d1 * norm[i, 1] + d2 * norm[i, 2]) > 0.0 else 0

    hit_sid[k] = hit
    hit_front[k] = front if hit >= 0 else 0
    any_hitmask[k] = any_hit


@cuda.jit
def kernel_trace_bvh_combined(
    orig,
    dirs,
    v0,
    e1,
    e2,
    norm,
    sid,
    bb_min,
    bb_max,
    left,
    right,
    start,
    cnt,
    emit_sid,
    matrix_min_sid,
    hit_sid,
    hit_front,
    any_hitmask,
):
    k = cuda.grid(1)
    if k >= orig.shape[0]:
        return

    o0 = orig[k, 0]
    o1 = orig[k, 1]
    o2 = orig[k, 2]
    d0 = dirs[k, 0]
    d1 = dirs[k, 1]
    d2 = dirs[k, 2]

    inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
    inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
    inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

    root_t = _aabb_tmin_dev(
        o0,
        o1,
        o2,
        inv0,
        inv1,
        inv2,
        bb_min[0, 0],
        bb_min[0, 1],
        bb_min[0, 2],
        bb_max[0, 0],
        bb_max[0, 1],
        bb_max[0, 2],
    )
    if root_t >= INF:
        hit_sid[k] = -1
        hit_front[k] = 0
        any_hitmask[k] = 0
        return

    stack = cuda.local.array(STACK_SIZE, nb.int32)
    tstack = cuda.local.array(STACK_SIZE, nb.float32)
    sp = 0
    stack[sp] = 0
    tstack[sp] = root_t
    sp += 1

    best_matrix = INF
    hit = -1
    front = 0
    any_hit = 0

    while sp > 0:
        sp -= 1
        node = stack[sp]
        node_t = tstack[sp]
        if node_t >= best_matrix:
            continue

        if cnt[node] > 0:
            for t in range(cnt[node]):
                tri = start[node] + t
                surf = sid[tri]
                if surf == emit_sid:
                    continue

                px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
                py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
                pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
                det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
                if abs(det) < 1e-7:
                    continue

                inv_det = 1.0 / det
                tx = o0 - v0[tri, 0]
                ty = o1 - v0[tri, 1]
                tz = o2 - v0[tri, 2]
                u = (tx * px + ty * py + tz * pz) * inv_det
                if u < 0.0 or u > 1.0:
                    continue

                qx = ty * e1[tri, 2] - tz * e1[tri, 1]
                qy = tz * e1[tri, 0] - tx * e1[tri, 2]
                qz = tx * e1[tri, 1] - ty * e1[tri, 0]
                v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
                if v < 0.0 or u + v > 1.0:
                    continue

                tparam = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                if tparam <= 1e-6:
                    continue

                any_hit = 1
                if surf < matrix_min_sid:
                    continue
                if tparam < best_matrix:
                    best_matrix = tparam
                    hit = surf
                    front = 1 if -(d0 * norm[tri, 0] + d1 * norm[tri, 1] + d2 * norm[tri, 2]) > 0.0 else 0
        else:
            ln = left[node]
            rn = right[node]
            tl = _aabb_tmin_dev(
                o0,
                o1,
                o2,
                inv0,
                inv1,
                inv2,
                bb_min[ln, 0],
                bb_min[ln, 1],
                bb_min[ln, 2],
                bb_max[ln, 0],
                bb_max[ln, 1],
                bb_max[ln, 2],
            )
            tr = _aabb_tmin_dev(
                o0,
                o1,
                o2,
                inv0,
                inv1,
                inv2,
                bb_min[rn, 0],
                bb_min[rn, 1],
                bb_min[rn, 2],
                bb_max[rn, 0],
                bb_max[rn, 1],
                bb_max[rn, 2],
            )

            if tl < tr:
                if tr < best_matrix and sp < STACK_SIZE:
                    stack[sp] = rn
                    tstack[sp] = tr
                    sp += 1
                if tl < best_matrix and sp < STACK_SIZE:
                    stack[sp] = ln
                    tstack[sp] = tl
                    sp += 1
            else:
                if tl < best_matrix and sp < STACK_SIZE:
                    stack[sp] = ln
                    tstack[sp] = tl
                    sp += 1
                if tr < best_matrix and sp < STACK_SIZE:
                    stack[sp] = rn
                    tstack[sp] = tr
                    sp += 1

    hit_sid[k] = hit
    hit_front[k] = front if hit >= 0 else 0
    any_hitmask[k] = any_hit


@cuda.jit
def kernel_reduce_hits(hit_sid, hit_front, hits_f, hits_b, n_surf):
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    if n_surf <= MAX_SHARED_SURF:
        shared_f = cuda.shared.array(MAX_SHARED_SURF, nb.int32)
        shared_b = cuda.shared.array(MAX_SHARED_SURF, nb.int32)

        i = tid
        while i < n_surf:
            shared_f[i] = 0
            shared_b[i] = 0
            i += block_size
        cuda.syncthreads()

        i = idx
        while i < hit_sid.shape[0]:
            hit = hit_sid[i]
            if 0 <= hit < n_surf:
                if hit_front[i] != 0:
                    cuda.atomic.add(shared_f, hit, 1)
                else:
                    cuda.atomic.add(shared_b, hit, 1)
            i += stride
        cuda.syncthreads()

        i = tid
        while i < n_surf:
            val_f = shared_f[i]
            val_b = shared_b[i]
            if val_f != 0:
                cuda.atomic.add(hits_f, i, val_f)
            if val_b != 0:
                cuda.atomic.add(hits_b, i, val_b)
            i += block_size
    else:
        i = idx
        while i < hit_sid.shape[0]:
            hit = hit_sid[i]
            if 0 <= hit < n_surf:
                if hit_front[i] != 0:
                    cuda.atomic.add(hits_f, hit, 1)
                else:
                    cuda.atomic.add(hits_b, hit, 1)
            i += stride


@cuda.jit
def kernel_accumulate_hits(hits_f_iter, hits_b_iter, hits_f_total, hits_b_total):
    i = cuda.grid(1)
    if i >= hits_f_iter.shape[0]:
        return
    hits_f_total[i] += hits_f_iter[i]
    hits_b_total[i] += hits_b_iter[i]


@cuda.jit
def kernel_accumulate_hits_stats(
    hits_f_iter,
    hits_b_iter,
    hits_f_total,
    hits_b_total,
    sum_f,
    sumsq_f,
    sum_b,
    sumsq_b,
    inv_rays,
):
    i = cuda.grid(1)
    if i >= hits_f_iter.shape[0]:
        return

    hf = hits_f_iter[i]
    hb = hits_b_iter[i]
    xf = float(hf) * inv_rays
    xb = float(hb) * inv_rays

    hits_f_total[i] += hf
    hits_b_total[i] += hb
    sum_f[i] += xf
    sumsq_f[i] += xf * xf
    sum_b[i] += xb
    sumsq_b[i] += xb * xb


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


@cuda.jit
def kernel_build_rays(
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
    k = cuda.grid(1)
    n_rays = orig.shape[0]
    if k >= n_rays:
        return

    cell = k // rays_per_cell
    ug = (u_grid[cell] + cp_grid[0]) % 1.0
    vg = (v_grid[cell] + cp_grid[1]) % 1.0

    q_tri = (halton_tri[k] + cp_dims[0]) % 1.0
    tri = _binary_search_cdf(cdf, q_tri)

    ur = (halton_u[k] + cp_dims[1] + ug) % 1.0
    vr = (halton_v[k] + cp_dims[2] + vg) % 1.0

    s = math.sqrt(ur)
    mix_b = s * vr
    mix_c = s * (1.0 - vr)

    ax = tri_a[tri, 0]
    ay = tri_a[tri, 1]
    az = tri_a[tri, 2]

    px = ax + mix_b * tri_e1[tri, 0] + mix_c * tri_e2[tri, 0]
    py = ay + mix_b * tri_e1[tri, 1] + mix_c * tri_e2[tri, 1]
    pz = az + mix_b * tri_e1[tri, 2] + mix_c * tri_e2[tri, 2]

    r1 = (halton_r1[k] + cp_dims[3]) % 1.0
    r2 = (halton_r2[k] + cp_dims[4]) % 1.0
    sin_t = math.sqrt(1.0 - r1)
    phi = 6.283185307179586 * r2
    x = sin_t * math.cos(phi)
    y = sin_t * math.sin(phi)
    z = math.sqrt(r1)

    dx = x * tri_u[tri, 0] + y * tri_v[tri, 0] + z * tri_n[tri, 0]
    dy = x * tri_u[tri, 1] + y * tri_v[tri, 1] + z * tri_n[tri, 1]
    dz = x * tri_u[tri, 2] + y * tri_v[tri, 2] + z * tri_n[tri, 2]
    eps = tri_origin_eps[tri]

    orig[k, 0] = px + eps * tri_n[tri, 0]
    orig[k, 1] = py + eps * tri_n[tri, 1]
    orig[k, 2] = pz + eps * tri_n[tri, 2]
    dire[k, 0] = dx
    dire[k, 1] = dy
    dire[k, 2] = dz


@cuda.jit(device=True, inline=True)
def _tregenza_patch_id(dx, dy, dz):
    if dz <= 0.0:
        return -1

    ring_hi_sin = (
        0.20791169081775934,
        0.40673664307580015,
        0.5877852522924731,
        0.7431448254773942,
        0.8660254037844386,
        0.9510565162951535,
        0.9945218953682733,
        1.0,
    )
    ring_n = (30, 30, 24, 24, 18, 12, 6, 1)
    ring_start = (0, 30, 60, 84, 108, 126, 138, 144)

    ridx = 7
    for j in range(8):
        if dz < ring_hi_sin[j] or j == 7:
            ridx = j
            break

    n_az = ring_n[ridx]
    base = ring_start[ridx]
    if n_az == 1:
        return base

    az = math.atan2(dy, dx) * 57.29577951308232
    if az < 0.0:
        az += 360.0
    width = 360.0 / n_az
    off = (180.0 / n_az) if (ridx & 1) == 1 else 0.0
    t = az - off
    if t < 0.0:
        t += 360.0
    elif t >= 360.0:
        t -= 360.0
    aidx = int(t // width)
    if aidx >= n_az:
        aidx = n_az - 1
    return base + aidx


@cuda.jit
def kernel_bin_tregenza(dirs, any_hitmask, counts):
    shared_counts = cuda.shared.array(TREGENZA_BINS, nb.int32)
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    i = tid
    while i < TREGENZA_BINS:
        shared_counts[i] = 0
        i += block_size
    cuda.syncthreads()

    i = idx
    while i < dirs.shape[0]:
        if any_hitmask[i] == 0:
            pid = _tregenza_patch_id(dirs[i, 0], dirs[i, 1], dirs[i, 2])
            if pid >= 0:
                cuda.atomic.add(shared_counts, pid, 1)
        i += stride
    cuda.syncthreads()

    i = tid
    while i < TREGENZA_BINS:
        val = shared_counts[i]
        if val != 0:
            cuda.atomic.add(counts, i, val)
        i += block_size


@cuda.jit
def kernel_count_upward_misses(dirs, any_hitmask, count):
    shared_total = cuda.shared.array(1, nb.int32)
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid == 0:
        shared_total[0] = 0
    cuda.syncthreads()

    i = idx
    while i < dirs.shape[0]:
        if any_hitmask[i] == 0 and dirs[i, 2] > 0.0:
            cuda.atomic.add(shared_total, 0, 1)
        i += stride
    cuda.syncthreads()

    if tid == 0 and shared_total[0] != 0:
        cuda.atomic.add(count, 0, shared_total[0])


@cuda.jit
def kernel_trace_tregenza(orig, dirs, v0, e1, e2, sid, emit_sid, min_sid, counts):
    shared_counts = cuda.shared.array(TREGENZA_BINS, nb.int32)
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    i = tid
    while i < TREGENZA_BINS:
        shared_counts[i] = 0
        i += block_size
    cuda.syncthreads()

    i = idx
    while i < orig.shape[0]:
        o0 = orig[i, 0]
        o1 = orig[i, 1]
        o2 = orig[i, 2]
        d0 = dirs[i, 0]
        d1 = dirs[i, 1]
        d2 = dirs[i, 2]

        hit_any = 0
        for tri in range(v0.shape[0]):
            surf = sid[tri]
            if _skip_surface(surf, emit_sid, min_sid):
                continue

            px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
            py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
            pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
            det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
            if abs(det) < 1e-7:
                continue

            inv_det = 1.0 / det
            tx = o0 - v0[tri, 0]
            ty = o1 - v0[tri, 1]
            tz = o2 - v0[tri, 2]
            u = (tx * px + ty * py + tz * pz) * inv_det
            if u < 0.0 or u > 1.0:
                continue

            qx = ty * e1[tri, 2] - tz * e1[tri, 1]
            qy = tz * e1[tri, 0] - tx * e1[tri, 2]
            qz = tx * e1[tri, 1] - ty * e1[tri, 0]
            v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue

            tparam = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
            if tparam > 1e-6:
                hit_any = 1
                break

        if hit_any == 0:
            pid = _tregenza_patch_id(d0, d1, d2)
            if pid >= 0:
                cuda.atomic.add(shared_counts, pid, 1)
        i += stride
    cuda.syncthreads()

    i = tid
    while i < TREGENZA_BINS:
        val = shared_counts[i]
        if val != 0:
            cuda.atomic.add(counts, i, val)
        i += block_size


@cuda.jit
def kernel_trace_count_upward(orig, dirs, v0, e1, e2, sid, emit_sid, min_sid, count):
    shared_total = cuda.shared.array(1, nb.int32)
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid == 0:
        shared_total[0] = 0
    cuda.syncthreads()

    i = idx
    while i < orig.shape[0]:
        o0 = orig[i, 0]
        o1 = orig[i, 1]
        o2 = orig[i, 2]
        d0 = dirs[i, 0]
        d1 = dirs[i, 1]
        d2 = dirs[i, 2]

        hit_any = 0
        for tri in range(v0.shape[0]):
            surf = sid[tri]
            if _skip_surface(surf, emit_sid, min_sid):
                continue

            px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
            py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
            pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
            det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
            if abs(det) < 1e-7:
                continue

            inv_det = 1.0 / det
            tx = o0 - v0[tri, 0]
            ty = o1 - v0[tri, 1]
            tz = o2 - v0[tri, 2]
            u = (tx * px + ty * py + tz * pz) * inv_det
            if u < 0.0 or u > 1.0:
                continue

            qx = ty * e1[tri, 2] - tz * e1[tri, 1]
            qy = tz * e1[tri, 0] - tx * e1[tri, 2]
            qz = tx * e1[tri, 1] - ty * e1[tri, 0]
            v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue

            tparam = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
            if tparam > 1e-6:
                hit_any = 1
                break

        if hit_any == 0 and d2 > 0.0:
            cuda.atomic.add(shared_total, 0, 1)
        i += stride
    cuda.syncthreads()

    if tid == 0 and shared_total[0] != 0:
        cuda.atomic.add(count, 0, shared_total[0])


@cuda.jit
def kernel_trace_bvh_tregenza(
    orig,
    dirs,
    v0,
    e1,
    e2,
    sid,
    bb_min,
    bb_max,
    left,
    right,
    start,
    cnt,
    emit_sid,
    min_sid,
    counts,
):
    shared_counts = cuda.shared.array(TREGENZA_BINS, nb.int32)
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    i = tid
    while i < TREGENZA_BINS:
        shared_counts[i] = 0
        i += block_size
    cuda.syncthreads()

    i = idx
    while i < orig.shape[0]:
        o0 = orig[i, 0]
        o1 = orig[i, 1]
        o2 = orig[i, 2]
        d0 = dirs[i, 0]
        d1 = dirs[i, 1]
        d2 = dirs[i, 2]

        inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
        inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
        inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

        root_t = _aabb_tmin_dev(
            o0,
            o1,
            o2,
            inv0,
            inv1,
            inv2,
            bb_min[0, 0],
            bb_min[0, 1],
            bb_min[0, 2],
            bb_max[0, 0],
            bb_max[0, 1],
            bb_max[0, 2],
        )

        hit_any = 0
        if root_t < INF:
            stack = cuda.local.array(STACK_SIZE, nb.int32)
            tstack = cuda.local.array(STACK_SIZE, nb.float32)
            sp = 0
            stack[sp] = 0
            tstack[sp] = root_t
            sp += 1

            while sp > 0 and hit_any == 0:
                sp -= 1
                node = stack[sp]

                if cnt[node] > 0:
                    for t in range(cnt[node]):
                        tri = start[node] + t
                        surf = sid[tri]
                        if _skip_surface(surf, emit_sid, min_sid):
                            continue

                        px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
                        py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
                        pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
                        det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
                        if abs(det) < 1e-7:
                            continue

                        inv_det = 1.0 / det
                        tx = o0 - v0[tri, 0]
                        ty = o1 - v0[tri, 1]
                        tz = o2 - v0[tri, 2]
                        u = (tx * px + ty * py + tz * pz) * inv_det
                        if u < 0.0 or u > 1.0:
                            continue

                        qx = ty * e1[tri, 2] - tz * e1[tri, 1]
                        qy = tz * e1[tri, 0] - tx * e1[tri, 2]
                        qz = tx * e1[tri, 1] - ty * e1[tri, 0]
                        v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
                        if v < 0.0 or u + v > 1.0:
                            continue

                        tparam = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                        if tparam > 1e-6:
                            hit_any = 1
                            break
                else:
                    ln = left[node]
                    rn = right[node]
                    tl = _aabb_tmin_dev(
                        o0,
                        o1,
                        o2,
                        inv0,
                        inv1,
                        inv2,
                        bb_min[ln, 0],
                        bb_min[ln, 1],
                        bb_min[ln, 2],
                        bb_max[ln, 0],
                        bb_max[ln, 1],
                        bb_max[ln, 2],
                    )
                    tr = _aabb_tmin_dev(
                        o0,
                        o1,
                        o2,
                        inv0,
                        inv1,
                        inv2,
                        bb_min[rn, 0],
                        bb_min[rn, 1],
                        bb_min[rn, 2],
                        bb_max[rn, 0],
                        bb_max[rn, 1],
                        bb_max[rn, 2],
                    )

                    if tl < tr:
                        if tr < INF and sp < STACK_SIZE:
                            stack[sp] = rn
                            tstack[sp] = tr
                            sp += 1
                        if tl < INF and sp < STACK_SIZE:
                            stack[sp] = ln
                            tstack[sp] = tl
                            sp += 1
                    else:
                        if tl < INF and sp < STACK_SIZE:
                            stack[sp] = ln
                            tstack[sp] = tl
                            sp += 1
                        if tr < INF and sp < STACK_SIZE:
                            stack[sp] = rn
                            tstack[sp] = tr
                            sp += 1

        if hit_any == 0:
            pid = _tregenza_patch_id(d0, d1, d2)
            if pid >= 0:
                cuda.atomic.add(shared_counts, pid, 1)
        i += stride
    cuda.syncthreads()

    i = tid
    while i < TREGENZA_BINS:
        val = shared_counts[i]
        if val != 0:
            cuda.atomic.add(counts, i, val)
        i += block_size


@cuda.jit
def kernel_trace_bvh_count_upward(
    orig,
    dirs,
    v0,
    e1,
    e2,
    sid,
    bb_min,
    bb_max,
    left,
    right,
    start,
    cnt,
    emit_sid,
    min_sid,
    count,
):
    shared_total = cuda.shared.array(1, nb.int32)
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid == 0:
        shared_total[0] = 0
    cuda.syncthreads()

    i = idx
    while i < orig.shape[0]:
        o0 = orig[i, 0]
        o1 = orig[i, 1]
        o2 = orig[i, 2]
        d0 = dirs[i, 0]
        d1 = dirs[i, 1]
        d2 = dirs[i, 2]

        inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
        inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
        inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

        root_t = _aabb_tmin_dev(
            o0,
            o1,
            o2,
            inv0,
            inv1,
            inv2,
            bb_min[0, 0],
            bb_min[0, 1],
            bb_min[0, 2],
            bb_max[0, 0],
            bb_max[0, 1],
            bb_max[0, 2],
        )

        hit_any = 0
        if root_t < INF:
            stack = cuda.local.array(STACK_SIZE, nb.int32)
            tstack = cuda.local.array(STACK_SIZE, nb.float32)
            sp = 0
            stack[sp] = 0
            tstack[sp] = root_t
            sp += 1

            while sp > 0 and hit_any == 0:
                sp -= 1
                node = stack[sp]

                if cnt[node] > 0:
                    for t in range(cnt[node]):
                        tri = start[node] + t
                        surf = sid[tri]
                        if _skip_surface(surf, emit_sid, min_sid):
                            continue

                        px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
                        py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
                        pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
                        det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
                        if abs(det) < 1e-7:
                            continue

                        inv_det = 1.0 / det
                        tx = o0 - v0[tri, 0]
                        ty = o1 - v0[tri, 1]
                        tz = o2 - v0[tri, 2]
                        u = (tx * px + ty * py + tz * pz) * inv_det
                        if u < 0.0 or u > 1.0:
                            continue

                        qx = ty * e1[tri, 2] - tz * e1[tri, 1]
                        qy = tz * e1[tri, 0] - tx * e1[tri, 2]
                        qz = tx * e1[tri, 1] - ty * e1[tri, 0]
                        v = (d0 * qx + d1 * qy + d2 * qz) * inv_det
                        if v < 0.0 or u + v > 1.0:
                            continue

                        tparam = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                        if tparam > 1e-6:
                            hit_any = 1
                            break
                else:
                    ln = left[node]
                    rn = right[node]
                    tl = _aabb_tmin_dev(
                        o0,
                        o1,
                        o2,
                        inv0,
                        inv1,
                        inv2,
                        bb_min[ln, 0],
                        bb_min[ln, 1],
                        bb_min[ln, 2],
                        bb_max[ln, 0],
                        bb_max[ln, 1],
                        bb_max[ln, 2],
                    )
                    tr = _aabb_tmin_dev(
                        o0,
                        o1,
                        o2,
                        inv0,
                        inv1,
                        inv2,
                        bb_min[rn, 0],
                        bb_min[rn, 1],
                        bb_min[rn, 2],
                        bb_max[rn, 0],
                        bb_max[rn, 1],
                        bb_max[rn, 2],
                    )

                    if tl < tr:
                        if tr < INF and sp < STACK_SIZE:
                            stack[sp] = rn
                            tstack[sp] = tr
                            sp += 1
                        if tl < INF and sp < STACK_SIZE:
                            stack[sp] = ln
                            tstack[sp] = tl
                            sp += 1
                    else:
                        if tl < INF and sp < STACK_SIZE:
                            stack[sp] = ln
                            tstack[sp] = tl
                            sp += 1
                        if tr < INF and sp < STACK_SIZE:
                            stack[sp] = rn
                            tstack[sp] = tr
                            sp += 1

        if hit_any == 0 and d2 > 0.0:
            cuda.atomic.add(shared_total, 0, 1)
        i += stride
    cuda.syncthreads()

    if tid == 0 and shared_total[0] != 0:
        cuda.atomic.add(count, 0, shared_total[0])


__all__ = [
    "kernel_trace_firsthit",
    "kernel_trace_bvh_firsthit",
    "kernel_trace_combined",
    "kernel_trace_bvh_combined",
    "kernel_reduce_hits",
    "kernel_accumulate_hits",
    "kernel_accumulate_hits_stats",
    "kernel_zero_i64",
    "kernel_zero_i32",
    "kernel_build_rays",
    "kernel_bin_tregenza",
    "kernel_count_upward_misses",
    "kernel_trace_tregenza",
    "kernel_trace_bvh_tregenza",
    "kernel_trace_count_upward",
    "kernel_trace_bvh_count_upward",
]
