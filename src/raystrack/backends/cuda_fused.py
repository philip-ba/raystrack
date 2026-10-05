"""Fused CUDA traversal and block reductions for the common v2 workspace.

The flat/BVH device routines preserve the validated combined kernels' traversal
and intersection operations. Instances and Tregenza bins reuse their original
device functions. No per-ray hit/mask arrays or separate reduction launches are
needed; each replicate produces one compact int64 histogram row.
"""
from __future__ import annotations

import numba as nb
from numba import cuda
from ..utils.cuda_trace import (
    INF, STACK_SIZE, MAX_SHARED_SURF, TREGENZA_BINS,
    _aabb_tmin_dev, _instance_ray_dev, _tregenza_patch_id,
)

MAX_FUSED_COLUMNS = 2 * MAX_SHARED_SURF + TREGENZA_BINS

@cuda.jit(device=True, inline=True)
def _flat_ray(k, orig, dirs, v0, e1, e2, norm, sid, surf_active, emit_sid):
    matrix_min_sid = 0
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
        if surf == emit_sid or surf_active[surf] == 0:
            continue
        px = d1 * e2[i, 2] - d2 * e2[i, 1]
        py = d2 * e2[i, 0] - d0 * e2[i, 2]
        pz = d0 * e2[i, 1] - d1 * e2[i, 0]
        det = e1[i, 0] * px + e1[i, 1] * py + e1[i, 2] * pz
        if abs(det) < 1e-07:
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
        if tparam <= 1e-06:
            continue
        any_hit = 1
        if surf < matrix_min_sid:
            continue
        if tparam < best_matrix:
            best_matrix = tparam
            hit = surf
            front = 1 if -(d0 * norm[i, 0] + d1 * norm[i, 1] + d2 * norm[i, 2]) > 0.0 else 0
    return (hit, front, any_hit)

@cuda.jit(device=True, inline=True)
def _bvh_ray(k, orig, dirs, v0, e1, e2, norm, sid, surf_active, bb_min, bb_max, left, right, start, cnt, emit_sid):
    if cnt.shape[0] == 0:
        return (-1, 0, 0)
    matrix_min_sid = 0
    o0 = orig[k, 0]
    o1 = orig[k, 1]
    o2 = orig[k, 2]
    d0 = dirs[k, 0]
    d1 = dirs[k, 1]
    d2 = dirs[k, 2]
    inv0 = 1.0 / d0 if abs(d0) > 1e-09 else 10000000000.0
    inv1 = 1.0 / d1 if abs(d1) > 1e-09 else 10000000000.0
    inv2 = 1.0 / d2 if abs(d2) > 1e-09 else 10000000000.0
    root_t = _aabb_tmin_dev(o0, o1, o2, inv0, inv1, inv2, bb_min[0, 0], bb_min[0, 1], bb_min[0, 2], bb_max[0, 0], bb_max[0, 1], bb_max[0, 2])
    if root_t >= INF:
        return (-1, 0, 0)
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
                if surf == emit_sid or surf_active[surf] == 0:
                    continue
                px = d1 * e2[tri, 2] - d2 * e2[tri, 1]
                py = d2 * e2[tri, 0] - d0 * e2[tri, 2]
                pz = d0 * e2[tri, 1] - d1 * e2[tri, 0]
                det = e1[tri, 0] * px + e1[tri, 1] * py + e1[tri, 2] * pz
                if abs(det) < 1e-07:
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
                if tparam <= 1e-06:
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
            tl = _aabb_tmin_dev(o0, o1, o2, inv0, inv1, inv2, bb_min[ln, 0], bb_min[ln, 1], bb_min[ln, 2], bb_max[ln, 0], bb_max[ln, 1], bb_max[ln, 2])
            tr = _aabb_tmin_dev(o0, o1, o2, inv0, inv1, inv2, bb_min[rn, 0], bb_min[rn, 1], bb_min[rn, 2], bb_max[rn, 0], bb_max[rn, 1], bb_max[rn, 2])
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
    return (hit, front, any_hit)

@cuda.jit(device=True, inline=True)
def _initialise_counts(shared, columns, use_shared):
    if use_shared:
        i = cuda.threadIdx.x
        while i < columns:
            shared[i] = 0
            i += cuda.blockDim.x
        cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def _count_ray(hit, front, any_hit, dx, dy, dz, n_surf, discrete,
               include_sky, shared, counts, use_shared):
    column = -1
    if hit >= 0:
        column = hit if front else n_surf + hit
    elif include_sky and any_hit == 0 and dz > 0:
        patch = _tregenza_patch_id(dx, dy, dz) if discrete else 0
        if patch >= 0:
            column = 2 * n_surf + patch
    if column >= 0:
        if use_shared:
            cuda.atomic.add(shared, column, 1)
        else:
            cuda.atomic.add(counts, column, 1)


@cuda.jit(device=True, inline=True)
def _publish_counts(shared, counts, use_shared):
    if use_shared:
        cuda.syncthreads()
        i = cuda.threadIdx.x
        while i < counts.shape[0]:
            count = shared[i]
            if count:
                cuda.atomic.add(counts, i, count)
            i += cuda.blockDim.x


@cuda.jit
def kernel_fused_flat(orig, dirs, v0, e1, e2, norm, sid, surf_active,
                      emit_sid, n_surf, discrete, include_sky, counts):
    shared = cuda.shared.array(MAX_FUSED_COLUMNS, nb.int32)
    use_shared = n_surf <= MAX_SHARED_SURF
    _initialise_counts(shared, counts.shape[0], use_shared)
    k = cuda.grid(1)
    if k < orig.shape[0]:
        hit, front, any_hit = _flat_ray(k, orig, dirs, v0, e1, e2, norm,
                                        sid, surf_active, emit_sid)
        _count_ray(hit, front, any_hit, dirs[k, 0], dirs[k, 1], dirs[k, 2],
                   n_surf, discrete, include_sky, shared, counts, use_shared)
    _publish_counts(shared, counts, use_shared)


@cuda.jit
def kernel_fused_bvh(orig, dirs, v0, e1, e2, norm, sid, surf_active,
                     bb_min, bb_max, left, right, start, cnt,
                     emit_sid, n_surf, discrete, include_sky, counts):
    shared = cuda.shared.array(MAX_FUSED_COLUMNS, nb.int32)
    use_shared = n_surf <= MAX_SHARED_SURF
    _initialise_counts(shared, counts.shape[0], use_shared)
    k = cuda.grid(1)
    if k < orig.shape[0]:
        hit, front, any_hit = _bvh_ray(k, orig, dirs, v0, e1, e2, norm, sid,
            surf_active, bb_min, bb_max, left, right, start, cnt, emit_sid)
        _count_ray(hit, front, any_hit, dirs[k, 0], dirs[k, 1], dirs[k, 2],
                   n_surf, discrete, include_sky, shared, counts, use_shared)
    _publish_counts(shared, counts, use_shared)


@cuda.jit
def kernel_fused_instanced(orig, dirs, geom, blo, bhi, bleft, bstart, bcount,
                          bescape, roots, rotations, translations, tlo, thi,
                          tleft, tstart, tcount, tescape, instance_order,
                          surf_active, emit_sid, n_surf, discrete,
                          include_sky, counts):
    shared = cuda.shared.array(MAX_FUSED_COLUMNS, nb.int32)
    use_shared = n_surf <= MAX_SHARED_SURF
    _initialise_counts(shared, counts.shape[0], use_shared)
    k = cuda.grid(1)
    if k < orig.shape[0]:
        o = (orig[k, 0], orig[k, 1], orig[k, 2])
        d = (dirs[k, 0], dirs[k, 1], dirs[k, 2])
        hit, front, any_hit = _instance_ray_dev(o, d, geom, blo, bhi, bleft,
            bstart, bcount, bescape, roots, rotations, translations, tlo, thi,
            tleft, tstart, tcount, tescape, instance_order, surf_active, emit_sid, 0)
        _count_ray(hit, front, any_hit, d[0], d[1], d[2], n_surf, discrete,
                   include_sky, shared, counts, use_shared)
    _publish_counts(shared, counts, use_shared)


__all__ = ["kernel_fused_flat", "kernel_fused_bvh", "kernel_fused_instanced"]
