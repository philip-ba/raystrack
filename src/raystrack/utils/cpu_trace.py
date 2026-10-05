from __future__ import annotations

import math

import numba as nb
import numpy as np

INF = 1.0e20
STACK_SIZE = 64


@nb.njit(inline="always", cache=True)
def _instance_aabb(o, d, lower, upper, node):
    near, far = 0.0, INF
    for axis in range(3):
        if abs(d[axis]) <= 1e-9:
            if o[axis] < lower[node, axis] or o[axis] > upper[node, axis]:
                return INF
        else:
            a = (lower[node, axis] - o[axis]) / d[axis]
            b = (upper[node, axis] - o[axis]) / d[axis]
            near, far = max(near, min(a, b)), min(far, max(a, b))
    return near if near <= far else INF


@nb.njit(inline="always", cache=True)
def _instance_intersect(o, d, geom, tri):
    px = d[1] * geom[tri, 8] - d[2] * geom[tri, 7]
    py = d[2] * geom[tri, 6] - d[0] * geom[tri, 8]
    pz = d[0] * geom[tri, 7] - d[1] * geom[tri, 6]
    det = geom[tri, 3] * px + geom[tri, 4] * py + geom[tri, 5] * pz
    if abs(det) < 1e-7:
        return INF
    inv = 1.0 / det
    tx, ty, tz = o[0] - geom[tri, 0], o[1] - geom[tri, 1], o[2] - geom[tri, 2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0 or u > 1:
        return INF
    qx = ty * geom[tri, 5] - tz * geom[tri, 4]
    qy = tz * geom[tri, 3] - tx * geom[tri, 5]
    qz = tx * geom[tri, 4] - ty * geom[tri, 3]
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    t = (geom[tri, 6] * qx + geom[tri, 7] * qy + geom[tri, 8] * qz) * inv
    return t if v >= 0 and u + v <= 1 and t > 1e-6 else INF


@nb.njit(inline="always", cache=True)
def _instance_ray(o, d, geom, blo, bhi, bleft, bstart, bcount, bescape,
                  roots, rotations, translations, tlo, thi, tleft, tstart,
                  tcount, tescape, instance_order, active, emit_sid, min_sid):
    best, hit, front, any_hit = INF, -1, 0, 0
    node = 0 if len(tcount) else -1
    while node >= 0:
        successor = tescape[node]
        if _instance_aabb(o, d, tlo, thi, node) < best:
            if tcount[node] == 0:
                successor = tleft[node]
            else:
                for position in range(tstart[node], tstart[node] + tcount[node]):
                    surface = instance_order[position]
                    if surface == emit_sid or active[surface] == 0:
                        continue
                    dx, dy, dz = (o[0] - translations[surface, 0],
                                  o[1] - translations[surface, 1],
                                  o[2] - translations[surface, 2])
                    # A proper rigid inverse preserves ray parameter distances.
                    local_o = (rotations[surface, 0, 0] * dx + rotations[surface, 1, 0] * dy + rotations[surface, 2, 0] * dz,
                               rotations[surface, 0, 1] * dx + rotations[surface, 1, 1] * dy + rotations[surface, 2, 1] * dz,
                               rotations[surface, 0, 2] * dx + rotations[surface, 1, 2] * dy + rotations[surface, 2, 2] * dz)
                    local_d = (rotations[surface, 0, 0] * d[0] + rotations[surface, 1, 0] * d[1] + rotations[surface, 2, 0] * d[2],
                               rotations[surface, 0, 1] * d[0] + rotations[surface, 1, 1] * d[1] + rotations[surface, 2, 1] * d[2],
                               rotations[surface, 0, 2] * d[0] + rotations[surface, 1, 2] * d[1] + rotations[surface, 2, 2] * d[2])
                    blas_node = roots[surface]
                    while blas_node >= 0:
                        next_blas = bescape[blas_node]
                        if _instance_aabb(local_o, local_d, blo, bhi, blas_node) < best:
                            if bcount[blas_node] == 0:
                                next_blas = bleft[blas_node]
                            else:
                                for tri in range(bstart[blas_node], bstart[blas_node] + bcount[blas_node]):
                                    distance = _instance_intersect(local_o, local_d, geom, tri)
                                    if distance < INF:
                                        any_hit = 1
                                        if surface >= min_sid and distance < best:
                                            best, hit = distance, surface
                                            front = int(-(local_d[0] * geom[tri, 9] + local_d[1] * geom[tri, 10]
                                                          + local_d[2] * geom[tri, 11]) > 0)
                        blas_node = next_blas
        node = successor
    return hit, front, any_hit


@nb.njit(parallel=True, cache=True)
def _trace_cpu_instanced_arrays(orig, dirs, geom, blo, bhi, bleft, bstart, bcount, bescape,
                               roots, rotations, translations, tlo, thi, tleft, tstart,
                               tcount, tescape, instance_order, active, emit_sid, min_sid,
                               hit, front, mask):
    for k in nb.prange(len(orig)):
        o = (orig[k, 0], orig[k, 1], orig[k, 2])
        d = (dirs[k, 0], dirs[k, 1], dirs[k, 2])
        hit[k], front[k], mask[k] = _instance_ray(
            o, d, geom, blo, bhi, bleft, bstart, bcount, bescape,
            roots, rotations, translations, tlo, thi, tleft, tstart,
            tcount, tescape, instance_order, active, emit_sid, min_sid)


def trace_cpu_instanced_combined(orig, dirs, scene, active, emit_sid, min_sid, hit, front, mask):
    """Nearest local-BLAS hits plus full-scene occlusion, through a world TLAS."""
    _trace_cpu_instanced_arrays(orig, dirs, *scene.traversal_arrays(), active,
                               emit_sid, min_sid, hit, front, mask)


def trace_cpu_instanced_firsthit(orig, dirs, scene, active, emit_sid, min_sid, hit, front):
    trace_cpu_instanced_combined(orig, dirs, scene, active, emit_sid, min_sid,
                                hit, front, np.empty(len(orig), np.uint8))


@nb.njit(inline="always", cache=True, fastmath=True)
def _aabb_tmin(o0, o1, o2, inv0, inv1, inv2, bmin0, bmin1, bmin2, bmax0, bmax1, bmax2):
    tmin = (bmin0 - o0) * inv0
    tmax = (bmax0 - o0) * inv0
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    tymin = (bmin1 - o1) * inv1
    tymax = (bmax1 - o1) * inv1
    if tymin > tymax:
        tymin, tymax = tymax, tymin
    if (tmin > tymax) or (tymin > tmax):
        return INF
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (bmin2 - o2) * inv2
    tzmax = (bmax2 - o2) * inv2
    if tzmin > tzmax:
        tzmin, tzmax = tzmax, tzmin
    if (tmin > tzmax) or (tzmin > tmax):
        return INF
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax
    if tmax < 0.0:
        return INF
    return tmin if tmin > 0.0 else 0.0


@nb.njit(inline="always", cache=True)
def _skip_surface(surface_id: int, surf_active, emit_sid: int, min_sid: int) -> bool:
    if surf_active[surface_id] == 0:
        return True
    if surface_id < min_sid:
        return True
    return surface_id == emit_sid


@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_firsthit(
    orig,
    dirs,
    v0,
    e1,
    e2,
    norm,
    sid,
    surf_active,
    emit_sid,
    min_sid,
    out_hit_sid,
    out_front,
):
    n_rays = orig.shape[0]
    n_tri = v0.shape[0]
    for k in nb.prange(n_rays):
        o0 = orig[k, 0]
        o1 = orig[k, 1]
        o2 = orig[k, 2]
        d0 = dirs[k, 0]
        d1 = dirs[k, 1]
        d2 = dirs[k, 2]

        best = INF
        hit = -1
        front = 0

        for i in range(n_tri):
            surf = sid[i]
            if _skip_surface(surf, surf_active, emit_sid, min_sid):
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

            t_param = (e2[i, 0] * qx + e2[i, 1] * qy + e2[i, 2] * qz) * inv_det
            if 1e-6 < t_param < best:
                best = t_param
                hit = surf
                front = 1 if -(d0 * norm[i, 0] + d1 * norm[i, 1] + d2 * norm[i, 2]) > 0.0 else 0

        out_hit_sid[k] = hit
        out_front[k] = front if hit >= 0 else 0


@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_bvh_firsthit(
    orig,
    dirs,
    v0,
    e1,
    e2,
    norm,
    sid,
    surf_active,
    bb_min,
    bb_max,
    left,
    right,
    start,
    count,
    emit_sid,
    min_sid,
    out_hit_sid,
    out_front,
):
    n_rays = orig.shape[0]
    for k in nb.prange(n_rays):
        o0 = orig[k, 0]
        o1 = orig[k, 1]
        o2 = orig[k, 2]
        d0 = dirs[k, 0]
        d1 = dirs[k, 1]
        d2 = dirs[k, 2]

        inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
        inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
        inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

        root_t = _aabb_tmin(
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
            out_hit_sid[k] = -1
            out_front[k] = 0
            continue

        stack = np.empty(STACK_SIZE, np.int32)
        tstack = np.empty(STACK_SIZE, np.float32)
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

            if count[node] > 0:
                for t in range(count[node]):
                    tri = start[node] + t
                    surf = sid[tri]
                    if _skip_surface(surf, surf_active, emit_sid, min_sid):
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

                    t_param = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                    if 1e-6 < t_param < best:
                        best = t_param
                        hit = surf
                        front = 1 if -(d0 * norm[tri, 0] + d1 * norm[tri, 1] + d2 * norm[tri, 2]) > 0.0 else 0
            else:
                ln = left[node]
                rn = right[node]
                tl = _aabb_tmin(
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
                tr = _aabb_tmin(
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

        out_hit_sid[k] = hit
        out_front[k] = front if hit >= 0 else 0


@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_combined(
    orig,
    dirs,
    v0,
    e1,
    e2,
    norm,
    sid,
    surf_active,
    emit_sid,
    matrix_min_sid,
    out_hit_sid,
    out_front,
    out_any_hitmask,
):
    n_rays = orig.shape[0]
    n_tri = v0.shape[0]
    for k in nb.prange(n_rays):
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

        for i in range(n_tri):
            surf = sid[i]
            if surf == emit_sid or surf_active[surf] == 0:
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

            t_param = (e2[i, 0] * qx + e2[i, 1] * qy + e2[i, 2] * qz) * inv_det
            if t_param <= 1e-6:
                continue

            any_hit = 1
            if surf < matrix_min_sid:
                continue
            if t_param < best_matrix:
                best_matrix = t_param
                hit = surf
                front = 1 if -(d0 * norm[i, 0] + d1 * norm[i, 1] + d2 * norm[i, 2]) > 0.0 else 0

        out_hit_sid[k] = hit
        out_front[k] = front if hit >= 0 else 0
        out_any_hitmask[k] = any_hit


@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_bvh_combined(
    orig,
    dirs,
    v0,
    e1,
    e2,
    norm,
    sid,
    surf_active,
    bb_min,
    bb_max,
    left,
    right,
    start,
    count,
    emit_sid,
    matrix_min_sid,
    out_hit_sid,
    out_front,
    out_any_hitmask,
):
    n_rays = orig.shape[0]
    for k in nb.prange(n_rays):
        o0 = orig[k, 0]
        o1 = orig[k, 1]
        o2 = orig[k, 2]
        d0 = dirs[k, 0]
        d1 = dirs[k, 1]
        d2 = dirs[k, 2]

        inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
        inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
        inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

        root_t = _aabb_tmin(
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
            out_hit_sid[k] = -1
            out_front[k] = 0
            out_any_hitmask[k] = 0
            continue

        stack = np.empty(STACK_SIZE, np.int32)
        tstack = np.empty(STACK_SIZE, np.float32)
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

            if count[node] > 0:
                for t in range(count[node]):
                    tri = start[node] + t
                    surf = sid[tri]
                    if surf == emit_sid or surf_active[surf] == 0:
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

                    t_param = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                    if t_param <= 1e-6:
                        continue

                    any_hit = 1
                    if surf < matrix_min_sid:
                        continue
                    if t_param < best_matrix:
                        best_matrix = t_param
                        hit = surf
                        front = 1 if -(d0 * norm[tri, 0] + d1 * norm[tri, 1] + d2 * norm[tri, 2]) > 0.0 else 0
            else:
                ln = left[node]
                rn = right[node]
                tl = _aabb_tmin(
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
                tr = _aabb_tmin(
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

        out_hit_sid[k] = hit
        out_front[k] = front if hit >= 0 else 0
        out_any_hitmask[k] = any_hit


@nb.njit(cache=True)
def reduce_first_hits(hit_sid, front_flag, out_front, out_back):
    for i in range(out_front.shape[0]):
        out_front[i] = 0
        out_back[i] = 0
    for i in range(hit_sid.shape[0]):
        hit = hit_sid[i]
        if hit < 0:
            continue
        if front_flag[i] != 0:
            out_front[hit] += 1
        else:
            out_back[hit] += 1


@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_hitmask(orig, dirs, v0, e1, e2, sid, surf_active, emit_sid, min_sid, out_hitmask):
    n_rays = orig.shape[0]
    n_tri = v0.shape[0]
    for k in nb.prange(n_rays):
        o0 = orig[k, 0]
        o1 = orig[k, 1]
        o2 = orig[k, 2]
        d0 = dirs[k, 0]
        d1 = dirs[k, 1]
        d2 = dirs[k, 2]

        out_hitmask[k] = 0
        for i in range(n_tri):
            surf = sid[i]
            if _skip_surface(surf, surf_active, emit_sid, min_sid):
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

            t_param = (e2[i, 0] * qx + e2[i, 1] * qy + e2[i, 2] * qz) * inv_det
            if t_param > 1e-6:
                out_hitmask[k] = 1
                break


@nb.njit(parallel=True, fastmath=True, cache=True)
def trace_cpu_bvh_hitmask(
    orig,
    dirs,
    v0,
    e1,
    e2,
    sid,
    surf_active,
    bb_min,
    bb_max,
    left,
    right,
    start,
    count,
    emit_sid,
    min_sid,
    out_hitmask,
):
    n_rays = orig.shape[0]
    for k in nb.prange(n_rays):
        o0 = orig[k, 0]
        o1 = orig[k, 1]
        o2 = orig[k, 2]
        d0 = dirs[k, 0]
        d1 = dirs[k, 1]
        d2 = dirs[k, 2]

        inv0 = 1.0 / d0 if abs(d0) > 1e-9 else 1e10
        inv1 = 1.0 / d1 if abs(d1) > 1e-9 else 1e10
        inv2 = 1.0 / d2 if abs(d2) > 1e-9 else 1e10

        root_t = _aabb_tmin(
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
            out_hitmask[k] = 0
            continue

        stack = np.empty(STACK_SIZE, np.int32)
        tstack = np.empty(STACK_SIZE, np.float32)
        sp = 0
        stack[sp] = 0
        tstack[sp] = root_t
        sp += 1

        hit_any = 0
        while sp > 0 and hit_any == 0:
            sp -= 1
            node = stack[sp]

            if count[node] > 0:
                for t in range(count[node]):
                    tri = start[node] + t
                    surf = sid[tri]
                    if _skip_surface(surf, surf_active, emit_sid, min_sid):
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

                    t_param = (e2[tri, 0] * qx + e2[tri, 1] * qy + e2[tri, 2] * qz) * inv_det
                    if t_param > 1e-6:
                        hit_any = 1
                        break
            else:
                ln = left[node]
                rn = right[node]
                tl = _aabb_tmin(
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
                tr = _aabb_tmin(
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

        out_hitmask[k] = hit_any


@nb.njit(inline="always", cache=True)
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

    az = math.degrees(math.atan2(dy, dx))
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


@nb.njit(cache=True)
def bin_tregenza_cpu(dirs, hitmask, counts):
    for i in range(counts.shape[0]):
        counts[i] = 0
    for i in range(dirs.shape[0]):
        if hitmask[i] != 0:
            continue
        pid = _tregenza_patch_id(dirs[i, 0], dirs[i, 1], dirs[i, 2])
        if pid >= 0:
            counts[pid] += 1


@nb.njit(cache=True)
def count_upward_misses_cpu(dirs, hitmask):
    total = 0
    for i in range(dirs.shape[0]):
        if hitmask[i] == 0 and dirs[i, 2] > 0.0:
            total += 1
    return total


__all__ = [
    "trace_cpu_firsthit",
    "trace_cpu_bvh_firsthit",
    "trace_cpu_combined",
    "trace_cpu_bvh_combined",
    "reduce_first_hits",
    "trace_cpu_hitmask",
    "trace_cpu_bvh_hitmask",
    "bin_tregenza_cpu",
    "count_upward_misses_cpu",
]
