from __future__ import annotations
import numpy as np
import numba as nb
from numba import cuda

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
