from __future__ import annotations
import numpy as np

LEAF_SIZE = 8  # triangles per leaf


def _tri_bounds(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    bmin = np.minimum(np.minimum(p0, p1), p2)
    bmax = np.maximum(np.maximum(p0, p1), p2)
    centroid = (p0 + p1 + p2) / 3.0
    return bmin.astype(np.float32), bmax.astype(np.float32), centroid.astype(np.float32)


def build_bvh(v0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
              leaf_size: int = LEAF_SIZE):
    """Return BVH arrays and triangle permutation."""
    m = v0.shape[0]
    tmin = np.empty((m, 3), np.float32)
    tmax = np.empty((m, 3), np.float32)
    cent = np.empty((m, 3), np.float32)
    for i in range(m):
        p0 = v0[i]
        p1 = v0[i] + e1[i]
        p2 = v0[i] + e2[i]
        bmin, bmax, c = _tri_bounds(p0, p1, p2)
        tmin[i] = bmin
        tmax[i] = bmax
        cent[i] = c

    bb_min, bb_max = [], []
    left, right = [], []
    start, count = [], []
    order: list[int] = []

    def add_node(idxs: np.ndarray) -> int:
        node = len(bb_min)
        bb_min.append(np.zeros(3, np.float32))
        bb_max.append(np.zeros(3, np.float32))
        left.append(-1)
        right.append(-1)
        start.append(0)
        count.append(0)

        bmin = tmin[idxs].min(axis=0)
        bmax = tmax[idxs].max(axis=0)
        bb_min[node][:] = bmin
        bb_max[node][:] = bmax

        if len(idxs) <= leaf_size:
            start[node] = len(order)
            count[node] = len(idxs)
            order.extend(idxs.tolist())
        else:
            ext = bmax - bmin
            axis = int(np.argmax(ext))
            idxs_sorted = idxs[np.argsort(cent[idxs, axis])]
            mid = idxs_sorted.size // 2
            l = add_node(idxs_sorted[:mid])
            r = add_node(idxs_sorted[mid:])
            left[node] = l
            right[node] = r
        return node

    add_node(np.arange(m, dtype=np.int32))
    perm = np.array(order, dtype=np.int32)
    return (np.asarray(bb_min, np.float32),
            np.asarray(bb_max, np.float32),
            np.asarray(left, np.int32),
            np.asarray(right, np.int32),
            np.asarray(start, np.int32),
            np.asarray(count, np.int32),
            perm)

__all__ = ["build_bvh", "LEAF_SIZE"]
