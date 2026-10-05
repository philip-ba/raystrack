from __future__ import annotations
import numpy as np
from numba import njit

LEAF_SIZE = 8  # triangles per leaf


def _tri_bounds(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    bmin = np.minimum(np.minimum(p0, p1), p2)
    bmax = np.maximum(np.maximum(p0, p1), p2)
    centroid = (p0 + p1 + p2) / 3.0
    return bmin.astype(np.float32), bmax.astype(np.float32), centroid.astype(np.float32)


def build_bvh(v0: np.ndarray, e1: np.ndarray, e2: np.ndarray,
              leaf_size: int = LEAF_SIZE):
    """Return BVH arrays and triangle permutation."""
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive")
    m = v0.shape[0]
    if m == 0:
        empty3 = np.empty((0, 3), np.float32)
        empty1 = np.empty(0, np.int32)
        return (empty3, empty3.copy(), empty1, empty1.copy(),
                empty1.copy(), empty1.copy(), empty1.copy())
    p1, p2 = v0 + e1, v0 + e2
    tmin = np.minimum(np.minimum(v0, p1), p2).astype(np.float32, copy=False)
    tmax = np.maximum(np.maximum(v0, p1), p2).astype(np.float32, copy=False)
    cent = ((v0 + p1 + p2) / 3.0).astype(np.float32, copy=False)

    return _build_bounds(tmin, tmax, cent, leaf_size)


def _build_bounds(tmin, tmax, cent, leaf_size):
    m = len(tmin)

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


def build_aabb_bvh(lower, upper, leaf_size=2):
    """Build a BVH over boxes directly, without rounding them into triangles."""
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive")
    lower, upper = np.asarray(lower, np.float32), np.asarray(upper, np.float32)
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[1] != 3:
        raise ValueError("bounds must have matching (n, 3) shapes")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("bounds must be finite with minimum <= maximum")
    if not len(lower):
        return build_bvh(lower, lower, lower, leaf_size)
    centers = ((lower.astype(np.float64) + upper) * 0.5).astype(np.float32)
    return _build_bounds(lower, upper, centers, leaf_size)


@njit(cache=True)
def refit_aabb_bvh(lower, upper, left, right, start, count):
    """Refit a box BVH in packed leaf order, preserving exact input bounds."""
    lo, hi = np.empty((len(count), 3), np.float32), np.empty((len(count), 3), np.float32)
    for node in range(len(count) - 1, -1, -1):
        if count[node] > 0:
            for axis in range(3):
                minimum, maximum = np.inf, -np.inf
                for item in range(start[node], start[node] + count[node]):
                    minimum = min(minimum, lower[item, axis])
                    maximum = max(maximum, upper[item, axis])
                lo[node, axis], hi[node, axis] = minimum, maximum
        else:
            for axis in range(3):
                lo[node, axis] = min(lo[left[node], axis], lo[right[node], axis])
                hi[node, axis] = max(hi[left[node], axis], hi[right[node], axis])
    return lo, hi

@njit(cache=True)
def _refit_bounds(v0, e1, e2, left, right, start, count):
    n_nodes = len(left)
    bb_min = np.empty((n_nodes, 3), dtype=np.float32)
    bb_max = np.empty_like(bb_min)
    for node in range(n_nodes - 1, -1, -1):
        size = int(count[node])
        if size > 0:
            first = int(start[node])
            if first < 0 or first + size > len(v0):
                raise ValueError("BVH leaf range is outside the triangle arrays")
            for axis in range(3):
                lo, hi = np.inf, -np.inf
                for triangle in range(first, first + size):
                    a = v0[triangle, axis]
                    b = a + e1[triangle, axis]
                    c = a + e2[triangle, axis]
                    lo = min(lo, min(a, min(b, c)))
                    hi = max(hi, max(a, max(b, c)))
                bb_min[node, axis], bb_max[node, axis] = lo, hi
        else:
            l, r = int(left[node]), int(right[node])
            if not (node < l < n_nodes and node < r < n_nodes):
                raise ValueError("BVH children must follow their parent")
            for axis in range(3):
                bb_min[node, axis] = min(bb_min[l, axis], bb_min[r, axis])
                bb_max[node, axis] = max(bb_max[l, axis], bb_max[r, axis])
    return bb_min, bb_max


def refit_bvh(
    v0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    start: np.ndarray,
    count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recalculate bounds without repartitioning a tree from :func:`build_bvh`.

    Triangles must retain their packed order. Node indices from ``build_bvh``
    place parents before children, so one compiled reverse pass updates leaves
    before their ancestors. Tree topology and existing bounds are never mutated.
    Large motion can make the original partition less efficient; rebuild the
    tree when that tradeoff matters.
    """
    n_nodes = len(left)
    if not (len(right) == len(start) == len(count) == n_nodes):
        raise ValueError("BVH node arrays must have equal lengths")
    if v0.shape != e1.shape or v0.shape != e2.shape or v0.ndim != 2 or v0.shape[1] != 3:
        raise ValueError("triangle arrays must have matching (n, 3) shapes")
    return _refit_bounds(v0, e1, e2, left, right, start, count)


__all__ = ["build_bvh", "refit_bvh", "build_aabb_bvh", "refit_aabb_bvh", "LEAF_SIZE"]
