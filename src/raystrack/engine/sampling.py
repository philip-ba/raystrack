"""Shared sampling and visibility rules preserved from the v1 baseline."""
from __future__ import annotations
import numpy as np
from ..utils.prepared import PreparedEmitter

_BVH_AUTO_THRESHOLD = 512

def _select_bvh(bvh: str | None, total_faces: int) -> bool:
    mode = (bvh or "auto").lower()
    if mode not in ("auto", "off", "builtin"):
        raise ValueError(f"bvh must be 'auto', 'off', or 'builtin' (got {bvh!r})")
    if mode == "builtin":
        return True
    if mode == "off":
        return False
    return total_faces >= _BVH_AUTO_THRESHOLD


def _build_emitter_surface_mask(
    idx_emit: int,
    emitter: PreparedEmitter,
    bounds_center: np.ndarray,
    bounds_extent: np.ndarray,
) -> np.ndarray:
    n_surf = int(bounds_center.shape[0])
    surf_active = np.ones(n_surf, dtype=np.uint8)
    if not emitter.plane_is_planar:
        # A nonplanar surface may see or occlude itself. Rays start slightly
        # off their emitting triangle, so the entire mesh must remain visible.
        return surf_active

    # A planar surface cannot see itself; keep that inexpensive rejection.
    if 0 <= idx_emit < n_surf:
        surf_active[idx_emit] = 0

    plane_origin = emitter.plane_origin
    plane_normal = emitter.plane_normal
    plane_tol = float(emitter.plane_tol)
    abs_n = np.abs(plane_normal)

    for j in range(n_surf):
        if j == idx_emit:
            continue
        dx = bounds_center[j, 0] - plane_origin[0]
        dy = bounds_center[j, 1] - plane_origin[1]
        dz = bounds_center[j, 2] - plane_origin[2]
        signed = (
            dx * plane_normal[0]
            + dy * plane_normal[1]
            + dz * plane_normal[2]
        )
        radius = (
            abs_n[0] * bounds_extent[j, 0]
            + abs_n[1] * bounds_extent[j, 1]
            + abs_n[2] * bounds_extent[j, 2]
        )
        if signed + radius <= plane_tol:
            surf_active[j] = 0
    return surf_active


def _convergence_checkpoint(iters_done: int, *, min_iters: int, interval: int, max_iters: int,
                            needs_variance: bool = False, total_rays: int = 0,
                            min_total_rays: int = 0) -> bool:
    if iters_done < max(1, int(min_iters)):
        return False
    if total_rays < min_total_rays:
        return False
    if needs_variance and iters_done <= 1:
        return False
    if iters_done >= int(max_iters):
        return True
    span = max(1, int(interval))
    if span <= 1:
        return True
    start = max(1, int(min_iters))
    return ((iters_done - start) % span) == 0
