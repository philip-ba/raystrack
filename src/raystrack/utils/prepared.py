from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .bvh import build_bvh
from .halton import cached_halton, cached_halton_dims
from .helpers import grid_from_density


@dataclass(frozen=True)
class PreparedScene:
    v0: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    normals: np.ndarray
    sid: np.ndarray
    bb_min: Optional[np.ndarray]
    bb_max: Optional[np.ndarray]
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    start: Optional[np.ndarray]
    count: Optional[np.ndarray]
    use_bvh: bool


@dataclass(frozen=True)
class PreparedEmitter:
    tri_a: np.ndarray
    tri_e1: np.ndarray
    tri_e2: np.ndarray
    tri_u: np.ndarray
    tri_v: np.ndarray
    tri_n: np.ndarray
    cdf: np.ndarray
    total_area: float
    g: int
    u_grid: np.ndarray
    v_grid: np.ndarray
    halton_tri: np.ndarray
    halton_u: np.ndarray
    halton_v: np.ndarray
    halton_r1: np.ndarray
    halton_r2: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.u_grid.shape[0])


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def _triangle_frames(tri_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri_u = np.empty_like(tri_n, dtype=np.float32)
    tri_v = np.empty_like(tri_n, dtype=np.float32)

    axis_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    axis_y = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(tri_n.shape[0]):
        n = tri_n[i]
        ref = axis_x if abs(float(n[0])) < 0.9 else axis_y
        u = np.cross(ref, n).astype(np.float32)
        u_len = float(np.linalg.norm(u))
        if u_len <= 1e-12:
            ref = axis_y if ref is axis_x else axis_x
            u = np.cross(ref, n).astype(np.float32)
            u_len = float(np.linalg.norm(u))
        if u_len <= 1e-12:
            tri_u[i] = axis_x
            tri_v[i] = axis_y
            continue
        u /= u_len
        tri_u[i] = u
        tri_v[i] = np.cross(n, u).astype(np.float32)
    return tri_u, tri_v


def prepare_scene(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    use_bvh: bool,
) -> PreparedScene:
    v0s = []
    e1s = []
    e2s = []
    normals = []
    sids = []

    for sid, (_, V, F) in enumerate(meshes):
        tri_a = np.asarray(V[F[:, 0]], dtype=np.float32)
        tri_b = np.asarray(V[F[:, 1]], dtype=np.float32)
        tri_c = np.asarray(V[F[:, 2]], dtype=np.float32)

        tri_e1 = (tri_b - tri_a).astype(np.float32, copy=False)
        tri_e2 = (tri_c - tri_a).astype(np.float32, copy=False)
        tri_n = np.cross(tri_e1, tri_e2).astype(np.float32, copy=False)
        tri_n = _safe_normalize(tri_n).astype(np.float32, copy=False)

        v0s.append(tri_a)
        e1s.append(tri_e1)
        e2s.append(tri_e2)
        normals.append(tri_n)
        sids.append(np.full(F.shape[0], sid, dtype=np.int32))

    if not v0s:
        empty3 = np.empty((0, 3), dtype=np.float32)
        empty1 = np.empty((0,), dtype=np.int32)
        return PreparedScene(
            empty3,
            empty3,
            empty3,
            empty3,
            empty1,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )

    v0 = np.concatenate(v0s, axis=0)
    e1 = np.concatenate(e1s, axis=0)
    e2 = np.concatenate(e2s, axis=0)
    tri_n = np.concatenate(normals, axis=0)
    sid = np.concatenate(sids, axis=0)

    bb_min = bb_max = left = right = start = count = None
    if use_bvh and v0.shape[0] > 0:
        bb_min, bb_max, left, right, start, count, perm = build_bvh(v0, e1, e2)
        v0 = v0[perm]
        e1 = e1[perm]
        e2 = e2[perm]
        tri_n = tri_n[perm]
        sid = sid[perm]

    return PreparedScene(
        v0=v0,
        e1=e1,
        e2=e2,
        normals=tri_n,
        sid=sid,
        bb_min=bb_min,
        bb_max=bb_max,
        left=left,
        right=right,
        start=start,
        count=count,
        use_bvh=bool(use_bvh and v0.shape[0] > 0),
    )


def prepare_emitters(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    samples: int,
    rays: int,
    flip_faces: bool,
) -> List[PreparedEmitter]:
    emitters: List[PreparedEmitter] = []
    for _, V, F in meshes:
        F_emit = F[:, [0, 2, 1]] if flip_faces else F
        tri_a = np.asarray(V[F_emit[:, 0]], dtype=np.float32)
        tri_b = np.asarray(V[F_emit[:, 1]], dtype=np.float32)
        tri_c = np.asarray(V[F_emit[:, 2]], dtype=np.float32)

        tri_e1 = (tri_b - tri_a).astype(np.float32, copy=False)
        tri_e2 = (tri_c - tri_a).astype(np.float32, copy=False)

        tri_n_raw = np.cross(tri_e1, tri_e2).astype(np.float32, copy=False)
        twice_area = np.linalg.norm(tri_n_raw, axis=1)
        tri_n = _safe_normalize(tri_n_raw).astype(np.float32, copy=False)
        tri_u, tri_v = _triangle_frames(tri_n)

        areas = 0.5 * twice_area
        total_area = float(areas.sum())
        if total_area <= 0.0:
            cdf = np.ones(F_emit.shape[0], dtype=np.float32)
            g = 4
            u_grid = np.zeros(g * g, dtype=np.float32)
            v_grid = np.zeros_like(u_grid)
            halton_tri = np.zeros(g * g * rays, dtype=np.float32)
            halton_u = np.zeros_like(halton_tri)
            halton_v = np.zeros_like(halton_tri)
            halton_r1 = np.zeros_like(halton_tri)
            halton_r2 = np.zeros_like(halton_tri)
        else:
            cdf = np.cumsum(areas, dtype=np.float64)
            cdf = (cdf / cdf[-1]).astype(np.float32)
            g = grid_from_density(total_area, samples)
            u_grid, v_grid = cached_halton(g)
            n_rays_once = g * g * rays
            halton_tri, halton_u, halton_v, halton_r1, halton_r2 = cached_halton_dims(n_rays_once)

        emitters.append(
            PreparedEmitter(
                tri_a=tri_a,
                tri_e1=tri_e1,
                tri_e2=tri_e2,
                tri_u=tri_u.astype(np.float32, copy=False),
                tri_v=tri_v.astype(np.float32, copy=False),
                tri_n=tri_n,
                cdf=cdf,
                total_area=total_area,
                g=g,
                u_grid=u_grid,
                v_grid=v_grid,
                halton_tri=halton_tri,
                halton_u=halton_u,
                halton_v=halton_v,
                halton_r1=halton_r1,
                halton_r2=halton_r2,
            )
        )
    return emitters


__all__ = ["PreparedScene", "PreparedEmitter", "prepare_scene", "prepare_emitters"]
