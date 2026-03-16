from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class PreparedDeviceScene:
    v0: Any
    e1: Any
    e2: Any
    normals: Any
    sid: Any
    bb_min: Optional[Any]
    bb_max: Optional[Any]
    left: Optional[Any]
    right: Optional[Any]
    start: Optional[Any]
    count: Optional[Any]
    use_bvh: bool


@dataclass(frozen=True)
class PreparedDeviceEmitter:
    u_grid: Any
    v_grid: Any
    halton_tri: Any
    halton_u: Any
    halton_v: Any
    halton_r1: Any
    halton_r2: Any
    cdf: Any
    tri_a: Any
    tri_e1: Any
    tri_e2: Any
    tri_u: Any
    tri_v: Any
    tri_n: Any


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


class PreparedSolver:
    """Cache prepared geometry, ray-generation tables and CUDA uploads.

    Reusing a single instance across repeated solves on the same mesh set
    avoids rebuilding triangle buffers, BVHs, Halton tables and device copies.
    """

    def __init__(self, meshes: List[Tuple[str, np.ndarray, np.ndarray]]):
        self.meshes = list(meshes)
        self.total_faces = int(sum(F.shape[0] for _, _, F in self.meshes))
        self._scene_cache: Dict[bool, PreparedScene] = {}
        self._emitter_cache: Dict[Tuple[int, int, bool], List[PreparedEmitter]] = {}
        self._device_scene_cache: Dict[Tuple[int, bool], PreparedDeviceScene] = {}
        self._device_emitter_cache: Dict[Tuple[int, int, int, int, bool], PreparedDeviceEmitter] = {}

    def get_scene(self, *, use_bvh: bool) -> PreparedScene:
        key = bool(use_bvh)
        scene = self._scene_cache.get(key)
        if scene is None:
            scene = prepare_scene(self.meshes, use_bvh=key)
            self._scene_cache[key] = scene
        return scene

    def get_emitters(self, *, samples: int, rays: int, flip_faces: bool) -> List[PreparedEmitter]:
        key = (int(samples), int(rays), bool(flip_faces))
        emitters = self._emitter_cache.get(key)
        if emitters is None:
            emitters = prepare_emitters(self.meshes, samples=samples, rays=rays, flip_faces=flip_faces)
            self._emitter_cache[key] = emitters
        return emitters

    def get_emitter(self, index: int, *, samples: int, rays: int, flip_faces: bool) -> PreparedEmitter:
        return self.get_emitters(samples=samples, rays=rays, flip_faces=flip_faces)[int(index)]

    def clear_device_cache(self) -> None:
        self._device_scene_cache.clear()
        self._device_emitter_cache.clear()

    def get_device_scene(self, *, use_bvh: bool) -> PreparedDeviceScene:
        from numba import cuda

        key = (int(cuda.get_current_device().id), bool(use_bvh))
        scene = self._device_scene_cache.get(key)
        if scene is None:
            host_scene = self.get_scene(use_bvh=use_bvh)
            scene = PreparedDeviceScene(
                v0=cuda.to_device(host_scene.v0),
                e1=cuda.to_device(host_scene.e1),
                e2=cuda.to_device(host_scene.e2),
                normals=cuda.to_device(host_scene.normals),
                sid=cuda.to_device(host_scene.sid),
                bb_min=None if host_scene.bb_min is None else cuda.to_device(host_scene.bb_min),
                bb_max=None if host_scene.bb_max is None else cuda.to_device(host_scene.bb_max),
                left=None if host_scene.left is None else cuda.to_device(host_scene.left),
                right=None if host_scene.right is None else cuda.to_device(host_scene.right),
                start=None if host_scene.start is None else cuda.to_device(host_scene.start),
                count=None if host_scene.count is None else cuda.to_device(host_scene.count),
                use_bvh=host_scene.use_bvh,
            )
            self._device_scene_cache[key] = scene
        return scene

    def get_device_emitter(self, index: int, *, samples: int, rays: int, flip_faces: bool) -> PreparedDeviceEmitter:
        from numba import cuda

        idx = int(index)
        key = (int(cuda.get_current_device().id), idx, int(samples), int(rays), bool(flip_faces))
        emitter = self._device_emitter_cache.get(key)
        if emitter is None:
            host_emitter = self.get_emitter(idx, samples=samples, rays=rays, flip_faces=flip_faces)
            emitter = PreparedDeviceEmitter(
                u_grid=cuda.to_device(host_emitter.u_grid),
                v_grid=cuda.to_device(host_emitter.v_grid),
                halton_tri=cuda.to_device(host_emitter.halton_tri),
                halton_u=cuda.to_device(host_emitter.halton_u),
                halton_v=cuda.to_device(host_emitter.halton_v),
                halton_r1=cuda.to_device(host_emitter.halton_r1),
                halton_r2=cuda.to_device(host_emitter.halton_r2),
                cdf=cuda.to_device(host_emitter.cdf),
                tri_a=cuda.to_device(host_emitter.tri_a),
                tri_e1=cuda.to_device(host_emitter.tri_e1),
                tri_e2=cuda.to_device(host_emitter.tri_e2),
                tri_u=cuda.to_device(host_emitter.tri_u),
                tri_v=cuda.to_device(host_emitter.tri_v),
                tri_n=cuda.to_device(host_emitter.tri_n),
            )
            self._device_emitter_cache[key] = emitter
        return emitter


__all__ = [
    "PreparedScene",
    "PreparedEmitter",
    "PreparedDeviceScene",
    "PreparedDeviceEmitter",
    "PreparedSolver",
    "prepare_scene",
    "prepare_emitters",
]
