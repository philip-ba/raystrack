from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bvh import build_bvh, refit_bvh
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
    # Packed triangle index -> triangle index in concatenated mesh face order.
    permutation: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PreparedEmitter:
    tri_a: np.ndarray
    tri_e1: np.ndarray
    tri_e2: np.ndarray
    tri_u: np.ndarray
    tri_v: np.ndarray
    tri_n: np.ndarray
    tri_origin_eps: np.ndarray
    plane_origin: np.ndarray
    plane_normal: np.ndarray
    plane_tol: float
    plane_is_planar: bool
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
    tri_origin_eps: Any


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def _triangle_frames(tri_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Batch the same canonical reference-axis rule used by the scalar builder.
    # This avoids two Python np.cross calls and a norm call per triangle.
    refs = np.zeros_like(tri_n, dtype=np.float32)
    # The scalar rule converted nx to Python float before comparing 0.9;
    # retain that threshold for float32 normals exactly on its rounded value.
    use_x = np.abs(tri_n[:, 0].astype(np.float64)) < 0.9
    refs[use_x, 0] = 1
    refs[~use_x, 1] = 1
    tri_u = np.cross(refs, tri_n).astype(np.float32)
    lengths = np.linalg.norm(tri_u, axis=1)
    retry = lengths <= 1e-12
    if np.any(retry):
        refs[retry] = refs[retry][:, [1, 0, 2]]
        tri_u[retry] = np.cross(refs[retry], tri_n[retry]).astype(np.float32)
        lengths[retry] = np.linalg.norm(tri_u[retry], axis=1)
    degenerate = lengths <= 1e-12
    valid = ~degenerate
    tri_u[valid] /= lengths[valid, None]
    tri_v = np.cross(tri_n, tri_u).astype(np.float32)
    tri_u[degenerate] = [1, 0, 0]
    tri_v[degenerate] = [0, 1, 0]
    return tri_u, tri_v


def _triangle_origin_eps(tri_e1: np.ndarray, tri_e2: np.ndarray) -> np.ndarray:
    edge_a = np.linalg.norm(tri_e1, axis=1)
    edge_b = np.linalg.norm(tri_e2, axis=1)
    edge_c = np.linalg.norm(tri_e2 - tri_e1, axis=1)
    scale = np.maximum(edge_a, np.maximum(edge_b, edge_c))
    return np.maximum(scale * 1.0e-6, 1.0e-8).astype(np.float32, copy=False)


def _emitter_plane(
    tri_a: np.ndarray,
    tri_e1: np.ndarray,
    tri_e2: np.ndarray,
    tri_n: np.ndarray,
    tri_origin_eps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    plane_origin = np.zeros(3, dtype=np.float32)
    plane_normal = np.zeros(3, dtype=np.float32)
    plane_tol = float(max(1.0e-7, np.max(tri_origin_eps) if tri_origin_eps.size else 0.0))

    if tri_a.shape[0] == 0:
        return plane_origin, plane_normal, plane_tol, False

    plane_origin = np.asarray(tri_a[0], dtype=np.float32)
    plane_normal = np.asarray(tri_n[0], dtype=np.float32)
    normal_len = float(np.linalg.norm(plane_normal))
    if normal_len <= 1.0e-12:
        return plane_origin, plane_normal, plane_tol, False

    plane_normal = (plane_normal / normal_len).astype(np.float32, copy=False)
    normal_align = tri_n @ plane_normal
    if np.any(normal_align < (1.0 - 1.0e-4)):
        return plane_origin, plane_normal, plane_tol, False

    offsets = (
        np.abs((tri_a - plane_origin) @ plane_normal),
        np.abs((tri_a + tri_e1 - plane_origin) @ plane_normal),
        np.abs((tri_a + tri_e2 - plane_origin) @ plane_normal),
    )
    max_dev = max(float(np.max(arr)) if arr.size else 0.0 for arr in offsets)
    if max_dev > plane_tol:
        return plane_origin, plane_normal, plane_tol, False

    return plane_origin, plane_normal, plane_tol, True


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
    perm = np.arange(v0.shape[0], dtype=np.int32)
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
        permutation=perm,
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
        tri_origin_eps = _triangle_origin_eps(tri_e1, tri_e2)
        plane_origin, plane_normal, plane_tol, plane_is_planar = _emitter_plane(
            tri_a,
            tri_e1,
            tri_e2,
            tri_n,
            tri_origin_eps,
        )

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
            u_grid, v_grid = _immutable_halton_grid(g)
            n_rays_once = g * g * rays
            halton_tri, halton_u, halton_v, halton_r1, halton_r2 = _immutable_halton_dimensions(n_rays_once)

        emitters.append(
            PreparedEmitter(
                tri_a=tri_a,
                tri_e1=tri_e1,
                tri_e2=tri_e2,
                tri_u=tri_u.astype(np.float32, copy=False),
                tri_v=tri_v.astype(np.float32, copy=False),
                tri_n=tri_n,
                tri_origin_eps=tri_origin_eps,
                plane_origin=plane_origin,
                plane_normal=plane_normal,
                plane_tol=plane_tol,
                plane_is_planar=plane_is_planar,
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


def _owned_readonly(array: np.ndarray) -> np.ndarray:
    """Hide a read-only owned allocation behind a read-only view."""
    array.setflags(write=False)
    view = array.view()
    view.setflags(write=False)
    return view


@lru_cache(maxsize=128)
def _immutable_halton_grid(g: int):
    return tuple(_owned_readonly(array) for array in cached_halton(g))


@lru_cache(maxsize=128)
def _immutable_halton_dimensions(n: int):
    return tuple(_owned_readonly(array) for array in cached_halton_dims(n))


def _readonly_prepared(prepared):
    return type(prepared)(**{
        field.name: (_owned_readonly(value) if isinstance(value, np.ndarray) and value.flags.writeable else value)
        for field in fields(prepared)
        for value in [getattr(prepared, field.name)]
    })


def _checked_vertices(vertices: np.ndarray) -> np.ndarray:
    v = np.array(vertices, dtype=np.float32, order="C", copy=True)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if not np.isfinite(v).all():
        raise ValueError("vertices must be finite")
    return _owned_readonly(v)


def _checked_geometry(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = _checked_vertices(vertices)
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (n, 3)")
    if f.dtype.kind not in "iu":
        raise ValueError("faces must contain integer vertex indices")
    if f.size and (np.min(f) < 0 or np.max(f) >= len(v)):
        raise ValueError("face vertex index is out of bounds")
    f = np.array(f, dtype=np.int32, order="C", copy=True)
    return v, _owned_readonly(f)


def _checked_transform(matrix):
    transform = np.array(matrix, dtype=np.float64, copy=True)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-7)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("transform must be a proper rigid rotation and translation (no scale or shear)")
    if not np.isfinite(transform.astype(np.float32)).all():
        raise ValueError("transform must fit finite float32 coordinates")
    return transform


def _local_bounds(vertices):
    if not len(vertices):
        return np.zeros(3, np.float64), np.zeros(3, np.float64)
    lower, upper = vertices.min(axis=0).astype(np.float64), vertices.max(axis=0).astype(np.float64)
    return (lower + upper) * 0.5, (upper - lower) * 0.5


def _check_transformed_bounds(bounds, transform):
    center, extent = bounds
    # Leave one float32 step for the outward rounding of instance AABBs.
    limit = float(np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0)))
    for rotation in (transform[:3, :3], transform[:3, :3].astype(np.float32).astype(np.float64)):
        world_center = rotation @ center + transform[:3, 3]
        world_extent = np.abs(rotation) @ extent
        if np.any(np.abs(world_center) + world_extent > limit):
            raise ValueError("transformed instance bounds must fit finite float32 coordinates")


def _mesh_triangles(vertices: np.ndarray, faces: np.ndarray):
    a = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - a
    e2 = vertices[faces[:, 2]] - a
    n = _safe_normalize(np.cross(e1, e2)).astype(np.float32, copy=False)
    return a, e1, e2, n


class PreparedSolver:
    """Own a dynamic scene and cache geometry, sampling tables and CUDA uploads.

    Input arrays are copied. ``meshes`` exposes read-only current geometry;
    change it using ``update_transform``, ``update_vertices`` or ``update_mesh``.
    Pass ``prepared.meshes`` to solves after updates. External mesh lists must
    match the owned geometry, which ``validate_meshes`` checks explicitly.

    Rigid transforms are absolute relative to the vertices supplied at
    construction or the last ``update_vertices``/``update_mesh`` call. Stable
    face topology refits the existing BVH; changed faces rebuild it lazily.
    Opt into ``acceleration='instanced'`` for per-object local BVHs and a
    transform-only top-level refit. ``from_instances`` shares repeated prototype
    meshes. Current world meshes and emitter triangles are materialized lazily;
    traversal uses unchanged local geometry through rigid inverse transforms.
    ``version`` starts at zero and increments after every successful update.

    ``solve_lock`` serializes updates and solves sharing this instance. Hold it
    around a complete custom solve: refreshed CUDA allocations are reused and
    overwritten. Prepared CPU scene snapshots survive subsequent updates.
    """

    def __init__(self, meshes: List[Tuple[str, np.ndarray, np.ndarray]], *,
                 acceleration: str = "flat", transforms=None):
        if acceleration not in ("flat", "instanced"):
            raise ValueError("acceleration must be 'flat' or 'instanced'")
        if transforms is not None and acceleration != "instanced":
            raise ValueError("transforms require acceleration='instanced'")
        self.acceleration = acceleration
        self.solve_lock = RLock()
        self._meshes: List[Tuple[str, np.ndarray, np.ndarray]] = []
        names = set()
        geometry_copies = {}
        for name, vertices, faces in meshes:
            if not isinstance(name, str):
                raise TypeError("mesh names must be strings")
            if name in names:
                raise ValueError("mesh names must be unique")
            names.add(name)
            identity = (id(vertices), id(faces))
            if identity not in geometry_copies:
                v, f = _checked_geometry(vertices, faces)
                # Keep original references until construction ends, preventing
                # identity reuse when meshes arrive from a generator.
                geometry_copies[identity] = (vertices, faces, v, f)
            _, _, v, f = geometry_copies[identity]
            self._meshes.append((name, v, f))
        self._local_vertices = [v for _, v, _ in self._meshes]
        self._local_bounds = [_local_bounds(v) for v in self._local_vertices]
        n_instances = len(self._meshes)
        self._instance_transforms = np.tile(np.eye(4), (n_instances, 1, 1))
        if transforms is not None:
            if len(transforms) != n_instances:
                raise ValueError("one transform is required per instance")
            for idx, matrix in enumerate(transforms):
                self._instance_transforms[idx] = _checked_transform(matrix)
                _check_transformed_bounds(self._local_bounds[idx], self._instance_transforms[idx])
        self._world_dirty = {idx for idx in range(n_instances)
                             if not np.array_equal(self._instance_transforms[idx], np.eye(4))}
        self._instancing = None
        self._device_instanced_cache = {}
        self._device_instanced_hosts = {}
        self._version = 0
        self._mesh_versions = [0] * len(self._meshes)
        self.total_faces = int(sum(len(f) for _, _, f in self._meshes))
        self._scene_cache: Dict[bool, PreparedScene] = {}
        self._emitter_cache: Dict[Tuple[int, int, bool], List[PreparedEmitter]] = {}
        self._emitter_dirty: Dict[Tuple[int, int, bool], set[int]] = {}
        self._device_scene_cache: Dict[Tuple[int, int, bool], PreparedDeviceScene] = {}
        self._device_scene_hosts: Dict[Tuple[int, int, bool], PreparedScene] = {}
        self._device_scene_versions: Dict[Tuple[int, int, bool], int] = {}
        self._device_scene_dirty: Dict[Tuple[int, int, bool], set[int]] = {}
        self._device_emitter_cache: Dict[Tuple[int, int, int, int, int, bool], PreparedDeviceEmitter] = {}
        self._device_emitter_hosts: Dict[Tuple[int, int, int, int, int, bool], PreparedEmitter] = {}
        self._device_emitter_versions: Dict[Tuple[int, int, int, int, int, bool], int] = {}
        self._device_contexts: Dict[Tuple[int, int], Any] = {}
        self._mesh_bounds_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @classmethod
    def from_instances(cls, geometries, instances):
        """Create shared rigid instances of named/indexed local mesh prototypes.

        ``geometries`` uses ordinary ``(name, vertices, faces)`` tuples.
        ``instances`` contains ``(instance_name, geometry_name_or_index, matrix)``.
        Each instance is a distinct emitting/receiving surface. Repeated local
        geometry shares one owned mesh allocation and one BLAS.
        """
        geometry_list = list(geometries)
        geometry_names = [mesh[0] for mesh in geometry_list]
        if len(set(geometry_names)) != len(geometry_names):
            raise ValueError("geometry names must be unique")
        meshes, transforms = [], []
        for name, source, transform in instances:
            if isinstance(source, str):
                if source not in geometry_names:
                    raise KeyError(f"unknown geometry {source!r}")
                idx = geometry_names.index(source)
            elif isinstance(source, (int, np.integer)) and not isinstance(source, (bool, np.bool_)):
                idx = int(source)
                if not 0 <= idx < len(geometry_list):
                    raise IndexError("geometry index is out of bounds")
            else:
                raise TypeError("geometry must be a name or nonnegative integer index")
            _, vertices, faces = geometry_list[idx]
            meshes.append((name, vertices, faces))
            transforms.append(transform)
        return cls(meshes, acceleration="instanced", transforms=transforms)

    def _materialize_world(self):
        for idx in self._world_dirty:
            transform = self._instance_transforms[idx]
            name, _, faces = self._meshes[idx]
            world = self._local_vertices[idx].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
            self._meshes[idx] = (name, _checked_vertices(world), faces)
        self._world_dirty.clear()

    def get_instanced_scene(self):
        """Return shared local BLAS and a TLAS over rigid instance bounds."""
        if self.acceleration != "instanced":
            raise ValueError("get_instanced_scene requires acceleration='instanced'")
        with self.solve_lock:
            if self._instancing is None:
                from .instancing import InstancedAcceleration
                local = [(name, self._local_vertices[idx], faces)
                         for idx, (name, _, faces) in enumerate(self._meshes)]
                self._instancing = InstancedAcceleration(local, self._instance_transforms)
                if self.version:
                    self._instancing.update_transforms(self._instance_transforms, revision=self.version)
            return self._instancing.scene

    @property
    def meshes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Read-only arrays in mesh order; editing the returned list has no effect."""
        with self.solve_lock:
            self._materialize_world()
            return list(self._meshes)

    @property
    def version(self) -> int:
        return self._version

    def validate_meshes(self, meshes: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
        """Reject mismatches that would otherwise silently use stale geometry.

        Passing ``self.meshes`` takes an identity fast path. Other arrays are
        compared at the float32 precision used by the tracing kernels.
        """
        with self.solve_lock:
            self._materialize_world()
            if len(meshes) != len(self._meshes):
                raise ValueError("meshes do not match prepared geometry; pass prepared.meshes")
            for idx, (supplied, owned) in enumerate(zip(meshes, self._meshes)):
                name, vertices, faces = supplied
                expected_name, expected_v, expected_f = owned
                same_v = vertices is expected_v or np.array_equal(
                    np.asarray(vertices, dtype=np.float32), expected_v)
                same_f = faces is expected_f or np.array_equal(np.asarray(faces), expected_f)
                if name != expected_name or not same_v or not same_f:
                    raise ValueError(
                        f"mesh {idx} does not match prepared geometry; update the prepared "
                        "scene explicitly and pass prepared.meshes")

    def _mesh_index(self, mesh: str | int) -> int:
        if isinstance(mesh, str):
            indices = [i for i, (name, _, _) in enumerate(self._meshes) if name == mesh]
            if not indices:
                raise KeyError(f"unknown mesh {mesh!r}")
            if len(indices) != 1:
                raise ValueError(f"mesh name {mesh!r} is ambiguous; use its index")
            return indices[0]
        if isinstance(mesh, (int, np.integer)) and not isinstance(mesh, (bool, np.bool_)):
            index = int(mesh)
            if 0 <= index < len(self._meshes):
                return index
            raise IndexError("mesh index is out of bounds")
        raise TypeError("mesh must be a name or a nonnegative integer index")

    def update_transform(self, mesh: str | int, matrix: np.ndarray) -> int:
        """Apply an absolute proper rigid 4x4 transform; return the new version.

        Column-vector convention: ``world = R @ local + translation``.
        Scale, shear, reflection, projective and non-finite transforms fail
        before changing any geometry or caches.
        """
        transform = _checked_transform(matrix)
        rotation = transform[:3, :3]
        with self.solve_lock:
            idx = self._mesh_index(mesh)
            if self.acceleration == "instanced":
                _check_transformed_bounds(self._local_bounds[idx], transform)
                transforms = self._instance_transforms.copy()
                transforms[idx] = transform
                # Refresh acceleration before committing the public version.
                # BLAS geometry is untouched and world triangles stay lazy.
                if self._instancing is not None:
                    self._instancing.update_transforms(transforms, revision=self.version + 1)
                self._instance_transforms = transforms
                self._world_dirty.add(idx)
                self._invalidate_instance(idx)
                return self.version
            vertices = self._local_vertices[idx].astype(np.float64) @ rotation.T + transform[:3, 3]
            v = _checked_vertices(vertices)
            self._apply_update(idx, v, self._meshes[idx][2])
            return self.version

    def update_vertices(self, mesh: str | int, vertices: np.ndarray) -> int:
        """Replace world vertices, retain faces and reset the local transform basis."""
        return self.update_mesh(mesh, vertices)

    def update_mesh(self, mesh: str | int, vertices: np.ndarray, faces: Optional[np.ndarray] = None) -> int:
        """Replace one mesh, optionally changing faces; return the new version.

        Faces omitted or unchanged keep the existing tree partition. Supplying
        different face indices or counts invalidates it for a fresh build.
        Input arrays are copied and become the new local transform basis.
        """
        with self.solve_lock:
            idx = self._mesh_index(mesh)
            old_faces = self._meshes[idx][2]
            v, f = _checked_geometry(vertices, old_faces if faces is None else faces)
            if np.array_equal(f, old_faces):
                f = old_faces
            if self.acceleration == "instanced":
                bounds = _local_bounds(v)
                _check_transformed_bounds(bounds, np.eye(4))
                transforms = self._instance_transforms.copy()
                transforms[idx] = np.eye(4)
                if self._instancing is not None:
                    self._instancing.update_geometry(idx, v, f, transforms,
                        revision=self.version + 1, same_faces=f is old_faces)
                self._instance_transforms = transforms
                self._local_vertices[idx] = v
                self._local_bounds[idx] = bounds
                self._meshes[idx] = (self._meshes[idx][0], v, f)
                self._world_dirty.discard(idx)
                self.total_faces = int(sum(len(faces) for _, _, faces in self._meshes))
                self._invalidate_instance(idx)
                return self.version
            self._apply_update(idx, v, f)
            self._local_vertices[idx] = v
            self._local_bounds[idx] = _local_bounds(v)
            return self.version

    def rebuild_bvh(self) -> int:
        """Repartition the current scene after large motion; return its version.

        Refitting preserves correctness but can reduce traversal efficiency as
        objects leave their original partitions. This explicit rebuild restores
        a partition based on current geometry, while keeping emitter tables and
        compatible device allocations. It increments ``version`` to invalidate
        any queued work for the previous acceleration structure.
        """
        with self.solve_lock:
            if self.acceleration == "instanced":
                self.get_instanced_scene()
                self._instancing.update_transforms(self._instance_transforms,
                    revision=self.version + 1, force_rebuild=True)
                self._version += 1
                return self.version
            self._materialize_world()
            scene = _readonly_prepared(prepare_scene(self._meshes, use_bvh=True))
            self._scene_cache[True] = scene
            self._version += 1
            return self.version

    def _invalidate_instance(self, idx):
        self._scene_cache.clear()
        self._mesh_bounds_cache = None
        for key in self._emitter_cache:
            self._emitter_dirty.setdefault(key, set()).add(idx)
        for key in self._device_scene_cache:
            self._device_scene_dirty.setdefault(key, set()).add(idx)
        self._version += 1
        self._mesh_versions[idx] += 1

    def _apply_update(self, idx: int, vertices: np.ndarray, faces: np.ndarray) -> None:
        name, _, old_faces = self._meshes[idx]
        stable_topology = faces is old_faces
        # Build replacement snapshots first. Existing snapshots remain valid if
        # a caller retains them, and an invalid update cannot partly mutate one.
        updated_scenes = {}
        if stable_topology:
            offset = sum(len(f) for _, _, f in self._meshes[:idx])
            a, edge1, edge2, normal = _mesh_triangles(vertices, faces)
            for key, scene in self._scene_cache.items():
                perm = scene.permutation
                if perm is None:  # Only the entirely empty scene has no mapping.
                    updated_scenes[key] = scene
                    continue
                packed = np.flatnonzero((perm >= offset) & (perm < offset + len(faces)))
                local = perm[packed] - offset
                v0, e1, e2, normals = (array.copy() for array in
                                        (scene.v0, scene.e1, scene.e2, scene.normals))
                v0[packed], e1[packed], e2[packed], normals[packed] = (
                    a[local], edge1[local], edge2[local], normal[local])
                bb_min, bb_max = scene.bb_min, scene.bb_max
                if scene.use_bvh:
                    bb_min, bb_max = refit_bvh(
                        v0, e1, e2, scene.left, scene.right, scene.start, scene.count)
                updated_scenes[key] = _readonly_prepared(PreparedScene(
                    v0, e1, e2, normals, scene.sid, bb_min, bb_max,
                    scene.left, scene.right, scene.start, scene.count,
                    scene.use_bvh, perm))
        self._meshes[idx] = (name, vertices, faces)
        self._scene_cache = updated_scenes
        self.total_faces = int(sum(len(f) for _, _, f in self._meshes))
        self._mesh_bounds_cache = None
        for key in self._emitter_cache:
            self._emitter_dirty.setdefault(key, set()).add(idx)
        for key in self._device_scene_cache:
            self._device_scene_dirty.setdefault(key, set()).add(idx)
        self._version += 1
        self._mesh_versions[idx] += 1

    def get_scene(self, *, use_bvh: bool) -> PreparedScene:
        with self.solve_lock:
            self._materialize_world()
            key = bool(use_bvh)
            scene = self._scene_cache.get(key)
            if scene is None:
                scene = _readonly_prepared(prepare_scene(self._meshes, use_bvh=key))
                self._scene_cache[key] = scene
            return scene

    def get_emitters(self, *, samples: int, rays: int, flip_faces: bool) -> List[PreparedEmitter]:
        with self.solve_lock:
            self._materialize_world()
            key = (int(samples), int(rays), bool(flip_faces))
            emitters = self._emitter_cache.get(key)
            if emitters is None:
                emitters = [_readonly_prepared(emitter) for emitter in
                            prepare_emitters(self._meshes, samples=samples, rays=rays, flip_faces=flip_faces)]
                self._emitter_cache[key] = emitters
            dirty = self._emitter_dirty.pop(key, set())
            if dirty:
                emitters = list(emitters)
                for idx in dirty:
                    emitters[idx] = _readonly_prepared(prepare_emitters(
                        [self._meshes[idx]], samples=samples, rays=rays, flip_faces=flip_faces)[0])
                self._emitter_cache[key] = emitters
            return list(emitters)

    def get_emitter(self, index: int, *, samples: int, rays: int, flip_faces: bool) -> PreparedEmitter:
        return self.get_emitters(samples=samples, rays=rays, flip_faces=flip_faces)[int(index)]

    def get_mesh_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.solve_lock:
            bounds = self._mesh_bounds_cache
            if bounds is None:
                if self.acceleration == "instanced":
                    scene = self.get_instanced_scene()
                    lower, upper = scene.object_min.astype(np.float64), scene.object_max.astype(np.float64)
                    bounds = (_owned_readonly(((lower + upper) * 0.5).astype(np.float32)),
                              _owned_readonly(((upper - lower) * 0.5).astype(np.float32)))
                    self._mesh_bounds_cache = bounds
                    return bounds
                n_mesh = len(self._meshes)
                centers = np.zeros((n_mesh, 3), dtype=np.float32)
                extents = np.zeros((n_mesh, 3), dtype=np.float32)
                for idx, (_, v, _) in enumerate(self._meshes):
                    if v.size == 0:
                        continue
                    vmin = np.min(v, axis=0)
                    vmax = np.max(v, axis=0)
                    centers[idx] = 0.5 * (vmin.astype(np.float64) + vmax)
                    extents[idx] = 0.5 * (vmax.astype(np.float64) - vmin)
                bounds = (_owned_readonly(centers), _owned_readonly(extents))
                self._mesh_bounds_cache = bounds
            return bounds

    def clear_device_cache(self) -> None:
        with self.solve_lock:
            self._device_scene_cache.clear()
            self._device_scene_hosts.clear()
            self._device_scene_versions.clear()
            self._device_scene_dirty.clear()
            self._device_emitter_cache.clear()
            self._device_emitter_hosts.clear()
            self._device_emitter_versions.clear()
            self._device_contexts.clear()
            self._device_instanced_cache.clear()
            self._device_instanced_hosts.clear()
            workspace_cache = getattr(self, "_execution_workspace_cache", None)
            if workspace_cache is not None:
                workspace_cache.clear()

    def _cuda_key(self, cuda) -> Tuple[int, int]:
        from numba import config

        device_id = int(cuda.get_current_device().id)
        context = cuda.current_context()
        if config.ENABLE_CUDASIM:
            token = device_id
        else:
            handle = getattr(context, "handle", None)
            token = int(handle.value) if handle is not None else id(context)
        key = (device_id, token)
        self._device_contexts[key] = context
        return key

    @staticmethod
    def _upload(cuda, previous, host: Optional[np.ndarray], old_host: Optional[np.ndarray], rows=None):
        if host is None:
            return None
        if previous is not None and host is old_host:
            return previous
        if previous is None or previous.shape != host.shape or previous.dtype != host.dtype:
            return cuda.to_device(host)
        if rows is None or len(rows) >= len(host) // 2:
            previous.copy_to_device(host)
        elif len(rows):
            # BVH partitions may scatter one object's triangles. Limit transfer
            # submissions, using its bounding range if it has many tiny runs.
            splits = np.flatnonzero(np.diff(rows) != 1) + 1
            runs = np.split(rows, splits)
            if len(runs) > 32:
                runs = [np.asarray([rows[0], rows[-1]])]
            for run in runs:
                first, last = int(run[0]), int(run[-1]) + 1
                previous[first:last].copy_to_device(host[first:last])
        return previous

    def get_device_scene(self, *, use_bvh: bool) -> PreparedDeviceScene:
        from numba import cuda

        with self.solve_lock:
            key = (*self._cuda_key(cuda), bool(use_bvh))
            previous = self._device_scene_cache.get(key)
            if previous is not None and self._device_scene_versions[key] == self.version:
                return previous
            host = self.get_scene(use_bvh=use_bvh)
            old_host = self._device_scene_hosts.get(key)
            triangle_rows = None
            if old_host is not None and host.permutation is old_host.permutation:
                changed = self._device_scene_dirty.get(key, set())
                triangle_rows = np.flatnonzero(np.isin(host.sid, list(changed)))
            uploaded = {}
            for field in fields(PreparedDeviceScene):
                name = field.name
                if name == "use_bvh":
                    uploaded[name] = host.use_bvh
                    continue
                array = getattr(host, name)
                rows = triangle_rows if name in ("v0", "e1", "e2", "normals") else None
                if name in ("bb_min", "bb_max") and old_host is not None:
                    old_array = getattr(old_host, name)
                    if array is not None and old_array is not None and array.shape == old_array.shape:
                        rows = np.flatnonzero(np.any(array != old_array, axis=1))
                uploaded[name] = self._upload(
                    cuda, None if previous is None else getattr(previous, name), array,
                    None if old_host is None else getattr(old_host, name), rows)
            scene = PreparedDeviceScene(**uploaded)
            self._device_scene_cache[key] = scene
            self._device_scene_hosts[key] = host
            self._device_scene_versions[key] = self.version
            self._device_scene_dirty.pop(key, None)
            return scene

    def get_device_instanced_scene(self):
        """Return persistent CUDA traversal arrays for the current BLAS/TLAS.

        The positional tuple matches ``InstancedScene.traversal_arrays()``.
        Rigid updates retain BLAS allocations and upload transforms/TLAS only.
        """
        from numba import cuda

        with self.solve_lock:
            key = self._cuda_key(cuda)
            scene = self.get_instanced_scene()
            arrays = scene.traversal_arrays()
            previous = self._device_instanced_cache.get(key)
            old_arrays = self._device_instanced_hosts.get(key)
            uploaded = tuple(self._upload(cuda, None if previous is None else previous[idx], array,
                                         None if old_arrays is None else old_arrays[idx])
                             for idx, array in enumerate(arrays))
            self._device_instanced_cache[key] = uploaded
            self._device_instanced_hosts[key] = arrays
            return uploaded

    def get_device_emitter(self, index: int, *, samples: int, rays: int, flip_faces: bool) -> PreparedDeviceEmitter:
        from numba import cuda

        with self.solve_lock:
            idx = self._mesh_index(index)
            key = (*self._cuda_key(cuda), idx, int(samples), int(rays), bool(flip_faces))
            previous = self._device_emitter_cache.get(key)
            if previous is not None and self._device_emitter_versions[key] == self._mesh_versions[idx]:
                return previous
            host = self.get_emitter(idx, samples=samples, rays=rays, flip_faces=flip_faces)
            old_host = self._device_emitter_hosts.get(key)
            uploaded = {
                field.name: self._upload(
                    cuda, None if previous is None else getattr(previous, field.name),
                    getattr(host, field.name), None if old_host is None else getattr(old_host, field.name))
                for field in fields(PreparedDeviceEmitter)
            }
            emitter = PreparedDeviceEmitter(**uploaded)
            self._device_emitter_cache[key] = emitter
            self._device_emitter_hosts[key] = host
            self._device_emitter_versions[key] = self._mesh_versions[idx]
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
