"""Software two-level acceleration: shared local BVHs and rigid instances.

No world triangle buffer is required for traversal. Rigid motion replaces only
instance transforms and top-level bounds. Local geometry buffers are immutable
and remain shared by CPU, CUDA and portable GPU scene snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .bvh import build_bvh, refit_bvh, build_aabb_bvh, refit_aabb_bvh


def readonly(array):
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    result = array.view()
    result.setflags(write=False)
    return result


def escape_links(left, right, count):
    """Stackless traversal successors for a pre-order BVH."""
    result = np.full(len(count), -1, np.int32)
    pending = [(0, -1)] if len(count) else []
    while pending:
        node, successor = pending.pop()
        result[node] = successor
        if count[node] == 0:
            pending.append((int(right[node]), successor))
            pending.append((int(left[node]), int(right[node])))
    return readonly(result)


@dataclass(frozen=True)
class LocalBVH:
    geom: np.ndarray
    bb_min: np.ndarray
    bb_max: np.ndarray
    left: np.ndarray
    right: np.ndarray
    start: np.ndarray
    count: np.ndarray
    escape: np.ndarray
    permutation: np.ndarray


@dataclass(frozen=True)
class PackedBLAS:
    geom: np.ndarray
    bb_min: np.ndarray
    bb_max: np.ndarray
    left: np.ndarray
    right: np.ndarray
    start: np.ndarray
    count: np.ndarray
    escape: np.ndarray


@dataclass(frozen=True)
class InstancedScene:
    blas: PackedBLAS
    roots: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    object_min: np.ndarray
    object_max: np.ndarray
    bb_min: np.ndarray
    bb_max: np.ndarray
    left: np.ndarray
    right: np.ndarray
    start: np.ndarray
    count: np.ndarray
    escape: np.ndarray
    instance_order: np.ndarray
    sid: np.ndarray
    revision: int
    tlas_rebuilds: int
    unique_geometries: int
    instanced: bool = True
    use_bvh: bool = True

    @property
    def local_triangle_count(self):
        return len(self.blas.geom)

    def traversal_arrays(self):
        """Stable positional array interface for compiled CPU/CUDA kernels."""
        b = self.blas
        return (b.geom, b.bb_min, b.bb_max, b.left, b.start, b.count, b.escape,
                self.roots, self.rotations, self.translations,
                self.bb_min, self.bb_max, self.left, self.start, self.count,
                self.escape, self.instance_order)


def _local_bvh(vertices, faces, previous=None):
    a = vertices[faces[:, 0]]
    e1, e2 = vertices[faces[:, 1]] - a, vertices[faces[:, 2]] - a
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if previous is None:
        lo, hi, left, right, start, count, perm = build_bvh(a, e1, e2)
        escape = escape_links(left, right, count)
    else:
        perm = previous.permutation
        left, right, start, count, escape = (previous.left, previous.right,
                                           previous.start, previous.count, previous.escape)
        lo, hi = refit_bvh(a[perm], e1[perm], e2[perm], left, right, start, count)
    geom = np.column_stack((a[perm], e1[perm], e2[perm], normals[perm])).astype(np.float32)
    return LocalBVH(*(readonly(x) for x in (geom, lo, hi, left, right, start, count, escape, perm)))


def _geometry_key(vertices, faces):
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(vertices.shape, np.int64).tobytes())
    digest.update(np.asarray(faces.shape, np.int64).tobytes())
    if vertices.size:
        digest.update(memoryview(np.ascontiguousarray(vertices)).cast("B"))
    if faces.size:
        digest.update(memoryview(np.ascontiguousarray(faces)).cast("B"))
    return digest.digest()


def _pack(geometries):
    pieces = {name: [] for name in PackedBLAS.__dataclass_fields__}
    roots = []
    node_offset = triangle_offset = 0
    for geometry in geometries:
        roots.append(node_offset if len(geometry.count) else -1)
        pieces["geom"].append(geometry.geom)
        pieces["bb_min"].append(geometry.bb_min)
        pieces["bb_max"].append(geometry.bb_max)
        for name in ("left", "right", "escape"):
            source = getattr(geometry, name)
            pieces[name].append(np.where(source >= 0, source + node_offset, -1).astype(np.int32))
        pieces["start"].append((geometry.start + triangle_offset).astype(np.int32))
        pieces["count"].append(geometry.count)
        node_offset += len(geometry.count)
        triangle_offset += len(geometry.geom)
    arrays = {}
    for name, chunks in pieces.items():
        width = 12 if name == "geom" else 3 if name in ("bb_min", "bb_max") else None
        empty = np.empty((0, width) if width else (0,), np.float32 if width else np.int32)
        arrays[name] = readonly(np.concatenate(chunks) if chunks else empty)
    return PackedBLAS(**arrays), np.asarray(roots, np.int32)


def _bounds(rotations, translations, roots, blas):
    lo = np.zeros((len(roots), 3), np.float32)
    hi = np.zeros_like(lo)
    valid = roots >= 0
    if np.any(valid):
        local_lo, local_hi = blas.bb_min[roots[valid]], blas.bb_max[roots[valid]]
        centers = (local_lo.astype(np.float64) + local_hi) * 0.5
        extents = (local_hi.astype(np.float64) - local_lo) * 0.5
        world_centers = np.einsum("nij,nj->ni", rotations[valid], centers) + translations[valid]
        world_extents = np.einsum("nij,nj->ni", np.abs(rotations[valid]), extents)
        # Outward rounding prevents a transformed boundary triangle escaping its box.
        lo[valid] = np.nextafter((world_centers - world_extents).astype(np.float32), -np.inf)
        hi[valid] = np.nextafter((world_centers + world_extents).astype(np.float32), np.inf)
        if not np.isfinite(lo[valid]).all() or not np.isfinite(hi[valid]).all():
            raise ValueError("transformed instance bounds must fit finite float32 coordinates")
    return readonly(lo), readonly(hi)


def _cost(lo, hi):
    if len(lo) <= 1:
        return 0.0
    extent = np.maximum(hi[1:].astype(np.float64) - lo[1:], 0)
    return float(np.sum(2 * (extent[:, 0] * extent[:, 1] + extent[:, 1] * extent[:, 2]
                            + extent[:, 2] * extent[:, 0])))


class InstancedAcceleration:
    """Own BLAS sharing and TLAS snapshots; geometry edits detach one instance."""

    def __init__(self, meshes, transforms, *, rebuild_ratio=1.75):
        self.rebuild_ratio = float(rebuild_ratio)
        self._geometry_cache = {}
        self._instance_geometries = []
        for _, vertices, faces in meshes:
            key = _geometry_key(vertices, faces)
            geometry = self._geometry_cache.get(key)
            if geometry is None:
                geometry = self._geometry_cache[key] = _local_bvh(vertices, faces)
            self._instance_geometries.append(geometry)
        self._scene = None
        self._rebuilds = 0
        self._baseline_cost = 0.0
        self._repack()
        self.update_transforms(transforms, revision=0, force_rebuild=True)

    def _repack(self):
        unique = []
        mapping = {}
        indices = []
        for geometry in self._instance_geometries:
            key = id(geometry)
            if key not in mapping:
                mapping[key] = len(unique)
                unique.append(geometry)
            indices.append(mapping[key])
        self._blas, roots = _pack(unique)
        self._roots = readonly(roots[np.asarray(indices, np.int32)])
        self._unique_count = len(unique)
        # Retain only referenced geometries; repeated topology/deformation edits
        # must not accumulate every historical mesh in the sharing cache.
        referenced = {id(geometry) for geometry in unique}
        self._geometry_cache = {key: geometry for key, geometry in self._geometry_cache.items()
                                if id(geometry) in referenced}

    def update_geometry(self, idx, vertices, faces, transforms, *, revision, same_faces):
        key = _geometry_key(vertices, faces)
        geometry = self._geometry_cache.get(key)
        if geometry is None:
            previous = self._instance_geometries[idx] if same_faces else None
            geometry = self._geometry_cache[key] = _local_bvh(vertices, faces, previous)
        self._instance_geometries[idx] = geometry
        self._repack()
        return self.update_transforms(transforms, revision=revision, force_rebuild=True)

    def update_transforms(self, transforms, *, revision, force_rebuild=False):
        rotations = readonly(np.asarray(transforms[:, :3, :3], np.float32))
        translations = readonly(np.asarray(transforms[:, :3, 3], np.float32))
        lo, hi = _bounds(rotations, translations, self._roots, self._blas)
        previous = self._scene
        valid = np.flatnonzero(self._roots >= 0).astype(np.int32)
        if previous is not None and not force_rebuild:
            order = previous.instance_order
            bb_min, bb_max = refit_aabb_bvh(lo[order], hi[order], previous.left,
                                          previous.right, previous.start, previous.count)
            current_cost = _cost(bb_min, bb_max)
            force_rebuild = current_cost > max(self._baseline_cost, 1e-20) * self.rebuild_ratio
        if previous is None or force_rebuild:
            bb_min, bb_max, left, right, start, count, perm = build_aabb_bvh(lo[valid], hi[valid], leaf_size=2)
            order = readonly(valid[perm])
            escape = escape_links(left, right, count)
            self._baseline_cost = _cost(bb_min, bb_max)
            self._rebuilds += 1
        else:
            left, right, start, count, escape = (previous.left, previous.right, previous.start,
                                               previous.count, previous.escape)
        self._scene = InstancedScene(
            self._blas, self._roots, rotations, translations, lo, hi,
            *(readonly(x) for x in (bb_min, bb_max, left, right, start, count, escape, order)),
            readonly(np.arange(len(self._roots), dtype=np.int32)), int(revision), self._rebuilds,
            self._unique_count)
        return self._scene

    @property
    def scene(self):
        return self._scene


__all__ = ["InstancedScene", "InstancedAcceleration"]
