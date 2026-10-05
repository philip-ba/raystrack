"""CPU segment visibility and direct area-pair geometry weights."""
from __future__ import annotations
import math
import numba as nb
import numpy as np
from ..utils.halton import _build_halton_dim

@nb.njit(parallel=True, cache=True)
def _unoccluded_segments(source, target, scene_v0, scene_e1, scene_e2,
                         source_triangle, target_triangle, eligible):
    visible = np.zeros(source.shape[0], dtype=np.uint8)
    for k in nb.prange(source.shape[0]):
        if not eligible[k]:
            continue
        dx = target[k, 0] - source[k, 0]
        dy = target[k, 1] - source[k, 1]
        dz = target[k, 2] - source[k, 2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= 1e-12:
            continue
        edge_tol = min(1e-4, 1e-7 / distance)
        blocked = False
        for i in range(scene_v0.shape[0]):
            if i == source_triangle[k] or i == target_triangle[k]:
                continue
            px = dy * scene_e2[i, 2] - dz * scene_e2[i, 1]
            py = dz * scene_e2[i, 0] - dx * scene_e2[i, 2]
            pz = dx * scene_e2[i, 1] - dy * scene_e2[i, 0]
            det = (scene_e1[i, 0] * px + scene_e1[i, 1] * py
                   + scene_e1[i, 2] * pz)
            if abs(det) <= 1e-12:
                continue
            inv_det = 1.0 / det
            tx = source[k, 0] - scene_v0[i, 0]
            ty = source[k, 1] - scene_v0[i, 1]
            tz = source[k, 2] - scene_v0[i, 2]
            u = (tx * px + ty * py + tz * pz) * inv_det
            if u < 0.0 or u > 1.0:
                continue
            qx = ty * scene_e1[i, 2] - tz * scene_e1[i, 1]
            qy = tz * scene_e1[i, 0] - tx * scene_e1[i, 2]
            qz = tx * scene_e1[i, 1] - ty * scene_e1[i, 0]
            v = (dx * qx + dy * qy + dz * qz) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue
            t = (scene_e2[i, 0] * qx + scene_e2[i, 1] * qy
                 + scene_e2[i, 2] * qz) * inv_det
            if edge_tol < t < 1.0 - edge_tol:
                blocked = True
                break
        if not blocked:
            visible[k] = 1
    return visible


def _sample_surface(mesh, uniforms: np.ndarray):
    _, vertices, faces = mesh
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if faces.size == 0 or np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("faces must contain valid vertex indices")
    a = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - a
    e2 = vertices[faces[:, 2]] - a
    cross = np.cross(e1, e2)
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    total_area = float(np.sum(areas))
    if total_area <= 0.0 or not np.isfinite(total_area):
        raise ValueError("sender and receiver must have positive finite area")
    cdf = np.cumsum(areas / total_area)
    cdf[-1] = 1.0
    indices = np.searchsorted(cdf, uniforms[:, 0], side="right")
    u = uniforms[:, 1]
    v = uniforms[:, 2]
    root = np.sqrt(u)
    bary_u = root * (1.0 - v)
    bary_v = root * v
    points = a[indices] + bary_u[:, None] * e1[indices] + bary_v[:, None] * e2[indices]
    normals = np.zeros_like(cross)
    valid = twice_area > 0.0
    normals[valid] = cross[valid] / twice_area[valid, None]
    return points, normals[indices], indices.astype(np.int32), total_area


def _pair_uniforms(samples: int, rng: np.random.Generator,
                   sequence: str) -> np.ndarray:
    if sequence == "random":
        return rng.random((samples, 6))
    if sequence == "shifted_halton":
        shifts = rng.random(6)
        return np.column_stack([
            (np.asarray(_build_halton_dim(samples, base), dtype=np.float64)
             + shifts[d]) % 1.0
            for d, base in enumerate((2, 3, 5, 7, 11, 13))
        ])
    raise ValueError("sequence must be 'random' or 'shifted_halton'")


class AreaPairWorkspace:
    """Cache a full shifted replicate; chunking never changes its samples."""
    def __init__(self):
        self.key = None

    def trace(self, prepared, source_index, target_index, cfg, iteration, ray_count, ray_offset):
        key = (id(prepared), prepared.version, source_index, target_index,
               cfg.pair_samples, cfg.seed, cfg.sequence, cfg.flip_faces, iteration)
        if key != self.key:
            meshes = prepared.meshes
            uniforms = _pair_uniforms(cfg.pair_samples, np.random.default_rng(cfg.seed + iteration), cfg.sequence)
            source, sn, source_tri, _ = _sample_surface(meshes[source_index], uniforms[:, :3])
            target, tn, target_tri, area = _sample_surface(meshes[target_index], uniforms[:, 3:])
            if cfg.flip_faces:
                sn = -sn
            disp = target - source
            d2 = np.sum(disp * disp, axis=1)
            distance = np.maximum(np.sqrt(d2), 1e-12)
            cs = np.einsum("ij,ij->i", sn, disp) / distance
            ct = -np.einsum("ij,ij->i", tn, disp) / distance
            eligible = (cs > 0) & (d2 > 1e-24) & (ct != 0)
            scene = prepared.get_scene(use_bvh=False)
            offsets = np.cumsum([0] + [len(f) for _, _, f in meshes])
            self.source, self.target = source, target
            self.st, self.tt = source_tri + offsets[source_index], target_tri + offsets[target_index]
            self.scene, self.eligible, self.ct = scene, eligible, ct
            self.weights = np.zeros(cfg.pair_samples, np.float64)
            self.weights[eligible] = area * cs[eligible] * np.abs(ct[eligible]) / (math.pi * d2[eligible])
            self.key = key
        sl = slice(ray_offset, ray_offset + ray_count)
        visible = _unoccluded_segments(self.source[sl], self.target[sl],
            np.asarray(self.scene.v0, np.float64), np.asarray(self.scene.e1, np.float64),
            np.asarray(self.scene.e2, np.float64), self.st[sl], self.tt[sl], self.eligible[sl])
        weights = self.weights[sl] * visible
        self.last_weights, self.last_sides = weights, self.ct[sl]
        front, back = np.zeros(len(prepared.meshes), np.float64), np.zeros(len(prepared.meshes), np.float64)
        front[target_index] = np.sum(weights[self.ct[sl] > 0])
        back[target_index] = np.sum(weights[self.ct[sl] < 0])
        return front, back, np.zeros(1, np.float64)
