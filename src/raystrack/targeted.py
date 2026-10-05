"""Direct area-pair sampling for a selected, potentially rare receiver."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numba as nb
import numpy as np

from .utils.prepared import PreparedSolver
from .utils.halton import _build_halton_dim
from .main import _prepared_locked


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


@_prepared_locked
def view_factor_targeted(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    sender: str,
    receiver: str,
    *,
    samples: int = 8192,
    seed: int = 1,
    sequence: str = "shifted_halton",
    prepared: PreparedSolver | None = None,
) -> Dict[str, float]:
    """Estimate one sender-to-receiver row with direct area-pair samples.

    Each sample connects uniformly sampled points on the two mesh surfaces.
    The cosine-distance geometry term and receiver area supply the sampling
    weight; every other triangle is tested for segment occlusion. This is a
    CPU estimator for selected receivers, especially those hit by few ordinary
    cosine rays. It does not compute the rest of the scene or sky matrix.
    ``sequence='shifted_halton'`` uses independently shifted low-discrepancy
    dimensions for each seed; ``'random'`` uses pseudorandom points.
    """
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    names = [name for name, _, _ in meshes]
    if len(set(names)) != len(names):
        raise ValueError("mesh names must be unique")
    if sender not in names or receiver not in names or sender == receiver:
        raise ValueError("sender and receiver must be different mesh names")
    if prepared is not None:
        if not isinstance(prepared, PreparedSolver) or len(prepared.meshes) != len(meshes):
            raise ValueError("prepared must contain the same ordered meshes")
        prepared.validate_meshes(meshes)

    rng = np.random.default_rng(seed)
    uniforms = _pair_uniforms(samples, rng, sequence)
    source_idx = names.index(sender)
    target_idx = names.index(receiver)
    source, source_normals, source_tri, _ = _sample_surface(meshes[source_idx], uniforms[:, :3])
    target, target_normals, target_tri, target_area = _sample_surface(meshes[target_idx], uniforms[:, 3:])
    displacement = target - source
    distance_sq = np.sum(displacement * displacement, axis=1)
    distance = np.sqrt(distance_sq)
    safe_distance = np.maximum(distance, 1e-12)
    cos_source = np.einsum("ij,ij->i", source_normals, displacement) / safe_distance
    cos_target = -np.einsum("ij,ij->i", target_normals, displacement) / safe_distance
    eligible = (cos_source > 0.0) & (distance_sq > 1e-24) & (cos_target != 0.0)
    if not np.any(eligible):
        return {}

    prepared_solver = prepared if prepared is not None else PreparedSolver(meshes)
    scene = prepared_solver.get_scene(use_bvh=False)
    offset = np.cumsum([0] + [len(faces) for _, _, faces in meshes])
    visible = _unoccluded_segments(
        source, target,
        np.asarray(scene.v0, dtype=np.float64),
        np.asarray(scene.e1, dtype=np.float64),
        np.asarray(scene.e2, dtype=np.float64),
        source_tri + offset[source_idx], target_tri + offset[target_idx], eligible,
    )
    weights = np.zeros(samples, dtype=np.float64)
    valid = eligible & (visible != 0)
    weights[valid] = (target_area * cos_source[valid] * np.abs(cos_target[valid])
                      / (math.pi * distance_sq[valid]))
    front = float(np.sum(weights[cos_target > 0.0]) / samples)
    back = float(np.sum(weights[cos_target < 0.0]) / samples)
    row: Dict[str, float] = {}
    if front > 0.0:
        row[f"{receiver}_front"] = front
    if back > 0.0:
        row[f"{receiver}_back"] = back
    return row


__all__ = ["view_factor_targeted"]
