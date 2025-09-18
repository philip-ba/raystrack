from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from .main import view_factor_matrix, view_factor_to_tregenza_sky


def _row_sum(row: Dict[str, float]) -> float:
    return float(sum(float(v) for v in row.values()))


def _keys_for_receiver(name: str) -> Tuple[str, str]:
    return f"{name}_front", f"{name}_back"


def _sum_receiver_contrib(row: Dict[str, float], name: str) -> float:
    kf, kb = _keys_for_receiver(name)
    return float(row.get(kf, 0.0)) + float(row.get(kb, 0.0))


def _scale_row_to_target_sum(row: Dict[str, float], target_sum: float) -> None:
    curr = _row_sum(row)
    if curr <= 0.0:
        return
    s = target_sum / curr
    for k in list(row.keys()):
        row[k] = float(row[k]) * s


def _find_ground_edge_meshes(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    angle_tol_deg: float = 5.0,
    z_eps: float = 1e-4,
    edge_eps: float = 1e-4,
) -> List[str]:
    """Return names of meshes likely representing outer ground surfaces.

    Heuristic:
    - Upward facing (average face normal within `angle_tol_deg` of +Z).
    - At the lowest scene Z (min vertex Z within `z_eps` of global min).
    - Touch the scene XY bounds within `edge_eps` (outer edges).
    """
    if not meshes:
        return []

    allV = np.concatenate([V for _, V, _ in meshes], axis=0)
    min_xyz = allV.min(axis=0)
    max_xyz = allV.max(axis=0)
    global_min_z = float(min_xyz[2])
    min_x, max_x = float(min_xyz[0]), float(max_xyz[0])
    min_y, max_y = float(min_xyz[1]), float(max_xyz[1])

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cos_tol = math.cos(math.radians(angle_tol_deg))

    ground_candidates: List[Tuple[str, np.ndarray]] = []  # (name, V)
    for name, V, F in meshes:
        if len(F) == 0:
            continue
        # Average face normal
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        if n.size == 0:
            continue
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        n_avg = n.mean(axis=0)
        n_avg /= (np.linalg.norm(n_avg) + 1e-12)
        if n_avg[2] < cos_tol:
            continue
        # Lowest z near global min
        if float(V[:, 2].min()) > global_min_z + z_eps:
            continue
        ground_candidates.append((name, V))

    if not ground_candidates:
        return []

    # Select those touching outer XY bounds
    edge_names: List[str] = []
    for name, V in ground_candidates:
        xs, ys = V[:, 0], V[:, 1]
        touches = (
            (abs(xs.min() - min_x) <= edge_eps)
            or (abs(xs.max() - max_x) <= edge_eps)
            or (abs(ys.min() - min_y) <= edge_eps)
            or (abs(ys.max() - max_y) <= edge_eps)
        )
        if touches:
            edge_names.append(name)
    # Fallback to all ground candidates if none are on the boundary
    if not edge_names:
        edge_names = [name for name, _ in ground_candidates]
    return edge_names


def view_factor_outside_workflow(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    matrix_params: Optional[Dict] = None,
    sky_params: Optional[Dict] = None,
    discrete: bool = False,
    threshold_multiplier: float = 5.0,
    ground_angle_tol_deg: float = 5.0,
    z_eps: float = 1e-3,
    edge_eps: float = 1e-3,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute scene VF matrix and Tregenza-sky VF, reconcile differences.

    Steps
    - Compute regular view-factor matrix (scene-to-scene). Do NOT enforce
      final row-sum in the call; rows are reconciled vs. sky later per emitter.
    - Compute Radiance-style sky view factor(s): merged (Sky) or 145 patches.
    - For each emitter, compare (1 - sum(scene VFs)) to sky total. If the
      absolute difference exceeds `threshold_multiplier` * max(tol), distribute
      the difference to ground-edge surfaces proportionally to their current
      VF contribution, then scale the emitter row so that:
         sum(scene VFs) == 1 - sky_total

    Parameters
    ----------
    meshes : list of (name, V, F)
        Scene meshes (float32/float64 vertices, int faces).
    matrix_params, sky_params : dict
        Passed through to view_factor_matrix and view_factor_to_tregenza_sky
        respectively. Provide tolerances via 'tol' and 'tol_mode'.
    discrete : bool
        When True, sky result contains 145 bins (Sky_Patch_i). When False,
        single key 'Sky' per emitter.
    threshold_multiplier : float
        Threshold factor applied to max(tol_matrix, tol_sky) to decide whether
        to reconcile a row.
    ground_angle_tol_deg : float
        Angle tolerance from +Z to classify a mesh as upward-facing ground.
    z_eps, edge_eps : float
        Tolerances for lowest-z and boundary touch checks.

    Returns
    -------
    vf_scene : dict
        Adjusted scene VF matrix where each emitter's row matches the sky
        total via sum(scene) == 1 - sky_total.
    sky_vf : dict
        Sky view factor(s): either {'Sky': vf} or {'Sky_Patch_i': vf} per emitter.
    """
    matrix_params = dict(matrix_params or {})
    sky_params = dict(sky_params or {})

    # Ensure we don't auto-enforce rows at matrix stage
    matrix_params.setdefault("enforce_reciprocity_rowsum", False)
    # Compute both
    vf_scene = view_factor_matrix(meshes, **matrix_params)
    sky_vf = view_factor_to_tregenza_sky(meshes, discrete=discrete, **sky_params)

    # Determine convergence tolerances
    tol_matrix = float(matrix_params.get("tol", 1e-5))
    tol_sky = float(sky_params.get("tol", 1e-5))
    threshold = abs(threshold_multiplier) * max(tol_matrix, tol_sky)

    # Pre-compute ground-edge mesh names
    ground_edge_names = _find_ground_edge_meshes(
        meshes,
        angle_tol_deg=ground_angle_tol_deg,
        z_eps=z_eps,
        edge_eps=edge_eps,
    )

    # Reconcile per emitter
    for emitter, row in vf_scene.items():
        # Scene sum and sky total for this emitter
        scene_sum = _row_sum(row)
        # Sum sky entries
        sky_row = sky_vf.get(emitter, {})
        if discrete:
            sky_total = float(sum(float(v) for v in sky_row.values()))
        else:
            sky_total = float(sky_row.get("Sky", 0.0))

        diff = (1.0 - scene_sum) - sky_total
        if abs(diff) <= threshold:
            # Scale row so 1 - sum == sky_total (remove small deviations)
            target_scene_sum = max(0.0, 1.0 - sky_total)
            _scale_row_to_target_sum(row, target_scene_sum)
            continue

        # Outside threshold: try to distribute difference to ground-edge meshes
        if ground_edge_names:
            # Build weights from current contributions
            weights = []
            for gname in ground_edge_names:
                weights.append(_sum_receiver_contrib(row, gname))
            wsum = float(sum(weights))
            if wsum <= 0.0:
                # Fallback to equal weights
                weights = [1.0] * len(ground_edge_names)
                wsum = float(len(ground_edge_names))

            # Amount to add/subtract to scene so that 1 - sum == sky
            delta = diff  # positive: increase scene; negative: decrease scene
            for gname, w in zip(ground_edge_names, weights):
                share = float(w) / wsum
                change = delta * share
                kf, kb = _keys_for_receiver(gname)
                f = float(row.get(kf, 0.0))
                b = float(row.get(kb, 0.0))
                s = f + b
                if s > 0.0:
                    rf = f / s
                    rb = 1.0 - rf
                else:
                    rf = 1.0
                    rb = 0.0
                f_new = max(0.0, f + change * rf)
                b_new = max(0.0, b + change * rb)
                row[kf] = f_new
                row[kb] = b_new

        # Finally, enforce sum(scene) == 1 - sky_total (exact)
        target_scene_sum = max(0.0, 1.0 - sky_total)
        _scale_row_to_target_sum(row, target_scene_sum)

    return vf_scene, sky_vf


__all__ = ["view_factor_outside_workflow"]

