from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .main import view_factor_matrix, view_factor_to_tregenza_sky


def _row_sum(row: Dict[str, float]) -> float:
    return float(sum(float(v) for v in row.values()))


def view_factor_outside_workflow(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    matrix_params: Optional[Dict] = None,
    sky_params: Optional[Dict] = None,
    discrete: bool = False,
    threshold: Optional[float] = 1e-6,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Compute scene VF matrix, sky VF and the residual fraction.

    Steps
    - Compute regular view-factor matrix (scene-to-scene).
    - Compute Radiance-style sky view factor(s): merged (Sky) or 145 patches.
    - For each emitter, compute the residual fraction required so that the
      total view factor sums to one: ``1 - sum(scene VFs) - sky_total``.

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
    threshold : float, optional
        Absolute tolerance used to treat residuals as numerical noise when
        enforcing ``scene + sky + rest = 1``. The default of ``1e-6`` mirrors
        the legacy behaviour while avoiding the old geometry-specific heuristics.
        Set to ``None`` to fall back to ``max(matrix_tol, sky_tol)``.

    Returns
    -------
    vf_scene : dict
        Scene view-factor matrix as returned by :func:`view_factor_matrix`.
    sky_vf : dict
        Sky view factor(s): either {'Sky': vf} or {'Sky_Patch_i': vf} per emitter.
    rest_vf : dict
        Residual view factor per emitter (``{"Rest": value}``) so that
        ``scene + sky + rest = 1``.
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
    threshold = abs(float(threshold)) if threshold is not None else max(tol_matrix, tol_sky)

    rest_vf: Dict[str, Dict[str, float]] = {}

    for emitter, row in vf_scene.items():
        scene_sum = _row_sum(row)
        sky_row = sky_vf.get(emitter, {})
        if discrete:
            sky_total = float(sum(float(v) for v in sky_row.values()))
        else:
            sky_total = float(sky_row.get("Sky", 0.0))

        residual = 1.0 - scene_sum - sky_total
        if abs(residual) <= threshold:
            residual = 0.0

        rest_vf[emitter] = {"REST": residual}

    return vf_scene, sky_vf, rest_vf


__all__ = ["view_factor_outside_workflow"]

