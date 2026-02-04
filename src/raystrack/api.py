from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

from .main import view_factor_matrix, view_factor_to_tregenza_sky
from .params import MatrixParams, SkyParams
from .utils.helpers import (
    enforce_reciprocity_and_rowsum as _enforce_reciprocity_and_rowsum,
    enforce_reciprocity_only as _enforce_reciprocity_only,
)


def _row_sum(row: Dict[str, float]) -> float:
    return float(sum(float(v) for v in row.values()))


def view_factor_outside_workflow(
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    matrix_params: MatrixParams,
    sky_params: SkyParams,
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
    matrix_params, sky_params : MatrixParams
        Passed through to view_factor_matrix and view_factor_to_tregenza_sky
        respectively. Provide tolerances via 'tol' and 'tol_mode'.

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
    if not isinstance(matrix_params, MatrixParams):
        raise TypeError("matrix_params must be a MatrixParams instance")
    if not isinstance(sky_params, SkyParams):
        raise TypeError("sky_params must be a SkyParams instance")
    threshold = 1e-6
    enforce_scene = bool(matrix_params.enforce_reciprocity_rowsum)
    reciprocity_flag = bool(matrix_params.reciprocity)

    # Ensure we don't auto-enforce rows at matrix stage
    matrix_defaults = MatrixParams(**matrix_params.as_dict())
    matrix_defaults.enforce_reciprocity_rowsum = False
    vf_scene = view_factor_matrix(meshes, params=matrix_defaults)
    sky_vf = view_factor_to_tregenza_sky(meshes, params=sky_params)

    # Determine convergence tolerances
    tol_matrix = float(matrix_params.tol)
    tol_sky = float(sky_params.tol)
    threshold = abs(float(threshold)) if threshold is not None else max(tol_matrix, tol_sky)

    mesh_names = [name for name, _, _ in meshes]
    scene_totals = {name: max(0.0, _row_sum(vf_scene.get(name, {}))) for name in mesh_names}

    if enforce_scene:
        row_targets = [scene_totals.get(name, 0.0) for name in mesh_names]
        _enforce_reciprocity_and_rowsum(vf_scene, meshes, None, row_targets=row_targets)

    rest_vf: Dict[str, Dict[str, float]] = {}

    mesh_names = [name for name, _, _ in meshes]
    sky_totals = {name: 0.0 for name in mesh_names}

    for emitter in mesh_names:
        row = vf_scene.get(emitter, {})
        scene_sum = _row_sum(row)
        sky_row = dict(sky_vf.get(emitter, {}))
        if sky_params.discrete:
            sky_total = float(sum(float(v) for v in sky_row.values()))
        else:
            sky_total = float(sky_row.get("Sky", 0.0))

        if scene_sum + sky_total > 1.0 + threshold:
            if sky_total > 0.0:
                allowed_sky = max(0.0, 1.0 - scene_sum)
                scale = min(1.0, allowed_sky / sky_total) if sky_total else 0.0
                if sky_params.discrete:
                    for key, value in list(sky_row.items()):
                        sky_row[key] = float(value) * scale
                    sky_total = float(sum(float(v) for v in sky_row.values()))
                else:
                    sky_row["Sky"] = float(sky_row.get("Sky", 0.0)) * scale
                    sky_total = float(sky_row.get("Sky", 0.0))
                sky_vf[emitter] = sky_row
            else:
                sky_total = 0.0

        sky_totals[emitter] = max(0.0, sky_total)

    if enforce_scene:
        row_targets = [max(0.0, 1.0 - sky_totals.get(name, 0.0)) for name in mesh_names]
        _enforce_reciprocity_and_rowsum(vf_scene, meshes, None, row_targets=row_targets)
    elif reciprocity_flag:
        _enforce_reciprocity_only(vf_scene, meshes)

    for emitter in mesh_names:
        row = vf_scene.get(emitter, {})
        scene_sum = _row_sum(row)
        sky_row = dict(sky_vf.get(emitter, {}))
        if sky_params.discrete:
            sky_total = float(sum(float(v) for v in sky_row.values()))
        else:
            sky_total = float(sky_row.get("Sky", 0.0))

        combined = scene_sum + sky_total
        if combined > 1.0 + threshold and sky_total > 0.0:
            allowed_sky = max(0.0, 1.0 - scene_sum)
            if allowed_sky <= 0.0:
                sky_row = {key: 0.0 for key in sky_row}
                sky_total = 0.0
            else:
                scale = min(1.0, allowed_sky / sky_total)
                if sky_params.discrete:
                    for key, value in list(sky_row.items()):
                        sky_row[key] = float(value) * scale
                    sky_total = float(sum(float(v) for v in sky_row.values()))
                else:
                    sky_row["Sky"] = float(sky_row.get("Sky", 0.0)) * scale
                    sky_total = float(sky_row.get("Sky", 0.0))
            sky_vf[emitter] = sky_row
            combined = scene_sum + sky_total

        residual = 1.0 - combined
        if abs(residual) <= threshold:
            residual = 0.0

        rest_vf[emitter] = {"Rest": residual}

    return vf_scene, sky_vf, rest_vf


__all__ = ["view_factor_outside_workflow"]

