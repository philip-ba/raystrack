from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def grid_from_density(area: float, density: float) -> int:
    """Return Halton grid size for a given surface area and sample density."""
    g = int(np.ceil(np.sqrt(max(area, 0.0) * density)))
    return max(g, 4)


def enforce_reciprocity_and_rowsum(
    result: Dict[str, Dict[str, float]],
    meshes: List[Tuple[str, np.ndarray, np.ndarray]],
    areas: List[float] | None,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> None:
    """In-place adjust result so that rows sum to 1 and reciprocity holds.

    Operates on totals per pair (front+back) using symmetric diagonal scaling
    of G = diag(A) F after symmetrization, then maps back to front/back totals
    proportionally to the original split.
    """
    n = len(meshes)
    names = [m[0] for m in meshes]
    name_to_idx = {name: i for i, name in enumerate(names)}

    if areas is None:
        areas = []
        for _, V_a, F_a in meshes:
            A_a = 0.5 * np.linalg.norm(
                np.cross(
                    V_a[F_a[:, 1]] - V_a[F_a[:, 0]],
                    V_a[F_a[:, 2]] - V_a[F_a[:, 0]],
                ),
                axis=1,
            )
            areas.append(float(A_a.sum()))
    A = np.asarray(areas, dtype=np.float64)

    F = np.zeros((n, n), dtype=np.float64)

    def base_of(key: str) -> str:
        if key.endswith("_front"):
            return key[:-6]
        if key.endswith("_back"):
            return key[:-5]
        return key

    for si, sname in enumerate(names):
        row = result.get(sname, {})
        if not isinstance(row, dict):
            continue
        accum: Dict[str, float] = {}
        for rkey, val in row.items():
            b = base_of(rkey)
            accum[b] = accum.get(b, 0.0) + float(val)
        for bname, v in accum.items():
            j = name_to_idx.get(bname, None)
            if j is None:
                continue
            F[si, j] = v

    G = (A[:, None] * F)
    G = 0.5 * (G + G.T)

    d = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        row = d * (G @ d)
        row = np.maximum(row, 1e-30)
        upd = A / row
        d_new = d * np.sqrt(upd)
        if np.max(np.abs(d_new - d)) < tol:
            d = d_new
            break
        d = d_new

    Gp = (d[:, None] * G) * d[None, :]
    Fp = Gp / A[:, None]

    for si, sname in enumerate(names):
        row = result.get(sname, {})
        fb: Dict[str, Tuple[float, float]] = {}
        for rkey, val in row.items():
            if rkey.endswith("_front"):
                base = rkey[:-6]
                cur_f, cur_b = fb.get(base, (0.0, 0.0))
                fb[base] = (cur_f + float(val), cur_b)
            elif rkey.endswith("_back"):
                base = rkey[:-5]
                cur_f, cur_b = fb.get(base, (0.0, 0.0))
                fb[base] = (cur_f, cur_b + float(val))
            else:
                base = rkey
                cur_f, cur_b = fb.get(base, (0.0, 0.0))
                fb[base] = (cur_f, cur_b + float(val))

        for bj, rname in enumerate(names):
            t_new = float(max(Fp[si, bj], 0.0))
            cur_f, cur_b = fb.get(rname, (0.0, 0.0))
            t_old = cur_f + cur_b
            if t_old > 0.0:
                s = t_new / t_old
                new_f = cur_f * s
                new_b = cur_b * s
            else:
                new_f = 0.0
                new_b = t_new

            if new_f > 0.0:
                row[f"{rname}_front"] = new_f
            elif f"{rname}_front" in row:
                del row[f"{rname}_front"]
            if new_b > 0.0:
                row[f"{rname}_back"] = new_b
            elif f"{rname}_back" in row:
                del row[f"{rname}_back"]

        result[sname] = row


__all__ = [
    "grid_from_density",
    "enforce_reciprocity_and_rowsum",
]

