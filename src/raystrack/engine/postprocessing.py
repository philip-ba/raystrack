"""Explicit reciprocity transforms, operating on structured surface channels."""
from __future__ import annotations
import numpy as np


def apply(values, errors, sender_ids, channels, areas, mode):
    if mode == "none":
        return values, errors, ()
    front = {c.surface_id: j for j, c in enumerate(channels) if c.kind == "surface" and c.side == "front"}
    back = {c.surface_id: j for j, c in enumerate(channels) if c.kind == "surface" and c.side == "back"}
    if set(front) != set(sender_ids) or (mode == "rowsum" and set(back) != set(sender_ids)):
        raise ValueError("Reciprocity requires complete matching sender and receiver selections")
    result, se = values.copy(), errors.copy()
    if mode in ("shortcut", "bidirectional"):
        for i, name_i in enumerate(sender_ids):
            for j in range(i + 1, len(sender_ids)):
                name_j = sender_ids[j]
                ai, aj = areas[name_i], areas[name_j]
                if ai <= 0 or aj <= 0:
                    continue
                ci, cj = front[name_j], front[name_i]
                exchange = ai * result[i, ci] if mode == "shortcut" else .5 * (ai * result[i, ci] + aj * result[j, cj])
                result[i, ci], result[j, cj] = exchange / ai, exchange / aj
                # Seeded streams may correlate; raw per-row errors do not
                # establish the covariance of this transformed estimate.
                se[i, ci] = se[j, cj] = np.nan
    else:
        n = len(sender_ids)
        a = np.asarray([areas[s] for s in sender_ids], np.float64)
        totals = np.asarray([[values[i, front[s]] + values[i, back[s]] for s in sender_ids] for i in range(n)])
        g = a[:, None] * totals
        g = .5 * (g + g.T)
        if np.any(a <= 0) or np.any(g.sum(axis=1) <= 0):
            raise ValueError("Row-sum reciprocity requires positive areas and exchanges in every row")
        d = np.ones(n)
        for _ in range(500):
            new = d * np.sqrt(a / np.maximum(d * (g @ d), 1e-30))
            if np.max(np.abs(new - d)) < 1e-10:
                d = new
                break
            d = new
        scaled = (d[:, None] * g * d[None, :]) / a[:, None]
        if np.max(np.abs(scaled.sum(axis=1) - 1)) > 1e-6:
            raise ValueError("Row-sum reciprocity did not converge for this exchange graph")
        for i in range(n):
            for j, s in enumerate(sender_ids):
                old = totals[i, j]
                result[i, front[s]] = scaled[i, j] * values[i, front[s]] / old if old else 0
                result[i, back[s]] = scaled[i, j] * values[i, back[s]] / old if old else scaled[i, j]
                se[i, front[s]] = se[i, back[s]] = np.nan
    return result, se, ({"algorithm": mode, "input": "raw cosine estimates",
                         "uncertainty": "unknown after postprocessing"},)
