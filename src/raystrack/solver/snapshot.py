from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .result import Channel, SparseValues, Result
from ..engine.postprocessing import apply


def snapshot(run, rays_used, elapsed_ms):
    query, options, scene, acc = run.query, run.options, run.solver.scene, run._accumulator
    senders, receivers = run._sender_ids, run._receiver_ids
    channels = []
    if query.scene:
        channels.extend(Channel("surface", receiver, side) for receiver in receivers for side in query.receiver_sides)
    if query.sky_mode is not None:
        channels.extend([Channel("sky", patch=i) for i in range(145)] if query.sky_mode == "tregenza145" else [Channel("sky")])
    if options.sampling.strategy == "cosine":
        channels.append(Channel("rest"))
        if not query.scene or set(receivers) != set(run._surface_ids) or len(query.receiver_sides) != 2:
            channels.append(Channel("unrequested"))
    channels = tuple(channels)
    values = np.zeros((len(senders), len(channels)), np.float64)
    errors = np.full_like(values, np.nan)
    coverage = np.zeros(len(senders), np.int8)
    ids = tuple(acc._names) if acc._names else run._surface_ids
    n = len(ids)
    for row, sender in enumerate(senders):
        state = acc._states.get(ids.index(sender))
        if state is None:
            continue
        coverage[row] = int(state.rays > 0 or state.matrix.empty and (state.sky is None or state.sky.empty))
        mv, me = state.matrix.values(), state.matrix.standard_errors()
        sv = state.sky.values() if state.sky is not None else None
        se = state.sky.standard_errors() if state.sky is not None else None
        if state.matrix.pending_rays:
            me = None
        if state.sky is not None and state.sky.pending_rays:
            se = None
        for col, channel in enumerate(channels):
            if channel.kind == "surface":
                j = ids.index(channel.surface_id) + (0 if channel.side == "front" else n)
                values[row, col] = mv[j]
                if me is not None:
                    errors[row, col] = me[j]
            elif channel.kind == "sky":
                j = channel.patch if channel.patch is not None else 0
                values[row, col] = sv[j]
                if se is not None:
                    errors[row, col] = se[j]
            else:
                j = 2*n + (0 if channel.kind == "unrequested" else 1)
                values[row, col] = max(0., mv[j])
                if me is not None:
                    errors[row, col] = me[j]
    status = "invalidated" if run._invalidated.is_set() else "cancelled" if run._cancelled.is_set() else acc.status
    if not senders:
        status = "converged"
    provenance = ()
    mode = options.postprocessing.reciprocity
    if mode != "none":
        if status not in ("converged", "max_iters") or np.any(coverage != 1):
            raise ValueError("Reciprocity postprocessing requires a complete solve; disable it for previews")
        if mode == "rowsum" and np.max(values[:, channels.index(Channel("rest"))], initial=0.) > 1e-8:
            raise ValueError("Row-sum reciprocity requires a closed scene with no sampled escape rays")
        emitters = run.solver._prepared.get_emitters(samples=options.sampling.density,
                     rays=options.sampling.rays_per_cell, flip_faces=options.sampling.flip_faces)
        areas = {sid: em.total_area for sid, em in zip(ids, emitters)}
        values, errors, provenance = apply(values, errors, senders, channels, areas, mode)
    stored = (values != 0) | (np.isfinite(errors) & (errors != 0))
    if provenance:
        # Postprocessed zeros may have unknown uncertainty even after raw
        # replicate statistics exist; record those explicitly.
        stored |= np.isnan(errors) & (coverage[:, None] == 1)
    rows, cols = np.nonzero(stored)
    statistics = acc.stats()
    statistics["options"] = asdict(options)
    return Result(senders, channels, SparseValues(rows, cols, values[rows, cols], errors[rows, cols]), coverage,
                  statistics=statistics, execution=acc.plan or {}, scene_revision=run.scene_revision,
                  rays_used=rays_used, cumulative_rays=acc.cumulative_rays, status=status,
                  converged=status == "converged", elapsed_ms=elapsed_ms, provenance=provenance)
