from __future__ import annotations
from typing import List, Tuple
import numpy as np



def flatten_receivers(meshes, idx_emit, skip: Iterable[int] = ()):  # type: ignore[assignment]
    """Return flattened receiver triangle arrays excluding the emitter.

    Parameters
    ----------
    meshes : list
        List of ``(name, V, F)`` mesh tuples.
    idx_emit : int
        Index of the emitting surface in ``meshes``.
    skip : Iterable[int], optional
        Additional surface indices to exclude from the receiver list.
    """
    v0s, e1s, e2s, sids, norms = [], [], [], [], []
    skip_set = set(skip)
    skip_set.add(idx_emit)
    for sid, (_, V, F) in enumerate(meshes):
        if sid in skip_set:
            continue
        v0 = V[F[:, 0]].astype(np.float32)
        v1 = V[F[:, 1]].astype(np.float32)
        v2 = V[F[:, 2]].astype(np.float32)

        v0s.append(v0)
        e1s.append(v1 - v0)
        e2s.append(v2 - v0)
        sids.append(np.full(len(F), sid, np.int32))

        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n, axis=1)[:, None]
        norms.append(n.astype(np.float32))

    if not v0s:
        return (
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.float32),
            np.empty((0,), np.int32),
            np.empty((0, 3), np.float32),
        )

    v0 = np.concatenate(v0s)
    e1 = np.concatenate(e1s)
    e2 = np.concatenate(e2s)
    sid = np.concatenate(sids)
    norm = np.concatenate(norms)
    return v0, e1, e2, sid, norm


def flip_meshes(meshes: List[Tuple[str, np.ndarray, np.ndarray]]
               ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Return a deep-copied list whose every triangle has reversed order."""
    flipped = []
    for name, V, F in meshes:
        F_flipped = F[:, [0, 2, 1]].copy()
        flipped.append((name, V.copy(), F_flipped))
    return flipped

__all__ = ["flatten_receivers", "flip_meshes"]
