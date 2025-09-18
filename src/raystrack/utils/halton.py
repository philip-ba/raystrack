from __future__ import annotations
import numpy as np
import numba as nb

@nb.njit(inline="always", cache=True)
def _halton(i: int, base: int) -> float:
    """Return the i-th value of a 1-D Halton sequence in the given base."""
    f = 1.0
    r = 0.0
    while i:
        f /= base
        r += f * (i % base)
        i //= base
    return r

@nb.njit(cache=True)
def cached_halton(samples: int):
    """Pre-compute one 2-D Halton jitter grid of size samples * samples."""
    g = samples
    cells = g * g
    u = np.empty(cells, np.float32)
    v = np.empty(cells, np.float32)
    for c in range(cells):
        i, j = divmod(c, g)
        u[c] = (_halton(c + 1, 2) + i) / g
        v[c] = (_halton(c + 1, 3) + j) / g
    return u, v

__all__ = ["cached_halton"]
