from __future__ import annotations

from functools import lru_cache

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
def _build_halton_grid(samples: int):
    g = samples
    cells = g * g
    u = np.empty(cells, np.float32)
    v = np.empty(cells, np.float32)
    for c in range(cells):
        i, j = divmod(c, g)
        u[c] = (_halton(c + 1, 2) + i) / g
        v[c] = (_halton(c + 1, 3) + j) / g
    return u, v


@nb.njit(cache=True)
def _build_halton_dim(length: int, base: int):
    out = np.empty(length, np.float32)
    for i in range(length):
        out[i] = _halton(i + 1, base)
    return out


@lru_cache(maxsize=128)
def cached_halton(samples: int):
    """Return a cached 2-D Halton jitter grid of size ``samples * samples``."""
    return _build_halton_grid(int(samples))


@lru_cache(maxsize=128)
def cached_halton_dims(length: int):
    """Return cached low-discrepancy dimensions for ray generation."""
    n = int(length)
    return (
        _build_halton_dim(n, 5),
        _build_halton_dim(n, 2),
        _build_halton_dim(n, 3),
        _build_halton_dim(n, 7),
        _build_halton_dim(n, 11),
    )


__all__ = ["cached_halton", "cached_halton_dims"]
