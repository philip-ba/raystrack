from __future__ import annotations

"""Utility subpackage exports.

This module re-exports commonly used helpers so callers can do:
    from raystrack.utils import cached_halton, flatten_receivers, build_rays

Keeping imports light: avoid importing CUDA-specific modules here so that the
package imports cleanly on systems without CUDA. Heavy modules are loaded on
first use by their direct importers.
"""

# Lightweight, CPU-safe helpers
from .halton import cached_halton  # noqa: F401
from .geometry import flatten_receivers, flip_meshes  # noqa: F401
from .ray_builder import build_rays  # noqa: F401

__all__ = [
    "cached_halton",
    "flatten_receivers",
    "flip_meshes",
    "build_rays",
]
