from __future__ import annotations
import os
import sys
import numpy as np


def grid_from_density(area: float, density: float) -> int:
    """Return Halton grid size for a given surface area and sample density."""
    g = int(np.ceil(np.sqrt(max(area, 0.0) * density)))
    return max(g, 4)


def hold_console_open(prompt: str = "Press Enter to close...") -> None:
    """Keep the console window open when running scripts directly.

    Controlled by the environment variable ``RAYSTRACK_HOLD_CONSOLE``.
    Set it to ``0`` or ``false`` to disable the prompt.
    """
    flag = os.environ.get("RAYSTRACK_HOLD_CONSOLE", "1").lower()
    if flag in {"0", "false", "no"}:
        return
    stdin = getattr(sys, "stdin", None)
    if stdin is None or not stdin.isatty():
        return
    try:
        input(prompt)
    except EOFError:
        pass


__all__ = [
    "grid_from_density",
    "hold_console_open",
]

