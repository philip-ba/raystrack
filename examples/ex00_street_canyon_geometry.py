#!/usr/bin/env python3
"""
ex00_street_canyon_geometry

Generates a simple street canyon geometry:
- Two opposing facades (east- and west-facing normals), each 10x4 m panels,
  stacked 5 stories high (total height 20 m). Facades are 8 m apart.
- A road surface between facades sized 10x8 m.

Saves meshes to "street_canyon.json" in this folder.

All inputs are defined directly in the script: `story_h`, `stories`, `facade_width`,
and `gap`. The function `rect_plane_x` builds one rectangular facade panel for a
given height range, while `rect_plane_z` creates the road plane. No command-line
arguments are required; edit the constants above to change the geometry.
"""
import os
import sys
from pathlib import Path
import numpy as np


def ensure_repo_on_path():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def rect_plane_x(x0: float, y0: float, y1: float, z0: float, z1: float, normal_sign: int):
    """Return V(4,3), F(2,3) for a rectangle on plane x=x0.

    normal_sign = +1 for +X (east), -1 for -X (west).
    y0<y1, z0<z1 are expected extents.
    Triangulation ensures requested normal orientation.
    """
    # Base vertex order for +X normal
    BL = (x0, y0, z0)
    BR = (x0, y1, z0)
    TR = (x0, y1, z1)
    TL = (x0, y0, z1)
    V = np.asarray([BL, BR, TR, TL], dtype=np.float32)
    if normal_sign >= 0:
        F = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    else:
        # Flip winding for -X normal
        F = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return V, F


def rect_plane_z(z0: float, x0: float, x1: float, y0: float, y1: float, normal_up: bool = True):
    """Return V(4,3), F(2,3) for a rectangle on plane z=z0.

    normal_up=True -> +Z normal; False -> -Z.
    """
    BL = (x0, y0, z0)
    BR = (x1, y0, z0)
    TR = (x1, y1, z0)
    TL = (x0, y1, z0)
    V = np.asarray([BL, BR, TR, TL], dtype=np.float32)
    if normal_up:
        F = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    else:
        F = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return V, F


def build_street_canyon():
    # Dimensions
    story_h = 4.0
    stories = 5
    facade_width = 10.0
    half_width = facade_width / 2.0
    gap = 8.0

    # Coordinates reference
    # x: East(+)/West(-), y: North(+)/South(-), z: Up(+)
    # Facades centered at y=0, span y in [-5, 5]
    y0, y1 = -half_width, +half_width
    # Heights for each story panel
    panels = [(i * story_h, (i + 1) * story_h) for i in range(stories)]

    # Facade planes at x = -gap/2 (west wall, normal +X => east_side)
    # and x = +gap/2 (east wall, normal -X => west_side)
    x_east_normal = -gap / 2.0
    x_west_normal = +gap / 2.0

    meshes = []
    for i, (z0, z1) in enumerate(panels):
        # East side (normals to +X)
        V, F = rect_plane_x(x_east_normal, y0, y1, z0, z1, normal_sign=+1)
        meshes.append((f"east_side_{i}", V, F))
        # West side (normals to -X)
        V2, F2 = rect_plane_x(x_west_normal, y0, y1, z0, z1, normal_sign=-1)
        meshes.append((f"west_side_{i}", V2, F2))

    # Road surface at z=0, spanning the canyon between the two walls
    x0, x1 = -gap / 2.0, +gap / 2.0
    V_road, F_road = rect_plane_z(0.0, x0, x1, y0, y1, normal_up=True)
    meshes.append(("road", V_road, F_road))

    return meshes


def main():
    ensure_repo_on_path()
    from raystrack.io import save_meshes_json

    meshes = build_street_canyon()
    here = Path(__file__).resolve().parent
    out = here / "street_canyon.json"
    save_path = save_meshes_json(meshes, str(out))
    print(f"Saved street canyon geometry to: {save_path}")
    print(f"Meshes: {[name for name,_,_ in meshes]}")


if __name__ == "__main__":
    main()
    from raystrack.utils.helpers import hold_console_open
    hold_console_open()
