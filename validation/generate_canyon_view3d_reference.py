#!/usr/bin/env python3
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

from common_validation import (
    VIEW3D_REFERENCE_ROOT,
    aggregate_view3d_faces,
    build_canyon_meshes,
    write_json,
)


VIEW3D_REPO = Path(r"C:\Users\Philip\Desktop\view3d")
VIEW3D_BIN = VIEW3D_REPO / "dist" / "windows" / "view3d.exe"


def main() -> None:
    if str(VIEW3D_REPO) not in sys.path:
        sys.path.insert(0, str(VIEW3D_REPO))

    from v3d_io import View3dIO

    if not VIEW3D_BIN.exists():
        raise FileNotFoundError(f"View3D executable not found: {VIEW3D_BIN}")

    VIEW3D_REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)

    meshes = build_canyon_meshes()
    vs3_path = VIEW3D_REFERENCE_ROOT / "canyon_view3d.vs3"
    txt_path = VIEW3D_REFERENCE_ROOT / "canyon_view3d.txt"

    View3dIO.generate_vs3(
        meshes,
        vs3_path,
        title="raystrack canyon validation reference",
        enclosure=False,
        out_format=0,
        surface_emissivity=0.999,
    )
    subprocess.run(
        [
            "cmd",
            "/c",
            VIEW3D_BIN.as_posix(),
            vs3_path.as_posix(),
            txt_path.as_posix(),
        ],
        check=True,
        cwd=VIEW3D_REFERENCE_ROOT,
    )
    raw_view3d = View3dIO.read(txt_path)
    base_view3d = aggregate_view3d_faces(raw_view3d, meshes)

    write_json(VIEW3D_REFERENCE_ROOT / "canyon_view3d_raw.json", raw_view3d)
    write_json(VIEW3D_REFERENCE_ROOT / "canyon_view3d_base.json", base_view3d)

    print(VIEW3D_REFERENCE_ROOT / "canyon_view3d_base.json")


if __name__ == "__main__":
    main()
