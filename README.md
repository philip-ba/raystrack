# raystrack

Lightweight Monte‑Carlo view‑factor solver for polygonal meshes.

Raystrack computes radiative view factors F(i->j) between triangulated surfaces
using quasi‑Monte‑Carlo ray tracing. It runs on plain CPU, can leverage
Numba/CUDA on NVIDIA GPUs when available, and optionally accelerates ray
intersection with a BVH. The repository includes Grasshopper user objects and
installers for Rhino 8 (Windows and macOS), as well as a pure‑Python API you
can use outside Rhino.

Features
- Efficient Monte‑Carlo view factors: front/back hits, optional reciprocity
- CPU and optional CUDA GPU backends (Numba)
- Optional BVH acceleration
- Grasshopper components + example file (`rhino/example.gh`)

Installation (Grasshopper)

Windows (Rhino 8)
1) Download or clone this repository.
2) Double‑click `grasshopper_win_installer.bat` (or run it from `cmd`).
   - The script installs the Python package into Rhino 8’s CPython
     (`%USERPROFILE%\.rhinocode\py*-rh8\python.exe`) and copies all `*.ghuser`
     files into `%APPDATA%\Grasshopper\UserObjects\raystrack`.
   - If your Rhino Python is in a non‑standard location, set the environment
     variable `RAYSTRACK_RHINO_PY` to its `python.exe` and re‑run.
3) Start Rhino 8 -> Grasshopper. The user objects appear under “raystrack”.

macOS (Rhino 8)
1) Download or clone this repository.
2) Make the installer executable in a terminal if needed: `chmod +x grasshopper_mac_installer.command`.
3) Double‑click `grasshopper_mac_installer.command` (or run it from Terminal).
   - The script installs the Python package into Rhino 8’s CPython under
     `~/.rhinocode/py*-rh8/…` and copies all `*.ghuser` files into
     `~/Library/Application Support/McNeel/Rhinoceros/8.0/Plug-ins/Grasshopper/UserObjects/raystrack`
     (with sensible fallbacks for other layouts).
   - You can override the interpreter by setting `RAYSTRACK_RHINO_PY` to the
     Rhino 8 Python path.
4) Launch Rhino 8 -> Grasshopper. The user objects appear under “raystrack”.

Notes
- CUDA is optional. On Windows with an NVIDIA GPU, install a recent NVIDIA
  driver; Numba uses the driver runtime (no full CUDA toolkit required). On
  macOS, CUDA is not available; the CPU backend will be used.
- An example Grasshopper file is provided at `rhino/example.gh`.

Installation (Python only)

Use Raystrack as a normal Python package outside Rhino/Grasshopper:

From a local clone of this repo
```
pip install .
```

Or from a local path
```
pip install /path/to/raystrack
```

Requires Python 3.9+ with `numpy` and `numba` (installed automatically).
CUDA acceleration is used automatically if `numba.cuda` detects a compatible GPU.

Quick start (Python)
```python
import numpy as np
from raystrack import view_factor_matrix

# Each mesh is a tuple: (name: str, V: (N,3) float32, F: (M,3) int32)
V_a = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
F_a = np.array([[0,1,2],[0,2,3]], dtype=np.int32)  # two triangles

V_b = np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
F_b = F_a.copy()

meshes = [
    ("A", V_a, F_a),
    ("B", V_b, F_b),
]

res = view_factor_matrix(
    meshes,
    samples=256,   # sampling density per unit area (QMC grid)
    rays=256,      # rays per cell
    bvh="builtin",  # optional BVH acceleration (auto|off|builtin)
    reciprocity=True,
)

print(res["A"])  # e.g. {"B_front": 0.5, "B_back": 0.0, ...}
```

License
GPL-3.0-only — see `LICENSE`.
