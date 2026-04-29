from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
VIEW3D_REFERENCE_ROOT = Path(__file__).resolve().parent / "view3d_reference"


def ensure_repo_on_path() -> None:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


Mesh = Tuple[str, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class RaystrackRun:
    vf: Dict[str, Dict[str, float]]
    iterations: Dict[str, int]
    converged_before_max: bool
    max_iters: int
    min_iters: int
    tol: float
    tol_mode: str


def rectangle_xy(
    name: str,
    width: float,
    depth: float,
    z: float,
    *,
    normal: int = 1,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    cx, cy = center
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    y0, y1 = cy - depth / 2.0, cy + depth / 2.0
    V = np.asarray(
        [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x0, y1, z],
        ],
        dtype=np.float32,
    )
    if normal >= 0:
        F = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    else:
        F = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return name, V, F


def rectangle_yz(
    name: str,
    length_y: float,
    height_z: float,
    x: float,
    *,
    normal: int = 1,
    y_center: float = 0.0,
    z_min: float = 0.0,
) -> Mesh:
    y0, y1 = y_center - length_y / 2.0, y_center + length_y / 2.0
    z0, z1 = z_min, z_min + height_z
    V = np.asarray(
        [
            [x, y0, z0],
            [x, y1, z0],
            [x, y1, z1],
            [x, y0, z1],
        ],
        dtype=np.float32,
    )
    if normal >= 0:
        F = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    else:
        F = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return name, V, F


def disk_xy(
    name: str,
    radius: float,
    z: float,
    *,
    segments: int = 128,
    normal: int = 1,
) -> Mesh:
    if segments < 8:
        raise ValueError("segments must be >= 8")
    vertices = [[0.0, 0.0, z]]
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        vertices.append([radius * math.cos(a), radius * math.sin(a), z])

    faces = []
    for i in range(segments):
        a = i + 1
        b = 1 + ((i + 1) % segments)
        faces.append([0, a, b] if normal >= 0 else [0, b, a])
    return (
        name,
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def run_raystrack(
    meshes: List[Mesh],
    *,
    samples: int,
    rays: int,
    max_iters: int,
    seed: int = 11,
    tol: float = 1.0e-4,
    min_iters: int = 5,
) -> RaystrackRun:
    ensure_repo_on_path()
    import raystrack.main as raystrack_main
    from raystrack import view_factor_matrix
    from raystrack.params import MatrixParams

    log_messages: List[str] = []
    old_log = raystrack_main._log
    raystrack_main._log = log_messages.append
    params = MatrixParams(
        samples=samples,
        rays=rays,
        seed=seed,
        bvh="builtin",
        device="cpu",
        cuda_async=False,
        gpu_raygen=False,
        max_iters=max_iters,
        min_iters=min_iters,
        tol=tol,
        tol_mode="stderr",
        convergence_interval=1,
        reciprocity=False,
        enforce_reciprocity_rowsum=False,
        flip_faces=False,
    )
    try:
        vf = view_factor_matrix(meshes, params=params)
    finally:
        raystrack_main._log = old_log

    iterations: Dict[str, int] = {}
    pattern = re.compile(r"\[\s*(?P<name>[^\]]+?)\s*\]\s+(?P<iters>\d+)\s+iter")
    for msg in log_messages:
        match = pattern.search(msg)
        if match:
            iterations[match.group("name")] = int(match.group("iters"))

    active_iters = [value for value in iterations.values() if value > 0]
    converged_before_max = bool(active_iters) and all(value < max_iters for value in active_iters)
    return RaystrackRun(
        vf=vf,
        iterations=iterations,
        converged_before_max=converged_before_max,
        max_iters=max_iters,
        min_iters=min_iters,
        tol=tol,
        tol_mode="stderr",
    )


def row_total_to(row: Dict[str, float], receiver: str) -> float:
    total = 0.0
    for key, value in row.items():
        if key == receiver or key == f"{receiver}_front" or key == f"{receiver}_back":
            total += float(value)
    return total


def row_front_to(row: Dict[str, float], receiver: str) -> float:
    return float(row.get(f"{receiver}_front", row.get(receiver, 0.0)))


def face_area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]),
        axis=1,
    )


def base_from_result_key(key: str) -> str:
    if key.endswith("_front"):
        return key[:-6]
    if key.endswith("_back"):
        return key[:-5]
    return key


def totals_by_base(row: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in row.items():
        base = base_from_result_key(key)
        out[base] = out.get(base, 0.0) + float(value)
    return out


def write_case_result(
    case_name: str,
    *,
    description: str,
    formula: str,
    analytical: float,
    raystrack: float,
    tolerance: float,
    settings: Dict[str, object],
    run: RaystrackRun | None = None,
) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    diff = abs(raystrack - analytical)
    passed = diff <= tolerance
    lines = [
        f"case: {case_name}",
        f"description: {description}",
        f"formula: {formula}",
        "",
        f"analytical: {analytical:.10f}",
        f"raystrack:  {raystrack:.10f}",
        f"abs_diff:   {diff:.10f}",
        f"tolerance:  {tolerance:.10f}",
        f"passed:     {passed}",
        "",
        "settings:",
    ]
    for key, value in settings.items():
        lines.append(f"  {key}: {value}")
    if run is not None:
        lines.extend(
            [
                "",
                "convergence:",
                f"  tol_mode: {run.tol_mode}",
                f"  tol: {run.tol:.10f}",
                f"  min_iters: {run.min_iters}",
                f"  max_iters: {run.max_iters}",
                f"  converged_before_max: {run.converged_before_max}",
                "  iterations:",
            ]
        )
        for name, iters in run.iterations.items():
            lines.append(f"    {name}: {iters}")
    result_path = RESULTS_ROOT / f"{case_name}.txt"
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_path


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_canyon_meshes() -> List[Mesh]:
    ensure_repo_on_path()
    from examples.ex00_street_canyon_geometry import build_street_canyon

    return build_street_canyon()


def aggregate_view3d_faces(
    raw: Dict[str, Dict[str, float]],
    meshes: List[Mesh],
) -> Dict[str, Dict[str, float]]:
    face_area_by_name: Dict[str, float] = {}
    face_names_by_base: Dict[str, List[str]] = {}
    for name, V, F in meshes:
        areas = face_area(V, F)
        face_names = []
        for i, area in enumerate(areas, start=1):
            fname = f"{name}_{i}"
            face_area_by_name[fname] = float(area)
            face_names.append(fname)
        face_names_by_base[name] = face_names

    out: Dict[str, Dict[str, float]] = {}
    for sender_base, sender_faces in face_names_by_base.items():
        areas = np.asarray([face_area_by_name[name] for name in sender_faces], dtype=float)
        total_area = float(areas.sum())
        if total_area <= 0.0:
            weights = np.full(len(sender_faces), 1.0 / max(1, len(sender_faces)))
        else:
            weights = areas / total_area

        row: Dict[str, float] = {}
        for weight, sender_face in zip(weights, sender_faces):
            raw_row = raw.get(sender_face, {})
            for receiver_face, value in raw_row.items():
                receiver_base = receiver_face.rsplit("_", 1)[0]
                row[receiver_base] = row.get(receiver_base, 0.0) + float(weight) * float(value)
        out[sender_base] = row
    return out


def raystrack_base_matrix(vf: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for sender, row in vf.items():
        out[sender] = totals_by_base(row)
    return out


def max_abs_pair_diff(
    left: Dict[str, Dict[str, float]],
    right: Dict[str, Dict[str, float]],
    *,
    names: Iterable[str],
) -> Tuple[float, Tuple[str, str], float, float]:
    max_diff = -1.0
    max_pair = ("", "")
    max_left = 0.0
    max_right = 0.0
    name_list = list(names)
    for sender in name_list:
        lrow = left.get(sender, {})
        rrow = right.get(sender, {})
        for receiver in name_list:
            lv = float(lrow.get(receiver, 0.0))
            rv = float(rrow.get(receiver, 0.0))
            diff = abs(lv - rv)
            if diff > max_diff:
                max_diff = diff
                max_pair = (sender, receiver)
                max_left = lv
                max_right = rv
    return max_diff, max_pair, max_left, max_right
