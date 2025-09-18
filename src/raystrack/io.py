from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np


VFDict = Dict[str, Dict[str, float]]
VFInput = Union[VFDict, List[VFDict]]
MeshTuple = Tuple[str, np.ndarray, np.ndarray]
Meshes = List[MeshTuple]


def _flatten_vf_matrix(vf_matrix: VFInput) -> VFDict:
    """Normalize input to a single dict {sender: {receiver: value}}.

    Accepts either a dict or a list of dicts (later entries overwrite earlier
    ones on the same sender key).
    """
    if isinstance(vf_matrix, list):
        flat: VFDict = {}
        for d in vf_matrix:
            if not isinstance(d, dict):
                raise TypeError("All elements of vf_matrix list must be dicts")
            flat.update(d)
        return flat
    if isinstance(vf_matrix, dict):
        return vf_matrix
    raise TypeError("vf_matrix must be a dict or list of dicts")


def save_vf_matrix_json(vf_matrix: VFInput, save_path: str) -> str:
    """Save a view-factor matrix to a JSON file.

    Receivers with value exactly 0.0 are omitted from the saved JSON to reduce
    file size and noise.
    """
    flat = _flatten_vf_matrix(vf_matrix)

    # Basic validation of structure and numeric values
    for sender, row in flat.items():
        if not isinstance(sender, str):
            raise TypeError("Sender keys must be strings")
        if not isinstance(row, dict):
            raise TypeError(f"Row for '{sender}' must be a dict mapping receiver->value")
        for recv, val in row.items():
            if not isinstance(recv, str):
                raise TypeError("Receiver keys must be strings")
            try:
                float(val)
            except Exception:
                raise TypeError(f"Value for '{sender}'->'{recv}' must be numeric")

    path = Path(save_path)
    if path.suffix.lower() == "":
        path = path.with_suffix(".json")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Drop zero-valued receivers
    cleaned: VFDict = {}
    for sender, row in flat.items():
        pruned = {k: float(v) for k, v in row.items() if float(v) != 0.0}
        cleaned[sender] = pruned

    with path.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, ensure_ascii=False, indent=2, sort_keys=True)

    return str(path.resolve())


def load_vf_matrix_json(load_path: str) -> VFDict:
    """Load a view-factor matrix JSON file and return a dictionary."""
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {load_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise TypeError("Loaded JSON must be a dict of dicts")

    # Validate and coerce values to float
    out: VFDict = {}
    for sender, row in data.items():
        if not isinstance(sender, str) or not isinstance(row, dict):
            raise TypeError("Invalid structure: expected {str: {str: number}}")
        new_row: Dict[str, float] = {}
        for recv, val in row.items():
            if not isinstance(recv, str):
                raise TypeError("Receiver keys must be strings")
            try:
                new_row[recv] = float(val)
            except Exception:
                raise TypeError(f"Value for '{sender}'->'{recv}' must be numeric")
        out[sender] = new_row

    return out


# ---------------------------------------------------------------
# Mesh geometry JSON IO
# ---------------------------------------------------------------

def save_meshes_json(meshes: Meshes, save_path: str) -> str:
    """Save meshes to a JSON file.

    The expected input format is a list of tuples:
        [(name: str, V: float32[N,3], F: int32[M,3]), ...]

    The JSON structure is:
        { "meshes": [
            {"name": str,
             "vertices": [[x,y,z], ...],
             "faces": [[i,j,k], ...]
            }, ...
        ]}
    """
    if not isinstance(meshes, list):
        raise TypeError("meshes must be a list of (name, V, F) tuples")

    payload = {"meshes": []}
    for item in meshes:
        if not (isinstance(item, tuple) and len(item) == 3):
            raise TypeError("Each mesh must be a (name, V, F) tuple")
        name, V, F = item
        if not isinstance(name, str) or name.strip() == "":
            raise TypeError("Mesh name must be a non-empty string")
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"Vertices for '{name}' must have shape (N,3)")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"Faces for '{name}' must have shape (M,3) of triangles")
        payload["meshes"].append(
            {
                "name": name,
                "vertices": V.tolist(),
                "faces": F.tolist(),
            }
        )

    path = Path(save_path)
    if path.suffix.lower() == "":
        path = path.with_suffix(".json")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return str(path.resolve())


def load_meshes_json(load_path: str) -> Meshes:
    """Load meshes from a JSON file saved by save_meshes_json.

    Returns a list of (name, V, F), where V is float32[N,3], F is int32[M,3].
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {load_path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "meshes" not in data:
        raise TypeError("Invalid mesh JSON: expected an object with 'meshes' list")
    meshes_raw = data["meshes"]
    if not isinstance(meshes_raw, list):
        raise TypeError("'meshes' must be a list")

    out: Meshes = []
    for i, entry in enumerate(meshes_raw):
        if not isinstance(entry, dict):
            raise TypeError("Each entry in 'meshes' must be an object")
        name = entry.get("name")
        V = entry.get("vertices")
        F = entry.get("faces")
        if not isinstance(name, str) or name.strip() == "":
            raise TypeError(f"Entry {i}: 'name' must be a non-empty string")
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"Entry {i} ('{name}'): vertices must have shape (N,3)")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"Entry {i} ('{name}'): faces must have shape (M,3)")
        out.append((name, V, F))

    return out
