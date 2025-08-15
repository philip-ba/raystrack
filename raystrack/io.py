from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union


VFDict = Dict[str, Dict[str, float]]
VFInput = Union[VFDict, List[VFDict]]


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
    """Save a view-factor matrix to a JSON file."""
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

    with path.open("w", encoding="utf-8") as fh:
        json.dump(flat, fh, ensure_ascii=False, indent=2, sort_keys=True)

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

