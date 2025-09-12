import sys
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_matrix_json(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root in {path}")
    # Coerce to float
    out: Dict[str, Dict[str, float]] = {}
    for s, row in data.items():
        if not isinstance(row, dict):
            raise ValueError(f"Row for {s} must be a dict")
        out[s] = {k: float(v) for k, v in row.items()}
    return out


def parse_name(name: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Return (base, direction, tri_index) parsed from a key.

    Parsing is from the end:
      - Optional '_front'/'_back' direction suffix.
      - Optional '_<digits>' triangle index suffix.
    Remaining leading part is treated as base (may contain underscores).
    """
    direction = None
    tri_index: Optional[int] = None
    parts = name.split("_")
    # First strip trailing integer id if present (triangle index)
    if parts and parts[-1].isdigit():
        try:
            tri_index = int(parts.pop())
        except Exception:
            tri_index = None
    # Then strip trailing direction suffix if present
    if parts and parts[-1] in {"front", "back"}:
        direction = parts.pop()
    base = "_".join(parts) if parts else name
    return base, direction, tri_index


def aggregate_v3d(mat: Dict[str, Dict[str, float]]) -> Dict[str, Dict[Tuple[str, Optional[str]], float]]:
    """Aggregate triangle-split v3d matrix into mesh-level per-direction totals.

    Returns: {sender_base: {(receiver_base, direction_or_None): value}}
    If receiver direction suffix is missing in v3d keys, the direction is None,
    and we will compare against front+back combined from the VF matrix.
    """
    agg: Dict[str, Dict[Tuple[str, Optional[str]], float]] = defaultdict(lambda: defaultdict(float))
    for s_name, row in mat.items():
        s_base, s_dir, s_idx = parse_name(s_name)
        # Sender direction is unused; we accumulate per sender_base.
        for r_name, val in row.items():
            r_base, r_dir, r_idx = parse_name(r_name)
            # Keep r_dir as None if not specified; compare will handle combining
            agg[s_base][(r_base, r_dir)] += float(val)
    return agg


def normalize_vf(mat: Dict[str, Dict[str, float]]) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Normalize vf_matrix rows into {sender_base: {(receiver_base, dir): value}}."""
    out: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    for s_name, row in mat.items():
        s_base, s_dir, _ = parse_name(s_name)
        for r_key, val in row.items():
            r_base, r_dir, _ = parse_name(r_key)
            direction = r_dir or "back"
            out[s_base][(r_base, direction)] = float(val)
    return out


def compare(v3d_path: Path, vf_path: Path, top_n: int = 20) -> None:
    v3d_raw = load_matrix_json(v3d_path)
    vf_raw = load_matrix_json(vf_path)

    def strip_dir(key: str) -> str:
        # Remove trailing _front/_back if present
        if key.endswith("_front"):
            return key[:-6]
        if key.endswith("_back"):
            return key[:-5]
        return key

    # Build vf matrix with receiver keys normalized by stripping direction and summing
    vf_flat = {}
    for s, row in vf_raw.items():
        acc = {}
        for rkey, val in row.items():
            base = strip_dir(rkey)
            acc[base] = acc.get(base, 0.0) + float(val)
        vf_flat[s] = acc

    # v3d is already per-face mesh names; use as-is
    v3d = v3d_raw

    senders = sorted(set(v3d.keys()) | set(vf_flat.keys()))
    diffs = []
    missing_in_v3d = []
    extra_in_v3d = []

    for s in senders:
        row_v3d = v3d.get(s, {})
        row_vf = vf_flat.get(s, {})
        rkeys = sorted(set(row_v3d.keys()) | set(row_vf.keys()))
        for r in rkeys:
            v3 = float(row_v3d.get(r, 0.0))
            vf = float(row_vf.get(r, 0.0))
            diffs.append((abs(vf - v3), s, r, v3, vf))
        for r in row_vf.keys() - row_v3d.keys():
            missing_in_v3d.append((s, r, float(row_vf[r])))
        for r in row_v3d.keys() - row_vf.keys():
            extra_in_v3d.append((s, r, float(row_v3d[r])))

    diffs.sort(reverse=True, key=lambda x: x[0])
    print("Direct comparison per-face (sender -> receiver):")
    for d, s, r, v3, vf in diffs[:top_n]:
        rel = d / max(abs(v3), 1e-12)
        print(f" sender={s:30s} receiver={r:30s} v3d={v3:.6g} vf={vf:.6g} |abs={d:.6g} rel={rel:.3g}")

    # Row-sum comparison per sender
    print("\nRow sum comparison per sender:")
    for s in senders:
        sum_v3d = float(sum(v3d.get(s, {}).values()))
        sum_vf = float(sum(vf_flat.get(s, {}).values()))
        d = abs(sum_vf - sum_v3d)
        rel = d / max(abs(sum_v3d), 1e-12)
        print(f" sender={s:30s} sum_v3d={sum_v3d:.6g} sum_vf={sum_vf:.6g} |abs={d:.6g} rel={rel:.3g}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    v3d_file = here / "v3d_results.json"
    vf_file = here / "vf_matrix.json"
    if len(sys.argv) >= 3:
        v3d_file = Path(sys.argv[1])
        vf_file = Path(sys.argv[2])
    compare(v3d_file, vf_file)
