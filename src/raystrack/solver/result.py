"""Indexed sparse snapshots with explicit sampling coverage and typed channels."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
import numpy as np
import math
from .options import _integer


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return readonly(value, value.dtype)
    return value


def readonly(value, dtype):
    array = np.asarray(value, dtype=dtype)
    # An immutable backing buffer prevents callers from re-enabling writes.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Channel:
    kind: str
    surface_id: str | None = None
    side: str | None = None
    patch: int | None = None

    def __post_init__(self):
        if self.kind not in ("surface", "sky", "rest", "unrequested"):
            raise ValueError("Unknown result channel kind")
        if self.kind == "surface":
            if not isinstance(self.surface_id, str) or not self.surface_id or self.patch is not None:
                raise ValueError("Surface channels require a surface ID and no patch")
            if self.side not in (None, "front", "back"):
                raise ValueError("side must be front, back, or unknown (None)")
        elif self.surface_id is not None or self.side is not None:
            raise ValueError("Only surface channels have a surface ID or side")
        if self.patch is not None:
            if self.kind != "sky" or isinstance(self.patch, bool) or not isinstance(self.patch, (int, np.integer)) or not 0 <= self.patch < 145:
                raise ValueError("Sky patch must be a zero-based index in 0..144")


@dataclass(frozen=True)
class SparseValues:
    row_indices: np.ndarray
    channel_indices: np.ndarray
    estimates: np.ndarray
    errors: np.ndarray

    def __post_init__(self):
        for name, dtype in (("row_indices", np.int64), ("channel_indices", np.int64),
                            ("estimates", np.float64), ("errors", np.float64)):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1:
                raise ValueError("Sparse arrays must be one-dimensional")
            if name.endswith("indices") and raw.dtype.kind not in "iu" and len(raw):
                raise ValueError("Sparse indices must be integers")
            object.__setattr__(self, name, readonly(raw, dtype))
        size = len(self.estimates)
        if any(len(getattr(self, name)) != size for name in ("row_indices", "channel_indices", "errors")):
            raise ValueError("Sparse arrays must have matching lengths")
        if np.any(self.row_indices < 0) or np.any(self.channel_indices < 0):
            raise ValueError("Sparse indices must be nonnegative")
        if not np.all(np.isfinite(self.estimates)) or np.any(self.estimates < 0):
            raise ValueError("Estimates must be finite and nonnegative")
        if np.any(np.isinf(self.errors)) or np.any(self.errors < 0):
            raise ValueError("Errors must be nonnegative or unknown (NaN)")
        pairs = list(zip(self.row_indices, self.channel_indices))
        if len(set(pairs)) != size:
            raise ValueError("Sparse entries must be unique")


@dataclass(frozen=True)
class Result:
    sender_ids: tuple[str, ...]
    channels: tuple[Channel, ...]
    data: SparseValues
    coverage: np.ndarray
    statistics: Mapping = field(default_factory=dict)
    execution: Mapping = field(default_factory=dict)
    scene_revision: int = 0
    rays_used: int | None = 0
    cumulative_rays: int | None = 0
    status: str = "not_started"
    converged: bool | None = False
    elapsed_ms: float = 0.0
    provenance: tuple[Mapping, ...] = ()

    def __post_init__(self):
        _integer("scene_revision", self.scene_revision, 0)
        for name in ("rays_used", "cumulative_rays"):
            if getattr(self, name) is not None:
                _integer(name, getattr(self, name), 0)
        if self.rays_used is not None and self.cumulative_rays is not None and self.rays_used > self.cumulative_rays:
            raise ValueError("rays_used cannot exceed cumulative_rays")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string")
        if self.converged is not None and not isinstance(self.converged, bool):
            raise ValueError("converged must be a bool or unknown (None)")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and nonnegative")
        object.__setattr__(self, "sender_ids", tuple(self.sender_ids))
        object.__setattr__(self, "channels", tuple(self.channels))
        if len(set(self.sender_ids)) != len(self.sender_ids) or any(not isinstance(s, str) or not s for s in self.sender_ids):
            raise ValueError("Sender IDs must be unique nonempty strings")
        if len(set(self.channels)) != len(self.channels) or any(not isinstance(c, Channel) for c in self.channels):
            raise ValueError("Result channels must be unique Channel objects")
        raw_coverage = np.asarray(self.coverage)
        if raw_coverage.shape != (len(self.sender_ids),) or not np.all(np.isin(raw_coverage, (-1, 0, 1))):
            raise ValueError("coverage must contain -1 (unknown), 0, or 1 per sender")
        object.__setattr__(self, "coverage", readonly(raw_coverage, np.int8))
        if not isinstance(self.data, SparseValues):
            raise TypeError("data must be SparseValues")
        if np.any(self.data.row_indices >= len(self.sender_ids)) or np.any(self.data.channel_indices >= len(self.channels)):
            raise ValueError("Sparse entry index exceeds result dimensions")
        if np.any(self.coverage[self.data.row_indices] == 0):
            raise ValueError("Uncomputed rows cannot contain stored estimates")
        object.__setattr__(self, "statistics", freeze(self.statistics))
        object.__setattr__(self, "execution", freeze(self.execution))
        object.__setattr__(self, "provenance", tuple(freeze(p) for p in self.provenance))
        object.__setattr__(self, "_sender_index", {s: i for i, s in enumerate(self.sender_ids)})
        object.__setattr__(self, "_channel_index", {c: i for i, c in enumerate(self.channels)})
        object.__setattr__(self, "_entries", {(int(r), int(c)): i for i, (r, c) in enumerate(zip(self.data.row_indices, self.data.channel_indices))})

    def value(self, sender, channel):
        row, col = self._sender_index[sender], self._channel_index[channel]
        entry = self._entries.get((row, col))
        if entry is not None:
            return float(self.data.estimates[entry])
        return 0.0 if self.coverage[row] == 1 else None

    def error(self, sender, channel):
        row, col = self._sender_index[sender], self._channel_index[channel]
        entry = self._entries.get((row, col))
        if entry is not None:
            error = float(self.data.errors[entry])
            return error if np.isfinite(error) else None
        if self.coverage[row] != 1:
            return None
        stats = self.statistics.get("emitters", {}).get(sender, {})
        kind = "sky" if channel.kind == "sky" else "matrix"
        est = stats.get(kind) or {}
        return 0.0 if est.get("replicates", 0) >= 2 and not est.get("pending_rays", 0) else None

    def row(self, sender):
        return MappingProxyType({c: self.value(sender, c) for c in self.channels})

    def dense(self, *, unknown=np.nan):
        values = np.full((len(self.sender_ids), len(self.channels)), unknown, np.float64)
        values[self.coverage == 1] = 0.0
        values[self.data.row_indices, self.data.channel_indices] = self.data.estimates
        return values
