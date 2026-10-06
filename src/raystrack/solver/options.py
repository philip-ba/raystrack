"""Immutable, shared controls for every estimator and output."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numbers


def _integer(name, value, minimum=1):
    """Validate integral settings without accepting Boolean values as counts."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Sampling:
    """Control the estimator, seed, sampling grid and emitter scheduling.

    Cosine tracing uses ``density`` and ``rays_per_cell``. CPU area-pair
    estimation uses ``pair_samples`` and its selected random sequence.
    Other faces of the emitting mesh participate in visibility and self view
    factors. ``flip_faces`` reverses emission only; receiver sides keep their
    original mesh winding (an outward box flipped inward receives on its back).
    """
    density: int = 16
    rays_per_cell: int = 128
    seed: int = 1
    mode: str = "fair"
    flip_faces: bool = False
    strategy: str = "cosine"
    sequence: str = "shifted_halton"
    pair_samples: int = 8192

    def __post_init__(self):
        """Validate sampling counts and the supported strategy combinations."""
        for name in ("density", "rays_per_cell", "pair_samples"):
            _integer(name, getattr(self, name))
        _integer("seed", self.seed, 0)
        if self.mode not in ("fair", "adaptive"):
            raise ValueError("mode must be fair or adaptive")
        if self.strategy not in ("cosine", "area_pair"):
            raise ValueError("strategy must be cosine or area_pair")
        if self.sequence not in ("shifted_halton", "random"):
            raise ValueError("sequence must be shifted_halton or random")
        if self.strategy == "cosine" and self.sequence != "shifted_halton":
            raise ValueError("cosine sampling requires shifted_halton")
        if not isinstance(self.flip_faces, bool):
            raise ValueError("flip_faces must be a bool")


@dataclass(frozen=True)
class Accuracy:
    """Control convergence checkpoints and the maximum randomized replicates.

    ``stderr`` compares sampling uncertainty to tolerance; ``delta`` compares
    successive estimates. A replicate limit can stop an unconverged result.
    """
    max_replicates: int = 100
    tolerance: float = 1e-4
    mode: str = "stderr"
    min_replicates: int = 5
    check_interval: int = 1
    min_rays: int = 0

    def __post_init__(self):
        """Validate convergence modes, finite tolerance and checkpoint counts."""
        for name in ("max_replicates", "min_replicates", "check_interval"):
            _integer(name, getattr(self, name))
        _integer("min_rays", self.min_rays, 0)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if self.mode not in ("stderr", "delta"):
            raise ValueError("accuracy mode must be stderr or delta")


@dataclass(frozen=True)
class Postprocessing:
    """Select explicit reciprocity processing for a complete scene result."""
    reciprocity: str = "none"

    def __post_init__(self):
        """Reject reciprocity modes outside the supported explicit algorithms."""
        if self.reciprocity not in ("none", "shortcut", "bidirectional", "rowsum"):
            raise ValueError("reciprocity must be none, shortcut, bidirectional, or rowsum")


@dataclass(frozen=True)
class SolveOptions:
    """Bundle independent sampling, accuracy and postprocessing settings.

    ``batch_size`` limits work per backend batch. It does not change the
    randomized sample prefix or the meaning of an additional ray budget.
    """
    sampling: Sampling = field(default_factory=Sampling)
    accuracy: Accuracy = field(default_factory=Accuracy)
    postprocessing: Postprocessing = field(default_factory=Postprocessing)
    batch_size: int = 65536

    def __post_init__(self):
        """Require typed option groups and a positive execution batch size."""
        for name, cls in (("sampling", Sampling), ("accuracy", Accuracy),
                          ("postprocessing", Postprocessing)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be a {cls.__name__}")
        _integer("batch_size", self.batch_size)


@dataclass(frozen=True)
class Budget:
    """Additional rays and a soft deadline, checked between bounded chunks."""
    rays: int | None = None
    time_ms: float | None = None

    def __post_init__(self):
        """Validate additional ray limits and finite soft deadlines."""
        if self.rays is not None:
            _integer("rays", self.rays, 0)
        if self.time_ms is not None and (not math.isfinite(self.time_ms) or self.time_ms < 0):
            raise ValueError("time_ms must be finite and nonnegative")
