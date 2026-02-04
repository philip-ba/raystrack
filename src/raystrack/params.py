from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class MatrixParams:
    """Configuration for scene-to-scene view-factor solves.

    Parameters
    ----------
    samples : int
        Quasi-Monte Carlo grid per side used for emission per emitter.
    rays : int
        Rays per grid cell.
    seed : int
        Base RNG seed. Each emitter/iteration derives its own sub-seed.
    bvh : {"auto","off","builtin"}
        Toggle a built-in BVH for faster visibility.
    device : {"auto","gpu","cpu"}
        Visibility device. GPU uses CUDA hitmask kernels. CPU uses numba njit.
    cuda_async : bool
        Use pinned memory + streams for overlap on CUDA.
    gpu_raygen : bool
        Generate rays on GPU to avoid H2D copies of rays.
    max_iters : int
        Maximum number of Monte-Carlo iterations.
    tol : float
        Convergence tolerance. Interpretation depends on ``tol_mode``.
    tol_mode : {"delta", "stderr"}
        - "delta": stop when successive cumulative estimates change by < tol.
        - "stderr": stop when per-iteration replicate standard error is <= tol.
    min_iters : int
        Minimum number of Monte-Carlo iterations before a convergence check.
    reciprocity : bool
        Also compute inverse view factors via reciprocity.
    enforce_reciprocity_rowsum : bool
        After computation, enforce reciprocity and make each row sum to 1 using
        symmetric diagonal scaling.
    """
    samples: int = 16
    rays: int = 128
    seed: int = 1
    bvh: str = "auto"
    device: str = "auto"
    cuda_async: bool = True
    gpu_raygen: bool = True
    max_iters: int = 100
    tol: float = 1e-4
    tol_mode: str = "stderr"
    min_iters: int = 5
    reciprocity: bool = True
    enforce_reciprocity_rowsum: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkyParams:
    """Configuration for merged sky view-factor solves.

    Parameters
    ----------
    samples : int
        Quasi-Monte Carlo grid per side used for emission per emitter.
    rays : int
        Rays per grid cell.
    seed : int
        Base RNG seed. Each emitter/iteration derives its own sub-seed.
    bvh : {"auto","off","builtin"}
        Toggle a built-in BVH for faster visibility.
    device : {"auto","gpu","cpu"}
        Visibility device. GPU uses CUDA hitmask kernels. CPU uses numba njit.
    cuda_async : bool
        Use pinned memory + streams for overlap on CUDA.
    gpu_raygen : bool
        Generate rays on GPU to avoid H2D copies of rays.
    max_iters : int
        Maximum number of Monte-Carlo iterations.
    tol : float
        Convergence tolerance. Interpretation depends on ``tol_mode``.
    tol_mode : {"delta", "stderr"}
        - "delta": stop when successive cumulative estimates change by < tol.
        - "stderr": stop when per-iteration replicate standard error is <= tol.
    min_iters : int
        Minimum number of Monte-Carlo iterations before a convergence check.
    """
    samples: int = 16
    rays: int = 128
    seed: int = 1
    bvh: str = "auto"
    device: str = "auto"
    cuda_async: bool = True
    gpu_raygen: bool = True
    max_iters: int = 100
    tol: float = 1e-4
    tol_mode: str = "stderr"
    min_iters: int = 5

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["MatrixParams", "SkyParams"]
