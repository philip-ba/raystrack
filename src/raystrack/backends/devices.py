"""Lazy device selection; portable GPU support is an optional dependency."""
from __future__ import annotations

import importlib.util


def resolve_backend(device: str = "auto") -> str:
    mode = (device or "auto").lower()
    if mode not in ("auto", "gpu", "cpu", "cuda", "taichi", "vulkan", "metal"):
        raise ValueError("device must be auto, gpu, cpu, cuda, taichi, vulkan, or metal")
    if mode == "cpu":
        return "cpu"
    if mode in ("taichi", "vulkan", "metal"):
        from ..utils.taichi_trace import get_taichi_backend
        get_taichi_backend(arch="auto" if mode == "taichi" else mode)
        return mode
    from numba import cuda
    if cuda.is_available():
        return "cuda"
    if mode == "cuda":
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if importlib.util.find_spec("taichi") is not None:
        from ..utils.taichi_trace import get_taichi_backend, TaichiUnavailableError
        try:
            get_taichi_backend()
            return "taichi"
        except TaichiUnavailableError:
            if mode == "gpu":
                raise
    if mode == "gpu":
        raise RuntimeError("No GPU backend available. Install raystrack[portable-gpu] for Vulkan/Metal support.")
    return "cpu"


def available_devices() -> dict:
    """Report CUDA and portable GPU availability, including failure reasons.

    Probing portable support loads its GPU drivers but does not initialize a runtime.
    Explicitly requested GPU devices never fall back to CPU.
    """
    from numba import cuda
    from numba import config
    report = {"cpu": {"available": True}}
    try:
        available = bool(cuda.is_available())
        report["cuda"] = {"available": available}
        if available:
            if config.ENABLE_CUDASIM:
                report["cuda"].update(name="CUDA simulator", id=0, simulated=True)
            else:
                dev = cuda.get_current_device()
                name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
                report["cuda"].update(name=name, id=int(dev.id))
    except Exception as exc:
        report["cuda"] = {"available": False, "reason": str(exc)}
    if importlib.util.find_spec("taichi") is None:
        report["taichi"] = {"available": False, "reason": "Install raystrack[portable-gpu]"}
    else:
        from ..utils.taichi_trace import probe_taichi
        report["taichi"] = probe_taichi()
    return report
