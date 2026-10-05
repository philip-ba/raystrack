"""Execution backends and capability reporting."""
from .devices import available_devices
from .calibration import ExecutionPlan

class CapabilityError(ValueError):
    """The requested estimator/output is unsupported by this backend."""

__all__ = ["available_devices", "ExecutionPlan", "CapabilityError"]
