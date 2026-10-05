"""Acceptance reports must contain evidence from the installed Grasshopper host."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "raystrack_gh_host_smoke", ROOT / "tools" / "grasshopper" / "host_smoke.py"
)
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "raystrack_gh_assembly_contract", ROOT / "tools" / "grasshopper" / "assembly_contract.py"
)
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
CONTRACT_SPEC.loader.exec_module(CONTRACT)


def complete_report():
    return {
        "phase": "passed", "rhino": "8.29", "assembly": "installed/Raystrack.Components.gha",
        "components": ["RS component " + str(index) for index in range(14)],
        "active_heartbeats": 4, "live_ray_counts": [4096, 8192], "result_rays": 12288,
        "pair_value": 0.1998, "icons_and_tooltips": True, "ribbon_icon": True, "instance_transform": True,
        "save_load_preserved": True, "cancelled_partial": True, "document_worker_stopped": True,
        "runtime_multiplexing": True, "reopen_disarmed": True, "owned_worker_pid": 12345, "document": "plates.gh",
    }


def test_host_report_accepts_complete_real_host_evidence():
    report = complete_report()
    assert HOST.validate_report(report) is report


def test_live_cancel_work_may_exceed_the_finished_analytic_solve():
    report = complete_report()
    report.update(result_rays=12288, live_ray_counts=[4096, 65536, 131072], cancelled_rays=135168)
    assert HOST.validate_report(report) is report


@pytest.mark.parametrize("field,value", [
    ("phase", "waiting"), ("active_heartbeats", 0), ("live_ray_counts", [4096, 4096]),
    ("result_rays", 1), ("pair_value", 0.5), ("save_load_preserved", False),
    ("cancelled_partial", False), ("document_worker_stopped", False), ("reopen_disarmed", False),
    ("ribbon_icon", False), ("owned_worker_pid", 0), ("document", None), ("runtime_multiplexing", False),
    ("instance_transform", False),
])
def test_host_report_rejects_incomplete_or_inconsistent_evidence(field, value):
    report = complete_report()
    report[field] = value
    with pytest.raises(AssertionError):
        HOST.validate_report(report)


@pytest.mark.skipif(
    sys.platform != "win32" or not CONTRACT.DEFAULT_COMPILER.is_file()
    or not (CONTRACT.DEFAULT_RHINO / "Plug-ins" / "Grasshopper" / "Grasshopper.dll").is_file(),
    reason="Managed GH contracts need Windows' local compiler and Rhino reference assemblies",
)
def test_actual_csharp_values_diagnoses_and_icon_rendering_without_gui():
    """Exercise the actual C# methods, not a Python copy of their behavior."""
    report = CONTRACT.run_contract()
    assert report["phase"] == "passed"
    assert report["host_gui_opened"] is False
    assert report["distribution_built"] is False
    assert report["installed"] is False
    assert len(report["summaries"]) == 13
    assert len(report["invalid_cases"]) == 10
    assert len(report["icons"]) >= 14
    assert len(report["component_help"]) == 14
    assert len(report["transforms"]) == 8
