"""Acceptance reports must contain evidence from the installed Grasshopper host."""
from __future__ import annotations

import importlib.util
import os
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
        "components": ["RS component " + str(index) for index in range(15)],
        "active_heartbeats": 4, "live_ray_counts": [4096, 8192], "result_rays": 12288,
        "pair_value": 0.1998, "icons_and_tooltips": True, "ribbon_icon": True, "surface_transform": True,
        "to_brep_geometry": True, "sky_labels": True,
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
    ("surface_transform", False), ("to_brep_geometry", False), ("sky_labels", False),
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
    assert len(report["summaries"]) == 15
    assert len(report["invalid_cases"]) == 13
    assert len(report["icons"]) >= 14
    assert len(report["component_help"]) == 15
    assert len(report["transforms"]) == 8
    # Compare the actual C# dome sectors against the numerical solver, including
    # wraparound sectors and the stagger between rings, not a second geometry implementation.
    from raystrack.utils.cpu_trace import _tregenza_patch_id
    import math
    patches = report["sky_and_devices"]["patch_bounds"]
    assert len(patches) == 145
    total_solid_angle = 0
    for patch in patches:
        assert _tregenza_patch_id(*patch["direction"]) == patch["index"]
        total_solid_angle += (patch["end"] - patch["start"]) * (math.sin(patch["upper"]) - math.sin(patch["lower"]))
        for elevation_fraction, azimuth_fraction in ((.01, .01), (.01, .99), (.99, .01), (.99, .99)):
            elevation = patch["lower"] + elevation_fraction * (patch["upper"] - patch["lower"])
            azimuth = patch["start"] + azimuth_fraction * (patch["end"] - patch["start"])
            direction = math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)
            assert _tregenza_patch_id(*direction) == patch["index"]
    assert total_solid_angle == pytest.approx(2 * math.pi)


@pytest.mark.skipif(os.environ.get("RAYSTRACK_TEST_NATIVE_GH") != "1",
                    reason="Opt in to licensed headless Rhino geometry with RAYSTRACK_TEST_NATIVE_GH=1")
def test_native_gh_connections_breps_tags_and_solver_patch_directions():
    report = CONTRACT.run_contract(native=True)
    native = report["native_geometry"]
    assert native["valid_spherical_patches"] == 145
    assert native["world_transform_once"] and native["native_text_tags"]
    assert native["components"]["query_accepts_sky_object"]
    assert native["components"]["aligned_brep_text_outputs"]
    assert native["components"]["merged_and_hidden_labels"]
    from raystrack.utils.cpu_trace import _tregenza_patch_id
    for patch, directions in enumerate(native["interior_directions"]):
        for direction in directions:
            assert _tregenza_patch_id(*direction) == patch
