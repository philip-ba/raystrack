"""Executable-backed clear-plate reference, including mixed and annual steps."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from raystrack import CapabilityError
from raystrack.backends.native import native_library


@pytest.mark.parametrize("backend", ["native-builtin", "embree", "vulkan-rt"])
def test_clear_plate_against_pinned_radiance_and_independent_quadrature(backend, tmp_path):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("set RAYSTRACK_NATIVE_LIBRARY to exercise compiled transport")
    if not Path("C:/Radiance/bin/rtrace.exe").exists():
        pytest.skip("Pinned Radiance installation unavailable")
    try:
        native_library(backend)
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    command = [sys.executable, "validation/validate_clear_glazing.py", "--output", str(tmp_path),
               "--backends", backend, "--samples", "8192"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["radiance_version"] == "RADIANCE 5.4 2023-11-05 LBNL (5.4.4ee32974b1)"
    assert report["native_library_sha256"]
    assert report["max_energy_residual"] < 1e-12
    result = report["backends"][backend]
    assert result["max_analytic_error_w_m2"] < .002
    assert result["max_radiance_error_w_m2"] < .002
    assert result["linearity_max_error_w_m2"] < .002
    np.testing.assert_allclose(np.sum(result["diffuse_expected"]), 100, atol=1e-10)
    assert len(result["annual_indices"]) == len(result["annual_timestamps"]) == 4
    annual = result["full_annual_direct"]
    assert annual["steps"] == 8760
    assert annual["max_abs_error_w_m2"] < .002
    assert max(annual["exposure_relative_error"]) < .02
    with np.load(tmp_path / annual["saved_arrays"]) as arrays:
        assert arrays["actual"].shape == (8760, 2, 3)
        np.testing.assert_allclose(arrays["actual"][:,:,0], arrays["expected"], rtol=2e-5, atol=.002)
        np.testing.assert_array_equal(arrays["actual"][arrays["dni"] == 0], 0)
