"""Real hardware traversal parity for shared host delta and finite transport."""

import os

import numpy as np
import pytest

from raystrack import CapabilityError, Mesh, Scene
from raystrack.backends.native import native_library
from raystrack.shortwave import ClearGlazing, DiffuseSheet, ShortwaveSolver


@pytest.fixture
def hardware():
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("set RAYSTRACK_NATIVE_LIBRARY to exercise Vulkan hardware")
    try:
        native_library("vulkan-rt")
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise


def panel(z, reverse=False):
    vertices = [[-20, -20, z], [20, -20, z], [20, 20, z], [-20, 20, z]]
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return Mesh(vertices, faces[:, ::-1] if reverse else faces)


@pytest.mark.parametrize("reverse,index", [(False, 1.52), (True, 1.52), (True, 1)])
def test_vulkan_mixed_diffuse_delta_stream_and_geometry_update(hardware, reverse, index):
    scene = Scene.from_meshes({"sheet": panel(0), "glass": panel(1, reverse)})
    materials = {
        "sheet": DiffuseSheet([.2, .3, .4], [.4, .2, .1], [.3, .4, .5]),
        "glass": ClearGlazing(index),
    }
    args = ([[.123, .234, .5], [.271, .163, -.5]], [[0, 0, -1], [0, 0, 1]],
            [[0, 0, 1], [.6, 0, .8]])
    options = dict(dni=[[800, 600, 400], [300, 400, 500]], dhi=80, samples=2048,
                   bounces=3, seed=29, channel_basis="controlled RGB parity")
    with ShortwaveSolver(scene, backend="vulkan-rt", materials=materials) as gpu:
        original = gpu.solve(*args, **options)
        with ShortwaveSolver(scene, backend="native-builtin", materials=materials) as cpu:
            reference = cpu.solve(*args, **options)
        # Both traversals consume the identical host sample stream. The fixture
        # avoids edges and coincidence; only float hit reconstruction differs.
        np.testing.assert_allclose(original.irradiance, reference.irradiance,
                                   rtol=2e-6, atol=2e-5)
        gpu.set_material("glass", ClearGlazing(1))
        material_edit = gpu.solve(*args, **options)
        scene.update_mesh("glass", panel(1.5, not reverse))
        updated = gpu.solve(*args, **options)
    fresh_materials = dict(materials, glass=ClearGlazing(1))
    with ShortwaveSolver(scene, backend="vulkan-rt", materials=fresh_materials) as fresh:
        rebuilt = fresh.solve(*args, **options)
    np.testing.assert_array_equal(updated.irradiance, rebuilt.irradiance)
    assert updated.backend == "vulkan-rt"
    assert original.material_revision < material_edit.material_revision
    assert original.geometry_revision == material_edit.geometry_revision < updated.geometry_revision
    assert np.isfinite(updated.irradiance).all()
    assert (updated.irradiance >= 0).all()
    assert np.any(updated.irradiance[:, :, 2:] > 0)
    assert not original.irradiance.flags.writeable
