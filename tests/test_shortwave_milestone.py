"""Integrated reciprocal sheets and material-frame finite lobe checks."""
import os

import numpy as np
import pytest

from raystrack import Mesh, Scene
from raystrack.backends.native import native_library
from raystrack.shortwave import DiffuseSheet, KlemsBSDF, ShortwaveSolver, ThinDiffuse


@pytest.fixture(params=("native-builtin", "embree", "vulkan-rt"))
def backend(request):
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("requires native library")
    native_library(request.param)
    return request.param


def panel(z=1, width=10000, reverse=False):
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return Mesh([[-width, -width, z], [width, -width, z], [width, width, z],
                 [-width, width, z]], faces[:, ::-1] if reverse else faces)


def test_reciprocal_sheet_sides_winding_edits_and_finite_escape(backend):
    # Large sheet analytic error <0.1 W/m2; no stochastic fit to reference.
    scene = Scene.from_meshes({"sheet": panel()})
    points, normals = [[0, .1, 2], [0, .1, 0]], [[0, 0, -1], [0, 0, 1]]
    options = dict(dni=800, dhi=100, samples=4096, seed=42)
    with ShortwaveSolver(scene, backend=backend,
                         materials={"sheet": DiffuseSheet(.3, .1, .4)}) as solver:
        original = solver.solve(points, normals, [0, 0, 1], **options)
        handle = solver._native
        solver.set_material("sheet", ThinDiffuse(.2, .4))
        symmetric = solver.solve(points, normals, [0, 0, 1], **options)
        assert handle is solver._native
        solver.set_material("sheet", DiffuseSheet(.3, .1, .4))
        scene.update_mesh("sheet", panel(reverse=True))
        reversed_result = solver.solve(points, normals, [0, 0, 1], **options)
        scene.update_mesh("sheet", panel(width=.1, reverse=True))
        finite = solver.solve(points, normals, [0, 0, 1], **options)
        with ShortwaveSolver(scene, backend=backend,
                             materials={"sheet": DiffuseSheet(.3, .1, .4)}) as fresh:
            np.testing.assert_array_equal(finite.irradiance,
                                          fresh.solve(points, normals, [0, 0, 1], **options).irradiance)
    np.testing.assert_allclose(original.total[0, :, 0], [270, 360], atol=.1)
    np.testing.assert_allclose(symmetric.total[0, :, 0], [180, 360], atol=.1)
    np.testing.assert_allclose(reversed_result.total[0, :, 0], [90, 360], atol=.1)
    assert finite.irradiance[0, 1, 3, 0] < 10
    assert original.geometry_revision == 0 and reversed_result.geometry_revision == 1
    assert not original.irradiance.flags.writeable


def test_anisotropic_finite_transmission_rotates_with_material_frame(backend):
    matrices = np.zeros((4, 145, 145))
    matrices[2, 5, 69] = .8
    scene = Scene.from_meshes({"sheet": panel()})
    angle = np.radians(10)
    directions = [[np.sin(angle), 0, np.cos(angle)], [0, np.sin(angle), np.cos(angle)]]
    options = dict(dni=800, samples=16384, replicates=3, seed=83)
    with ShortwaveSolver(scene, backend=backend,
                         materials={"sheet": KlemsBSDF(matrices)}) as solver:
        original = solver.solve([0, .1, 0], [0, 0, 1], directions, **options)
        rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        scene.update_transform("sheet", rotation)
        rotated = solver.solve([0, .1, 0], [0, 0, 1], directions, **options)
    assert original.total[0, 0, 0] > 1 and original.total[1, 0, 0] == 0
    assert rotated.total[1, 0, 0] > 1 and rotated.total[0, 0, 0] == 0
    error = np.hypot(original.standard_error[0, 0, 3, 0], rotated.standard_error[1, 0, 3, 0])
    assert abs(original.total[0, 0, 0] - rotated.total[1, 0, 0]) < 4 * error + .01
