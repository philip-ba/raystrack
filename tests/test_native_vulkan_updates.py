"""Real Vulkan geometry edits retain the device and preserve stable hit slots."""

import os

import numpy as np
import pytest

from raystrack.backends import CapabilityError
from raystrack.backends.native import NativeGeometry, native_library


def test_vulkan_updates_match_fresh_geometry_and_reject_invalid_topology():
    if not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"):
        pytest.skip("set RAYSTRACK_NATIVE_LIBRARY to exercise Vulkan hardware")
    try:
        library = native_library("vulkan-rt")
    except CapabilityError as error:
        if "absent from this build" in str(error):
            pytest.skip(str(error))
        raise
    if not hasattr(library, "rs_update"):
        pytest.skip("configured native library predates geometry updates")

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    triangles = np.array([[0, 1, 2]], np.int32)
    empty_vertices = np.empty((0, 3), np.float32)
    empty_triangles = np.empty((0, 3), np.int32)
    empty = ("empty", empty_vertices, empty_triangles)
    initial = [empty, ("panel", vertices, triangles)]

    def trace(geometry, surfaces):
        origins = np.array([[0.2, 0.3, 2], [2.2, 0.3, 2]], np.float32)
        directions = np.tile(np.array([0, 0, -1], np.float32), (2, 1))
        active = np.ones(surfaces, np.uint8)
        hits = np.empty(2, np.int32)
        fronts = np.empty(2, np.uint8)
        blocked = np.empty(2, np.uint8)
        geometry.longwave(origins, directions, active, -1, 0, hits, fronts, blocked)
        return hits, fronts, blocked

    native = NativeGeometry("vulkan-rt", initial)
    try:
        original_handle = native._handle.value
        np.testing.assert_array_equal(trace(native, 2)[0], [1, -1])
        shifted = vertices + np.array([2, 0, 0], np.float32)
        deformed = shifted.copy()
        deformed[2, 2] = 0.4
        revisions = [
            [empty, ("panel", shifted, triangles)],
            [empty, ("panel", deformed, triangles)],
            [("panel", vertices, triangles[:, ::-1].copy())],
            [],
            initial,
        ]
        for meshes in revisions:
            assert native.update(meshes)
            assert native._handle.value == original_handle
            fresh = NativeGeometry("vulkan-rt", meshes)
            try:
                for actual, expected in zip(trace(native, len(meshes)), trace(fresh, len(meshes))):
                    np.testing.assert_array_equal(actual, expected)
            finally:
                fresh.close()

        with pytest.raises(RuntimeError, match="triangle index"):
            native.update([("invalid", vertices, np.array([[0, 1, 99]], np.int32))])
        np.testing.assert_array_equal(trace(native, 2)[0], [1, -1])
    finally:
        native.close()
