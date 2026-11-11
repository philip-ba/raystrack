"""Private NumPy-to-C++ batch interface. Loading is always explicit."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path

from . import CapabilityError


BACKENDS = {"native-builtin": 0, "embree": 1, "vulkan-rt": 2}
_pointer = ct.c_void_p
_size = ct.c_size_t


class _Surface(ct.Structure):
    _fields_ = [("vertices", _pointer), ("triangles", _pointer),
                ("vertex_count", _size), ("triangle_count", _size)]


class Lighting(ct.Structure):
    _fields_ = [("sun", _pointer), ("dni", _pointer), ("sky", _pointer),
                ("ring_offsets", _pointer), ("rings", _size), ("sky_stride", _size),
                ("ring_height", ct.c_double), ("north_degrees", ct.c_double)]


_RGB = ct.c_double * 3
_VEC3 = ct.c_float * 3


class OpticalMaterial(ct.Structure):
    _fields_ = [("front", _RGB), ("back", _RGB), ("transmission", _RGB),
                ("klems", _pointer), ("reverse_cdf", _pointer),
                ("tangent", _VEC3), ("alternate", _VEC3),
                ("kind", ct.c_uint32), ("refractive_index", ct.c_double),
                ("delta_normals", _pointer), ("delta_normal_count", _size),
                ("absorption", _RGB)]


@lru_cache(maxsize=4)
def _load(path):
    try:
        library = ct.CDLL(path)
        signatures = (
            ("rs_abi", ct.c_uint, ()),
            ("rs_backends", ct.c_uint, ()),
            ("rs_error", ct.c_char_p, ()),
            ("rs_create", ct.c_int, (ct.c_int, ct.POINTER(_Surface), _size, ct.POINTER(_pointer))),
            ("rs_destroy", None, (_pointer,)),
            ("rs_shortwave", ct.c_int, (_pointer, _pointer, _size, ct.POINTER(Lighting),
                                        _size, _pointer, ct.c_uint32, ct.c_uint64,
                                        ct.c_uint32, ct.c_float, _pointer)),
            ("rs_longwave", ct.c_int, (_pointer, _pointer, _pointer, _size, _pointer,
                                       ct.c_int32, ct.c_int32, _pointer, _pointer, _pointer)),
        )
        for name, result, arguments in signatures:
            function = getattr(library, name)
            function.restype = result
            function.argtypes = arguments
        # Optional geometry-update entry point; compatible builds may rebuild.
        if hasattr(library, "rs_update"):
            library.rs_update.restype = ct.c_int
            library.rs_update.argtypes = (_pointer, ct.POINTER(_Surface), _size,
                                         ct.POINTER(ct.c_int))
        if library.rs_abi() != 4:
            raise CapabilityError(f"Native library has an incompatible ABI (requires material ABI 4): {path}")
    except (OSError, AttributeError) as error:
        raise CapabilityError(f"Cannot load Raystrack native library {path!r}: {error}") from error
    return library


def native_library(backend):
    path = os.environ.get("RAYSTRACK_NATIVE_LIBRARY")
    if not path:
        raise CapabilityError("Set RAYSTRACK_NATIVE_LIBRARY to the built Raystrack native library")
    library = _load(str(Path(path).resolve()))
    if not library.rs_backends() & (1 << BACKENDS[backend]):
        raise CapabilityError(f"Native backend {backend!r} is absent from this build")
    return library


class NativeGeometry:
    """Own one committed native geometry snapshot with ordered surface slots."""

    def __init__(self, backend, meshes):
        self._backend = backend
        self._handle = _pointer()
        self._library = native_library(backend)
        entries = (_Surface * len(meshes))(*(
            _Surface(vertices.ctypes.data, faces.ctypes.data, len(vertices), len(faces))
            for _, vertices, faces in meshes
        ))
        status = self._library.rs_create(BACKENDS[backend], entries, len(entries), ct.byref(self._handle))
        if status:
            raise CapabilityError(f"Could not initialize {backend!r}: {self._error()}")

    def update(self, meshes):
        """Commit ordered world geometry while retaining backend resources if supported.

        The caller serializes updates against tracing. Return whether the native
        traversal reused resources; unsupported topology/backend updates rebuild.
        """
        if not self._handle:
            raise RuntimeError("NativeGeometry is closed")
        if not hasattr(self._library, "rs_update"):
            replacement = NativeGeometry(self._backend, meshes)
            self.close()
            self._library = replacement._library
            self._handle, replacement._handle = replacement._handle, _pointer()
            return False
        entries = (_Surface * len(meshes))(*(
            _Surface(vertices.ctypes.data, faces.ctypes.data, len(vertices), len(faces))
            for _, vertices, faces in meshes
        ))
        retained = ct.c_int()
        status = self._library.rs_update(self._handle, entries, len(entries), ct.byref(retained))
        if status:
            raise RuntimeError(f"Could not update {self._backend!r}: {self._error()}")
        return bool(retained.value)

    def _error(self):
        return self._library.rs_error().decode("utf-8", errors="replace")

    def close(self):
        if self._handle:
            self._library.rs_destroy(self._handle)
            self._handle = _pointer()

    def __del__(self):
        self.close()

    def shortwave(self, sensors, light, steps, materials, samples, seed, bounces, epsilon, out):
        status = self._library.rs_shortwave(
            self._handle, sensors.ctypes.data, len(sensors), ct.byref(light), steps,
            materials, samples, seed, bounces, epsilon, out.ctypes.data,
        )
        if status:
            message = self._error()
            if message.startswith("capability:"):
                raise CapabilityError(message.removeprefix("capability:").strip())
            raise RuntimeError(message)

    def longwave(self, origins, directions, active, excluded, minimum, hits, fronts, blocked):
        status = self._library.rs_longwave(
            self._handle, origins.ctypes.data, directions.ctypes.data, len(origins),
            active.ctypes.data, excluded, minimum, hits.ctypes.data,
            fronts.ctypes.data, blocked.ctypes.data,
        )
        if status:
            raise RuntimeError(self._error())
