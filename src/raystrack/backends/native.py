"""Private NumPy-to-C++ batch interface. Loading is always explicit."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from . import CapabilityError


BACKENDS = {"native-builtin": 0, "embree": 1, "vulkan-rt": 2}
_pointer = ct.c_void_p
_size = ct.c_size_t


class _Surface(ct.Structure):
    _fields_ = [("vertices", _pointer), ("triangles", _pointer),
                ("vertex_count", _size), ("triangle_count", _size)]


class _Instance(ct.Structure):
    _fields_ = [("prototype", ct.c_uint32), ("transform", ct.c_float * 12)]


class _InstanceStatistics(ct.Structure):
    _fields_ = [(key, ct.c_uint64) for key in
               ("prototype_builds", "prototype_refits", "transform_commits", "scene_commits")]


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
        if hasattr(library, "rs_create_instances"):
            library.rs_create_instances.restype = ct.c_int
            library.rs_create_instances.argtypes = (ct.c_int, ct.POINTER(_Surface), _size,
                                                    ct.POINTER(_Instance), _size, ct.POINTER(_pointer))
            library.rs_update_instances.restype = ct.c_int
            library.rs_update_instances.argtypes = (_pointer, ct.POINTER(_Surface), _size,
                                                    ct.POINTER(_Instance), _size, ct.POINTER(ct.c_int))
            library.rs_instance_statistics.restype = ct.c_int
            library.rs_instance_statistics.argtypes = (_pointer, ct.POINTER(_InstanceStatistics))
        if hasattr(library, "rs_trace"):
            library.rs_trace.restype = ct.c_int
            library.rs_trace.argtypes = (_pointer, _pointer, _size, _pointer, _pointer)
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
        self.geometry_mode = "flat-world"
        self._handle = _pointer()
        self._library = native_library(backend)
        entries = (_Surface * len(meshes))(*(
            _Surface(vertices.ctypes.data, faces.ctypes.data, len(vertices), len(faces))
            for _, vertices, faces in meshes
        ))
        status = self._library.rs_create(BACKENDS[backend], entries, len(entries), ct.byref(self._handle))
        if status:
            raise CapabilityError(f"Could not initialize {backend!r}: {self._error()}")

    @staticmethod
    def _scene_entries(state):
        prototypes, slots = [], {}
        instances = (_Instance * len(state.surfaces))()
        for i, surface in enumerate(state.surfaces):
            key = id(surface.mesh)
            if key not in slots:
                slots[key] = len(prototypes)
                prototypes.append(surface.mesh)
            instances[i].prototype = slots[key]
            instances[i].transform[:] = surface.transform[:3].ravel()
        entries = (_Surface * len(prototypes))(*(
            _Surface(mesh.vertices.ctypes.data, mesh.faces.ctypes.data,
                     len(mesh.vertices), len(mesh.faces)) for mesh in prototypes))
        return entries, instances

    @classmethod
    def from_scene(cls, backend, state):
        """Use shared Embree prototypes, with explicit flat fallback elsewhere."""
        library = native_library(backend)
        if backend != "embree" or not hasattr(library, "rs_create_instances"):
            return cls(backend, [(s.surface_id, s.world_vertices(), s.mesh.faces) for s in state.surfaces])
        result = cls.__new__(cls)
        result._backend, result._library = backend, library
        result._handle = _pointer()
        result.geometry_mode = "shared-prototype-instances"
        entries, instances = result._scene_entries(state)
        status = library.rs_create_instances(BACKENDS[backend], entries, len(entries),
                                              instances, len(instances), ct.byref(result._handle))
        if status:
            raise CapabilityError(f"Could not initialize {backend!r} instances: {result._error()}")
        return result

    def update_scene(self, state):
        """Atomically submit one immutable scene boundary, retaining local BLAS."""
        if self.geometry_mode == "flat-world":
            return self.update([(s.surface_id, s.world_vertices(), s.mesh.faces) for s in state.surfaces])
        if not self._handle:
            raise RuntimeError("NativeGeometry is closed")
        entries, instances = self._scene_entries(state)
        retained = ct.c_int()
        status = self._library.rs_update_instances(self._handle, entries, len(entries),
                                                   instances, len(instances), ct.byref(retained))
        if status:
            raise RuntimeError(f"Could not update {self._backend!r} instances: {self._error()}")
        return bool(retained.value)

    @property
    def instance_statistics(self):
        """Current traversal counters; rebuilding resets them to a new lifetime."""
        if not self._handle:
            raise RuntimeError("NativeGeometry is closed")
        if not hasattr(self._library, "rs_instance_statistics"):
            return None
        stats = _InstanceStatistics()
        if self._library.rs_instance_statistics(self._handle, ct.byref(stats)):
            raise RuntimeError(self._error())
        return {key: getattr(stats, key) for key, _ in stats._fields_}

    def trace(self, origins, directions, *, t_min=1e-6, t_max=np.inf):
        """Diagnostic closest hits and bounded occlusion; input directions are unchanged."""
        if not self._handle:
            raise RuntimeError("NativeGeometry is closed")
        if not hasattr(self._library, "rs_trace"):
            raise CapabilityError("Native library lacks diagnostic rs_trace")
        origins, directions = np.asarray(origins, np.float32), np.asarray(directions, np.float32)
        if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
            raise ValueError("origins and directions must have shape (n, 3)")
        if (not np.isfinite(origins).all() or not np.isfinite(directions).all()
                or np.any(np.max(np.abs(directions), axis=1) == 0)):
            raise ValueError("origins and nonzero directions must be finite float32 values")
        if not np.isscalar(t_min) or not np.isscalar(t_max):
            raise ValueError("ray bounds must be scalars")
        if not np.isfinite(t_min) or t_min < 0 or np.isnan(t_max) or t_max < t_min:
            raise ValueError("ray bounds require finite 0 <= t_min <= t_max")
        rays = np.empty((len(origins), 8), np.float32)
        rays[:, :3], rays[:, 3:6], rays[:, 6], rays[:, 7] = origins, directions, t_min, t_max
        dtype = np.dtype([("surface", "i4"), ("primitive", "i4"), ("distance", "f4"),
                          ("u", "f4"), ("v", "f4"), ("normal", "f4", 3), ("point", "f4", 3)])
        hits, occluded = np.empty(len(origins), dtype), np.empty(len(origins), np.uint8)
        status = self._library.rs_trace(self._handle, rays.ctypes.data, len(rays),
                                        hits.ctypes.data, occluded.ctypes.data)
        if status:
            raise RuntimeError(self._error())
        return hits, occluded

    def update(self, meshes):
        """Commit ordered world geometry while retaining backend resources if supported.

        The caller serializes updates against tracing. Return whether the native
        traversal reused resources; unsupported topology/backend updates rebuild.
        """
        if not self._handle:
            raise RuntimeError("NativeGeometry is closed")
        if self.geometry_mode != "flat-world":
            raise RuntimeError("Shared instance geometry requires update_scene")
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
