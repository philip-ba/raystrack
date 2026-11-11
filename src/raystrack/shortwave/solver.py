"""Mutable-scene shortwave orchestration around batched C++ transport."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from threading import RLock
from typing import Optional

import numpy as np

from ..backends.native import BACKENDS, Lighting, NativeGeometry, OpticalMaterial
from ..backends import CapabilityError
from ..model import Scene
from ..model.geometry import _immutable_array
from .klems import KLEMS_PROJECTED_SOLID_ANGLES, KlemsBSDF
from .materials import ClearGlazing, DiffuseSheet, Opaque, SolidDielectric, ThinDiffuse
from .sky import PatchSky, PatchSkySeries
from .solar import SolarPosition, SolarSite, SolarTimeline


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real values")
    owned = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise ValueError(f"{name} must be finite")
    return owned


def _vectors(value, name, *, normalize=False):
    array = _array(value, name)
    if array.shape == (3,):
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if normalize:
        scale = np.max(np.abs(array), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError(f"{name} must be nonzero")
        array /= scale
        array /= np.linalg.norm(array, axis=1, keepdims=True)
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must fit finite float32 values")
    return array


def _forcing(value, steps, name):
    array = _array(value, name)
    if array.shape in ((), (steps,)):
        array = np.broadcast_to(array, (steps,))[:, None]
    if array.shape not in ((steps, 1), (steps, 3), (1, 3)) or np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative with scalar, (steps,), or (steps, 3) shape")
    return np.ascontiguousarray(np.broadcast_to(array, (steps, 3)))


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def _validate_solids(state, assignments, points, offset):
    """Conservatively establish one exterior air medium for convex solids.

    AABB separation deliberately rejects some nonintersecting configurations.
    Plane checks are bounded in memory and reject excessive validation work.
    """
    geometry = [(surface.world_vertices().astype(np.float64), surface.mesh.faces)
                for surface in state.surfaces]
    bounds = [(vertices.min(axis=0), vertices.max(axis=0)) if len(vertices) else None
              for vertices, _ in geometry]
    for i, token in enumerate(state.instances.values()):
        if not isinstance(assignments.get(token), SolidDielectric):
            continue
        vertices, faces = geometry[i]
        if not len(faces) or not len(vertices):
            raise CapabilityError("SolidDielectric requires a closed outward convex mesh")
        if len(faces)*len(vertices) > 100_000_000:
            raise CapabilityError("SolidDielectric convex validation exceeds 100000000 face/vertex checks")
        edges = {}
        for a, b, c in faces:
            for start, end in ((a, b), (b, c), (c, a)):
                key = (min(start, end), max(start, end))
                edges.setdefault(key, []).append((int(start), int(end)))
        if any(len(pair) != 2 or pair[0] != pair[1][::-1] for pair in edges.values()):
            raise CapabilityError("SolidDielectric requires a closed mesh with consistent outward winding")
        neighbors = {}
        for start, end in edges:
            neighbors.setdefault(start, []).append(end)
            neighbors.setdefault(end, []).append(start)
        visited, pending = set(), [next(iter(neighbors))]
        while pending:
            vertex = pending.pop()
            if vertex not in visited:
                visited.add(vertex)
                pending.extend(neighbors[vertex])
        if len(visited) != len(neighbors):
            raise CapabilityError("SolidDielectric requires a single connected boundary shell")
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0):
            raise CapabilityError("SolidDielectric requires nondegenerate boundary triangles")
        normals /= lengths[:, None]
        # Float ray geometry cannot resolve boundaries or clearances below this.
        tolerance = 8*np.finfo(np.float32).eps*max(1., np.abs(vertices).max())
        margin = float(offset)*4 + tolerance
        for begin in range(0, len(faces), 32):
            normal = normals[begin:begin+32]
            origin = triangles[begin:begin+32, 0]
            minimum = np.full(len(normal), np.inf)
            for first in range(0, len(vertices), 4096):
                distances = (vertices[first:first+4096] @ normal.T
                             - np.einsum("ij,ij->i", origin, normal))
                if np.any(distances > tolerance):
                    raise CapabilityError("SolidDielectric requires a convex mesh with outward winding")
                minimum = np.minimum(minimum, distances.min(axis=0))
            if np.any(-minimum <= 4*margin):
                raise CapabilityError("SolidDielectric thickness is unresolved relative to ray_offset and float precision")
        for point in points:
            if np.all(np.einsum("ij,ij->i", point-triangles[:, 0], normals) <= margin):
                raise CapabilityError("SolidDielectric sensors must lie strictly outside the solid")
        low, high = bounds[i]
        for j, other in enumerate(bounds):
            if i == j or other is None:
                continue
            if np.all(low-margin <= other[1]) and np.all(other[0]-margin <= high):
                raise CapabilityError("SolidDielectric requires disjoint bounding boxes from all other surfaces; nesting, overlap and interior objects are unsupported")


@dataclass(frozen=True, eq=False)
class ShortwaveResult:
    """An immutable irradiance snapshot indexed by timestep, sensor, component, RGB."""

    irradiance: np.ndarray
    standard_error: Optional[np.ndarray]
    timestamps: Optional[np.ndarray]
    durations_hours: Optional[np.ndarray]
    geometry_revision: int
    material_revision: int
    backend: str
    samples: int
    replicates: int
    seed: int
    bounces: int
    sky_model: str
    channel_basis: str
    solar_site: Optional[SolarSite]
    utc_offset_minutes: Optional[int]
    forcing_id: Optional[str]

    components = ("direct", "sky", "reflected", "transmitted")

    @property
    def total(self):
        """Sum irradiance components while retaining separate RGB channels."""
        return _immutable_array(self.irradiance.sum(axis=2))

    def exposure(self):
        """Integrate each sensor, component and RGB channel in Wh/m²."""
        if self.durations_hours is None:
            raise ValueError("Exposure requires interval durations or a SolarTimeline")
        return _immutable_array(np.einsum("t,tscr->scr", self.durations_hours, self.irradiance))


class ShortwaveSolver:
    """Solve sensor irradiance on a revisioned Scene with a native backend.

    Missing optical assignments are black opaque. A solve captures a consistent
    geometry/material snapshot. Updates affect subsequent solves and completed
    results retain their original revisions.
    """

    def __init__(self, scene, *, materials=None, backend="embree"):
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        if backend not in BACKENDS:
            raise ValueError(f"Unknown native backend {backend!r}")
        self._scene, self._backend = scene, backend
        self._lock = RLock()
        self._materials = {}
        self._cdf_cache = {}
        self._delta_normal_arrays = []
        self._material_revision = 0
        self._native = None
        self._geometry_revision = None
        self._closed = False
        if materials is not None:
            for surface_id, material in materials.items():
                self.set_material(surface_id, material)

    @property
    def scene(self):
        return self._scene

    @property
    def backend(self):
        return self._backend

    def set_material(self, surface_id, material):
        """Commit an optical assignment or remove it with ``None``."""
        if material is not None and not isinstance(material, (Opaque, ThinDiffuse, DiffuseSheet, ClearGlazing, SolidDielectric, KlemsBSDF)):
            raise TypeError("material must be Opaque, ThinDiffuse, DiffuseSheet, ClearGlazing, SolidDielectric, KlemsBSDF, or None")
        if isinstance(material, KlemsBSDF) and material.band.lower() != "solar":
            raise ValueError("shortwave transport requires the Solar Klems data block")
        with self._lock:
            self._check_open()
            token = self.scene._snapshot().instances[surface_id]
            if material is None:
                self._materials.pop(token, None)
            else:
                self._materials[token] = material
            self._material_revision += 1
            return self._material_revision

    def _check_open(self):
        if self._closed:
            raise RuntimeError("ShortwaveSolver is closed")

    def close(self):
        with self._lock:
            self._closed = True
            self._cdf_cache.clear()
            if self._native is not None:
                self._native.close()
                self._native = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def _optics(self, state):
        tokens = tuple(state.instances.values())
        self._materials = {token: self._materials[token] for token in tokens if token in self._materials}
        packed = (OpticalMaterial * len(tokens))()
        cdf_cache = {}
        delta_normal_arrays = []
        colored = False
        for i, (token, surface) in enumerate(zip(tokens, state.surfaces)):
            material = self._materials.get(token)
            if isinstance(material, Opaque):
                packed[i].front[:] = material.front
                packed[i].back[:] = material.back
                colored |= any(len(set(rgb)) > 1 for rgb in (material.front, material.back))
            elif isinstance(material, ThinDiffuse):
                packed[i].front[:] = material.reflectance
                packed[i].back[:] = material.reflectance
                packed[i].transmission[:] = material.transmittance
                colored |= any(len(set(rgb)) > 1 for rgb in (material.reflectance, material.transmittance))
            elif isinstance(material, DiffuseSheet):
                packed[i].front[:] = material.front_reflectance
                packed[i].back[:] = material.back_reflectance
                packed[i].transmission[:] = material.transmittance
                colored |= any(len(set(rgb)) > 1 for rgb in
                               (material.front_reflectance, material.back_reflectance, material.transmittance))
            elif isinstance(material, SolidDielectric):
                packed[i].kind = 2
                packed[i].refractive_index = material.refractive_index
                packed[i].absorption[:] = material.absorption
                colored |= len(set(material.absorption)) > 1
            elif isinstance(material, ClearGlazing):
                packed[i].kind = 1
                packed[i].refractive_index = material.refractive_index
                vertices = surface.world_vertices().astype(np.float64)
                triangles = vertices[surface.mesh.faces]
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                lengths = np.linalg.norm(normals, axis=1)
                normals = normals[lengths > 0] / lengths[lengths > 0, None]
                # An unoriented plane has the same reflection operator on both sides.
                if len(normals):
                    major = np.argmax(np.abs(normals), axis=1)
                    signs = np.sign(normals[np.arange(len(normals)), major])
                    normals *= signs[:, None]
                normals = np.ascontiguousarray(np.unique(normals.astype(np.float32), axis=0))
                delta_normal_arrays.append(normals)
                packed[i].delta_normals = normals.ctypes.data
                packed[i].delta_normal_count = len(normals)
            elif isinstance(material, KlemsBSDF):
                matrix = material.matrices
                key = id(material)
                entry = cdf_cache.get(key, self._cdf_cache.get(key))
                if entry is None:
                    masses = matrix.transpose(0, 2, 1) * KLEMS_PROJECTED_SOLID_ANGLES
                    entry = (material, np.ascontiguousarray(np.cumsum(masses, axis=-1)))
                cdf_cache[key] = entry
                packed[i].klems = matrix.ctypes.data
                packed[i].reverse_cdf = entry[1].ctypes.data
                packed[i].tangent[:] = surface.transform[:3, 0]
                packed[i].alternate[:] = surface.transform[:3, 1]
        self._cdf_cache = cdf_cache
        self._delta_normal_arrays = delta_normal_arrays
        return packed, colored

    def solve(self, points, normals, sun, *, dni, dhi=None, sky=None, samples=512,
              bounces=1, seed=0, replicates=1, durations_hours=None, ray_offset=1e-5,
              channel_basis="neutral", forcing_id=None):
        """Compute timestep irradiance in W/m² on oriented point sensors.

        ``sun`` accepts unit directions toward the sun or a SolarPosition.
        ``dni``/``dhi`` may be neutral scalars or timestep arrays. Use a PatchSky
        or PatchSkySeries as the alternative to ``dhi``. Set samples and bounces
        to zero for direct-only forcing. Use an explicit ``channel_basis`` label
        for non-neutral RGB. ``ray_offset`` is in scene coordinate units.
        ``forcing_id`` can record the weather file, sky matrix or reference case.
        """
        samples = _integer(samples, "samples", 0, 2**32 - 1)
        bounces = _integer(bounces, "bounces", 0, 32)
        replicates = _integer(replicates, "replicates", 1, 2**32 - 1)
        seed = _integer(seed, "seed", 0, 2**64 - 1)
        points, normals = _vectors(points, "points"), _vectors(normals, "normals", normalize=True)
        if points.shape != normals.shape:
            raise ValueError("points and normals must contain the same number of sensors")
        sensors = np.concatenate((points, normals), axis=1)
        offset = _array(ray_offset, "ray_offset")
        if offset.ndim:
            raise ValueError("ray_offset must be a scalar")
        with np.errstate(over="ignore", under="ignore"):
            offset = np.float32(offset)
        if not np.isfinite(offset) or offset <= 0:
            raise ValueError("ray_offset must be a positive finite float32 value")

        timestamps = None
        timeline = sun if isinstance(sun, SolarTimeline) else None
        if isinstance(sun, SolarPosition):
            timestamps = _immutable_array(np.asarray(sun.timestamps))
            if durations_hours is None and isinstance(sun, SolarTimeline):
                durations_hours = sun.durations_hours
            sun = sun.directions
        directions = _vectors(sun, "sun", normalize=True)
        steps = len(directions)
        if timestamps is not None and timestamps.shape != (steps,):
            raise ValueError("solar timestamps must match the number of directions")
        if durations_hours is not None:
            hours = _array(durations_hours, "durations_hours")
            if hours.shape != (steps,) or np.any(hours <= 0):
                raise ValueError("durations_hours must have one positive value per timestep")
            durations_hours = _immutable_array(hours)

        direct = _forcing(dni, steps, "dni")
        if sky is None:
            radiance = _forcing(0 if dhi is None else dhi, steps, "dhi") / np.pi
            offsets = None
            light = Lighting(directions.ctypes.data, direct.ctypes.data, radiance.ctypes.data,
                             None, 0, 1, 0, 0)
            sky_model = "isotropic"
        else:
            if dhi is not None or not isinstance(sky, (PatchSky, PatchSkySeries)):
                raise ValueError("use either dhi or a PatchSky/PatchSkySeries")
            if isinstance(sky, PatchSkySeries) and len(sky) != steps:
                raise ValueError("sky series length must match the solar timesteps")
            basis = sky.basis
            radiance = sky.radiance
            offsets = np.asarray(basis.ring_offsets, dtype=np.int32)
            stride = len(basis) if isinstance(sky, PatchSkySeries) else 0
            light = Lighting(directions.ctypes.data, direct.ctypes.data, radiance.ctypes.data,
                             offsets.ctypes.data, len(offsets)-1, stride,
                             basis.ring_height_rad, basis.north_deg)
            sky_model = f"reinhart-{basis.multiplier}"
        if samples == 0 and (bounces or np.any(radiance)):
            raise ValueError("diffuse sky or bounces require samples > 0")
        if not isinstance(channel_basis, str) or not channel_basis.strip():
            raise ValueError("channel_basis must identify the RGB convention")
        if forcing_id is not None and (not isinstance(forcing_id, str) or not forcing_id.strip()):
            raise ValueError("forcing_id must be a nonempty string")

        with self._lock:
            self._check_open()
            state = self.scene._snapshot()
            materials, colored = self._optics(state)
            if any(material.kind == 2 for material in materials):
                if np.any(direct):
                    raise CapabilityError("SolidDielectric supports sky forcing only; DNI must be zero")
                _validate_solids(state, self._materials, points, offset)
            if channel_basis == "neutral" and (colored or any(np.any(a != a[..., :1]) for a in (direct, radiance))):
                raise ValueError("non-neutral RGB requires an explicit channel_basis")
            if self._geometry_revision != state.revision:
                meshes = [(surface.surface_id, surface.world_vertices(), surface.mesh.faces)
                          for surface in state.surfaces]
                if self._native is None:
                    self._native = NativeGeometry(self.backend, meshes)
                else:
                    self._native.update(meshes)
                self._geometry_revision = state.revision

            shape = (steps, len(sensors), len(ShortwaveResult.components), 3)
            sample = np.empty(shape, dtype=np.float64)
            mean = sample if replicates == 1 else np.zeros(shape, dtype=np.float64)
            squared = None if replicates == 1 else np.zeros(shape, dtype=np.float64)
            seeds = np.random.SeedSequence(seed).generate_state(replicates, dtype=np.uint64)
            for count, replicate_seed in enumerate(seeds, 1):
                self._native.shortwave(sensors, light, steps, materials, samples, int(replicate_seed),
                                       bounces, float(offset), sample)
                if squared is not None:
                    delta = sample - mean
                    mean += delta / count
                    squared += delta * (sample - mean)
            if squared is None:
                error = None
            else:
                error = _immutable_array(np.sqrt(np.maximum(squared, 0) / (replicates * (replicates - 1))))
            return ShortwaveResult(
                _immutable_array(mean), error, timestamps, durations_hours, state.revision,
                self._material_revision, self.backend, samples, replicates, seed, bounces,
                sky_model, channel_basis, None if timeline is None else timeline.site,
                None if timeline is None else timeline.utc_offset_minutes, forcing_id,
            )
