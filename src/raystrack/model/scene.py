"""Ordered scene snapshots and centrally observable geometry revisions."""
from __future__ import annotations

from dataclasses import dataclass
from reprlib import repr as short_repr
from threading import RLock
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional, Tuple

from .geometry import Mesh, Surface, rigid_transform


@dataclass(frozen=True)
class _SceneState:
    surfaces: Tuple[Surface, ...]
    by_id: Mapping[str, Surface]
    revision: int


class Scene:
    """Own ordered instances and stable IDs independently of display labels.

    Readers receive immutable snapshots. Mutations replace a complete state
    while holding ``lock``, an RLock also available to a Solver for consistent
    execution. Internal subscribers run before waiting for that lock, allowing
    active work to be cancelled promptly. Subscribers must not raise or mutate
    the scene. Geometry changes increase ``revision``; label edits do not.
    """

    def __init__(self, surfaces: Iterable[Surface] = (), *, revision: int = 0):
        """Own an ordered sequence of immutable instances at a known revision."""
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        self.lock = RLock()
        self._observer_lock = RLock()
        self._observers = []
        self._state = self._make_state(tuple(surfaces), revision)

    @staticmethod
    def _make_state(surfaces, revision):
        by_id = {}
        for surface in surfaces:
            if not isinstance(surface, Surface):
                raise TypeError("scene entries must be Surface objects")
            if surface.surface_id in by_id:
                raise ValueError("surface IDs must be unique: {!r}".format(surface.surface_id))
            by_id[surface.surface_id] = surface
        return _SceneState(tuple(surfaces), MappingProxyType(by_id), revision)

    @classmethod
    def from_meshes(cls, meshes, *, labels: Optional[Mapping[str, str]] = None):
        """Create identity instances from a mapping or iterable of (ID, Mesh)."""
        entries = meshes.items() if isinstance(meshes, Mapping) else meshes
        labels = {} if labels is None else dict(labels)
        surfaces = tuple(Surface(surface_id, mesh, labels.get(surface_id))
                         for surface_id, mesh in entries)
        unknown = set(labels).difference(surface.surface_id for surface in surfaces)
        if unknown:
            raise KeyError("labels contain unknown surface IDs: {!r}".format(sorted(unknown)))
        return cls(surfaces)

    @classmethod
    def from_instances(cls, prototypes: Mapping[str, Mesh], instances,
                       *, labels: Optional[Mapping[str, str]] = None):
        """Create (ID, prototype key, transform) instances sharing Mesh identity."""
        if not isinstance(prototypes, Mapping):
            raise TypeError("prototypes must be a mapping of names to Mesh objects")
        if any(not isinstance(mesh, Mesh) for mesh in prototypes.values()):
            raise TypeError("prototypes must contain Mesh objects")
        labels = {} if labels is None else dict(labels)
        surfaces = []
        for surface_id, prototype_key, transform in instances:
            surfaces.append(Surface(surface_id, prototypes[prototype_key],
                                    labels.get(surface_id), transform))
        unknown = set(labels).difference(surface.surface_id for surface in surfaces)
        if unknown:
            raise KeyError("labels contain unknown surface IDs: {!r}".format(sorted(unknown)))
        return cls(surfaces)

    @property
    def surfaces(self) -> Tuple[Surface, ...]:
        """Return the current immutable, ordered instance snapshot."""
        return self._state.surfaces

    @property
    def surface_ids(self) -> Tuple[str, ...]:
        """Return stable IDs in the order used by queries and result rows."""
        return tuple(self._state.by_id)

    @property
    def revision(self) -> int:
        """Return the geometry revision; display label edits do not increase it."""
        return self._state.revision

    def __getitem__(self, surface_id: str) -> Surface:
        """Find an immutable instance by its stable surface ID."""
        return self._state.by_id[surface_id]

    def __len__(self) -> int:
        """Return the number of scene instances, including all occluders."""
        return len(self._state.surfaces)

    def __repr__(self):
        """Summarize instance and geometry counts without expanding meshes."""
        state = self._state
        unique_meshes = {id(surface.mesh) for surface in state.surfaces}
        triangles = sum(len(surface.mesh.faces) for surface in state.surfaces)
        return (f"Scene(surfaces={len(state.surfaces)}, meshes={len(unique_meshes)}, "
                f"triangles={triangles}, revision={state.revision}, "
                f"ids={short_repr(tuple(state.by_id))})")

    def world_meshes(self):
        """Return ordered (ID, immutable world vertices, immutable faces)."""
        state = self._state
        return [(surface.surface_id, surface.world_vertices(), surface.mesh.faces)
                for surface in state.surfaces]

    def _subscribe(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("subscriber must be callable")
        with self._observer_lock:
            if callback not in self._observers:
                self._observers.append(callback)

    def _unsubscribe(self, callback: Callable[[], None]) -> None:
        with self._observer_lock:
            if callback in self._observers:
                self._observers.remove(callback)

    def _notify(self) -> None:
        with self._observer_lock:
            observers = tuple(self._observers)
        for callback in observers:
            callback()

    def _replace(self, surface: Surface, *, geometry: bool) -> int:
        # Called with lock held; build the complete new snapshot before commit.
        state = self._state
        if surface.surface_id not in state.by_id:
            raise KeyError(surface.surface_id)
        surfaces = tuple(surface if old.surface_id == surface.surface_id else old
                         for old in state.surfaces)
        self._state = self._make_state(surfaces, state.revision + int(geometry))
        return self._state.revision

    def update_transform(self, surface_id: str, transform) -> int:
        """Apply an absolute rigid transform relative to the owned local Mesh."""
        matrix = rigid_transform(transform)
        current = self[surface_id]
        Surface(surface_id, current.mesh, current.label, matrix)  # Validate before notification.
        self._notify()
        with self.lock:
            current = self[surface_id]
            return self._replace(Surface(surface_id, current.mesh, current.label, matrix), geometry=True)

    def update_mesh(self, surface_id: str, mesh: Mesh) -> int:
        """Replace geometry and reset its local transform basis to identity."""
        current = self[surface_id]
        replacement = Surface(surface_id, mesh, current.label)
        self._notify()
        with self.lock:
            current = self[surface_id]
            if current.label != replacement.label:
                replacement = Surface(surface_id, mesh, current.label)
            return self._replace(replacement, geometry=True)

    def update_vertices(self, surface_id: str, world_vertices) -> int:
        """Replace world vertices, retain faces, and reset the local transform."""
        current = self[surface_id]
        initial_mesh = current.mesh
        mesh = Mesh(world_vertices, initial_mesh.faces)
        Surface(surface_id, mesh, current.label)
        self._notify()
        with self.lock:
            current = self[surface_id]
            if current.mesh is not initial_mesh:
                # Retain the topology actually being replaced if another
                # mutation completed while this call waited for the lock.
                mesh = Mesh(world_vertices, current.mesh.faces)
            return self._replace(Surface(surface_id, mesh, current.label), geometry=True)

    def set_label(self, surface_id: str, label: str) -> None:
        """Change a display label without invalidating numerical work."""
        with self.lock:
            current = self[surface_id]
            self._replace(Surface(surface_id, current.mesh, label, current.transform), geometry=False)

    def add(self, surface: Surface) -> int:
        """Append a new stable-ID surface and return the geometry revision."""
        if not isinstance(surface, Surface):
            raise TypeError("surface must be a Surface")
        if surface.surface_id in self._state.by_id:
            raise ValueError("surface IDs must be unique: {!r}".format(surface.surface_id))
        self._notify()
        with self.lock:
            state = self._state
            self._state = self._make_state(state.surfaces + (surface,), state.revision + 1)
            return self._state.revision

    def remove(self, surface_id: str) -> int:
        """Remove a surface without changing the remaining IDs or ordering."""
        self[surface_id]
        self._notify()
        with self.lock:
            self[surface_id]
            state = self._state
            surfaces = tuple(surface for surface in state.surfaces if surface.surface_id != surface_id)
            self._state = self._make_state(surfaces, state.revision + 1)
            return self._state.revision
