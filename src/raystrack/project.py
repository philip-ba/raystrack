"""Version-one, contained file inputs for dynamic shortwave projects.

Coordinates are metres, east/north/up. Weather timestamps are explicit UTC
interval midpoints; state timestamps are interval starts, held until replaced.
"""
from __future__ import annotations

import csv
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .io._arrays import contained_path, integer, json_value
from .model import Mesh, Scene
from .model.geometry import rigid_transform
from .shortwave import ClearGlazing, DiffuseSheet, KlemsBSDF, Opaque, SolarSite, SolarTimeline, solar_position

FORMAT = "raystrack.shortwave-project"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp(value):
    """Parse only explicit UTC, never assume the machine's local timezone."""
    if (not isinstance(value, str)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value) is None):
        raise ValueError("timestamp must be an explicit UTC ISO8601 string ending in Z")
    try:
        instant = datetime.fromisoformat(value[:-1] + "+00:00")
        return np.datetime64(instant.astimezone(timezone.utc).replace(tzinfo=None), "us")
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid UTC timestamp") from exc


def _name(value, role):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a nonempty string")
    return value


def _object(value, role):
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _list(value, role):
    if not isinstance(value, list):
        raise ValueError(f"{role} must be a list")
    return value


@dataclass(frozen=True)
class Project:
    path: Path
    manifest: dict
    scene: Scene
    materials: dict
    material_library: dict
    points: np.ndarray
    normals: np.ndarray
    sensor_ids: tuple
    sensor_groups: tuple
    timeline: SolarTimeline
    dni: np.ndarray
    dhi: np.ndarray
    states: tuple
    source_hashes: dict
    settings: dict


def load_project(path):
    """Load and validate all assets before any simulation/output mutation.

    State vertices and meshes are loaded as Mesh/array objects; transforms are
    validated proper rigid matrices. State ``timestamp_utc`` is datetime64[us].
    Repeated prototype references preserve Mesh object identity.
    """
    path = Path(path).resolve()
    manifest = _object(json.loads(path.read_text(encoding="utf-8")), "project")
    json_value(manifest)
    if manifest.get("schema") != FORMAT or type(manifest.get("version")) is not int or manifest.get("version") != 1:
        raise ValueError("unsupported project schema/version")
    if manifest.get("units") != {"length": "m", "irradiance": "W/m2"}:
        raise ValueError("units must be length=m and irradiance=W/m2")
    if manifest.get("coordinates") != "ENU":
        raise ValueError("coordinates must be ENU")
    hashes = {path.name: file_sha256(path)}

    def asset(relative):
        target = contained_path(path.parent, relative)
        if not target.is_file():
            raise FileNotFoundError(f"missing project asset: {relative}")
        hashes[relative] = file_sha256(target)
        return target

    def array(relative, dtype):
        value = np.load(asset(relative), allow_pickle=False, mmap_mode="r")
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
            raise ValueError(f"{relative}: expected {np.dtype(dtype)} NPY array")
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f"{relative}: expected (N,3) array")
        return value

    def mesh(entry):
        entry = _object(entry, "mesh")
        return Mesh(array(entry.get("vertices"), np.float32), array(entry.get("faces"), np.int32))

    prototypes = {}
    for item in _list(manifest.get("prototypes"), "prototypes"):
        item = _object(item, "prototype")
        name = _name(item.get("id"), "prototype ID")
        if name in prototypes:
            raise ValueError("duplicate prototype ID")
        prototypes[name] = mesh(item)
    library = {}
    for name, item in _object(manifest.get("materials"), "materials").items():
        _name(name, "material ID")
        item = _object(item, "material")
        kind = item.get("type")
        if kind == "opaque":
            material = Opaque(item.get("front"), item.get("back"))
        elif kind == "diffuse_sheet":
            material = DiffuseSheet(item.get("front_reflectance"), item.get("back_reflectance"), item.get("transmittance"))
        elif kind == "clear_glazing":
            material = ClearGlazing(item.get("refractive_index", 1.52))
        elif kind == "klems_xml":
            if item.get("band", "solar") != "solar":
                raise ValueError("district Klems inputs require explicit Solar band")
            material = KlemsBSDF.from_xml(asset(item.get("path")), band="solar")
        elif kind == "rough_transmission":
            material = KlemsBSDF.rough_transmission(item.get("transmittance"), item.get("exponent"),
                                                  reflectance=item.get("reflectance", 0))
        else:
            raise ValueError(f"unsupported material type: {kind}")
        library[name] = material
    instances, materials = [], {}
    for item in _list(manifest.get("instances"), "instances"):
        item = _object(item, "instance")
        name = _name(item.get("id"), "instance ID")
        if name in materials:
            raise ValueError("duplicate instance ID")
        if item.get("prototype") not in prototypes or item.get("material") not in library:
            raise ValueError("instance refers to an unknown prototype/material ID")
        instances.append((name, item["prototype"], rigid_transform(item.get("transform", np.eye(4)))))
        materials[name] = library[item["material"]]
    scene = Scene.from_instances(prototypes, instances)
    if not instances:
        raise ValueError("project requires at least one instance")
    sensors = manifest.get("sensors")
    if isinstance(sensors, dict):
        sensors = sensors.get("path")
    with asset(sensors).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "group", "x", "y", "z", "nx", "ny", "nz"]:
            raise ValueError("sensor CSV header must be id,group,x,y,z,nx,ny,nz")
        rows = list(reader)
    ids = tuple(_name(row["id"], "sensor ID") for row in rows)
    groups = tuple(_name(row["group"], "sensor group") for row in rows)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("sensor IDs must be unique and nonempty")
    points = np.array([[float(row[k]) for k in ("x", "y", "z")] for row in rows])
    normals = np.array([[float(row[k]) for k in ("nx", "ny", "nz")] for row in rows])
    if not np.isfinite(points).all() or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("sensor coordinates must be finite and normals nonzero")
    normals /= np.max(np.abs(normals), axis=1)[:, None]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    site_spec = _object(manifest.get("site"), "site")
    site = SolarSite(site_spec["latitude_deg"], site_spec["longitude_deg"], site_spec.get("north_deg", 0))
    weather = _object(manifest.get("weather"), "weather")
    if weather.get("type") == "synthetic":
        year = integer(weather.get("year", 2025), "year", minimum=1901)
        minutes = integer(weather.get("interval_minutes", 60), "interval_minutes", minimum=1)
        offset = site_spec.get("utc_offset_minutes", 0)
        full = SolarTimeline.for_year(year, site, step_minutes=minutes, utc_offset_minutes=offset)
        steps = integer(weather.get("steps", len(full.timestamps)), "steps", minimum=1)
        start = utc_timestamp(weather["start_utc"]) if "start_utc" in weather else full.timestamps[0] - np.timedelta64(minutes * 30, "s")
        timestamps = start + (np.arange(steps, dtype=np.int64)*minutes*60 + minutes*30).astype("timedelta64[s]")
        durations = np.full(steps, minutes / 60)
        position = solar_position(site, timestamps)
        timeline = SolarTimeline(position.timestamps, position.directions, position.elevation_deg, position.azimuth_deg,
                                 position.above_horizon, durations, site, year, minutes, offset)
        # Deterministic synthetic broadband forcing; deliberately not measured weather.
        modulation = .72 + .2*np.sin(np.arange(steps)*minutes/(60*24)*.37)
        sine = np.maximum(0, timeline.directions[:, 2])
        dni = np.where(sine > 0, 850*np.exp(-.16/np.maximum(sine, .02))*modulation, 0)
        dhi = np.where(sine > 0, 80 + 70*(1-modulation), 0)
    elif weather.get("type") == "csv":
        if weather.get("units") != "W/m2":
            raise ValueError("weather CSV units must be W/m2")
        with asset(weather.get("path")).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["timestamp_utc", "duration_hours", "dni", "dhi"]:
                raise ValueError("weather CSV header must be timestamp_utc,duration_hours,dni,dhi")
            rows = list(reader)
        timestamps = np.array([utc_timestamp(r["timestamp_utc"]) for r in rows], dtype="datetime64[us]")
        durations = np.array([float(r["duration_hours"]) for r in rows])
        dni, dhi = (np.array([float(r[k]) for r in rows]) for k in ("dni", "dhi"))
        position = solar_position(site, timestamps)
        year = int(str(timestamps[0])[:4]) if len(timestamps) else 2025
        timeline = SolarTimeline(position.timestamps, position.directions, position.elevation_deg, position.azimuth_deg,
                                 position.above_horizon, durations, site, year, 0, site_spec.get("utc_offset_minutes", 0))
    else:
        raise ValueError("weather type must be synthetic or csv")
    if (not len(timeline.timestamps) or np.any(np.diff(timeline.timestamps).astype(np.int64) <= 0)
            or not np.isfinite(durations).all() or np.any(durations <= 0)
            or not np.isfinite(dni).all() or not np.isfinite(dhi).all() or np.any(dni < 0) or np.any(dhi < 0)):
        raise ValueError("weather times must increase; durations positive; irradiance finite and nonnegative")
    half_us = np.rint(durations*3_600_000_000/2).astype("timedelta64[us]")
    starts, ends = timeline.timestamps-half_us, timeline.timestamps+half_us
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("weather intervals must not overlap")
    raw_states = _list(manifest.get("states", []), "states")
    if "state_tracks" in manifest:
        track = _object(manifest["state_tracks"], "state_tracks")
        if track.get("interpolation", "hold") != "hold":
            raise ValueError("only sampled-pose hold interpolation is supported")
        merged, previous = {}, None
        for raw in raw_states:
            raw = _object(raw, "state")
            stamp = utc_timestamp(raw.get("timestamp_utc"))
            if previous is not None and stamp <= previous:
                raise ValueError("state timestamps must strictly increase")
            merged[stamp] = copy.deepcopy(raw)
            previous = stamp
        with asset(track.get("path")).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["timestamp_utc", "instance_id", "x", "y", "z", "yaw_deg", "material"]:
                raise ValueError("pose CSV header must be timestamp_utc,instance_id,x,y,z,yaw_deg,material")
            previous, seen = None, set()
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("pose CSV row must match its header columns")
                stamp = utc_timestamp(row["timestamp_utc"])
                name = _name(row["instance_id"], "instance ID")
                if previous is not None and stamp < previous:
                    raise ValueError("pose CSV timestamps must not decrease")
                if (stamp, name) in seen:
                    raise ValueError("duplicate pose CSV timestamp/instance ID")
                seen.add((stamp, name))
                previous = stamp
                entry = merged.setdefault(stamp, {"timestamp_utc": str(stamp)+"Z"})
                fields = [row[key] for key in ("x", "y", "z", "yaw_deg")]
                if any(fields):
                    if not all(fields):
                        raise ValueError("pose CSV requires all x,y,z,yaw_deg values")
                    x, y, z, yaw = map(float, fields)
                    angle = np.radians(yaw)
                    c, s = np.cos(angle), np.sin(angle)
                    matrix = [[c,-s,0,x], [s,c,0,y], [0,0,1,z], [0,0,0,1]]
                    poses = _object(entry.setdefault("transforms", {}), "transforms")
                    if name in poses:
                        raise ValueError("duplicate inline/CSV transform assignment")
                    poses[name] = matrix
                if row["material"]:
                    optics = _object(entry.setdefault("materials", {}), "materials")
                    if name in optics:
                        raise ValueError("duplicate inline/CSV material assignment")
                    optics[name] = row["material"]
                if not any(fields) and not row["material"]:
                    raise ValueError("pose CSV row requires a pose or material assignment")
        raw_states = [merged[key] for key in sorted(merged)]
    states, previous = [], None
    state_meshes = {surface.surface_id: surface.mesh for surface in scene.surfaces}
    for raw in raw_states:
        raw = _object(raw, "state")
        stamp = utc_timestamp(raw.get("timestamp_utc"))
        if previous is not None and stamp <= previous:
            raise ValueError("state timestamps must strictly increase")
        if stamp not in starts:
            raise ValueError("state timestamp must match a simulation interval start boundary")
        if raw.get("interpolation", "hold") != "hold":
            raise ValueError("only sampled-pose hold interpolation is supported")
        state = {"timestamp_utc": stamp, "transforms": {}, "materials": {}, "vertices": {}, "meshes": {}}
        for category in ("transforms", "materials", "vertices", "meshes"):
            for name, value in _object(raw.get(category, {}), category).items():
                if name not in materials:
                    raise ValueError("state refers to unknown instance ID")
                if category == "transforms":
                    value = rigid_transform(value)
                elif category == "materials":
                    if value not in library:
                        raise ValueError("state refers to unknown material ID")
                elif category == "vertices":
                    value = array(value, np.float32)
                    if value.shape != state_meshes[name].vertices.shape:
                        raise ValueError("state deformation must preserve vertex count")
                    state_meshes[name] = Mesh(value, state_meshes[name].faces)
                else:
                    value = mesh(value)
                    state_meshes[name] = value
                state[category][name] = value
        if set(state["vertices"]) & set(state["meshes"]) or set(state["transforms"]) & (set(state["vertices"]) | set(state["meshes"])):
            raise ValueError("one state cannot transform and replace geometry for the same instance")
        states.append(state)
        previous = stamp
    return Project(path, manifest, scene, materials, library, points, normals, ids, groups,
                   timeline, dni, dhi, tuple(states), hashes, _object(manifest.get("settings", {}), "settings"))
