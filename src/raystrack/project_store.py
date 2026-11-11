"""Atomic bounded shortwave blocks with selective, verified numeric reads.

NPY blocks map without decompressing unselected sensors. NPZ is an explicit
compression option. Exposure-only retention keeps sufficient per-block sums,
while hourly readers fail explicitly when hourly values were not retained.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

from .io._arrays import contained_path, integer, json_value
from .project import file_sha256, utc_timestamp

FORMAT = "raystrack.shortwave-results"


def _publish(root, manifest):
    temporary = root / (".manifest-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(json_value(manifest), stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def _ids(values, role):
    values = list(values)
    if not values or any(not isinstance(x, str) or not x.strip() for x in values) or len(set(values)) != len(values):
        raise ValueError(f"{role} must contain unique nonempty strings")
    return values


def _load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    json_value(manifest)
    if not isinstance(manifest, dict) or manifest.get("schema") != FORMAT or type(manifest.get("version")) is not int or manifest.get("version") != 1:
        raise ValueError("unsupported result schema/version")
    if manifest.get("storage") not in ("npy", "npz") or manifest.get("retention") not in ("hourly", "exposure"):
        raise ValueError("invalid result storage/retention")
    if manifest.get("units") != {"values": "W/m2", "errors": "W/m2 one standard error", "exposure": "Wh/m2"}:
        raise ValueError("invalid result units")
    if not isinstance(manifest.get("configuration"), dict) or not isinstance(manifest.get("complete"), bool):
        raise ValueError("invalid result configuration/completion descriptor")
    axes = manifest.get("axes")
    if not isinstance(axes, dict) or set(axes) != {"sensors", "components", "bands"}:
        raise ValueError("invalid result axes")
    for key in axes:
        if not isinstance(axes[key], list):
            raise ValueError("result axes must be lists")
        _ids(axes[key], key)
    groups = manifest.get("sensor_groups")
    if not isinstance(groups, list) or len(groups) != len(axes["sensors"]) or any(not isinstance(x, str) for x in groups):
        raise ValueError("invalid sensor groups")
    exposure_shape = tuple(len(axes[key]) for key in ("sensors", "components", "bands"))
    for key in ("exposure_wh_m2", "exposure_one_se_conservative_bound_wh_m2"):
        value = np.asarray(manifest.get(key), dtype=np.float64)
        if value.shape != exposure_shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("invalid exposure descriptor")
    integer(manifest.get("completed_steps"), "completed steps")
    if not isinstance(manifest.get("blocks"), list):
        raise ValueError("result blocks must be a list")
    expected = 0
    previous_end = None
    paths = set()
    for block in manifest["blocks"]:
        if not isinstance(block, dict):
            raise ValueError("invalid result block")
        for key in ("start", "stop", "scene_revision", "material_revision"):
            integer(block.get(key), key)
        if block["start"] != expected or block["stop"] <= expected:
            raise ValueError("invalid result block interval sequence")
        expected = block["stop"]
        first, last = utc_timestamp(block.get("start_utc")), utc_timestamp(block.get("end_utc"))
        if first >= last or (previous_end is not None and first < previous_end):
            raise ValueError("invalid result block time sequence")
        previous_end = last
        keys = {"archive"} if manifest["storage"] == "npz" else {"timestamps", "durations_hours"} | (
            {"values", "errors"} if manifest["retention"] == "hourly" else {"exposure", "exposure_error"})
        if not isinstance(block.get("files"), dict) or set(block["files"]) != keys:
            raise ValueError("invalid result block file descriptor")
        for entry in block["files"].values():
            if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or re.fullmatch("[0-9a-f]{64}", entry["sha256"]) is None:
                raise ValueError("invalid result chunk checksum descriptor")
            contained_path(root, entry.get("path"))
            if entry["path"] in paths:
                raise ValueError("duplicate result chunk path")
            paths.add(entry["path"])
    if expected != manifest["completed_steps"]:
        raise ValueError("completed step count disagrees with blocks")
    return manifest


def _verify(root, descriptor):
    path = contained_path(root, descriptor["path"])
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"result chunk checksum mismatch: {descriptor['path']}")
    return path


def _validate_block_time(block, arrays):
    count = block["stop"]-block["start"]
    stamps, durations = arrays["timestamps"], arrays["durations_hours"]
    if (stamps.shape != (count,) or stamps.dtype != np.int64 or durations.shape != (count,)
            or durations.dtype != np.float64 or not np.isfinite(durations).all() or np.any(durations <= 0)
            or np.any(np.diff(stamps) <= 0)):
        raise ValueError("invalid chunk timestamps/durations shape, dtype or values")
    times = stamps.astype("datetime64[us]")
    if np.isnat(times).any():
        raise ValueError("invalid chunk timestamps")
    half = np.rint(durations*1_800_000_000).astype("timedelta64[us]")
    starts, ends = times-half, times+half
    if (np.any(starts[1:] < ends[:-1]) or starts[0] != utc_timestamp(block["start_utc"])
            or ends[-1] != utc_timestamp(block["end_utc"])):
        raise ValueError("block temporal metadata disagrees with chunk timestamps/durations")


class ResultWriter:
    """Publish completed blocks; resume validates configuration and every chunk.

    Concurrent writers to the same directory are unsupported. Atomic publication
    protects readers/crash recovery, not concurrent simulation processes.
    """
    def __init__(self, path, *, configuration, sensor_ids, components=("direct", "sky", "reflected", "transmitted"),
                 bands=("neutral_R", "neutral_G", "neutral_B"), sensor_groups=None,
                 storage="npy", retention="hourly", resume=False):
        self.path = Path(path).resolve()
        axes = {"sensors": _ids(sensor_ids, "sensor IDs"), "components": _ids(components, "components"), "bands": _ids(bands, "bands")}
        if storage not in ("npy", "npz") or retention not in ("hourly", "exposure"):
            raise ValueError("storage must be npy/npz and retention hourly/exposure")
        configuration = json_value(configuration)
        if not isinstance(configuration, dict):
            raise ValueError("configuration must be an object")
        groups = list(sensor_groups) if sensor_groups is not None else [""]*len(axes["sensors"])
        if len(groups) != len(axes["sensors"]) or any(not isinstance(x, str) for x in groups):
            raise ValueError("sensor groups must align with sensors")
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "manifest.json").exists():
            if not resume:
                raise FileExistsError("existing result requires resume=True")
            self.manifest = _load_manifest(self.path)
            if any(self.manifest[key] != value for key, value in (("configuration", configuration), ("axes", axes),
                    ("sensor_groups", groups), ("storage", storage), ("retention", retention))):
                raise ValueError("resume configuration/provenance mismatch")
            for block in self.manifest["blocks"]:
                for descriptor in block["files"].values():
                    _verify(self.path, descriptor)
            # Aggregates are derivable state, not a trusted checkpoint input.
            reader = ResultReader(self.path, verify=False)
            shape = tuple(len(axes[k]) for k in ("sensors", "components", "bands"))
            exposure, uncertainty = np.zeros(shape), np.zeros(shape)
            for block in self.manifest["blocks"]:
                names = ("values", "errors") if retention == "hourly" else ("exposure", "exposure_error")
                arrays = reader._arrays(block, ("timestamps", "durations_hours") + names)
                _validate_block_time(block, arrays)
                expected = (block["stop"]-block["start"],) + shape if retention == "hourly" else shape
                if any(arrays[key].shape != expected or arrays[key].dtype != np.float64
                       or not np.isfinite(arrays[key]).all() or np.any(arrays[key] < 0) for key in names):
                    raise ValueError("invalid checkpoint numeric chunk")
                if retention == "hourly":
                    exposure += np.sum(arrays["values"]*arrays["durations_hours"][:, None, None, None], axis=0)
                    uncertainty += np.sum(arrays["errors"]*arrays["durations_hours"][:, None, None, None], axis=0)
                else:
                    exposure += arrays["exposure"]
                    uncertainty += arrays["exposure_error"]
            for key, expected in (("exposure_wh_m2", exposure), ("exposure_one_se_conservative_bound_wh_m2", uncertainty)):
                if not np.allclose(self.manifest[key], expected, rtol=1e-13, atol=1e-13):
                    raise ValueError("checkpoint exposure aggregate disagrees with completed chunks")
        else:
            if resume:
                raise FileNotFoundError("no completed result manifest to resume")
            shape = tuple(len(axes[k]) for k in ("sensors", "components", "bands"))
            self.manifest = dict(schema=FORMAT, version=1, configuration=configuration, axes=axes,
                sensor_groups=groups, storage=storage, retention=retention, completed_steps=0, complete=False,
                units={"values": "W/m2", "errors": "W/m2 one standard error", "exposure": "Wh/m2"},
                timestamps="UTC interval midpoints", uncertainty="replicate standard error; integrated conservative linear sum; no independent-hour assumption",
                blocks=[], exposure_wh_m2=np.zeros(shape).tolist(), exposure_one_se_conservative_bound_wh_m2=np.zeros(shape).tolist())
            _publish(self.path, self.manifest)

    @property
    def completed_steps(self):
        return self.manifest["completed_steps"]

    def append(self, timestamps, durations_hours, values, errors, scene_revision, material_revision):
        if self.manifest["complete"]:
            raise ValueError("cannot append to completed result")
        stamps = np.asarray(timestamps)
        if stamps.dtype.kind != "M" or stamps.ndim != 1 or not len(stamps) or np.isnat(stamps).any():
            raise ValueError("timestamps must be nonempty finite UTC datetime64 array")
        stamps = stamps.astype("datetime64[us]")
        durations = np.asarray(durations_hours, dtype=np.float64)
        if durations.shape != stamps.shape or not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("durations must be positive finite hours aligned with timestamps")
        half = np.rint(durations*1_800_000_000).astype("timedelta64[us]")
        starts, ends = stamps-half, stamps+half
        if np.any(np.diff(stamps).astype(np.int64) <= 0) or np.any(starts[1:] < ends[:-1]):
            raise ValueError("timestamps must increase with nonoverlapping intervals")
        if self.manifest["blocks"] and starts[0] < utc_timestamp(self.manifest["blocks"][-1]["end_utc"]):
            raise ValueError("new block overlaps completed time")
        shape = (len(stamps),) + tuple(len(self.manifest["axes"][key]) for key in ("sensors", "components", "bands"))
        values, errors = np.asarray(values, dtype=np.float64), np.asarray(errors, dtype=np.float64)
        if (values.shape != shape or errors.shape != shape or not np.isfinite(values).all()
                or not np.isfinite(errors).all() or np.any(values < 0) or np.any(errors < 0)):
            raise ValueError("values/errors must be finite nonnegative arrays with axes (time,sensor,component,band)")
        integer(scene_revision, "scene revision")
        integer(material_revision, "material revision")
        if self.manifest["blocks"]:
            previous = self.manifest["blocks"][-1]
            if scene_revision < previous["scene_revision"] or material_revision < previous["material_revision"]:
                raise ValueError("scene/material revisions cannot decrease")
        integrated = np.sum(values*durations[:, None, None, None], axis=0)
        integrated_error = np.sum(errors*durations[:, None, None, None], axis=0)
        arrays = {"timestamps": stamps.astype(np.int64), "durations_hours": durations}
        if self.manifest["retention"] == "hourly":
            arrays.update(values=values, errors=errors)
        else:
            arrays.update(exposure=integrated, exposure_error=integrated_error)
        # Unique filenames allow recovery after interrupted/unpublished writes.
        prefix = f"block-{len(self.manifest['blocks']):06d}-{uuid.uuid4().hex}"
        files = {}
        if self.manifest["storage"] == "npz":
            target = self.path / (prefix + ".npz")
            with target.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            files["archive"] = {"path": target.name, "sha256": file_sha256(target)}
        else:
            for key, value in arrays.items():
                target = self.path / f"{prefix}-{key}.npy"
                with target.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                files[key] = {"path": target.name, "sha256": file_sha256(target)}
        next_manifest = json.loads(json.dumps(self.manifest))
        next_manifest["blocks"].append(dict(start=self.completed_steps, stop=self.completed_steps+len(stamps),
            start_utc=str(starts[0])+"Z", end_utc=str(ends[-1])+"Z", scene_revision=scene_revision,
            material_revision=material_revision, files=files))
        next_manifest["completed_steps"] += len(stamps)
        for name, addition in (("exposure_wh_m2", integrated), ("exposure_one_se_conservative_bound_wh_m2", integrated_error)):
            next_manifest[name] = (np.array(next_manifest[name]) + addition).tolist()
        _publish(self.path, next_manifest)
        self.manifest = next_manifest

    def finish(self):
        completed = {**self.manifest, "complete": True}
        _publish(self.path, completed)
        self.manifest = completed
        return self.manifest


class ResultReader:
    """Read only intersecting time blocks; NPY sensor selection uses memory maps.

    With verify=True (default), selected payload files are streamed once for
    checksum verification. This bounds RAM but necessarily reads their bytes.
    """
    def __init__(self, path, *, verify=True):
        self.path = Path(path).resolve()
        self.manifest = _load_manifest(self.path)
        self.verify = verify
        self.files_read = []

    def _arrays(self, block, keys):
        if self.manifest["storage"] == "npz":
            descriptor = block["files"]["archive"]
            path = _verify(self.path, descriptor) if self.verify else contained_path(self.path, descriptor["path"])
            self.files_read.append(descriptor["path"])
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key] for key in keys}
        result = {}
        for key in keys:
            descriptor = block["files"][key]
            path = _verify(self.path, descriptor) if self.verify else contained_path(self.path, descriptor["path"])
            self.files_read.append(descriptor["path"])
            result[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return result

    def read(self, start=0, stop=None, *, sensor_ids=None, components=None, bands=None):
        if self.manifest["retention"] != "hourly":
            raise ValueError("hourly values were not retained in exposure-only store")
        stop = self.manifest["completed_steps"] if stop is None else stop
        integer(start, "start")
        integer(stop, "stop")
        if stop < start or stop > self.manifest["completed_steps"]:
            raise ValueError("time window outside completed intervals")
        indices, labels = [], {}
        for axis, selected in (("sensors", sensor_ids), ("components", components), ("bands", bands)):
            available = self.manifest["axes"][axis]
            selected = available if selected is None else list(selected)
            if any(key not in available for key in selected) or len(set(selected)) != len(selected):
                raise ValueError(f"invalid selected {axis} IDs")
            labels[axis] = selected
            indices.append([available.index(key) for key in selected])
        shape = (stop-start,) + tuple(len(x) for x in indices)
        out = dict(values=np.empty(shape), errors=np.empty(shape), timestamps=np.empty(stop-start, dtype="datetime64[us]"),
                   durations_hours=np.empty(stop-start), axes=labels)
        self.files_read = []
        for block in self.manifest["blocks"]:
            first, last = max(start, block["start"]), min(stop, block["stop"])
            if first >= last:
                continue
            arrays = self._arrays(block, ("values", "errors", "timestamps", "durations_hours"))
            section = slice(first-block["start"], last-block["start"])
            output = slice(first-start, last-start)
            expected = (block["stop"]-block["start"],) + tuple(len(self.manifest["axes"][key]) for key in ("sensors", "components", "bands"))
            if any(arrays[key].shape != expected or arrays[key].dtype != np.float64 for key in ("values", "errors")):
                raise ValueError("chunk values/errors shape or dtype mismatch")
            _validate_block_time(block, arrays)
            for key in ("values", "errors"):
                # Select time and sensors first, so never copy all sensors.
                out[key][output] = arrays[key][section][:, indices[0]][:, :, indices[1]][:, :, :, indices[2]]
            out["timestamps"][output] = arrays["timestamps"][section].astype("datetime64[us]")
            out["durations_hours"][output] = arrays["durations_hours"][section]
        return out

    def read_times(self, start_utc, stop_utc, **selection):
        first, last = utc_timestamp(start_utc), utc_timestamp(stop_utc)
        if last < first:
            raise ValueError("time window stop precedes start")
        indices = []
        for block in self.manifest["blocks"]:
            # Time metadata is checked against the small numeric axis even for
            # excluded blocks, so corrupted bounds cannot hide valid samples.
            time_arrays = self._arrays(block, ("timestamps", "durations_hours"))
            _validate_block_time(block, time_arrays)
            if utc_timestamp(block["end_utc"]) <= first or utc_timestamp(block["start_utc"]) >= last:
                continue
            stamps = time_arrays["timestamps"].astype("datetime64[us]")
            indices.extend((np.flatnonzero((stamps >= first) & (stamps < last)) + block["start"]).tolist())
        return self.read(indices[0], indices[-1]+1, **selection) if indices else self.read(0, 0, **selection)

    def export_csv(self, path, start=0, stop=None, **selection):
        """Export a selected bounded window, one row per time/sensor/component/band."""
        result = self.read(start, stop, **selection)
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp_utc", "duration_hours", "sensor_id", "component", "band", "irradiance_w_m2", "standard_error_w_m2"])
            for time_index, stamp in enumerate(result["timestamps"]):
                for sensor_index, sensor in enumerate(result["axes"]["sensors"]):
                    for component_index, component in enumerate(result["axes"]["components"]):
                        for band_index, band in enumerate(result["axes"]["bands"]):
                            key = (time_index, sensor_index, component_index, band_index)
                            writer.writerow([str(stamp)+"Z", result["durations_hours"][time_index], sensor, component, band,
                                             result["values"][key], result["errors"][key]])
        return str(path)
