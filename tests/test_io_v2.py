"""Versioned storage preserves structured results and distrusts descriptors."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from raystrack.model import Mesh, Scene, Surface
from raystrack.solver import Channel, Result, SparseValues
from raystrack.io import import_v1_json, load, save


def mesh():
    return Mesh(np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32),
                np.asarray([[0, 1, 2], [0, 2, 3]], np.int32))


def scene_and_result():
    shared = mesh()
    transform = np.eye(4)
    transform[2, 3] = 1
    scene = Scene([Surface("A", shared, "Sender label"), Surface("B", shared, "Empfänger", transform)], revision=7)
    channels = (Channel("surface", "B", "front"), Channel("surface", "B", "back"),
                Channel("sky", patch=144), Channel("unrequested"), Channel("rest"))
    result = Result(sender_ids=("A", "B"), channels=channels,
                    data=SparseValues(np.asarray([0, 0], np.int64), np.asarray([0, 2], np.int64),
                                      np.asarray([0.25, 0.5]), np.asarray([0.01, np.nan])),
                    coverage=np.asarray([1, 0], np.int8), statistics={"emitters": {"A": {"replicates": 4}}},
                    execution={"device": "cpu", "calibrated": False}, scene_revision=7,
                    rays_used=64, cumulative_rays=128, status="budget_exhausted", converged=False,
                    elapsed_ms=2.5, provenance=({"operation": "reciprocity", "complete": False},))
    return scene, result


def read_manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def replace_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_v2_round_trip_shared_geometry_instances_and_sparse_result(tmp_path):
    scene, result = scene_and_result()
    path = Path(save(tmp_path / "case", scene, result, {"label": "測定", "tags": [1, True]}, chunk_bytes=12))
    manifest = read_manifest(path)
    assert manifest["version"] == 2
    assert len(manifest["scene"]["geometries"]) == 1
    assert len(manifest["scene"]["geometries"][0]["vertices"]["chunks"]) == 4
    loaded = load(path)
    assert loaded.scene.revision == 7
    first, second = loaded.scene.surfaces
    assert first.mesh is second.mesh
    assert first.surface_id == "A" and second.label == "Empfänger"
    np.testing.assert_array_equal(second.transform, scene.surfaces[1].transform)
    np.testing.assert_array_equal(first.mesh.vertices, scene.surfaces[0].mesh.vertices)
    actual = loaded.result
    assert actual.channels == result.channels and actual.sender_ids == result.sender_ids
    assert actual.value("A", result.channels[0]) == 0.25
    assert actual.value("A", result.channels[1]) == 0.0
    assert actual.value("B", result.channels[0]) is None
    assert actual.error("A", result.channels[2]) is None
    assert actual.rays_used == 64 and actual.cumulative_rays == 128
    assert actual.provenance == result.provenance
    np.testing.assert_array_equal(actual.data.errors, result.data.errors)
    assert loaded.metadata["tags"] == (1, True)
    with pytest.raises(TypeError):
        loaded.metadata["new"] = 1
    with pytest.raises(ValueError):
        actual.data.estimates.setflags(write=True)
    with pytest.raises(FileExistsError):
        save(path, scene, result)


def test_scene_only_round_trip_and_revision_mismatch(tmp_path):
    scene, result = scene_and_result()
    path = save(tmp_path / "scene.raystrack", scene)
    assert load(path).result is None
    different = Scene(scene.surfaces, revision=8)
    with pytest.raises(ValueError, match="revision"):
        save(tmp_path / "bad", different, result)
    assert not (tmp_path / "bad.raystrack").exists()


def test_failed_array_write_never_publishes_manifest(tmp_path):
    scene, _ = scene_and_result()
    path = tmp_path / "interrupted.raystrack"
    with mock.patch("raystrack.io._arrays.np.save", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            save(path, scene)
    assert not (path / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        load(path)


@pytest.mark.parametrize("relative", ["../outside.npy", "C:/outside.npy", "C:outside.npy", "//server/share/outside.npy", "geometry/../../outside.npy", "..\\outside.npy"])
def test_rejects_chunk_paths_outside_store(tmp_path, relative):
    scene, _ = scene_and_result()
    path = Path(save(tmp_path / "case", scene))
    manifest = read_manifest(path)
    manifest["scene"]["geometries"][0]["vertices"]["chunks"][0]["path"] = relative
    replace_manifest(path, manifest)
    with pytest.raises(ValueError, match="path"):
        load(path)


@pytest.mark.parametrize("change", ["shape", "dtype", "rows", "indices", "object", "version"])
def test_rejects_malformed_arrays_and_descriptors(tmp_path, change):
    scene, result = scene_and_result()
    path = Path(save(tmp_path / "case", scene, result))
    manifest = read_manifest(path)
    geometry = manifest["scene"]["geometries"][0]
    if change == "shape":
        geometry["vertices"]["shape"][1] = 4
    elif change == "dtype":
        geometry["vertices"]["dtype"] = "<f8"
    elif change == "rows":
        geometry["vertices"]["chunks"][0]["rows"] += 1
    elif change == "indices":
        filename = path / manifest["result"]["channel_indices"]["chunks"][0]["path"]
        np.save(filename, np.asarray([0, 99], np.int64), allow_pickle=False)
    elif change == "object":
        filename = path / geometry["vertices"]["chunks"][0]["path"]
        np.save(filename, np.asarray([["untrusted"] * 3] * 4, dtype=object))
    else:
        manifest["version"] = 999
    replace_manifest(path, manifest)
    with pytest.raises(ValueError):
        load(path)


def test_rejects_nonfinite_metadata_before_creating_store(tmp_path):
    scene, _ = scene_and_result()
    with pytest.raises(ValueError, match="finite"):
        save(tmp_path / "bad", scene, metadata={"x": np.inf})
    with pytest.raises(TypeError, match="keys"):
        save(tmp_path / "bad", scene, metadata={1: "wrong"})
    assert not (tmp_path / "bad.raystrack").exists()


def write_v1_fixture(path):
    path.mkdir()
    geometry = path / "geometry" / "000000"
    geometry.mkdir(parents=True)
    original = mesh()
    descriptors = {}
    for key, array in (("vertices", original.vertices), ("faces", original.faces)):
        filename = geometry / f"{key}-0000.npy"
        np.save(filename, array, allow_pickle=False)
        descriptors[key] = {"rows": len(array), "chunks": [{"path": filename.relative_to(path).as_posix(), "rows": len(array)}]}
    results = {kind: {"senders": [], "receivers": [], "chunks": []} for kind in ("scene", "sky", "rest")}
    for kind, receivers, values in (("scene", ["B_front", "B_back"], [0.25, 0.0]),
                                   ("sky", ["Sky_Patch_145"], [0.5])):
        directory = path / "results" / kind / "000000"
        directory.mkdir(parents=True)
        for name, array in (("offsets", np.asarray([0, len(values)], np.int64)),
                            ("columns", np.arange(len(values), dtype=np.int32)),
                            ("values", np.asarray(values, np.float64))):
            np.save(directory / f"{name}.npy", array, allow_pickle=False)
        results[kind] = {"senders": ["A"], "receivers": receivers,
                         "chunks": [{"path": directory.relative_to(path).as_posix(), "start": 0, "count": 1}]}
    manifest = {"format": "raystrack-store", "version": 1, "complete": True,
                "chunk_bytes": 48, "params": {"matrix": {"seed": 7}, "sky": None},
                "metadata": {"note": "legacy"}, "meshes": [{"name": "A", **descriptors}, {"name": "B", **descriptors}], "results": results}
    replace_manifest(path, manifest)
    return manifest


def test_v1_store_import_preserves_known_values_without_inventing_coverage(tmp_path):
    path = tmp_path / "legacy.raystrack"
    write_v1_fixture(path)
    loaded = load(path)
    result = loaded.result
    assert result.value("A", Channel("surface", "B", "front")) == 0.25
    assert result.value("A", Channel("surface", "B", "back")) == 0.0
    assert result.value("A", Channel("sky", patch=144)) == 0.5
    assert result.error("A", Channel("surface", "B", "front")) is None
    assert result.coverage.tolist() == [-1, -1]
    assert result.statistics == {} and result.execution == {}
    assert result.rays_used is None and result.cumulative_rays is None and result.converged is None
    assert loaded.metadata["v1_import"]["params"]["matrix"]["seed"] == 7
    converted = load(save(tmp_path / "converted", loaded.scene, loaded.result, loaded.metadata))
    assert converted.result.coverage.tolist() == [-1, -1]
    assert converted.result.value("A", Channel("surface", "B", "back")) == 0.0


@pytest.mark.parametrize("change", ["path", "offsets", "columns", "values"])
def test_v1_import_validates_legacy_sparse_arrays(tmp_path, change):
    path = tmp_path / "legacy.raystrack"
    manifest = write_v1_fixture(path)
    directory = path / "results" / "scene" / "000000"
    if change == "path":
        manifest["results"]["scene"]["chunks"][0]["path"] = "../external"
        replace_manifest(path, manifest)
    elif change == "offsets":
        np.save(directory / "offsets.npy", np.asarray([1, 2], np.int64))
    elif change == "columns":
        np.save(directory / "columns.npy", np.asarray([0, 999], np.int32))
    else:
        np.save(directory / "values.npy", np.asarray([np.nan, 0], np.float64))
    with pytest.raises(ValueError):
        load(path)


def test_v1_json_import_preserves_exact_mesh_names_before_parsing_suffixes(tmp_path):
    original = mesh()
    meshes_path, scene_path = tmp_path / "meshes.json", tmp_path / "scene.json"
    meshes_path.write_text(json.dumps({"meshes": [{"name": "panel_front", "vertices": original.vertices.tolist(),
                                                  "faces": original.faces.tolist()}]}), encoding="utf-8")
    scene_path.write_text(json.dumps({"panel_front": {"panel_front": 0.3, "unknown_back": 0.2}}), encoding="utf-8")
    loaded = import_v1_json(meshes_path, scene_path)
    assert loaded.result.value("panel_front", Channel("surface", "panel_front")) == 0.3
    assert loaded.result.value("panel_front", Channel("surface", "unknown", "back")) == 0.2
    assert loaded.result.coverage.tolist() == [-1]
    assert loaded.result.provenance[0]["source"] == "json"


def test_empty_geometry_and_empty_result_round_trip(tmp_path):
    scene = Scene.from_meshes({"empty": Mesh(np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))})
    result = Result(sender_ids=("empty",), channels=(Channel("sky"),),
                    data=SparseValues(np.empty(0, np.int64), np.empty(0, np.int64),
                                      np.empty(0), np.empty(0)), coverage=np.asarray([0], np.int8))
    loaded = load(save(tmp_path / "empty", scene, result))
    assert loaded.scene["empty"].mesh.vertices.shape == (0, 3)
    assert loaded.scene["empty"].mesh.faces.shape == (0, 3)
    assert loaded.result.value("empty", Channel("sky")) is None
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps({"meshes": [{"name": "empty", "vertices": [], "faces": []}]}), encoding="utf-8")
    imported = import_v1_json(json_path)
    assert imported.scene["empty"].mesh.vertices.shape == (0, 3)
    assert imported.result is None


def test_numpy_integer_patch_and_counts_serialize_as_json_numbers(tmp_path):
    scene = Scene.from_meshes({"A": mesh()})
    channel = Channel("sky", patch=np.int64(144))
    result = Result(sender_ids=("A",), channels=(channel,),
                    data=SparseValues(np.asarray([0], np.int64), np.asarray([0], np.int64),
                                      np.asarray([1.0]), np.asarray([0.0])), coverage=np.asarray([1], np.int8),
                    rays_used=np.int64(10), cumulative_rays=np.int64(10))
    loaded = load(save(tmp_path / "numpy-scalars", scene, result))
    assert loaded.result.channels[0].patch == 144
    assert loaded.result.rays_used == 10


def test_v2_requires_scene_geometry_for_all_result_surface_ids(tmp_path):
    from dataclasses import replace
    scene, result = scene_and_result()
    unknown = replace(result, sender_ids=("missing", "B"))
    with pytest.raises(ValueError, match="absent from scene geometry"):
        save(tmp_path / "missing", scene, unknown)
    path = Path(save(tmp_path / "valid", scene, result))
    manifest = read_manifest(path)
    manifest["result"]["channels"][0]["surface_id"] = "missing"
    replace_manifest(path, manifest)
    with pytest.raises(ValueError, match="absent from scene geometry"):
        load(path)
