from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from raystrack import (
    MatrixParams,
    SkyParams,
    load_meshes_json,
    load_vf_matrix_json,
    open_store,
    save_run,
)


def sample_mesh(name="A"):
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    return name, vertices, faces


class StoreTests(unittest.TestCase):
    def test_complete_run_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            meshes = [sample_mesh("Fenster_ä"), sample_mesh("B")]
            matrix = MatrixParams(seed=7)
            sky_params = SkyParams(discrete=True)
            scene = {"Fenster_ä": {"B_front": 0.25, "B_back": 0.0}, "B": {}}
            sky = {"Fenster_ä": {"Sky_Patch_1": 0.0, "Sky_Patch_145": 0.4}, "B": {"Sky": 0.2}}
            rest = {"Fenster_ä": {"Rest": 0.35}, "B": {"Rest": 0.8}}
            saved = save_run(path, meshes=meshes, matrix_params=matrix, sky_params=sky_params,
                             scene=scene, sky=sky, rest=rest,
                             metadata={"note": "測定", "tags": [1, True]}, chunk_bytes=48)
            self.assertEqual(saved, str(path.resolve()))
            with open_store(path) as store:
                self.assertTrue(store.complete)
                self.assertEqual(store.matrix_params, matrix)
                self.assertEqual(store.sky_params, sky_params)
                self.assertEqual(store.metadata, {"note": "測定", "tags": [1, True]})
                loaded = store.load_meshes()
                self.assertEqual([mesh[0] for mesh in loaded], ["Fenster_ä", "B"])
                for (_, expected_v, expected_f), (_, actual_v, actual_f) in zip(meshes, loaded):
                    np.testing.assert_array_equal(actual_v, expected_v)
                    np.testing.assert_array_equal(actual_f, expected_f)
                    self.assertEqual(actual_v.dtype, np.float32)
                    self.assertEqual(actual_f.dtype, np.int32)
                self.assertGreater(len(list(store.iter_mesh_chunks("B", "vertices"))), 1)
                self.assertEqual(store.load_result("scene"), scene)
                self.assertEqual(store.load_result("sky"), sky)
                self.assertEqual(store.load_result("rest"), rest)
                self.assertIn("B_back", store.load_result("scene")["Fenster_ä"])
                self.assertNotIn("B_front", store.load_result("scene")["B"])

    def test_append_resume_and_selective_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            with open_store(path, "w", chunk_bytes=512) as store:
                store.add_mesh(*sample_mesh("A"))
                store.add_mesh(*sample_mesh("B"))
                store.append_result_rows("scene", {"A": {"B_front": 0.3}})
                store.flush()
                self.assertFalse(store.complete)
            with open_store(path, "a") as store:
                store.append_result_rows("scene", [("B", {"A_back": 0.1})])
                store.finalize()
            with open_store(path) as store:
                self.assertEqual([name for name, _, _ in store.load_meshes(["B"])], ["B"])
                with mock.patch("raystrack.store.np.load", wraps=np.load) as loader:
                    self.assertEqual(store.load_result("scene", ["B"]), {"B": {"A_back": 0.1}})
                    self.assertEqual(loader.call_count, 3)
                self.assertEqual(list(store.iter_result_rows("scene", ["A"])),
                                 [("A", {"B_front": 0.3})])
            with self.assertRaisesRegex(ValueError, "finalized"):
                open_store(path, "a")

    def test_uncommitted_chunk_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            with self.assertRaisesRegex(OSError, "interrupted"):
                with open_store(path, "w") as writer:
                    writer.append_result_rows("scene", {"A": {"B_front": 99.0}})
                    with mock.patch.object(writer, "_commit", side_effect=OSError("interrupted")):
                        writer.flush()
            self.assertTrue((path / "results" / "scene" / "000000").is_dir())
            with open_store(path) as reader:
                self.assertEqual(reader.load_result("scene"), {})
            with open_store(path, "a") as writer:
                writer.append_result_rows("scene", {"A": {"B_front": 0.2}})
                writer.finalize()
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["results"]["scene"]["chunks"][0]["path"],
                             "results/scene/000001")
            with open_store(path) as reader:
                self.assertEqual(reader.load_result("scene"), {"A": {"B_front": 0.2}})

    def test_validation_and_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            with open_store(path, "w") as store:
                store.add_mesh(*sample_mesh())
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    store.add_mesh(*sample_mesh())
                with self.assertRaisesRegex(ValueError, "shape"):
                    store.add_mesh("bad", np.zeros((2, 2)), np.zeros((0, 3), dtype=np.int32))
                with self.assertRaisesRegex(ValueError, "out of bounds"):
                    store.add_mesh("bad", np.zeros((2, 3)), [[0, 1, 2]])
                with self.assertRaisesRegex(TypeError, "integers"):
                    store.add_mesh("bad", np.zeros((2, 3)), [[0.0, 1.0, 0.0]])
                store.append_result_rows("scene", {"A": {"B_front": 0.0}})
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    store.append_result_rows("scene", {"A": {}})
                with self.assertRaisesRegex(ValueError, "finite"):
                    store.append_result_rows("sky", {"A": {"Sky": float("nan")}})
                with self.assertRaisesRegex(RuntimeError, "Another writer"):
                    open_store(path, "a")
            with self.assertRaises(FileExistsError):
                open_store(path, "w")
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                open_store(path)

    def test_result_row_limit_creates_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            rows = {f"sender_{i:03d}": {} for i in range(257)}
            save_run(path, meshes=[], scene=rows)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([item["count"] for item in manifest["results"]["scene"]["chunks"]],
                             [256, 1])
            with open_store(path) as store:
                self.assertEqual(store.load_result("scene", ["sender_256"]),
                                 {"sender_256": {}})

    def test_existing_json_files_still_load(self):
        root = Path(__file__).resolve().parents[1]
        meshes = load_meshes_json(str(root / "examples" / "street_canyon.json"))
        matrix = load_vf_matrix_json(str(root / "examples" / "vf_matrix.json"))
        self.assertGreater(len(meshes), 0)
        self.assertGreater(len(matrix), 0)


if __name__ == "__main__":
    unittest.main()
