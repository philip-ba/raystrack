"""Ported v1 store behaviors under the v2 immutable-snapshot contract.

Each checkpoint is a separate store; shared geometry, chunking, committed-only
reads, and result coverage replace the old mutable append/finalize interface.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest import mock

import numpy as np

from raystrack.io import import_v1_json, load, save
from raystrack.model import Mesh, Scene
from raystrack.solver import Channel, Result, SparseValues


def sample_mesh():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    return Mesh(vertices, faces)


def values(rows=(), columns=(), estimates=()):
    return SparseValues(np.asarray(rows, np.int64), np.asarray(columns, np.int64),
                        np.asarray(estimates, np.float64), np.full(len(estimates), np.nan))


class StoreTests(unittest.TestCase):
    def test_complete_run_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            mesh = sample_mesh()
            scene = Scene.from_meshes({"Fenster_ä": mesh, "B": mesh}, labels={"B": "Receiver"})
            channels = (Channel("surface", "B", "front"), Channel("surface", "B", "back"),
                        Channel("sky"), Channel("sky", patch=144), Channel("rest"))
            result = Result(sender_ids=scene.surface_ids, channels=channels,
                            data=values([0, 0, 0, 0, 1, 1], [0, 1, 3, 4, 2, 4],
                                        [0.25, 0.0, 0.4, 0.35, 0.2, 0.8]),
                            coverage=np.asarray([1, 1], np.int8), rays_used=64, cumulative_rays=64,
                            statistics={"emitters": {"Fenster_ä": {"matrix": {"replicates": 4}}}},
                            execution={"device": "cpu", "seed": 7},
                            status="converged", converged=True)
            metadata = {"note": "測定", "tags": [1, True]}
            saved = save(path, scene, result, metadata, chunk_bytes=48)
            self.assertEqual(saved, str(path.resolve()))
            stored = load(path)
            self.assertEqual(stored.metadata["note"], "測定")
            self.assertEqual(stored.metadata["tags"], (1, True))
            self.assertEqual(stored.scene.surface_ids, ("Fenster_ä", "B"))
            self.assertEqual(stored.scene["B"].label, "Receiver")
            self.assertIs(stored.scene["Fenster_ä"].mesh, stored.scene["B"].mesh)
            for surface in stored.scene.surfaces:
                np.testing.assert_array_equal(surface.mesh.vertices, mesh.vertices)
                np.testing.assert_array_equal(surface.mesh.faces, mesh.faces)
                self.assertEqual(surface.mesh.vertices.dtype, np.float32)
                self.assertEqual(surface.mesh.faces.dtype, np.int32)
            actual = stored.result
            self.assertEqual(actual.execution, result.execution)
            self.assertEqual(actual.statistics, result.statistics)
            self.assertEqual(actual.status, "converged")
            self.assertTrue(actual.converged)
            self.assertEqual(actual.cumulative_rays, 64)
            self.assertEqual(actual.value("Fenster_ä", channels[0]), 0.25)
            self.assertEqual(actual.value("Fenster_ä", channels[1]), 0.0)
            self.assertEqual(actual.value("Fenster_ä", channels[3]), 0.4)
            self.assertEqual(actual.value("B", channels[4]), 0.8)
            self.assertEqual(actual.value("B", channels[0]), 0.0)
            # The explicitly recorded zero is retained, independently of an
            # absent sparse entry whose sampled coverage also implies zero.
            np.testing.assert_array_equal(actual.data.estimates, result.data.estimates)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], 2)
            self.assertTrue(manifest["complete"])
            self.assertGreater(len(manifest["scene"]["geometries"][0]["vertices"]["chunks"]), 1)

    def test_refined_snapshots_preserve_partial_coverage_and_selected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Scene.from_meshes({"A": sample_mesh(), "B": sample_mesh()})
            channels = (Channel("surface", "B", "front"), Channel("surface", "A", "back"))
            partial = Result(sender_ids=scene.surface_ids, channels=channels,
                             data=values([0], [0], [0.3]), coverage=np.asarray([1, 0], np.int8),
                             rays_used=16, cumulative_rays=16, status="budget_exhausted")
            refined = Result(sender_ids=scene.surface_ids, channels=channels,
                             data=values([0, 1], [0, 1], [0.3, 0.1]), coverage=np.asarray([1, 1], np.int8),
                             rays_used=16, cumulative_rays=32, status="budget_exhausted")
            first_path = save(Path(directory) / "preview", scene, partial)
            second_path = save(Path(directory) / "refined", scene, refined)
            first, second = load(first_path).result, load(second_path).result
            self.assertEqual(first.value("A", channels[0]), 0.3)
            self.assertIsNone(first.value("B", channels[1]))
            self.assertEqual(second.row("B")[channels[1]], 0.1)
            self.assertEqual(second.row("B")[channels[0]], 0.0)
            self.assertEqual(first.cumulative_rays, 16)
            self.assertEqual(second.rays_used, 16)
            self.assertEqual(second.cumulative_rays, 32)
            # Checkpoints are immutable stores; saving a refined run uses a new
            # path and never changes the preview already available to readers.
            with self.assertRaises(FileExistsError):
                save(first_path, scene, refined)
            self.assertIsNone(load(first_path).result.value("B", channels[1]))

    def test_uncommitted_chunks_are_ignored_and_failed_snapshot_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Scene.from_meshes({"A": sample_mesh()})
            channel = Channel("sky")
            result = Result(sender_ids=("A",), channels=(channel,), data=values([0], [0], [0.2]),
                            coverage=np.asarray([1], np.int8))
            interrupted = Path(directory) / "interrupted.raystrack"
            with mock.patch("raystrack.io.write_manifest", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save(interrupted, scene, result)
            self.assertTrue((interrupted / "result" / "estimates" / "000000.npy").is_file())
            self.assertFalse((interrupted / "manifest.json").exists())
            with self.assertRaises(FileNotFoundError):
                load(interrupted)
            committed = Path(save(Path(directory) / "recovered", scene, result))
            # Unlisted numerical files, even invalid payloads, are invisible to
            # readers. Only a published manifest grants a chunk membership.
            np.save(committed / "result" / "estimates" / "uncommitted.npy",
                    np.asarray(["invalid"], dtype=object))
            geometry_directory = committed / "geometry" / "orphan"
            geometry_directory.mkdir()
            np.save(geometry_directory / "uncommitted.npy", np.asarray([99.0]))
            self.assertEqual(load(committed).result.value("A", channel), 0.2)

    def test_validation_and_exclusive_store_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.raystrack"
            scene = Scene.from_meshes({"A": sample_mesh()})
            with self.assertRaisesRegex(ValueError, "unique"):
                Scene.from_meshes([("A", sample_mesh()), ("A", sample_mesh())])
            with self.assertRaisesRegex(ValueError, "shape"):
                Mesh(np.zeros((2, 2)), np.zeros((0, 3), dtype=np.int32))
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                Mesh(np.zeros((2, 3)), [[0, 1, 2]])
            with self.assertRaisesRegex(ValueError, "integer"):
                Mesh(np.zeros((2, 3)), [[0.0, 1.0, 0.0]])
            with self.assertRaisesRegex(ValueError, "unique"):
                values([0, 0], [0, 0], [0.0, 0.1])
            with self.assertRaisesRegex(ValueError, "finite"):
                values([0], [0], [float("nan")])
            with self.assertRaisesRegex(ValueError, "chunk_bytes"):
                save(path, scene, chunk_bytes=0)
            self.assertFalse(path.exists())

            import raystrack.io as storage
            original_write = storage.write_array
            writing, release = Event(), Event()
            errors = []

            def paused_write(*args, **kwargs):
                writing.set()
                if not release.wait(5):
                    raise RuntimeError("concurrent writer test timed out")
                return original_write(*args, **kwargs)

            def first_writer():
                try:
                    save(path, scene)
                except BaseException as error:
                    errors.append(error)

            with mock.patch("raystrack.io.write_array", side_effect=paused_write):
                worker = Thread(target=first_writer)
                worker.start()
                try:
                    self.assertTrue(writing.wait(5))
                    with self.assertRaises(FileExistsError):
                        save(path, scene)
                finally:
                    release.set()
                    worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(load(path).scene.surface_ids, ("A",))
            with self.assertRaises(FileExistsError):
                save(path, scene)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load(path)

    def test_result_row_limit_creates_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = sample_mesh()
            scene = Scene.from_meshes({f"sender_{i:03d}": mesh for i in range(257)})
            channel = Channel("sky")
            result = Result(sender_ids=scene.surface_ids, channels=(channel,), data=values(),
                            coverage=np.ones(257, np.int8), rays_used=257, cumulative_rays=257)
            path = Path(save(Path(directory) / "case", scene, result, chunk_bytes=256))
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([item["rows"] for item in manifest["result"]["coverage"]["chunks"]], [256, 1])
            stored = load(path)
            self.assertEqual(len(stored.scene), 257)
            self.assertEqual(stored.result.row("sender_256")[channel], 0.0)
            self.assertEqual(stored.result.coverage.shape, (257,))

    def test_existing_json_files_still_import(self):
        root = Path(__file__).resolve().parents[1]
        stored = import_v1_json(root / "examples" / "street_canyon.json", root / "examples" / "vf_matrix.json")
        self.assertGreater(len(stored.scene), 0)
        self.assertGreater(len(stored.result.sender_ids), 0)
        self.assertTrue(np.all(stored.result.coverage == -1))
        self.assertEqual(stored.result.statistics, {})
        self.assertIsNone(stored.result.cumulative_rays)


if __name__ == "__main__":
    unittest.main()
