"""Meaningful protocol and lifecycle checks without a Rhino installation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import ast
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from threading import Event, Thread
import time

import numpy as np
import pytest

from raystrack import (Accuracy, Budget, Channel, Query, Result, Sampling,
                       Run, SolveOptions, Solver, SparseValues)
from raystrack.integrations.grasshopper.serialization import (
    options_from_json, query_from_json, result_from_json, result_to_json,
    scene_fingerprint, scene_from_json, scene_to_json)
from raystrack.integrations.grasshopper.worker import WorkerService


def square(z=0, *, down=False, radius=.5):
    faces = [[0, 1, 2], [0, 2, 3]]
    return {"vertices": [[-radius, -radius, z], [radius, -radius, z],
                         [radius, radius, z], [-radius, radius, z]],
            "faces": [face[::-1] for face in faces] if down else faces}


def scene(*, blocked=False):
    surfaces = [{"id": "A", "label": "Emitter", "mesh": square()},
                {"id": "B", "label": "Receiver", "mesh": square(1, down=True)}]
    if blocked:
        surfaces.append({"id": "blocker", "label": "Occluder", "mesh": square(.5, radius=10)})
    return {"kind": "scene", "surfaces": surfaces}


def arguments(*, key="component", launch="first", rays=127, blocked=False):
    options = SolveOptions(Sampling(density=1, rays_per_cell=7, seed=19),
                           Accuracy(max_replicates=4, min_replicates=1, tolerance=0, mode="delta"),
                           batch_size=19)
    return {"key": key, "launch": launch, "scene": scene(blocked=blocked),
            "query": {"kind": "query", "senders": ["A"], "receivers": ["B"],
                      "scene": True, "sky_mode": None, "receiver_sides": ["front", "back"]},
            "options": dict(asdict(options), kind="options"), "device": "cpu",
            "acceleration": "instanced", "ray_budget": rays}


def wait(service, args, *, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = service.request("status", {"key": args["key"], "launch": args["launch"]})
        if state["status"] in ("succeeded", "paused", "cancelled", "failed"):
            assert state["status"] != "failed", state
            return state
        time.sleep(.01)
    raise AssertionError("Worker did not finish its bounded job")


def assert_same(actual, expected):
    restored = result_from_json(actual)
    assert restored.sender_ids == expected.sender_ids
    assert restored.channels == expected.channels
    np.testing.assert_array_equal(restored.coverage, expected.coverage)
    np.testing.assert_array_equal(restored.dense(), expected.dense())
    for sender in expected.sender_ids:
        for channel in expected.channels:
            assert restored.error(sender, channel) == expected.error(sender, channel)
    assert restored.statistics == expected.statistics
    assert restored.cumulative_rays == expected.cumulative_rays


def test_pause_resume_is_exact_and_launch_is_idempotent():
    args = arguments(rays=127)
    with WorkerService() as service:
        initial = service.request("solve", args)
        assert initial["status"] == "preparing"
        partial = wait(service, args)
        assert partial["status"] == "paused"
        assert partial["result"]["rays_used"] == 127
        assert partial["cumulative_rays"] == 127
        assert service.request("solve", args) == partial
        changed = deepcopy(args)
        changed["ray_budget"] = 1
        with pytest.raises(ValueError, match="Arguments changed"):
            service.request("solve", changed)
        args["launch"], args["ray_budget"] = "resumed", 321
        service.request("solve", args)
        final = wait(service, args)
        assert final["status"] == "succeeded"
        assert final["cumulative_rays"] == 448
        assert final["result"]["rays_used"] == 321
        assert final["result"]["status"] in ("max_iters", "converged")
        with pytest.raises(ValueError, match="Stale"):
            service.request("status", {"key": args["key"], "launch": "first"})
        with Solver(scene_from_json(args["scene"]), device="cpu", acceleration="instanced", auto_tune=False) as solver:
            expected = solver.solve(query_from_json(args["query"]), options_from_json(args["options"]), Budget(rays=448))
        assert_same(final["result"], expected)


def test_completed_new_launch_starts_new_run_and_retains_solver():
    args = arguments(rays=0)
    with WorkerService() as service:
        service.request("solve", args)
        first = wait(service, args)
        cache = service._caches[args["key"]]
        solver, run = cache.solver, cache.run
        args["launch"], args["ray_budget"] = "another", 11
        service.request("solve", args)
        second = wait(service, args)
        assert second["cumulative_rays"] == 11
        assert first["cumulative_rays"] == 448
        assert cache.solver is solver and cache.run is not run


def test_cancel_a_paused_run_starts_fresh_on_the_next_launch():
    args = arguments(rays=13)
    with WorkerService() as service:
        service.request("solve", args)
        wait(service, args)
        cancelled = service.request("cancel", {"key": args["key"], "launch": args["launch"]})
        assert cancelled["status"] == "cancelled"
        assert cancelled["result"]["status"] == "cancelled"
        assert cancelled["cumulative_rays"] == 13
        args["launch"] = "new-after-cancel"
        service.request("solve", args)
        assert wait(service, args)["cumulative_rays"] == 13


def test_receiver_filter_keeps_full_scene_occlusion():
    args = arguments(rays=448, blocked=True)
    with WorkerService() as service:
        service.request("solve", args)
        result = result_from_json(wait(service, args)["result"])
    assert result.value("A", Channel("surface", "B", "front")) == 0
    assert result.value("A", Channel("unrequested")) > .99
    assert result.sender_ids == ("A",)
    assert sum(result.row("A").values()) == pytest.approx(1)


def test_transform_update_invalidates_run_and_label_update_can_resume():
    args = arguments(rays=31)
    with WorkerService() as service:
        service.request("solve", args)
        wait(service, args)
        cache = service._caches[args["key"]]
        solver, first_run = cache.solver, cache.run
        args["launch"] = "renamed"
        args["scene"]["surfaces"][0]["label"] = "A renamed label"
        service.request("solve", args)
        renamed = wait(service, args)
        assert cache.run is first_run
        assert renamed["cumulative_rays"] == 62
        assert cache.scene.revision == 0
        args["launch"] = "moved"
        transform = np.eye(4)
        transform[0, 3] = 10
        args["scene"]["surfaces"][1]["transform"] = transform.tolist()
        service.request("solve", args)
        moved = wait(service, args)
        assert cache.solver is solver and cache.run is not first_run
        assert cache.scene.revision == 1
        assert moved["cumulative_rays"] == 31
        assert moved["result"]["scene_revision"] == 1
        assert result_from_json(moved["result"]).value("A", Channel("surface", "B", "front")) == 0


def test_scene_geometry_deduplicates_and_uses_explicit_ids():
    description = scene()
    description["surfaces"][1]["mesh"] = deepcopy(description["surfaces"][0]["mesh"])
    transform = np.eye(4)
    transform[2, 3] = 1
    description["surfaces"][1]["transform"] = transform.tolist()
    restored = scene_from_json(description)
    assert restored["A"].mesh is restored["B"].mesh
    assert scene_fingerprint(scene_from_json(scene_to_json(restored))) == scene_fingerprint(restored)
    assert restored.surface_ids == ("A", "B")


def test_cancel_during_preparation_is_retained_without_tracing(monkeypatch):
    entered, proceed = Event(), Event()
    original = Solver.start

    def held_start(self, *args, **kwargs):
        entered.set()
        assert proceed.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Solver, "start", held_start)
    args = arguments()
    with WorkerService() as service:
        service.request("solve", args)
        assert entered.wait(10)
        cancelled = service.request("cancel", {"key": args["key"], "launch": args["launch"]})
        assert cancelled["message"] == "Cancellation requested"
        proceed.set()
        final = wait(service, args)
        assert final["status"] == "cancelled"
        assert final["cumulative_rays"] == 0
        assert final["result"]["status"] == "cancelled"


def test_invalid_inputs_fail_only_the_job_and_release_cancels_cache():
    args = arguments()
    args["query"]["receivers"] = ["absent"]
    with WorkerService() as service:
        service.request("solve", args)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = service.request("status", {"key": args["key"], "launch": args["launch"]})
            if state["status"] == "failed":
                break
            time.sleep(.01)
        assert state["error"]["type"] == "ValueError"
        assert "Unknown surface IDs" in state["message"]
        service.request("release", {"key": args["key"]})
        with pytest.raises(KeyError):
            service.request("status", {"key": args["key"], "launch": args["launch"]})


def test_bad_relaunch_does_not_poison_the_next_valid_run():
    args = arguments(rays=13)
    with WorkerService() as service:
        service.request("solve", args)
        wait(service, args)
        invalid = deepcopy(args)
        invalid["launch"] = "bad"
        invalid["query"]["receivers"] = ["absent"]
        service.request("solve", invalid)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            failed = service.request("status", {"key": args["key"], "launch": "bad"})
            if failed["status"] == "failed":
                break
            time.sleep(.01)
        assert failed["status"] == "failed"
        args["launch"] = "valid-again"
        service.request("solve", args)
        assert wait(service, args)["cumulative_rays"] == 13


def test_cancel_during_advance_and_shutdown_wait_for_the_owning_thread(monkeypatch):
    entered, proceed = Event(), Event()
    original = Run.advance

    def held_advance(self, *args, **kwargs):
        entered.set()
        assert proceed.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Run, "advance", held_advance)
    args = arguments(rays=100_000)
    service = WorkerService()
    service.request("solve", args)
    assert entered.wait(10)
    closing = Thread(target=service.close)
    closing.start()
    assert service._jobs[args["key"]].cancelled.wait(10)
    proceed.set()
    closing.join(10)
    assert not closing.is_alive()
    assert service._jobs[args["key"]].payload["status"] == "cancelled"
    assert service._jobs[args["key"]].payload["cumulative_rays"] == 0
    assert not service._caches


def test_snapshot_preserves_known_zero_and_unknown_coverage():
    result = Result(("A", "B", "C"), (Channel("surface", "A", "front"), Channel("rest")),
                    SparseValues(np.array([0, 2]), np.array([0, 0]),
                                 np.array([0., .25]), np.array([np.nan, np.nan])),
                    np.array([1, 0, -1]), rays_used=None, cumulative_rays=None, converged=None)
    payload = result_to_json(result)
    assert payload["values"] == [[0, 0], [None, None], [.25, None]]
    assert payload["errors"] == [[None, None], [None, None], [None, None]]
    assert_same(payload, result)
    json.dumps(payload, allow_nan=False)


def test_save_load_uses_v2_and_checks_geometry_after_transform_update(tmp_path):
    args = arguments(rays=31)
    with WorkerService() as service:
        service.request("solve", args)
        wait(service, args)
        transform = np.eye(4)
        transform[0, 3] = 1
        args["launch"] = "moved"
        args["scene"]["surfaces"][1]["transform"] = transform.tolist()
        service.request("solve", args)
        final = wait(service, args)
        path = str(tmp_path / "moving.raystrack")
        service.request("save", {"path": path, "scene": args["scene"], "result": final["result"],
                                 "metadata": {"origin": "Grasshopper"}}).result(10)
        loaded = service.request("load", {"path": path}).result(10)
        assert loaded["scene"]["revision"] == 1
        assert loaded["metadata"] == {"origin": "Grasshopper"}
        assert loaded["result"] == final["result"]
        bad_scene = scene()
        with pytest.raises(ValueError, match="different scene geometry"):
            service.request("save", {"path": str(tmp_path / "bad.raystrack"), "scene": bad_scene,
                                     "result": final["result"]}).result(10)
    assert json.loads((Path(path) / "manifest.json").read_text())["version"] == 2


def test_reciprocity_is_only_applied_after_full_completion():
    args = arguments(rays=47)
    args["query"]["senders"], args["query"]["receivers"] = None, None
    args["options"]["postprocessing"]["reciprocity"] = "bidirectional"
    with WorkerService() as service:
        service.request("solve", args)
        partial = wait(service, args)
        assert partial["status"] == "paused"
        assert partial["result"]["provenance"] == []
        args["launch"], args["ray_budget"] = "finished", 0
        service.request("solve", args)
        final = wait(service, args)
        assert final["status"] == "succeeded"
        assert final["result"]["provenance"]
        assert final["result"]["statistics"]["options"]["postprocessing"]["reciprocity"] == "bidirectional"


class ProtocolClient:
    def __init__(self, python=None):
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), NUMBA_NUM_THREADS="2")
        self.process = subprocess.Popen([python or sys.executable, "-u", "-m",
                                         "raystrack.integrations.grasshopper.worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", env=environment)
        self.lines, self.errors = queue.Queue(), []
        self.identifier = 0
        self.reader = Thread(target=self._read, daemon=True)
        self.stderr_reader = Thread(target=self._read_errors, daemon=True)
        self.reader.start()
        self.stderr_reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)

    def _read_errors(self):
        for line in self.process.stderr:
            self.errors.append(line)

    def request(self, operation, arguments=None, *, raw=None, timeout=90):
        self.identifier += 1
        payload = {"protocol": 2, "id": self.identifier, "operation": operation,
                   "arguments": arguments or {}}
        self.process.stdin.write((json.dumps(payload) if raw is None else raw) + "\n")
        self.process.stdin.flush()
        line = self.lines.get(timeout=timeout)
        response = json.loads(line)
        assert response["protocol"] == 2
        assert response["id"] == (self.identifier if raw is None else None)
        return response

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            raise AssertionError("Worker did not cancel and exit on EOF")
        self.reader.join(10)
        self.stderr_reader.join(10)
        assert self.process.returncode == 0, "".join(self.errors)


def test_subprocess_protocol_rejects_malformed_requests_and_exits_on_eof():
    client = ProtocolClient()
    try:
        error = client.request("runtime", raw="{bad json")
        assert error["ok"] is False
        unknown = client.request("unknown")
        assert unknown["ok"] is False
        runtime = client.request("runtime")
        assert runtime["ok"] is True
        assert runtime["result"]["protocol"] == 2
        assert runtime["result"]["devices"]["cpu"]["available"] is True
        args = arguments(rays=29)
        initial = client.request("solve", args)
        assert initial["result"]["status"] == "preparing"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = client.request("status", {"key": args["key"], "launch": args["launch"]})
            assert state["ok"] is True
            if state["result"]["status"] in ("paused", "failed"):
                break
            time.sleep(.02)
        assert state["result"]["status"] == "paused", state
        assert state["result"]["cumulative_rays"] == 29
    finally:
        client.close()


def test_subprocess_eof_cancels_an_active_job():
    client = ProtocolClient()
    args = arguments(rays=0)
    args["options"]["accuracy"]["max_replicates"] = 100_000_000
    args["options"]["sampling"]["rays_per_cell"] = 128
    args["options"]["batch_size"] = 65_536
    initial = client.request("solve", args)
    assert initial["result"]["status"] == "preparing"
    state = client.request("status", {"key": args["key"], "launch": args["launch"]})
    assert state["ok"]
    client.close()


def test_native_stdout_is_redirected_away_from_protocol(tmp_path):
    script = tmp_path / "native_stdout.py"
    script.write_text("from raystrack.integrations.grasshopper.worker import _protocol_stream\n"
                      "import os,sys\n"
                      "protocol=_protocol_stream()\n"
                      "print('python banner')\n"
                      "os.write(1,b'native banner\\n')\n"
                      "protocol.write('{\\\"protocol\\\":2}\\n')\n"
                      "protocol.close()\n", encoding="utf-8")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            encoding="utf-8", env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"protocol": 2}
    assert "python banner" in result.stderr and "native banner" in result.stderr


@pytest.mark.skipif(__import__("importlib").util.find_spec("taichi") is None, reason="Optional Taichi runtime")
def test_real_vulkan_worker_keeps_native_logs_off_json_stdout():
    client = ProtocolClient()
    try:
        runtime = client.request("runtime")
        devices = runtime["result"]["devices"]
        if "vulkan" not in devices.get("taichi", {}).get("architectures", []):
            pytest.skip("Vulkan unavailable on this test device")
        args = arguments(rays=37)
        args["device"] = "vulkan"
        assert client.request("solve", args)["ok"]
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = client.request("status", {"key": args["key"], "launch": args["launch"]})["result"]
            if state["status"] in ("paused", "failed"):
                break
            time.sleep(.02)
        assert state["status"] == "paused", state
        assert state["cumulative_rays"] == 37
        assert state["result"]["execution"]["backend"] == "vulkan"
        assert any("Taichi" in line for line in client.errors)
    finally:
        client.close()


@pytest.mark.parametrize("mutate, expected", [
    (lambda value: value["surfaces"][0]["mesh"]["vertices"].__setitem__(0, [0, 1]),
     "scene.surfaces[0] (id='A').mesh.vertices[0]"),
    (lambda value: value["surfaces"][1]["mesh"]["faces"][0].__setitem__(2, 99),
     "scene.surfaces[1] (id='B').mesh.faces[0][2]=99"),
    (lambda value: value["surfaces"][1].__setitem__("id", "A"),
     "duplicates scene.surfaces[0].id"),
    (lambda value: value["surfaces"][0].__setitem__("label", 9),
     "scene.surfaces[0] (id='A').label"),
    (lambda value: value["surfaces"][1].__setitem__("transform", np.diag([2., 1., 1., 1.]).tolist()),
     "scene.surfaces[1] (id='B').transform: transform must be a proper rigid"),
])
def test_scene_validation_identifies_surface_field_and_repair(mutate, expected):
    description = scene()
    mutate(description)
    with pytest.raises(ValueError) as error:
        scene_from_json(description)
    assert expected in str(error.value)


@pytest.mark.parametrize("value, expected", [
    (False, "options must be a JSON object"),
    ({"sampling": {"density": 0}}, "options.sampling.density must be an integer"),
    ({"accuracy": {"tolerance": float("inf")}}, "options.accuracy.tolerance must be finite"),
    ({"sampling": {"ray_count": 1}}, "Unknown options.sampling fields"),
])
def test_options_validation_names_the_nested_setting(value, expected):
    with pytest.raises(ValueError) as error:
        options_from_json(value)
    assert expected in str(error.value)


def test_query_and_result_validation_report_selection_and_dimensions():
    with pytest.raises(ValueError, match=r"query.senders\[0\]"):
        query_from_json({"senders": [None]})
    with pytest.raises(ValueError, match="query.senders must be a JSON list"):
        query_from_json({"senders": "A"})
    snapshot = {"kind": "result", "sender_ids": ["A"], "channels": [{"kind": "rest"}],
                "coverage": [1], "values": [[None]], "errors": [[None]]}
    with pytest.raises(ValueError, match="use 0 for a sampled zero"):
        result_from_json(snapshot)
    snapshot["values"] = [[.25, .5]]
    with pytest.raises(ValueError, match=r"result.values\[0\].*1 columns"):
        result_from_json(snapshot)
    snapshot["values"] = [[float("inf")]]
    with pytest.raises(ValueError, match=r"result.values\[0\]\[0\].*finite nonnegative"):
        result_from_json(snapshot)


def test_every_grasshopper_python_callable_has_an_explanatory_docstring():
    root = Path(__file__).resolve().parents[1] / "src" / "raystrack" / "integrations" / "grasshopper"
    missing = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    missing.append(f"{path.name}:{node.lineno} {node.name}")
    assert not missing, "Missing callable documentation: " + ", ".join(missing)

