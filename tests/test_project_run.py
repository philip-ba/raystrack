"""File-driven state boundaries, replay and selected completed output."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from raystrack import Mesh, Scene
from raystrack.project import load_project
from raystrack.project_run import run_project
from raystrack.project_store import ResultReader
from raystrack.shortwave import ShortwaveSolver
from validation.create_district_project import create_project, pose


def test_atomic_transform_batch_validates_before_notifying_and_skips_noops():
    mesh = Mesh([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]])
    scene = Scene.from_instances({"m":mesh}, [(key,"m",np.eye(4)) for key in ("a","b")])
    notifications = []
    scene._subscribe(lambda: notifications.append(scene.revision))
    original = scene._snapshot()
    with pytest.raises(KeyError):
        scene.update_transforms({"a":pose(2),"missing":pose(3)})
    assert scene._snapshot() is original and not notifications
    with pytest.raises(ValueError):
        scene.update_transforms({"a":pose(2),"b":np.zeros((4,4))})
    assert scene._snapshot() is original and not notifications
    assert scene.update_transforms({"a":pose(),"b":pose()}) == 0
    assert not notifications
    assert scene.update_transforms({"a":pose(2),"b":pose(3)}) == 1
    assert notifications == [0]
    assert scene["a"].mesh is scene["b"].mesh is mesh
    assert scene._snapshot().instances["a"] is original.instances["a"]
    assert scene.update_transforms({"a":pose(2),"b":pose(3)}) == 1
    assert notifications == [0]


def test_checked_project_shared_people_original_solar_annual_midpoints():
    project = load_project("examples/dynamic_district/project.json")
    assert len(project.timeline.timestamps) == 8760
    assert project.timeline.timestamps[0] == np.datetime64("2024-12-31T23:30:00")
    assert project.scene["t0/person0"].mesh is project.scene["t0/person17"].mesh
    assert not np.array_equal(project.states[0]["transforms"]["t0/shade0"],
                              project.states[1]["transforms"]["t0/shade0"])
    assert project.source_hashes["assets/2011-SA1.XML"] == "5f398e08a81fecd57f9e566fa2a16f08bb0f2ca2a22fdba04d5d949c818b3336"


native = pytest.mark.skipif(not os.environ.get("RAYSTRACK_NATIVE_LIBRARY"), reason="optional native library required")


@native
def test_motion_changes_people_shade_shadows_and_old_results_immutable(tmp_path):
    project = load_project(create_project(tmp_path/"inputs",busy=True))
    points = [[1,5.4,.01],[3.9,7.7,.01]]
    with ShortwaveSolver(project.scene,materials=project.materials,backend="native-builtin") as solver:
        before = solver.solve(points,[[0,0,1]]*2,[[0,0,1]],dni=800,samples=0,bounces=0)
        saved = before.irradiance.copy()
        project.scene.update_transforms({"t0/person0":pose(1,6.8),"t0/shade2":pose(3.9,10,3.2)})
        after = solver.solve(points,[[0,0,1]]*2,[[0,0,1]],dni=800,samples=0,bounces=0)
    assert np.all(before.irradiance[0,:,0,0] == 0)
    assert np.all(after.irradiance[0,:,0,0] == 800)
    np.testing.assert_array_equal(before.irradiance,saved)
    assert not before.irradiance.flags.writeable


@native
def test_busy_resume_all_edits_selected_export_and_duration_integration(tmp_path):
    path = create_project(tmp_path/"inputs",busy=True)
    options = dict(backend="native-builtin",samples=8,replicates=2,block_steps=7,fresh_probes=False)
    partial = run_project(path,tmp_path/"resume",max_blocks=44,**options)
    assert partial["intervals"] == 44
    final = run_project(path,tmp_path/"resume",resume=True,**options)
    whole = run_project(path,tmp_path/"whole",**options)
    assert final["complete"] and whole["intervals"] == 96
    reader = ResultReader(tmp_path/"resume")
    actual,expected = reader.read(),ResultReader(tmp_path/"whole").read()
    np.testing.assert_array_equal(actual["values"],expected["values"])
    integrated = np.einsum("t,tscb->scb",actual["durations_hours"],actual["values"])
    np.testing.assert_allclose(reader.manifest["exposure_wh_m2"],integrated,atol=1e-9)
    selection = reader.read(start=49,stop=52,sensor_ids=["t0/sheet","t0/person"])
    np.testing.assert_array_equal(selection["values"],actual["values"][49:52][:,[1,4]])
    reader.export_csv(tmp_path/"selected.csv",start=49,stop=52,sensor_ids=["t0/person"])
    assert "t0/person" in (tmp_path/"selected.csv").read_text()


@native
def test_time_partition_invariance_exposure_only(tmp_path):
    path = create_project(tmp_path/"inputs",busy=True)
    raw = json.loads(path.read_text())
    raw["weather"]["steps"] = 8
    raw["weather"]["start_utc"] = "2025-06-21T10:00:00Z"
    raw["states"] = []
    raw.pop("state_tracks")
    path.write_text(json.dumps(raw))
    options = dict(backend="native-builtin",samples=16,replicates=2,fresh_probes=False)
    run_project(path,tmp_path/"a",block_steps=3,**options)
    run_project(path,tmp_path/"b",block_steps=8,**options)
    run_project(path,tmp_path/"c",block_steps=8,retention="exposure",**options)
    a,b = ResultReader(tmp_path/"a"),ResultReader(tmp_path/"b")
    np.testing.assert_allclose(a.read()["values"],b.read()["values"],rtol=1e-12,atol=1e-9)
    np.testing.assert_allclose(a.manifest["exposure_wh_m2"],b.manifest["exposure_wh_m2"],atol=1e-9)
    c = ResultReader(tmp_path/"c")
    np.testing.assert_allclose(a.manifest["exposure_wh_m2"],c.manifest["exposure_wh_m2"],atol=1e-9)
    with pytest.raises(ValueError):
        c.read(start=0,stop=1)


@native
def test_cli_partial_resume_inspect_and_selected_csv_round_trip(tmp_path):
    path=create_project(tmp_path/"inputs",busy=True,inline_states=True)
    raw=json.loads(path.read_text())
    raw["weather"]["steps"]=2
    raw["states"]=raw["states"][:2]
    raw["settings"].update(samples=4,bounces=1,block_steps=1)
    path.write_text(json.dumps(raw))
    repo=Path(__file__).resolve().parents[1]
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(repo/"src"),str(repo)))}
    base=[sys.executable,"-m","raystrack.project_run"]
    output=tmp_path/"results"
    command=base+["run",str(path),"--backend","native-builtin","--output",str(output)]
    first=subprocess.run(command+["--max-blocks","1"],env=env,capture_output=True,text=True,check=True,timeout=60)
    assert json.loads(first.stdout)["intervals"]==1
    final=subprocess.run(command+["--resume"],env=env,capture_output=True,text=True,check=True,timeout=60)
    assert json.loads(final.stdout)["complete"]
    inspect=subprocess.run(base+["inspect",str(output)],env=env,capture_output=True,text=True,check=True,timeout=60)
    assert json.loads(inspect.stdout)["completed_steps"]==2
    target=tmp_path/"selected.csv"
    subprocess.run(base+["export",str(output),"--start","0","--stop","1","--sensors","t0/person", "--output",str(target)],
                   env=env,capture_output=True,text=True,check=True,timeout=60)
    lines=target.read_text().splitlines()
    assert len(lines)==13 and all("t0/person" in line for line in lines[1:])
