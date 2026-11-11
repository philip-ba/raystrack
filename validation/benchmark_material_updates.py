"""Isolated optical edit costs on equal district geometry and controlled forcing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from raystrack.project import file_sha256, load_project
from raystrack.shortwave import ShortwaveSolver
from validation.annual_urban import process_peak_bytes


def run(project_path, backend):
    project = load_project(project_path)
    options = dict(dni=700, dhi=100, samples=256, replicates=2, bounces=3, seed=17)
    args = (project.points, project.normals, [[0,0,1]])
    records = []
    with ShortwaveSolver(project.scene, materials=project.materials, backend=backend) as solver:
        tick = time.perf_counter()
        original = solver.solve(*args, **options)
        cold_s = time.perf_counter()-tick
        snapshot = original.irradiance.copy()
        for material_id in ("sheet_open", "sheet", "sheet_open", "sheet", "sheet_open"):
            geometry = solver._native.instance_statistics
            revision = project.scene.revision
            material = project.material_library[material_id]
            tick = time.perf_counter()
            solver.set_material("t0/sheet", material)
            edit_s = time.perf_counter()-tick
            tick = time.perf_counter()
            result = solver.solve(*args, **options)
            solve_s = time.perf_counter()-tick
            assert project.scene.revision == revision
            assert solver._native.instance_statistics == geometry
            assigned = {**project.materials, "t0/sheet": material}
            tick = time.perf_counter()
            with ShortwaveSolver(project.scene, materials=assigned, backend=backend) as fresh:
                rebuilt = fresh.solve(*args, **options)
            rebuild_s = time.perf_counter()-tick
            np.testing.assert_array_equal(result.irradiance, rebuilt.irradiance)
            np.testing.assert_array_equal(original.irradiance, snapshot)
            if material_id == "sheet_open":
                assert np.max(abs(result.irradiance-original.irradiance)) > 1
            records.append(dict(material=material_id, edit_s=edit_s, solve_s=solve_s,
                                fresh_build_and_solve_s=rebuild_s,
                                geometry_revision=revision, material_revision=result.material_revision,
                                acceleration_unchanged=True, fresh_equal=True))
        mode = solver._native.geometry_mode
    return dict(backend=backend, geometry_mode=mode, instances=len(project.scene),
                triangles=sum(len(surface.mesh.faces) for surface in project.scene.surfaces),
                sensors=len(project.sensor_ids), timesteps=1, options=options,
                forcing="controlled vertical sun700W/m2 and isotropic DHI100W/m2",
                cold_s=cold_s, edits=records, peak_host_bytes=process_peak_bytes(),
                native_library_sha256=file_sha256(os.environ["RAYSTRACK_NATIVE_LIBRARY"]),
                source_hashes=project.source_hashes,
                scope="material edits reuse geometry; no persistent transfer cache; not independent material accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",default="examples/dynamic_district/project.json")
    parser.add_argument("--backend",default="embree",choices=("embree","vulkan-rt"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report = run(args.project,args.backend)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("backend","geometry_mode","cold_s","peak_host_bytes")}))


if __name__ == "__main__":
    main()
