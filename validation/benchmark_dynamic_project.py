"""One isolated execution with selected reads and analytic open-sensor checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from raystrack.project import file_sha256, load_project
from raystrack.project_run import run_project
from raystrack.project_store import ResultReader
from validation.annual_urban import process_peak_bytes


def benchmark(project_path, output, backend, *, probes=False):
    report = run_project(project_path, output, backend=backend, fresh_probes=probes)
    project = load_project(project_path)
    reader = ResultReader(output)
    count = report["intervals"]
    open_ids = [key for key in project.sensor_ids if key.endswith("/open")]
    max_sun_error, max_sky_error = 0., 0.
    for start in range(0, count, 168):
        stop = min(count, start+168)
        selected = reader.read(start, stop, sensor_ids=open_ids, components=["direct", "sky"], bands=["neutral_R"])
        expected_sun = project.dni[start:stop]*np.maximum(0,project.timeline.directions[start:stop,2])
        max_sun_error = max(max_sun_error,float(np.max(abs(selected["values"][:,:,0,0]-expected_sun[:,None]))))
        max_sky_error = max(max_sky_error,float(np.max(abs(selected["values"][:,:,1,0]-project.dhi[start:stop,None]))))
    if max_sun_error > .002 or max_sky_error > .002:
        raise AssertionError(f"elevated reference mismatch sun={max_sun_error}, sky={max_sky_error}")
    window_start = min(4100,max(0,count//2))
    tick = time.perf_counter()
    window = reader.read(window_start,min(count,window_start+4),sensor_ids=[project.sensor_ids[0]],bands=["neutral_R"])
    read_s = time.perf_counter()-tick
    tick = time.perf_counter()
    reader.export_csv(Path(output)/"selected.csv",window_start,min(count,window_start+4),
                      sensor_ids=[project.sensor_ids[0]],bands=["neutral_R"])
    export_s = time.perf_counter()-tick
    report.update(process_peak_resident_bytes=process_peak_bytes(),
                  memory_measurement="OS process-lifetime peak resident working set, includes native allocations",
                  selected_read_s=read_s, selected_array_bytes=window["values"].nbytes,
                  selected_export_s=export_s,
                  bytes_on_disk=sum(path.stat().st_size for path in Path(output).iterdir() if path.is_file()),
                  native_library_sha256=file_sha256(os.environ["RAYSTRACK_NATIVE_LIBRARY"]),
                  analytic_open_sun_max_abs_w_m2=max_sun_error,
                  analytic_open_sky_max_abs_w_m2=max_sky_error,
                  timing_scope="one fresh process; includes completed block writes; fresh probes optional",
                  cold_block_s=report["timings"][0]["solve_s"],
                  warm_block_s=[entry["solve_s"] for entry in report["timings"][1:]],
                  edit_s=sum(entry["edit_s"] for entry in report["timings"]),
                  solve_s=sum(entry["solve_s"] for entry in report["timings"]),
                  write_s=sum(entry["write_s"] for entry in report["timings"]))
    (Path(output)/"benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--output",required=True)
    parser.add_argument("--backend",choices=("embree","vulkan-rt","native-builtin"),default="embree")
    parser.add_argument("--probes",action="store_true")
    args = parser.parse_args()
    report = benchmark(args.project,args.output,args.backend,probes=args.probes)
    print(json.dumps({key:report[key] for key in ("intervals","sensors","instances","triangles","solve_s","write_s",
        "process_peak_resident_bytes","analytic_open_sun_max_abs_w_m2","analytic_open_sky_max_abs_w_m2")}))


if __name__ == "__main__":
    main()
