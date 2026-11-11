"""Compare equal numeric NPY/NPZ blocks; run when tracing agents are idle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from raystrack.project_store import ResultReader, ResultWriter
from validation.annual_urban import process_peak_bytes


def run_case(output, storage, *, steps=168, sensors=256, block_steps=24, seed=17):
    if min(steps, sensors, block_steps) < 1:
        raise ValueError('steps, sensors and block_steps must be positive')
    root = Path(output)/storage
    writer = ResultWriter(root, configuration=dict(seed=seed, steps=steps), sensor_ids=[f's{i}' for i in range(sensors)], storage=storage)
    rng = np.random.default_rng(seed)
    write_seconds = 0.
    values_bytes = 0
    for first in range(0, steps, block_steps):
        count = min(block_steps,steps-first)
        # Random values with exact dark intervals model both compressible nights
        # and finite-precision stochastic daylight components.
        values = rng.random((count,sensors,4,3))*500
        values[np.arange(first,first+count)%24 < 6] = 0
        errors = np.sqrt(values)*.01
        stamps = np.datetime64('2025-01-01T00:30','us') + np.arange(first,first+count)*np.timedelta64(1,'h')
        before = time.perf_counter()
        writer.append(stamps,np.ones(count),values,errors,0,0)
        write_seconds += time.perf_counter()-before
        values_bytes += values.nbytes+errors.nbytes
    writer.finish()
    reader = ResultReader(root)
    first, last = min(37, steps-1), min(43, steps)
    selected_sensors = ['s0',f's{min(7,sensors-1)}'] if sensors > 1 else ['s0']
    before = time.perf_counter()
    selected = reader.read(first,last,sensor_ids=selected_sensors,components=['sky'])
    verified_read_seconds = time.perf_counter()-before
    reader = ResultReader(root,verify=False)
    before = time.perf_counter()
    reader.read(first,last,sensor_ids=selected_sensors,components=['sky'])
    unchecked_read_seconds = time.perf_counter()-before
    return dict(storage=storage,intervals=steps,sensors=sensors,block_steps=block_steps,
                numeric_bytes=values_bytes,stored_bytes=sum(p.stat().st_size for p in root.iterdir() if p.is_file()),
                write_seconds=write_seconds,verified_selected_read_seconds=verified_read_seconds,
                unchecked_selected_read_seconds=unchecked_read_seconds,selected_array_bytes=selected['values'].nbytes,
                process_peak_bytes=process_peak_bytes(),seed=seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--storage',choices=['npy','npz'],required=True)
    parser.add_argument('--steps',type=int,default=168)
    parser.add_argument('--sensors',type=int,default=256)
    args=parser.parse_args()
    result=run_case(args.output,args.storage,steps=args.steps,sensors=args.sensors)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/(args.storage+'-report.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
