"""Compare v2 with a fixed exported v1 tree without changing the workspace.

Runs separate subprocesses with identical geometry, seeds, additional budgets,
and explicit backends. Warmed timings include ray generation, tracing, reduction
and Result construction; cold compilation is reported separately. Moving scenes
include transform commits plus lazy emitter/scene preparation. Hardware timing
is local evidence, not a general speed guarantee.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np

BASELINE='1757748cea4f523324f6ba44235699390b9a826c'
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default=BASELINE)
    p.add_argument('--gpu-device',choices=('vulkan','cuda','metal'),default='vulkan')
    p.add_argument('--cases',nargs='+',choices=('small_cpu','large_gpu','moving_instances'),
                   default=['small_cpu','large_gpu','moving_instances'])
    p.add_argument('--subdivisions',type=int,default=64)
    p.add_argument('--repeats',type=int,default=15)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--json',type=Path,default=ROOT/'validation/results/v2_baseline_comparison.json')
    args=p.parse_args()
    if args.subdivisions < 1 or args.repeats < 1 or args.trials < 1 or args.warmups < 1:
        p.error('subdivisions and repeats must be positive')
    report={'baseline_sha':args.baseline,'workspace_api':'v2','notes':[
        'Baseline exported with git archive to a temporary directory; working tree is never checked out.',
        'Separate warmed subprocesses use the same geometry, seed 13, work budgets and explicit device.',
        'Unrestricted calibration disabled. Bounded solves use the same per-chunk ray cap.',
        'Three separate-process trials per API/case by default, alternating v1/v2 process order, with five untimed warmups and 15 timed repeats.',
        'Cold first-transform and lazy refresh measured separately; warmed motion runs before all timed moving frames.',
        'Aggregate medians summarize per-trial medians; raw trials retained. These are local timings, not confidence intervals.',
        'Equality checks estimates on fixed seeded workloads; this is not uncertainty coverage validation.'],
        'comparisons':[]}
    with tempfile.TemporaryDirectory(prefix='raystrack-v1-baseline-') as directory:
        temp=Path(directory); archive=temp/'baseline.zip'; exported=temp/'baseline'
        subprocess.run(['git','archive','--format=zip','-o',str(archive),args.baseline],cwd=ROOT,check=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(exported)
        worker=ROOT/'validation/_benchmark_workload.py'
        for case in args.cases:
            reports={'v1':[],'v2':[]}
            for trial in range(args.trials):
                order=(('v1',exported),('v2',ROOT)) if trial%2 == 0 else (('v2',ROOT),('v1',exported))
                for api,tree in order:
                    output=temp/f'{case}-{api}-{trial}.json'
                    env=os.environ.copy(); env['PYTHONPATH']=str(tree/'src'); env['NUMBA_NUM_THREADS']='4'
                    subprocess.run([sys.executable,str(worker),'--api',api,'--case',case,'--device',args.gpu_device,
                        '--subdivisions',str(args.subdivisions),'--repeats',str(args.repeats),
                        '--warmups',str(args.warmups),'--json',str(output)],cwd=tree,env=env,check=True)
                    reports[api].append(json.loads(output.read_text(encoding='utf-8')))
            def aggregate(api):
                first=dict(reports[api][0])
                for key in ('median_solve_ms','median_frame_ms','median_update_and_prepare_ms'):
                    first[key]=float(np.median([r[key] for r in reports[api]]))
                first['trials']=reports[api]
                return first
            before,after=aggregate('v1'),aggregate('v2')
            pairs=[(a,b) for old,new in zip(reports['v1'],reports['v2'])
                   for a,b in zip(old['measurements'],new['measurements'])]
            error=max(float(np.max(np.abs(np.asarray(a['values'])-np.asarray(b['values'])))) for a,b in pairs)
            same_rays=all(a['rays'] == b['rays'] for a,b in pairs)
            comparison={'case':case,'baseline':before,'v2':after,'max_abs_estimate_difference':error,
                        'equal_ray_counts':same_rays,'numerically_equal':bool(error <= 1e-6 and same_rays),
                        'v2_vs_v1_median_solve_ratio':after['median_solve_ms']/before['median_solve_ms'],
                        'v2_vs_v1_median_frame_ratio':after['median_frame_ms']/before['median_frame_ms']}
            report['comparisons'].append(comparison)
            print(case,'max difference',error,'frame ratio',comparison['v2_vs_v1_median_frame_ratio'],flush=True)
    report['all_numerically_equal']=all(row['numerically_equal'] for row in report['comparisons'])
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not report['all_numerically_equal']:
        raise SystemExit('Baseline workload differs')


if __name__ == '__main__':
    main()
