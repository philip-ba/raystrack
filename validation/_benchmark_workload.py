"""Isolated equal-workload worker for benchmark_v2_vs_v1.py.

The v1 branch runs exclusively against the exported baseline tree. It is a
benchmark fixture, not a production compatibility API.
"""
import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np


def plane(subdivisions,down=False):
    axis=np.linspace(-1,1,subdivisions+1,dtype=np.float32)
    x,y=np.meshgrid(axis,axis)
    vertices=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size,np.float32))).astype(np.float32)
    faces=[]; width=subdivisions+1
    for row in range(subdivisions):
        for col in range(subdivisions):
            a=row*width+col
            faces.extend(((a,a+1,a+width+1),(a,a+width+1,a+width)))
    faces=np.asarray(faces,np.int32)
    return vertices,faces[:,::-1].copy() if down else faces


def transform(x=0,z=0):
    t=np.eye(4); t[0,3]=x; t[2,3]=z
    return t


def timed(call):
    started=time.perf_counter(); value=call()
    return value,(time.perf_counter()-started)*1000


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--api',choices=('v1','v2'),required=True)
    p.add_argument('--case',choices=('small_cpu','large_gpu','moving_instances'),required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--subdivisions',type=int,default=64)
    p.add_argument('--repeats',type=int,default=15)
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--json',type=Path,required=True)
    args=p.parse_args()
    device='cpu' if args.case == 'small_cpu' else args.device
    sub=1 if args.case == 'small_cpu' else args.subdivisions
    instanced=args.case == 'moving_instances'
    acceleration='instanced' if instanced else 'flat'
    up,down=plane(sub),plane(sub,True)
    instances=[('emitter',0,transform()),('receiver',1,transform(z=1 if args.case == 'small_cpu' else 2))]
    if args.case != 'small_cpu':
        instances.append(('blocker',0,transform(z=1)))
    if instanced:
        instances.extend((f'far{i}',0,transform(x=4+3*i,z=1)) for i in range(8))
    budget=512 if args.case == 'small_cpu' else 262144 if args.case == 'large_gpu' else 4096
    density=256 if args.case == 'large_gpu' else 4
    rays=256 if args.case == 'large_gpu' else 32 if args.case == 'small_cpu' else 64
    batch=65536 if args.case == 'large_gpu' else 1024
    max_iters=16
    if args.api == 'v1':
        from raystrack import PreparedSolver,PreviewSession,MatrixParams,SkyParams
        import raystrack.main
        raystrack.main._log=lambda message:None
        meshes=[('up',*up),('down',*down)]
        if instanced:
            prepared=PreparedSolver.from_instances(meshes,instances)
        else:
            prepared=PreparedSolver([(sid,(up if key == 0 else down)[0]@t[:3,:3].T+t[:3,3],
                                      (up if key == 0 else down)[1]) for sid,key,t in instances])
        shared=dict(samples=density,rays=rays,seed=13,device=device,bvh='builtin',
                    min_iters=max_iters,max_iters=max_iters,tol=0,auto_tune=False,
                    emitter_names=['emitter'],ray_batch_size=batch)
        session=PreviewSession(prepared,MatrixParams(**shared,reciprocity=False),SkyParams(**shared))
        def refresh():
            prepared.get_instanced_scene() if instanced else prepared.get_scene(use_bvh=True)
            prepared.get_emitters(samples=density,rays=rays,flip_faces=False)
            prepared.get_mesh_bounds()
        def solve():
            return session.solve(max_total_rays=budget,max_time_ms=None)
        def move(x):
            session.update_transform('blocker',transform(x=x,z=1 if instanced else 0))
        def vector(result):
            row=result.scene.get('emitter',{})
            return [row.get('receiver_front',0),row.get('receiver_back',0),
                    row.get('blocker_front',0),row.get('blocker_back',0),
                    result.sky.get('emitter',{}).get('Sky',0),result.rest.get('emitter',{}).get('Rest',0)]
        def execution(result):
            return getattr(prepared,'_last_execution_plan',{})
        def close():
            session.close()
    else:
        from raystrack import Mesh,Scene,Solver,Query,SolveOptions,Sampling,Accuracy,Budget,Channel
        prototypes={'up':Mesh(*up),'down':Mesh(*down)}
        if instanced:
            scene=Scene.from_instances(prototypes,[(sid,'up' if key == 0 else 'down',t) for sid,key,t in instances])
        else:
            scene=Scene.from_meshes({sid:Mesh((up if key == 0 else down)[0]@t[:3,:3].T+t[:3,3],
                                           (up if key == 0 else down)[1]) for sid,key,t in instances})
        solver=Solver(scene,device=device,acceleration=acceleration,bvh='builtin',auto_tune=False)
        opts=SolveOptions(Sampling(density=density,rays_per_cell=rays,seed=13),
                          Accuracy(max_replicates=max_iters,min_replicates=max_iters,tolerance=0),batch_size=batch)
        query=Query.row('emitter',sky='merged')
        def refresh():
            prepared=solver._sync_scene()
            prepared.get_instanced_scene() if instanced else prepared.get_scene(use_bvh=True)
            prepared.get_emitters(samples=density,rays=rays,flip_faces=False)
            prepared.get_mesh_bounds()
        def solve():
            return solver.solve(query,opts,Budget(rays=budget))
        def move(x):
            scene.update_transform('blocker',transform(x=x,z=1 if instanced else 0))
        def vector(result):
            return [result.value('emitter',Channel('surface',sid,side)) if sid in scene.surface_ids else 0
                    for sid,side in (('receiver','front'),('receiver','back'),('blocker','front'),('blocker','back'))]+[
                    result.value('emitter',Channel('sky')),result.value('emitter',Channel('rest'))]
        def execution(result):
            return dict(result.execution)
        def close():
            solver.close()
    try:
        _,prepare_ms=timed(refresh)
        cold,cold_ms=timed(solve)
        cold_transform_ms=cold_refresh_ms=0.
        if instanced:
            _,cold_transform_ms=timed(lambda:move(.125))
            _,cold_refresh_ms=timed(refresh)
        for i in range(args.warmups):
            if instanced:
                move(float(i)/max(1,args.warmups-1)*3)
                refresh()
            solve()
        measurements=[]
        for i in range(args.repeats):
            update_ms=refresh_ms=0.
            x=float(i)/(max(1,args.repeats-1))*3
            if instanced:
                _,update_ms=timed(lambda:move(x))
                _,refresh_ms=timed(refresh)
            result,solve_ms=timed(solve)
            assert result.rays_used == budget,(result.rays_used,budget)
            measurements.append({'iteration':i,'x':x if instanced else None,
                'update_ms':update_ms,'lazy_prepare_ms':refresh_ms,'solve_ms':solve_ms,
                'frame_ms':update_ms+refresh_ms+solve_ms,'rays':result.rays_used,'values':vector(result)})
        report={'api':args.api,'case':args.case,'device':device,'acceleration':acceleration,
            'triangles':len(instances)*2*sub*sub,'unique_geometry_triangles':4*sub*sub if instanced else len(instances)*2*sub*sub,
            'budget':budget,'density':density,'rays_per_cell':rays,'batch_size':batch,
            'cold_prepare_ms':prepare_ms,'cold_solve_ms':cold_ms,
            'cold_first_transform_ms':cold_transform_ms,'cold_first_transform_refresh_ms':cold_refresh_ms,
            'untimed_warmups':args.warmups,'execution':execution(result),
            'median_solve_ms':float(np.median([r['solve_ms'] for r in measurements])),
            'median_frame_ms':float(np.median([r['frame_ms'] for r in measurements])),
            'median_update_and_prepare_ms':float(np.median([r['update_ms']+r['lazy_prepare_ms'] for r in measurements])),
            'measurements':measurements,'python':platform.python_version(),'platform':platform.platform(),
            'versions':{name:importlib.metadata.version(name) for name in ('numpy','numba')}}
        args.json.write_text(json.dumps(report,indent=2,default=dict)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('measurements','execution')},indent=2,default=dict))
    finally:
        close()


if __name__ == '__main__':
    main()
