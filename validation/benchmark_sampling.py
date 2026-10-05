"""Compare ordinary cosine rays with targeted sampling for a rare receiver.

Run with ``PYTHONPATH=src python validation/benchmark_sampling.py``.
The reference uses the point-receiver limit for a 1 cm square at 1 m height.
"""

from __future__ import annotations

import time

import numpy as np

from raystrack import Mesh, Scene, Solver, Query, SolveOptions, Sampling, Accuracy, Postprocessing, Budget, Channel

def square(name: str, z: float, half_width: float, downward: bool = False):
    h = half_width
    vertices = np.array([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]],
                        dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    if downward:
        faces = faces[:, ::-1].copy()
    return name, vertices, faces


def main():
    scene=Scene.from_meshes({sid:Mesh(v,f) for sid,v,f in [
        square("emitter",0,1),square("tiny",1,.005,downward=True)]})
    grid=(np.arange(500)+.5)/250-1
    x,y=np.meshgrid(grid,grid)
    reference=.01**2/np.pi*np.mean(1/(1+x*x+y*y)**2)
    def opts(method,seed):
        sampling=(Sampling(density=4,rays_per_cell=32,seed=seed) if method == "cosine"
                  else Sampling(strategy="area_pair",pair_samples=512,seed=seed,
                                sequence="random" if method == "area_pair_random" else "shifted_halton"))
        return SolveOptions(sampling,Accuracy(max_replicates=1,min_replicates=1,tolerance=0))
    with Solver(scene,device="cpu",bvh="off",auto_tune=False) as solver:
        for method in ("cosine","area_pair_random","area_pair_halton"):
            solver.solve(Query.pair("emitter","tiny"),opts(method,0),Budget(rays=512))
            estimates=[]; started=time.perf_counter()
            for seed in range(20):
                result=solver.solve(Query.pair("emitter","tiny"),opts(method,seed),Budget(rays=512))
                assert result.rays_used == 512
                estimates.append(result.value("emitter",Channel("surface","tiny","front")))
            error=np.mean(np.abs(np.asarray(estimates)-reference))
            print(method,"rays/seed=512","mean_abs_error",float(error),
                  "zero_estimates",sum(v == 0 for v in estimates),"elapsed_s",time.perf_counter()-started)
    print("Point-receiver reference",float(reference))
    facing=Scene.from_meshes({sid:Mesh(v,f) for sid,v,f in [
        square("lower",0,1),square("upper",1,1,downward=True)]})
    delta=(np.arange(1000)+.5)*4/1000-2
    dx,dy=np.meshgrid(delta,delta)
    reference=np.sum((2-np.abs(dx))*(2-np.abs(dy))/(1+dx*dx+dy*dy)**2)*(4/1000)**2/(4*np.pi)
    with Solver(facing,device="cpu",bvh="off",auto_tune=False) as solver:
        for method in ("one_direction","bidirectional"):
            estimates=[]
            for seed in range(50):
                query=Query.row("lower") if method == "one_direction" else Query.matrix()
                options=SolveOptions(Sampling(density=4,rays_per_cell=64 if method == "one_direction" else 32,seed=seed),
                    Accuracy(max_replicates=1,min_replicates=1,tolerance=0),
                    Postprocessing("none" if method == "one_direction" else "bidirectional"))
                result=solver.solve(query,options)
                assert result.rays_used == 1024
                estimates.append(result.value("lower",Channel("surface","upper","front")))
            print(method,"rays/pair=1024","RMSE",float(np.sqrt(np.mean((np.asarray(estimates)-reference)**2))))


if __name__ == "__main__":
    main()
