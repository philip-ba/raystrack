"""Compare escape above a finite ground plane with directional sky tracing."""
import numpy as np
from _support import canyon, options
from raystrack import Mesh, Scene, Surface, Solver, Query, Channel


def main():
    scene=canyon()
    vertices=np.concatenate([s.world_vertices() for s in scene.surfaces])
    lo,hi=vertices.min(axis=0),vertices.max(axis=0)
    x0,y0=lo[:2]-100; x1,y1=hi[:2]+100; z=lo[2]-.001
    ground=Mesh([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],[[0,1,2],[0,2,3]])
    grounded=Scene((*scene.surfaces,Surface("ground",ground)))
    query=Query.matrix(senders=scene.surface_ids)
    with Solver(grounded,device="cpu") as solver:
        matrix=solver.solve(query,options(seed=20))
    with Solver(scene,device="cpu") as solver:
        sky=solver.solve(Query.sky(),options(seed=20))
    print("sender / escape above finite ground / upward sky / difference")
    for sender in scene.surface_ids:
        escape=matrix.value(sender,Channel("rest"))
        upward=sky.value(sender,Channel("sky"))
        print(sender,f"{escape:.6f}",f"{upward:.6f}",f"{upward-escape:+.6f}")
    print("A finite ground plane can miss shallow downward rays; this is an approximate comparison.")


if __name__ == "__main__":
    main()
