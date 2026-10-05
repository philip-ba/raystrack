"""Reuse one Solver while independent runs vary only the randomized seed."""
from dataclasses import replace
import numpy as np
from _support import canyon, options
from raystrack import Solver, Query, Accuracy


def main():
    scene=canyon()
    base=replace(options(),accuracy=Accuracy(max_replicates=5,min_replicates=5,tolerance=0))
    with Solver(scene,device="cpu") as solver:
        reference=None
        for seed in (1,2,3,4,5):
            opts=replace(base,sampling=replace(base.sampling,seed=seed))
            result=solver.solve(Query.matrix(),opts)
            if reference is None:
                reference=result.dense()
            print("seed",seed,"rays",result.rays_used,
                  "mean difference",float(np.mean(np.abs(result.dense()-reference))))
    print("Sampling settings stay fixed within a Run; changing seed starts a fresh Run.")


if __name__ == "__main__":
    main()
