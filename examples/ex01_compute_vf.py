"""Compute all sender/receiver channels and store an immutable v2 snapshot."""
from _support import canyon, options, save_snapshot
from raystrack import Solver, Query


def main():
    scene=canyon()
    with Solver(scene,device="cpu") as solver:
        result=solver.solve(Query.matrix(),options())
    for sender in result.sender_ids:
        hits={str(channel):value for channel,value in result.row(sender).items() if value}
        print(sender,hits)
    print("Status:",result.status,"rays:",result.rays_used)
    print("Saved:",save_snapshot("vf_matrix",scene,result))


if __name__ == "__main__":
    main()
