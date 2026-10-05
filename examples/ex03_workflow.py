"""Trace scene and Tregenza sky together with one shared sampling stream."""
from _support import canyon, options, save_snapshot
from raystrack import Solver, Query, Channel
from raystrack.io import load


def main():
    scene=canyon()
    with Solver(scene,device="cpu") as solver:
        result=solver.solve(Query.matrix(sky="tregenza145"),options())
    for sender in result.sender_ids:
        sky=sum(value for channel,value in result.row(sender).items() if channel.kind == "sky")
        print(sender,"sky",round(sky,6),"rest",result.value(sender,Channel("rest")))
    path=save_snapshot("combined_workflow",scene,result)
    stored=load(path)
    print("Saved:",path,"revision:",stored.scene.revision,"rays:",stored.result.cumulative_rays)


if __name__ == "__main__":
    main()
