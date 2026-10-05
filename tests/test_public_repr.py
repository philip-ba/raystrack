"""Readable object summaries for Python scripting and Grasshopper inspection."""
from __future__ import annotations

import inspect

import numpy as np

from raystrack import (Accuracy, Budget, Channel, Mesh, Postprocessing, Query,
                       Result, Run, Sampling, Scene, SolveOptions, Solver,
                       SparseValues, Surface)


def test_large_geometry_and_results_have_bounded_informative_summaries():
    mesh = Mesh(np.zeros((10_000, 3), np.float32), np.zeros((2_000, 3), np.int32))
    transform = np.eye(4)
    transform[0, 3] = 1.25
    surface = Surface("A", mesh, "Wall", transform)
    scene = Scene(Surface(f"surface-{index}", mesh) for index in range(100))
    data = SparseValues(np.arange(100), np.zeros(100, np.int64),
                        np.full(100, .25), np.full(100, np.nan))
    result = Result(scene.surface_ids, (Channel("rest"),), data,
                    np.ones(100, np.int8), cumulative_rays=12_345, status="sampling")
    with np.printoptions(threshold=np.inf):
        summaries = [repr(item) for item in (mesh, surface, scene, data, result)]
    assert all(len(summary) < 400 for summary in summaries)
    assert all("array(" not in summary and "object at 0x" not in summary for summary in summaries)
    assert "vertices=10000" in summaries[0] and "triangles=2000" in summaries[0]
    assert "label='Wall'" in summaries[1] and "translation=(1.25, 0.0, 0.0)" in summaries[1]
    assert "surfaces=100" in summaries[2] and "meshes=1" in summaries[2]
    assert "entries=100" in summaries[3] and "known_errors=0" in summaries[3]
    assert "sampled=100" in summaries[4] and "cumulative_rays=12345" in summaries[4]


def test_query_selection_summary_stays_bounded_without_changing_ids():
    ids = tuple(f"surface-{index:06d}-" + "x" * 100 for index in range(10_000))
    query = Query.matrix(ids, ids, sky="tregenza145")
    summary = repr(query)
    assert len(summary) < 650
    assert "tregenza145" in summary and "front" in summary and "back" in summary
    assert query.senders == ids and query.receivers == ids
    assert "senders=all" in repr(Query.matrix())


def test_solver_and_run_summaries_report_progress_and_closure():
    mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    scene = Scene.from_meshes({"A": mesh})
    solver = Solver(scene, device="cpu", auto_tune=False)
    run = solver.start(Query.sky(("A",)), SolveOptions(Sampling(density=1, rays_per_cell=1),
                                                     Accuracy(max_replicates=2, tolerance=0, min_replicates=1)))
    assert "surfaces=1" in repr(solver) and "device='cpu'" in repr(solver)
    assert "cumulative_rays=0" in repr(run)
    run.advance(Budget(rays=13))
    assert "cumulative_rays=13" in repr(run)
    assert len(repr(run)) < 400
    solver.close()
    assert "closed=True" in repr(solver)


def test_scalar_option_summaries_and_public_objects_are_documented():
    objects = (Sampling(), Accuracy(), Postprocessing(), SolveOptions(), Budget(rays=100, time_ms=50))
    assert all("array(" not in repr(item) and len(repr(item)) < 700 for item in objects)
    assert "seed=1" in repr(objects[0]) and "tolerance=" in repr(objects[1])
    assert "reciprocity='none'" in repr(objects[2]) and "batch_size=" in repr(objects[3])
    assert "rays=100" in repr(objects[4]) and "time_ms=50" in repr(objects[4])
    for cls in (Mesh, Surface, Scene, Query, Sampling, Accuracy, Postprocessing,
                SolveOptions, Budget, Channel, SparseValues, Result, Solver, Run):
        assert inspect.getdoc(cls), cls.__name__
        for name, attribute in vars(cls).items():
            if name.startswith("_"):
                continue
            if isinstance(attribute, (classmethod, staticmethod)):
                attribute = attribute.__func__
            if callable(attribute) or isinstance(attribute, property):
                assert inspect.getdoc(attribute), f"{cls.__name__}.{name}"
