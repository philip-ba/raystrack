"""Single-mesh enclosures used by CPU, CUDA and portable GPU regressions."""
from __future__ import annotations

import numpy as np

from raystrack import Accuracy, Budget, Channel, Mesh, Query, Sampling, Scene, SolveOptions, Solver, Surface


def box_mesh(*, inward=True, open_top=False):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    if open_top:
        faces = np.delete(faces, [2, 3], axis=0)
    if inward:
        faces = faces[:, ::-1].copy()
    return Mesh(vertices, faces)


def joined_plates():
    """Two facing unit squares, one mesh: analytical aggregate self VF 0.1998248957."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    return Mesh(vertices, [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]])


def fixed_options(*, flip=False, density=2, rays=16, replicas=2, batch=65536):
    return SolveOptions(Sampling(density=density, rays_per_cell=rays, seed=19, flip_faces=flip),
                        Accuracy(max_replicates=replicas, min_replicates=replicas, tolerance=0),
                        batch_size=batch)


def backend_summary(device):
    """Exact enclosure and convex controls, with CPU parity for an open enclosure."""
    report = []
    for acceleration, bvh in (("flat", "off"), ("flat", "builtin"), ("instanced", "builtin")):
        for inward, flip, side in ((True, False, "front"), (False, True, "back"), (False, False, None)):
            scene = Scene([Surface("box", box_mesh(inward=inward))])
            with Solver(scene, device=device, acceleration=acceleration, bvh=bvh, auto_tune=False) as solver:
                result = solver.solve(Query.row("box", sky="tregenza145"), fixed_options(flip=flip))
                front = result.value("box", Channel("surface", "box", "front"))
                back = result.value("box", Channel("surface", "box", "back"))
                sky = sum(result.value("box", Channel("sky", patch=i)) for i in range(145))
                escape = result.value("box", Channel("rest"))
                assert abs(sum(result.row("box").values()) - 1) < 1e-12
                if side:
                    assert result.value("box", Channel("surface", "box", side)) == 1, (device, acceleration, side, front, back, sky, escape)
                    assert sky == escape == 0
                else:
                    assert front == back == 0, (device, acceleration, front, back)
                    assert sky + escape == 1
                report.append(dict(acceleration=acceleration, bvh=bvh, inward=inward, flip=flip,
                                   self_front=front, self_back=back, sky=sky, escape=escape))
        scene = Scene([Surface("box", box_mesh(open_top=True))])
        options = fixed_options(rays=8)
        results = []
        for backend in ("cpu", device):
            with Solver(scene, device=backend, acceleration=acceleration, bvh=bvh, auto_tune=False) as solver:
                results.append(solver.solve(Query.row("box", sky="merged"), options, Budget()))
        np.testing.assert_array_equal(results[0].dense(), results[1].dense())
        assert results[1].value("box", Channel("surface", "box", "front")) > 0
    return report
