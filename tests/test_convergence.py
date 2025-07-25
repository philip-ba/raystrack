import numpy as np
from raystrack.main import view_factor_matrix, view_factor

V_em = np.array([
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
], np.float32)
F_em = np.array([[0,1,2],[0,2,3]], np.int32)
sender = ("emitter", V_em, F_em)

V_rc = np.array([
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
], np.float32)
F_rc = np.array([[0,1,2],[0,2,3]], np.int32)
receiver = ("receiver", V_rc, F_rc)

meshes = [sender, receiver]
params = dict(samples=16, rays=256, use_bvh=False)

def test_max_iters_tol_break():
    # With a large tolerance the iterative run stops after two iterations,
    # producing the same result as ``max_iters=2``.
    ref = view_factor_matrix(meshes, max_iters=2, tol=0, **params)
    multi = view_factor_matrix(meshes, max_iters=5, tol=1.0, **params)
    assert abs(ref["emitter"]["receiver_back"] - multi["emitter"]["receiver_back"]) < 1e-3




def test_convergence_decreasing_error():
    vf1 = view_factor_matrix(meshes, max_iters=50, tol=1e-5, **params, seed=1)["emitter"]["receiver_back"]
    print("vf1", vf1)
    vf2 = view_factor_matrix(meshes, max_iters=50, tol=1e-5, **params, seed=2)["emitter"]["receiver_back"]
    print("vf2", vf2)
    vf3 = view_factor_matrix(meshes, max_iters=50, tol=1e-5, **params, seed=201)["emitter"]["receiver_back"]
    print("vf3", vf3)
    print(vf3-vf2, vf2-vf1, vf3-vf1)
    assert abs(vf3 - vf2) < 0.05
if __name__ == "__main__":
    test_convergence_decreasing_error()