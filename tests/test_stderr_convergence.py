import numpy as np
from raystrack.main import view_factor_matrix


def make_simple_pair():
    V_em = np.array([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ], np.float32)
    F_em = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    sender = ("emitter", V_em, F_em)

    V_rc = np.array([
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ], np.float32)
    F_rc = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    receiver = ("receiver", V_rc, F_rc)
    return [sender, receiver]


def test_cross_seed_within_stderr():
    meshes = make_simple_pair()
    params = dict(samples=16, rays=256, use_bvh=False)
    # Use stderr-based stopping with a modest tolerance
    tol = 3e-4

    res1, stats1 = view_factor_matrix(
        meshes,
        max_iters=1000,
        tol=tol,
        tol_mode="stderr",
        min_iters=20,
        **params,
        seed=1,
        return_stats=True,
    )
    res2, stats2 = view_factor_matrix(
        meshes,
        max_iters=1000,
        tol=tol,
        tol_mode="stderr",
        min_iters=20,
        **params,
        seed=12,
        return_stats=True,
    )

    v1 = res1["emitter"]["receiver_back"]
    v2 = res2["emitter"]["receiver_back"]
    s1 = stats1["emitter"]["receiver_back"]
    s2 = stats2["emitter"]["receiver_back"]
    diff = abs(v1 - v2)
    bound = 3.0 * max(s1, s2)
    print("Diff:", diff, "Bound:", bound, "Values:", v1, v2, "Stderr:", s1, s2)

    # Allow a small absolute fallback to avoid flakiness in very low-variance cases
    assert diff <= max(bound, 3e-4)

if __name__ == "__main__":
    test_cross_seed_within_stderr()

