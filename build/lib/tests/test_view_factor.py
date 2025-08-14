import numpy as np
from raystrack.main import view_factor_matrix, view_factor


# original emitter
V_em = np.array([
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
], np.float32)
F_em = np.array([[0,1,2], [0,2,3]], np.int32)
sender = ("emitter", V_em, F_em)

# first receiver at z=1
V_rc1 = np.array([
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
], np.float32)
F_rc1 = np.array([[0,1,2], [0,2,3]], np.int32)
receiver1 = ("receiver1", V_rc1, F_rc1)

# second receiver at z=2 (shifted further away)
V_rc2 = np.array([
    (0.0, 0.0, 2.0),
    (5.0, 0.0, 2.0),
    (5.0, 1.0, 2.0),
    (0.0, 1.0, 2.0),
], np.float32)
F_rc2 = np.array([[0,1,2], [0,2,3]], np.int32)
receiver2 = ("receiver2", V_rc2, F_rc2)


def test_view_factor_two_receivers():
    params = dict(samples=64, rays=64, seed=42, use_bvh=False)

    # Build the full matrix with emitter + 2 receivers
    meshes = [sender, receiver1, receiver2]
    mat = view_factor_matrix(meshes, **params)

    # Compute per-sender→per-receiver direct VF
    vf = view_factor(sender, [receiver1, receiver2], **params)

    # Extract from the matrix the total emitter→receiver1 and emitter→receiver2
    vf1_from_mat = mat["emitter"].get("receiver1_front", 0.0) + mat["emitter"].get("receiver1_back", 0.0)
    vf2_from_mat = mat["emitter"].get("receiver2_front", 0.0) + mat["emitter"].get("receiver2_back", 0.0)

    # Extract from direct view_factor call
    vf1 = vf["emitter"]["receiver1_back"]
    vf2 = vf["emitter"]["receiver2_back"]

    print("Receiver1 VF — mat:", vf1_from_mat, "direct:", vf1)
    print("Receiver2 VF — mat:", vf2_from_mat, "direct:", vf2)

    # Assert they agree to within tolerance
    assert abs(vf1_from_mat - vf1) < 1e-2
    assert abs(vf2_from_mat - vf2) < 1e-2

test_view_factor_two_receivers()
