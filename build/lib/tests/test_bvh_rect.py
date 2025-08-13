import time
import numpy as np
from raystrack.main import view_factor_matrix

# simple rectangle setup from test_diff_rect

e = 1e-7
V_em = np.array([
    (-e/2, -e/2, 0.0),
    ( e/2, -e/2, 0.0),
    ( e/2,  e/2, 0.0),
    (-e/2,  e/2, 0.0),
], np.float32)
F_em = np.array([[0,1,2], [0,2,3]], np.int32)

V_rc = np.array([
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
], np.float32)
F_rc = np.array([[0,1,2], [0,2,3]], np.int32)

meshes = [
    ("emitter", V_em, F_em),
    ("rectangle", V_rc, F_rc),
]


def compute(use_bvh: bool):
    t0 = time.time()
    res = view_factor_matrix(meshes, samples=64, rays=64, use_bvh=use_bvh)
    return res, time.time() - t0


if __name__ == "__main__":
    brute, t_brute = compute(False)
    bvh, t_bvh = compute(True)
    print("brute", t_brute, "bvh", t_bvh)
    for surf in brute:
        print(brute[surf].items())
        print(bvh[surf].items())
        print("-------------")
        for k, v in brute[surf].items():
            assert abs(v - bvh[surf].get(k, 0.0)) < 1e-2
