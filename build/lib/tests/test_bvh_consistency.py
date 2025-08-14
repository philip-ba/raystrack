import time
import numpy as np
from raystrack.main import view_factor_matrix


def square(center, normal):
    cx, cy, cz = center
    h = 0.5
    if normal[2]:
        V = [(cx-h, cy-h, cz), (cx+h, cy-h, cz), (cx+h, cy+h, cz), (cx-h, cy+h, cz)]
        if normal[2] < 0:
            V.reverse()
    elif normal[0]:
        V = [(cx, cy-h, cz-h), (cx, cy-h, cz+h), (cx, cy+h, cz+h), (cx, cy+h, cz-h)]
        if normal[0] > 0:
            V.reverse()
    else:
        V = [(cx-h, cy, cz-h), (cx-h, cy, cz+h), (cx+h, cy, cz+h), (cx+h, cy, cz-h)]
        if normal[1] < 0:
            V.reverse()
    return np.asarray(V, np.float32), np.asarray([[0,1,2],[0,2,3]], np.int32)

cube = [
    ("bottom", *square((0,0,-0.5),(0,0, 1))),
    ("top",    *square((0,0, 0.5),(0,0,-1))),
    ("right",  *square((0.5,0,0),(-1,0,0))),
    ("left",   *square((-0.5,0,0),(1,0,0))),
    ("front",  *square((0,0.5,0),(0,-1,0))),
    ("back",   *square((0,-0.5,0),(0, 1,0))),
]


def compute(use_bvh: bool):
    t0 = time.time()
    res = view_factor_matrix(cube, samples=64, rays=64, use_bvh=use_bvh)
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
            
