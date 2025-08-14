import time
from numba import cuda
import numpy as np

from raystrack import view_factor_matrix


def square(center, normal):
    cx,cy,cz=center; h=0.5
    if normal[2]:
        V=[(cx-h,cy-h,cz),(cx+h,cy-h,cz),(cx+h,cy+h,cz),(cx-h,cy+h,cz)]
        if normal[2]<0: V.reverse()
    elif normal[0]:
        V=[(cx,cy-h,cz-h),(cx,cy-h,cz+h),(cx,cy+h,cz+h),(cx,cy+h,cz-h)]
        if normal[0]>0: V.reverse()
    else:
        V=[(cx-h,cy,cz-h),(cx-h,cy,cz+h),(cx+h,cy,cz+h),(cx+h,cy,cz-h)]
        if normal[1]<0: V.reverse()
    return np.asarray(V,np.float32), np.asarray([[0,1,2],[0,2,3]],np.int32)

cube=[("bottom", *square((0,0,-0.5),(0,0, 1))),
          ("top",    *square((0,0, 0.5),(0,0,-1))),
          ("right",  *square((0.5,0,0),(-1,0,0))),
          ("left",   *square((-0.5,0,0),(1,0,0))),
          ("front",  *square((0,0.5,0),(0,-1,0))),
          ("back",   *square((0,-0.5,0),(0, 1,0)))]

t0=time.time()
VF=view_factor_matrix(cube, samples=256, rays=256)
print("\nCUDA" if cuda.is_available() else "\nCPU",
        "total elapsed", round(time.time()-t0,3),"s")
for src,row in VF.items():
    print(src,row)
