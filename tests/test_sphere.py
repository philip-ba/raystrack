import math, time
import numpy as np
from raystrack.main import view_factor_matrix

def gen_sphere_patches(lat_n: int, lon_n: int, radius: float = 1.0):
    """
    Return a list of (name, V, F) for each triangular patch on a sphere of
    given radius, subdivided in latitude (lat_n) and longitude (lon_n).
    Normals are flipped so they point inward (toward the center).
    """
    meshes = []
    idx = 0
    for i in range(lat_n):
        phi0 = math.pi *  i   / lat_n
        phi1 = math.pi * (i+1)/ lat_n
        for j in range(lon_n):
            th0 = 2*math.pi *  j    / lon_n
            th1 = 2*math.pi * (j+1) / lon_n

            # four corner points of this quad (on the sphere surface)
            p00 = np.array([
                math.sin(phi0)*math.cos(th0),
                math.sin(phi0)*math.sin(th0),
                math.cos(phi0)
            ])*radius
            p01 = np.array([
                math.sin(phi0)*math.cos(th1),
                math.sin(phi0)*math.sin(th1),
                math.cos(phi0)
            ])*radius
            p10 = np.array([
                math.sin(phi1)*math.cos(th0),
                math.sin(phi1)*math.sin(th0),
                math.cos(phi1)
            ])*radius
            p11 = np.array([
                math.sin(phi1)*math.cos(th1),
                math.sin(phi1)*math.sin(th1),
                math.cos(phi1)
            ])*radius

            # split quad into two triangles
            for tri in ((p00,p10,p01), (p11,p01,p10)):
                V = np.vstack(tri).astype(np.float32)   # (3,3)
                # ensure winding so normal = cross(v1-v0, v2-v0) points inward
                n = np.cross(V[1]-V[0], V[2]-V[0])
                # inward direction is toward -V[0] for a sphere at origin
                if np.dot(n, -V[0]) < 0:
                    # flip the last two vertices
                    V = V[[0,2,1]]
                F = np.array([[0,1,2]], np.int32)
                meshes.append((f"patch_{idx}", V, F))
                idx += 1

    return meshes

if __name__ == "__main__":
    # build ~400 patches
    LAT = 10
    LON = 20
    meshes = gen_sphere_patches(LAT, LON)

    print(f"Generated {len(meshes)} triangular patches on the sphere.")

    # Monte-Carlo parameters
    SAMPLES = 128   # grid per side
    RAYS    = 128  # rays per cell
    total_rays = SAMPLES*SAMPLES*RAYS

    print(f"Tracing from patch_0 with {SAMPLES**2*RAYS:,d} rays…")
    t0 = time.time()
    VF = view_factor_matrix(meshes, samples=SAMPLES, rays=RAYS, use_bvh=False)
    elapsed = time.time() - t0

    # collect all front+back view-factors from patch_1
    row = VF["patch_1"]
    sum_vf = sum(row.values())

    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Sum of view-factors from patch_1 to all others: {sum_vf:.6f}")