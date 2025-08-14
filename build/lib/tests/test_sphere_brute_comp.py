import math, time, numpy as np
from raystrack.main import (
    view_factor_matrix,        # BVH-accelerated (default)
    view_factor_matrix_brute   # pure triangle loop
)

# ----------------------------------------------------------------------
# Geometry helper  (unchanged)
# ----------------------------------------------------------------------
def gen_sphere_patches(lat_n: int, lon_n: int, radius: float = 1.0):
    meshes = []
    idx = 0
    for i in range(lat_n):
        phi0 = math.pi *  i    / lat_n
        phi1 = math.pi * (i+1) / lat_n
        for j in range(lon_n):
            th0 = 2*math.pi *  j    / lon_n
            th1 = 2*math.pi * (j+1) / lon_n

            p00 = np.array([math.sin(phi0)*math.cos(th0),
                            math.sin(phi0)*math.sin(th0),
                            math.cos(phi0)]) * radius
            p01 = np.array([math.sin(phi0)*math.cos(th1),
                            math.sin(phi0)*math.sin(th1),
                            math.cos(phi0)]) * radius
            p10 = np.array([math.sin(phi1)*math.cos(th0),
                            math.sin(phi1)*math.sin(th0),
                            math.cos(phi1)]) * radius
            p11 = np.array([math.sin(phi1)*math.cos(th1),
                            math.sin(phi1)*math.sin(th1),
                            math.cos(phi1)]) * radius

            for tri in ((p00, p10, p01), (p11, p01, p10)):
                V = np.vstack(tri).astype(np.float32)
                # flip if the normal points outward
                n = np.cross(V[1]-V[0], V[2]-V[0])
                if np.dot(n, -V[0]) < 0:
                    V = V[[0, 2, 1]]
                meshes.append((f"patch_{idx}", V, np.array([[0, 1, 2]], np.int32)))
                idx += 1
    return meshes

# ----------------------------------------------------------------------
# Build sphere
# ----------------------------------------------------------------------
LAT, LON = 10, 20          # ≈ 400 patches
meshes   = gen_sphere_patches(LAT, LON)
print(f"Generated {len(meshes)} triangular patches on the sphere.\n")

# Monte-Carlo parameters
SAMPLES, RAYS = 64, 64
SEED          = 123        # SAME seed -> identical rays
print(f"Parameters: {SAMPLES=}  {RAYS=}  total rays/patch = {SAMPLES**2*RAYS:,}\n")

# ----------------------------------------------------------------------
# BVH run
# ----------------------------------------------------------------------
t0 = time.perf_counter()
VF_bvh = view_factor_matrix(
    meshes,
    samples=SAMPLES,
    rays=RAYS,
    seed=SEED,          # identical rays
    use_bvh=True
)
t_bvh = time.perf_counter() - t0
print(f"BVH pass finished in {t_bvh:.3f} s")

# ----------------------------------------------------------------------
# Brute run
# ----------------------------------------------------------------------
t1 = time.perf_counter()
VF_brute = view_factor_matrix_brute(
    meshes,
    samples=SAMPLES,
    rays=RAYS,
    seed=SEED          # identical rays
)
t_brute = time.perf_counter() - t1
print(f"Brute pass finished in {t_brute:.3f} s\n")

# ----------------------------------------------------------------------
# Compare results for one sample emitter (patch_1)
# ----------------------------------------------------------------------
row_bvh   = VF_bvh  ["patch_1"]
row_brute = VF_brute["patch_1"]

sum_bvh   = sum(row_bvh.values())
sum_brute = sum(row_brute.values())

max_delta = 0.0
for k in set(row_bvh) | set(row_brute):
    d = abs(row_bvh.get(k, 0.0) - row_brute.get(k, 0.0))
    if d > max_delta:
        max_delta = d

print("patch_1 → * view-factor sums")
print(f"  BVH   : {sum_bvh  :.6f}")
print(f"  Brute : {sum_brute:.6f}")
print(f"\nBVH vs Brute maximum per-receiver Delta: {max_delta:.3e}")
print(f"Speed-up (Brute / BVH): {t_brute / t_bvh if t_bvh else float('inf'):.2f}×")
