import numpy as np, math, time
from raystrack.main import view_factor_matrix, view_factor_matrix_brute

# Radiance reference value
F_ref = 0.0557
print(f"Reference F = {F_ref:.6f}")

# -------------------------------------------------------------------
# Hard-coded meshes
# -------------------------------------------------------------------
e = 1e-7

V_em = np.array([
    (-e/2, -e/2, 0.0),
    ( e/2, -e/2, 0.0),
    ( e/2,  e/2, 0.0),
    (-e/2,  e/2, 0.0),
], dtype=np.float32)
F_em = np.array([[0,1,2], [0,2,3]], dtype=np.int32)

V_rc = np.array([
    (-1.0, 1.0, 0.0),
    (-1.0, 1.0, 1.0),
    (-1.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
], dtype=np.float32)
F_rc = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
meshes = [
    ("emitter",  V_em, F_em),
    ("rectangle", V_rc, F_rc),
]

# -------------------------------------------------------------------
# Monte-Carlo run
# -------------------------------------------------------------------
SAMPLES = 254
RAYS    = 254

print(f"Running Monte-Carlo: samples={SAMPLES}, rays={RAYS}")
t0 = time.time()
VF = view_factor_matrix(meshes, samples=SAMPLES, rays=RAYS)
dt = time.time() - t0
print(f"Done in {dt:.3f}s\n")

# Extract the computed view-factor
F_mc = VF["emitter"].get("rectangle_front", 0.0)
print(f"MC   F = {F_mc:.6f}")
print(f"Ref  F = {F_ref:.6f}")
print(f"Error = {abs(F_mc - F_ref):.6e}")


t0 = time.time()
VF_BRUTE = view_factor_matrix_brute(meshes, samples=SAMPLES, rays=RAYS)
dt = time.time() - t0
print(f"Done in {dt:.3f}s\n")










