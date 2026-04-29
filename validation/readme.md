# Raystrack validation

This folder contains standalone validation scripts for Raystrack view factors.

Analytical cases from `Radiation View factors.pdf`:

- `validate_01_parallel_equal_square.py`
- `validate_02_parallel_equal_rectangle.py`
- `validate_03_equal_coaxial_discs.py`
- `validate_04_patch_to_disc.py`
- `validate_05_perpendicular_square_rectangle.py`

Each script writes a flat result file under `validation/results/`, for example
`validation/results/01_parallel_equal_square.txt`.

All validation scripts use `tol=1e-4`, `tol_mode="stderr"`, `min_iters=40`,
and `max_iters=500`. The result files report the actual iteration count for
each emitter and whether the run converged before the 500-iteration cap.

The canyon cross-check is:

- `validate_06_canyon_view3d_compare.py`

It loads the saved View3D canyon reference from `validation/view3d_reference/`,
runs Raystrack for the same canyon geometry, and writes
`validation/results/06_canyon_view3d.txt`.

Regenerate the saved View3D canyon reference:

```powershell
python validation\generate_canyon_view3d_reference.py
```

Run all validations:

```powershell
python validation\run_all.py
```
