# 2D Sloan2013 seepage

This benchmark defines a 2D heterogeneous seepage case with seepage.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `run_2D_sloan2013_seepage_capture`
- PETSc config: [`case.toml`](case.toml)

## Asset Definition

- Asset: `2d_sloan2013`
- Mesh variant: `default.msh`
- Profile: default
- Analysis: `seepage`
- Element: `P1`

Geometry, materials, hydraulics, and boundary conditions are defined in
[`../../meshes/2d_sloan2013/definition.py`](../../meshes/2d_sloan2013/definition.py).

## Notes

Current PETSc seepage implementation is scalar; under MPI the benchmark executes on rank 0
only.
