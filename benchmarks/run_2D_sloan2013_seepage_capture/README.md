# 2D Sloan2013 seepage

This 2D case runs a config-driven seepage analysis using asset `2d_sloan2013` and mesh variant `default.msh`. It is part of the MATLAB-parity benchmark suite.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_sloan2013`
- Mesh variant: `default.msh`
- Profile: `default`
- Analysis: `seepage`
- Element order: `P1`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/2d_sloan2013/definition.py`](../../meshes/2d_sloan2013/definition.py).

## Reference

- MATLAB driver: `run_2D_sloan2013_seepage_capture`

## Notes

Current PETSc seepage implementation is scalar; under MPI the benchmark executes on rank 0 only.
