# 3D heterogeneous seepage

This 3D case runs a config-driven seepage analysis using asset `3d_hetero_seepage` and mesh variant `concave_family_b.msh`. It is part of the MATLAB-parity benchmark suite.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_hetero_seepage`
- Mesh variant: `concave_family_b.msh`
- Profile: `default`
- Analysis: `seepage`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/3d_hetero_seepage/definition.py`](../../meshes/3d_hetero_seepage/definition.py).

## Reference

- MATLAB driver: `run_3D_hetero_seepage_capture`

## Notes

Current PETSc seepage implementation is scalar; under MPI the benchmark executes on rank 0 only.
