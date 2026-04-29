# 3D heterogeneous seepage

Capture seepage-only results for the 3D heterogeneous water-level problem.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `run_3D_hetero_seepage_capture`
- PETSc config: [`case.toml`](case.toml)

## Asset Definition

- Asset: `3d_hetero_seepage`
- Mesh variant: `concave_family_b.msh`
- Profile: default
- Analysis: `seepage`
- Element: `P2`

Geometry, materials, hydraulics, and boundary conditions are defined in
[`../../meshes/3d_hetero_seepage/definition.py`](../../meshes/3d_hetero_seepage/definition.py).

## Notes

Current PETSc seepage implementation is scalar; under MPI the benchmark executes on rank 0
only.
