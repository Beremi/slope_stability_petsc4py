# 3D heterogeneous SSR

This 3D case runs a config-driven shear strength reduction (SSR) analysis using asset `3d_hetero_slope` and mesh variant `adaptive_family_a_l1.msh`. It is part of the MATLAB-parity benchmark suite.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_hetero_slope`
- Mesh variant: `adaptive_family_a_l1.msh`
- Profile: `default`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/3d_hetero_slope/definition.py`](../../meshes/3d_hetero_slope/definition.py).

## Reference

- MATLAB driver: `run_3D_hetero_SSR_capture`
