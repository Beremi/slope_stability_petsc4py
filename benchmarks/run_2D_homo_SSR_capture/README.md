# 2D homogeneous SSR

This 2D case runs a config-driven shear strength reduction (SSR) analysis using asset `2d_homo_slope` and mesh variant `h1.0.msh`. It is part of the MATLAB-parity benchmark suite.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_homo_slope`
- Mesh variant: `h1.0.msh`
- Profile: `default`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/2d_homo_slope/definition.py`](../../meshes/2d_homo_slope/definition.py).

## Reference

- MATLAB driver: `run_2D_homo_SSR_capture`
