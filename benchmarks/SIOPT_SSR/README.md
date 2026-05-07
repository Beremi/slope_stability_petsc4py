# 3D SIOPT SSR

This 3D case runs a config-driven shear strength reduction (SSR) analysis using asset `3d_siopt` and mesh variant `reference_l0.msh`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_siopt`
- Mesh variant: `reference_l0.msh`
- Profile: `fixed_base`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/3d_siopt/definition.py`](../../meshes/3d_siopt/definition.py).

## Reference

- MATLAB driver: `SIOPT_SSR.m`
