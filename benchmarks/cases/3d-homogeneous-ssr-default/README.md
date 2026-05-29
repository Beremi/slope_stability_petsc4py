# 3D homogeneous SSR default

This 3D case runs a config-driven shear strength reduction (SSR) analysis using asset `3d_homo_slope` and mesh variant `adaptive_family_a_l1`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_homo_slope`
- Mesh variant: `adaptive_family_a_l1`
- Solver profile: `pmg-deflated-baseline`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/3d_homo_slope/definition.py`](../../../meshes/3d_homo_slope/definition.py).
