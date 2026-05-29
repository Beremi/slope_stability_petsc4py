# 2D homogeneous SSR

This 2D case runs a config-driven shear strength reduction (SSR) analysis using asset `2d_homo_slope` and mesh variant `h1.0`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_homo_slope`
- Mesh variant: `h1.0`
- Solver profile: `pmg-deflated-baseline`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/2d_homo_slope/definition.py`](../../../meshes/2d_homo_slope/definition.py).
