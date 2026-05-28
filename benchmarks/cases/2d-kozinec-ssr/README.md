# 2D Kozinec SSR

This 2D case runs a config-driven shear strength reduction (SSR) analysis using asset `2d_kozinec` and mesh variant `default`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_kozinec`
- Mesh variant: `default`
- Solver profile: `baseline-pmg-deflated`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/2d_kozinec/definition.py`](../../../meshes/2d_kozinec/definition.py).
