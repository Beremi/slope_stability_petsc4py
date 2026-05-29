# 2D Kozinec LL

This 2D case runs a config-driven limit-load (LL) analysis using asset `2d_kozinec` and mesh variant `default`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_kozinec`
- Mesh variant: `default`
- Solver profile: `pmg-deflated-baseline`
- Analysis: `ll`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/2d_kozinec/definition.py`](../../../meshes/2d_kozinec/definition.py).
