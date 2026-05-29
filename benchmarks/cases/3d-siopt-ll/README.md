# 3D SIOPT LL

This 3D case runs a config-driven limit-load (LL) analysis using asset `3d_siopt` and mesh variant `reference_l0`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_siopt`
- Mesh variant: `reference_l0`
- Solver profile: `pmg-deflated-baseline`
- Analysis: `ll`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/3d_siopt/definition.py`](../../../meshes/3d_siopt/definition.py).
