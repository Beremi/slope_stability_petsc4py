# 2D Sloan2013 seepage

This 2D case runs a config-driven seepage analysis using asset `2d_sloan2013` and mesh variant `default`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_sloan2013`
- Mesh variant: `default`
- Solver profile: `baseline-pmg-deflated`
- Analysis: `seepage`
- Element order: `P1`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/2d_sloan2013/definition.py`](../../../meshes/2d_sloan2013/definition.py).
