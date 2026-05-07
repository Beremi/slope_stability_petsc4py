# 2D homogeneous LL

This 2D case runs a config-driven limit-load (LL) analysis using asset `2d_homo_slope` and mesh variant `h0.5.msh`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_homo_slope`
- Mesh variant: `h0.5.msh`
- Profile: `default`
- Analysis: `ll`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/2d_homo_slope/definition.py`](../../meshes/2d_homo_slope/definition.py).

## Reference

- MATLAB driver: `slope_stability_2D_homo_LL.m`
