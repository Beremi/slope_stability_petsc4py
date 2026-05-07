# 2D Franz dam SSR

This 2D case runs a config-driven shear strength reduction (SSR) analysis using asset `2d_franz_dam` and mesh variant `default.msh`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `2d_franz_dam`
- Mesh variant: `default.msh`
- Profile: `default`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/2d_franz_dam/definition.py`](../../meshes/2d_franz_dam/definition.py).

## Reference

- MATLAB driver: `slope_stability_2D_Franz_dam_SSR.m`
