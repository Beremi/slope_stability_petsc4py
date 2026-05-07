# 3D heterogeneous seepage SSR COMSOL

This 3D case runs a config-driven shear strength reduction (SSR) analysis using asset `3d_hetero_seepage_transition` and mesh variant `transition_default.msh`. It is part of the MATLAB-parity benchmark suite.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_hetero_seepage_transition`
- Mesh variant: `transition_default.msh`
- Profile: `fixed_base`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../meshes/3d_hetero_seepage_transition/definition.py`](../../meshes/3d_hetero_seepage_transition/definition.py).

## Reference

- MATLAB driver: `run_3D_hetero_seepage_SSR_comsol_capture`
