# 3D concave seepage SSR

This 3D case runs a config-driven shear strength reduction (SSR) analysis using asset `3d_hetero_seepage_transition` and mesh variant `transition_default`.

## Run

```bash
./run.sh
```

## Case Inputs

- Case config: [`case.toml`](case.toml)
- Asset: `3d_hetero_seepage_transition`
- Mesh variant: `transition_default`
- Solver profile: `pmg-deflated-baseline`
- Analysis: `ssr`
- Element order: `P2`

Geometry, materials, hydraulic behavior, and boundary conditions are defined in [`../../../meshes/3d_hetero_seepage_transition/definition.py`](../../../meshes/3d_hetero_seepage_transition/definition.py).
