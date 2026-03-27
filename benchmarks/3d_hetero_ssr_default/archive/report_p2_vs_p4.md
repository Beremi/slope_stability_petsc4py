# 3D Hetero SSR: P2 vs P4

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- MPI ranks: `2`
- `step_max`: `1`
- Runner: `slope_stability.cli.run_3D_hetero_SSR_capture`

## Results

| Metric | P2 | P4 | P4 / P2 |
| --- | ---: | ---: | ---: |
| Mesh nodes | 27605 | 208549 | 7.555x |
| Unknowns | 80362 | 616322 | 7.669x |
| Runtime [s] | 13.750 | 449.070 | 32.660x |
| Accepted continuation steps | 1 | 1 | 1.000x |
| Init linear iterations | 61 | 141 | 2.311x |
| Step linear iterations total | 55 | 167 | 3.036x |
| Step Newton iterations total | 6 | 7 | 1.167x |
| Final lambda | 1.159862541 | 1.160363642 | 1.000x |
| Final omega | 6243526.799569955 | 6244976.113431931 | 1.000x |
| Final Umax | 0.829960933 | 0.829356998 | 0.999x |

## Notes

- The comparison uses the same tet4 `.msh` source mesh and elevates it in-memory to `tet10`/`tri6` for `P2` and `tet35`/`tri15` for `P4` after loading.
- The current VTU/export path linearizes higher-order simplex cells for visualization. Solver-side assembly still uses the full elevated connectivity.
- The 3D `P4` path uses a 24-point Keast tetrahedron rule, exact for total polynomial degree 6 on the reference tetrahedron.
