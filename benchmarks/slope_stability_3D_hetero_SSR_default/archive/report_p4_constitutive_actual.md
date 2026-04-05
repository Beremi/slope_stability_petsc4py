# 3D Hetero SSR P4 Constitutive Comparison

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- Tangent kernel: `rows`
- `step_max`: `2`
- Constitutive modes: `overlap, unique_exchange`

## `overlap`

- Rank-1 runtime: `644.763` s
- Rank-1 `local_constitutive`: `1.289` s
- Rank-1 `local_constitutive_comm`: `0.000` s
- Best runtime rank: `8`

## `unique_exchange`

- Rank-1 runtime: `620.310` s
- Rank-1 `local_constitutive`: `1.278` s
- Rank-1 `local_constitutive_comm`: `1.264` s
- Best runtime rank: `8`

## `unique_exchange` vs `overlap`

| Ranks | Runtime baseline [s] | Runtime unique_exchange [s] | Runtime speedup | Local constitutive baseline [s] | Local constitutive unique_exchange [s] | Local comm baseline [s] | Local comm unique_exchange [s] | |delta lambda| | |delta omega| | |delta Umax| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 644.763 | 620.310 | 1.039x | 1.289 | 1.278 | 0.000 | 1.264 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 2 | 400.058 | 403.808 | 0.991x | 1.011 | 0.983 | 0.000 | 1.290 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 4 | 308.130 | 311.155 | 0.990x | 0.953 | 0.963 | 0.000 | 1.817 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 8 | 264.676 | 267.173 | 0.991x | 0.505 | 0.455 | 0.000 | 2.160 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## Artifact Layout

- Per-mode results are written under `mode_<name>/kernel_<name>/` when multiple constitutive modes are requested.
- This report compares constitutive modes with the tangent kernel held fixed.
