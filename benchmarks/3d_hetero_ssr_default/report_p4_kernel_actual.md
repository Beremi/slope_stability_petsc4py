# 3D Hetero SSR P4 Kernel Comparison

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- `step_max`: `2`
- Constitutive mode: `overlap`
- Kernels: `legacy, rows`

## `legacy`

- Rank-1 runtime: `702.030` s
- Rank-1 `build_tangent_local`: `135.022` s
- Rank-1 `local_constitutive_comm`: `0.000` s
- Best runtime rank: `8`

## `rows`

- Rank-1 runtime: `644.763` s
- Rank-1 `build_tangent_local`: `61.328` s
- Rank-1 `local_constitutive_comm`: `0.000` s
- Best runtime rank: `8`

## `rows` vs `legacy`

| Ranks | Runtime baseline [s] | Runtime rows [s] | Runtime speedup | Tangent baseline [s] | Tangent rows [s] | Tangent speedup | |delta lambda| | |delta omega| | |delta Umax| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 702.030 | 644.763 | 1.089x | 135.022 | 61.328 | 2.202x | 3.952e-14 | 1.863e-09 | 8.948e-14 |
| 2 | 436.841 | 400.058 | 1.092x | 73.525 | 34.819 | 2.112x | 2.465e-14 | 0.000e+00 | 4.419e-14 |
| 4 | 327.511 | 308.130 | 1.063x | 43.722 | 23.594 | 1.853x | 2.975e-14 | 2.794e-09 | 1.205e-13 |
| 8 | 272.450 | 264.676 | 1.029x | 26.339 | 18.138 | 1.452x | 4.135e-12 | 0.000e+00 | 9.899e-12 |

## Artifact Layout

- Per-kernel summaries, plots, and rank subdirectories are written under `kernel_<name>/` inside the output root.
- This root report compares the kernels directly while preserving the original per-kernel scaling artifacts.
