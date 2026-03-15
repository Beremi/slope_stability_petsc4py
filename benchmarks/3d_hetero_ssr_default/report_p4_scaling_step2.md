# 3D Hetero SSR P4 Scaling

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- `step_max`: `2`
- Ranks tested: `1, 2, 4, 8, 16`
- Runner: `slope_stability.cli.run_3D_hetero_SSR_capture`
- Mesh nodes: `208549`
- Mesh elements: `18419`
- Unknowns: `616322`

## Strong Scaling

| Ranks | Runtime [s] | Speedup vs 1 | Efficiency | Final accepted states | Continuation advances | Init lin iters | Step lin iters | Step Newton iters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 748.878 | 1.000x | 1.000 | 3 | 1 | 140 | 166 | 7 |
| 2 | 443.372 | 1.689x | 0.845 | 3 | 1 | 141 | 167 | 7 |
| 4 | 330.044 | 2.269x | 0.567 | 3 | 1 | 141 | 166 | 7 |
| 8 | 273.377 | 2.739x | 0.342 | 3 | 1 | 139 | 159 | 7 |
| 16 | 287.330 | 2.606x | 0.163 | 3 | 1 | 132 | 166 | 7 |

## Continuation Reach

| Ranks | Init accepted states | Final accepted states | Successful continuation attempts |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 3 | 1 / 1 |
| 2 | 2 | 3 | 1 / 1 |
| 4 | 2 | 3 | 1 / 1 |
| 8 | 2 | 3 | 1 / 1 |
| 16 | 2 | 3 | 1 / 1 |

## Solution Reach

| Ranks | Final lambda | Final omega | Final Umax | |lambda-rank1| | |omega-rank1| | |Umax-rank1| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.160363588 | 6244976.095602410 | 0.829357048 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 2 | 1.160363642 | 6244976.113431931 | 0.829356998 | 5.422e-08 | 1.783e-02 | 4.987e-08 |
| 4 | 1.160363267 | 6244976.019935260 | 0.829356878 | 3.211e-07 | 7.567e-02 | 1.702e-07 |
| 8 | 1.160364429 | 6244976.078749288 | 0.829359313 | 8.413e-07 | 1.685e-02 | 2.265e-06 |
| 16 | 1.160364359 | 6244976.098163762 | 0.829357886 | 7.706e-07 | 2.561e-03 | 8.373e-07 |

## Timing Breakdown

| Ranks | Init solve [s] | Attempt solves [s] | Attempt prec [s] | Tangent local [s] | Build F [s] | Local strain [s] | Local constitutive [s] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 236.746 | 283.757 | 19.875 | 140.328 | 5.301 | 4.161 | 1.290 |
| 2 | 144.695 | 162.779 | 12.752 | 76.026 | 4.452 | 3.134 | 1.016 |
| 4 | 106.897 | 126.232 | 8.730 | 45.140 | 5.977 | 4.586 | 0.948 |
| 8 | 92.353 | 106.351 | 8.194 | 26.940 | 5.472 | 4.040 | 0.495 |
| 16 | 92.223 | 115.698 | 10.487 | 18.106 | 5.991 | 4.949 | 0.416 |

## Notes

- Strong-scaling speedup and efficiency are computed against the `1`-rank runtime.
- Best runtime in this sweep was at `8` ranks.
- Each run started from `2` accepted initialization states (`lambda = 1.0` and `lambda = 1.1`) and then accepted `1` continuation advance; allowing `2` continuation steps did not produce a second accepted advance on this configuration.
- Maximum solution drift across ranks was `|delta lambda| = 8.413e-07`, `|delta omega| = 7.567e-02`, and `|delta Umax| = 2.265e-06` relative to rank `1`.
- The elevated mesh and unknown count are constant across ranks; only the parallel decomposition changes.
