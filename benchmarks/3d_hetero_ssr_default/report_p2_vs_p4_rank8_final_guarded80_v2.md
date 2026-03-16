# 3D Hetero SSR: P2 vs P4 Final-State Comparison

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- MPI ranks: `8`
- `step_max`: `100`
- Runner: `slope_stability.cli.run_3D_hetero_SSR_capture`
- Raw artifacts: `artifacts/p2_p4_compare_rank8_final_guarded80_v2`

## Headline Metrics

| Metric | P2 | P4 | P4 / P2 |
| --- | ---: | ---: | ---: |
| Mesh nodes | 27605 | 208549 | 7.555x |
| Unknowns | 80362 | 616322 | 7.669x |
| Runtime [s] | 121.804 | 5908.031 | 48.505x |
| Final accepted states | 14 | 12 | 0.857x |
| Continuation advances after init | 12 | 10 | 0.833x |
| Init linear iterations | 61 | 93 | 1.525x |
| Attempt linear iterations total | 2301 | 3715 | 1.615x |
| Accepted-step Newton iterations total | 143 | 329 | 2.301x |

## Final State

| Metric | P2 | P4 | Absolute difference |
| --- | ---: | ---: | ---: |
| Final lambda | 1.666095401 | 1.570620085 | 9.548e-02 |
| Final omega | 12000000.000000000 | 9922242.674696889 | 2.078e+06 |
| Final Umax | 132.877693723 | 84.475075541 | 4.840e+01 |

## Continuation Reach

| Order | Init accepted states | Final accepted states | Successful attempts |
| --- | ---: | ---: | ---: |
| P2 | 2 | 14 | 12 / 12 |
| P4 | 2 | 12 | 10 / 10 |

## Timing Breakdown

| Metric | P2 [s] | P4 [s] | P4 / P2 |
| --- | ---: | ---: | ---: |
| Init solve | 1.929 | 59.629 | 30.915x |
| Attempt solves total | 77.284 | 2556.719 | 33.082x |
| Attempt preconditioner total | 6.678 | 2084.681 | 312.192x |
| Tangent local | 9.061 | 377.725 | 41.687x |
| Build F | 5.565 | 120.891 | 21.723x |
| Local strain | 5.397 | 101.613 | 18.827x |
| Local constitutive | 2.811 | 11.802 | 4.199x |

## Memory Guard

| Metric | P4 guarded run |
| --- | ---: |
| Peak RSS [GiB] | 45.806 |
| Minimum MemAvailable [GiB] | 44.444 |
| Samples | 597 |
| Guard triggered | no |
| Guard log | `artifacts/p2_p4_compare_rank8_final_guarded80_v2/memory_guard_p4_no_recycle.jsonl` |

## Plots

![Continuation curves](../../artifacts/p2_p4_compare_rank8_final_guarded80_v2/plots/continuation_curves.png)

![Iteration comparison](../../artifacts/p2_p4_compare_rank8_final_guarded80_v2/plots/iterations.png)

![Timing breakdown](../../artifacts/p2_p4_compare_rank8_final_guarded80_v2/plots/timing_breakdown.png)

## Notes

- This comparison uses the same tet4 `.msh` source mesh and elevates it in-memory to `tet10`/`tri6` for `P2` and `tet35`/`tri15` for `P4` after loading.
- The current VTU/export path linearizes higher-order simplex cells for visualization. Solver-side assembly still uses the full elevated connectivity.
- On this mesh, both orders terminated naturally after the same continuation reach. The large difference is computational cost, not a different final branch point.
