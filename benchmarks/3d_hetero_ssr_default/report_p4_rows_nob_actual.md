# P4 `rows+overlap` Without `overlap_B`: Actual Scaling Validation

- Date: `2026-03-16`
- Python: [`.venv/bin/python`](/home/beremi/repos/slope_stability-1/.venv/bin/python)
- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- Tangent kernel: `rows`
- Constitutive mode: `overlap`
- `step_max = 2`
- Ranks tested: `1, 2, 4, 8`

## Source Artifacts

- New sweep summary: [kernel_rows/summary.json](/home/beremi/repos/slope_stability-1/artifacts/p4_scaling_rows_nob_actual/kernel_rows/summary.json)
- New sweep report: [kernel_rows/report.md](/home/beremi/repos/slope_stability-1/artifacts/p4_scaling_rows_nob_actual/kernel_rows/report.md)
- Previous actual `rows+overlap` baseline: [report_p4_actual_scaling_vs_previous.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/report_p4_actual_scaling_vs_previous.md)
- Previous memory smoke: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/petsc_smoke_rows_overlap_memfix_step1/data/run_info.json)
- New memory smoke: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/petsc_smoke_rows_overlap_nob_step1/data/run_info.json)

## Rank-1 Smoke Gate

| Metric | Previous smoke | New smoke | Delta |
| --- | ---: | ---: | ---: |
| Runtime [s] | 627.199 | 624.000 | -0.51% |
| `build_tangent_local` [s] | 59.580 | 61.125 | +2.59% |
| `build_F` [s] | 5.101 | 9.427 | +84.79% |
| `local_strain` [s] | 4.043 | 2.123 | -47.48% |
| Final lambda | 1.160363588144855 | 1.160363588144844 | 1.110e-14 |
| Final omega | 6244976.095602412 | 6244976.095602410 | 1.863e-09 |
| Final Umax | 0.829357048318336 | 0.829357048318393 | 5.751e-14 |
| `overlap_B_bytes` | 1681224196 | 0 | removed |
| `scatter_bytes` | 0 | 0 | unchanged |

### Smoke Result

- The production `rows+overlap` path now records `owned_tangent_pattern.stats_max.overlap_B_bytes = 0.0`.
- Final-state drift versus the previous smoke is at roundoff level.
- End-to-end runtime did not regress on the rank-1 smoke.

## Actual `1/2/4/8` Sweep

| Ranks | Previous runtime [s] | New runtime [s] | Delta | Previous tangent [s] | New tangent [s] | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 644.763 | 635.147 | -1.49% | 61.328 | 61.262 | -0.11% |
| 2 | 400.058 | 385.356 | -3.67% | 34.819 | 34.120 | -2.01% |
| 4 | 308.130 | 290.995 | -5.56% | 23.594 | 22.408 | -5.03% |
| 8 | 264.676 | 251.315 | -5.05% | 18.138 | 17.506 | -3.49% |

## Strong Scaling

| Ranks | Runtime [s] | Speedup vs 1 | Efficiency | Final lambda | Final omega | Final Umax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 635.147 | 1.000x | 1.000 | 1.160363588 | 6244976.095602410 | 0.829357048 |
| 2 | 385.356 | 1.648x | 0.824 | 1.160363642 | 6244976.113431931 | 0.829356998 |
| 4 | 290.995 | 2.183x | 0.546 | 1.160363267 | 6244976.019935260 | 0.829356878 |
| 8 | 251.315 | 2.527x | 0.316 | 1.160364429 | 6244976.078749287 | 0.829359313 |

### Observations

- The no-`overlap_B` path is faster than the previous actual `rows+overlap` run at every tested rank.
- The gain is modest at rank `1` and larger at `2/4/8` ranks.
- Solution drift remains in the same small range as the previous actual scaling report:
  - max `|delta lambda| = 8.413e-07`
  - max `|delta omega| = 7.567e-02`
  - max `|delta Umax| = 2.265e-06`

## Owned-Pattern Bytes

| Ranks | `scatter_bytes` | `row_slot_bytes` | `overlap_B_bytes` | `unique_B_bytes` | `dphi_bytes` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 814743042 | 0 | 4 | 371327040 |
| 2 | 0 | 411277267 | 0 | 4 | 195229440 |
| 4 | 0 | 207170548 | 0 | 4 | 103440960 |
| 8 | 0 | 105921082 | 0 | 4 | 58766400 |

### Memory Interpretation

- The persistent sparse overlap strain matrix is gone from the production path at every tested rank.
- The dominant remaining Python-side footprint is now `row_slot_bytes + dphi_bytes`.
- On the step-2 sweep, removing `overlap_B` did not cause a runtime regression; the sweep improved slightly instead.

## Recommendation

- Keep `tangent_kernel="rows"` and `constitutive_mode="overlap"` as the production defaults.
- Keep the geometry-only no-`overlap_B` path in production for 3D `rows`.
- The next memory target is not `overlap_B` anymore. It is the remaining large footprint from row-slot metadata, `dphi*`, and the PETSc/preconditioner hierarchy visible on the long rank-8 run.
