# Rank-8 `P2` vs `P4` Memory-Fix Validation

- Date: `2026-03-16`
- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- MPI ranks: `8`
- Tangent kernel: `rows`
- Constitutive mode: `overlap`
- `step_max = 100`

## What Was Validated

There are two distinct outcomes for the fresh rank-8 runs:

1. The requested recycle-enabled `P4` configuration (`recycle_preconditioner=true`, `max_deflation_basis_vectors=16`) did **not** complete under the `80 GiB` guard.
2. The no-recycle fallback completed successfully and is the fresh apples-to-apples comparison against the old guarded baseline.

## Fresh Artifacts

- Completed comparison report: [report_p2_vs_p4_rank8_final_memfix.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/report_p2_vs_p4_rank8_final_memfix.md)
- Completed comparison summary: [summary.json](/home/beremi/repos/slope_stability-1/artifacts/p2_p4_compare_rank8_final_memfix/summary.json)
- Recycle-enabled failure note: [report_p4_rank8_recycle_guard80_failed.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/report_p4_rank8_recycle_guard80_failed.md)
- Previous guarded comparison baseline: [report_p2_vs_p4_rank8_final_guarded80_v2.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/report_p2_vs_p4_rank8_final_guarded80_v2.md)

## Requested Recycle-Enabled `P4` Run

| Metric | Value |
| --- | ---: |
| Completed | no |
| Guard triggered | yes |
| Peak RSS [GiB] | 80.162 |
| Last accepted state | 8 |
| Last accepted lambda | 1.565494461 |

Interpretation:

- The new geometry-only `rows` path removed `overlap_B`, but that did **not** solve the rank-8 memory problem for the recycle-enabled configuration.
- The dominant memory growth on that path is elsewhere, most likely retained solver/preconditioner state plus the still-large row-slot and `dphi*` data.

## Completed No-Recycle Comparison vs Old Baseline

| Metric | Old P2 | New P2 | Delta | Old P4 | New P4 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime [s] | 121.804 | 108.002 | -11.33% | 5908.031 | 5542.992 | -6.18% |
| Final accepted states | 14 | 14 | unchanged | 12 | 12 | unchanged |
| Final lambda | 1.666095401 | 1.666089658 | 5.742e-06 | 1.570620085 | 1.570672856 | 5.277e-05 |
| Final omega | 12000000.000000000 | 12000000.000000000 | 0 | 9922242.674696889 | 9922242.674347960 | 3.489e-04 |
| Final Umax | 132.877693723 | 132.839467255 | 3.823e-02 | 84.475075541 | 83.781198088 | 6.939e-01 |
| Peak RSS [GiB] | - | - | - | 45.806 | 42.676 | -6.83% |

## Owned-Pattern Bytes on the Completed Run

| Metric | P2 | P4 |
| --- | ---: | ---: |
| `scatter_bytes` | 0 | 0 |
| `row_slot_bytes` | 9164279 | 105921082 |
| `overlap_B_bytes` | 0 | 0 |
| `unique_B_bytes` | 4 | 4 |
| `dphi_bytes` | 7988640 | 58766400 |

## Conclusion

- The 3D `rows` production path is now genuinely `overlap_B`-free.
- On the completed rank-8 no-recycle comparison, the memory fix was successful:
  - `P4` peak RSS dropped from `45.806 GiB` to `42.676 GiB`
  - `P4` runtime dropped from `5908.031 s` to `5542.992 s`
  - continuation reach stayed the same at `12` accepted states
- The remaining open problem is the recycle-enabled rank-8 configuration. That path still grows to the guard limit and needs separate solver/preconditioner-focused work.
