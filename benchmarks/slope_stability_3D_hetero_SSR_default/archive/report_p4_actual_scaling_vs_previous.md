# P4 Actual Scaling Validation vs Previous Baseline

- Date: `2026-03-15`
- Python: [`.venv/bin/python`](/home/beremi/repos/slope_stability-1/.venv/bin/python)
- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- `step_max = 2`
- Ranks tested: `1, 2, 4, 8`

## Source Artifacts

- Old baseline report: [report_p4_scaling_step2.md](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_p4_scaling_step2.md)
- Phase-1 kernel report: [report_p4_kernel_actual.md](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_p4_kernel_actual.md)
- Phase-2 constitutive report: [report_p4_constitutive_actual.md](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_p4_constitutive_actual.md)
- Phase-1 summaries:
  - [kernel_legacy/summary.json](/home/beremi/repos/slope_stability-1/artifacts/p4_scaling_kernel_actual/kernel_legacy/summary.json)
  - [kernel_rows/summary.json](/home/beremi/repos/slope_stability-1/artifacts/p4_scaling_kernel_actual/kernel_rows/summary.json)
- Phase-2 summaries:
  - [mode_overlap/kernel_rows/summary.json](/home/beremi/repos/slope_stability-1/artifacts/p4_scaling_constitutive_actual/mode_overlap/kernel_rows/summary.json)
  - [mode_unique_exchange/kernel_rows/summary.json](/home/beremi/repos/slope_stability-1/artifacts/p4_scaling_constitutive_actual/mode_unique_exchange/kernel_rows/summary.json)

## Phase 1: `legacy` vs `rows` with `constitutive_mode=overlap`

| Ranks | Legacy runtime [s] | Rows runtime [s] | Rows speedup | Legacy tangent [s] | Rows tangent [s] | Tangent speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 702.030 | 644.763 | 1.089x | 135.022 | 61.328 | 2.202x |
| 2 | 436.841 | 400.058 | 1.092x | 73.525 | 34.819 | 2.112x |
| 4 | 327.511 | 308.130 | 1.063x | 43.722 | 23.594 | 1.853x |
| 8 | 272.450 | 264.676 | 1.029x | 26.339 | 18.138 | 1.452x |

### Observations

- `rows` is faster than `legacy` at every tested rank.
- The largest end-to-end gain is at `1` and `2` ranks, about `8-9%`.
- The local tangent build improvement is much larger than the total runtime improvement:
  - `2.20x` at rank `1`
  - `2.11x` at rank `2`
  - `1.85x` at rank `4`
  - `1.45x` at rank `8`
- Final `lambda`, `omega`, and `Umax` stayed numerically aligned with `legacy`; the phase-1 report shows only roundoff-level differences.

## Phase 2: `rows+overlap` vs `rows+unique_exchange`

| Ranks | Overlap runtime [s] | Unique-exchange runtime [s] | Unique-exchange speedup | Overlap local constitutive [s] | Unique-exchange local constitutive [s] | Unique-exchange comm [s] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 644.763 | 620.310 | 1.039x | 1.289 | 1.278 | 1.264 |
| 2 | 400.058 | 403.808 | 0.991x | 1.011 | 0.983 | 1.290 |
| 4 | 308.130 | 311.155 | 0.990x | 0.953 | 0.963 | 1.817 |
| 8 | 264.676 | 267.173 | 0.991x | 0.505 | 0.455 | 2.160 |

### Observations

- `unique_exchange` is numerically identical to `overlap` on this test set; the phase-2 report shows zero drift in final `lambda`, `omega`, and `Umax`.
- On rank `1`, `unique_exchange` is faster by about `3.9%`.
- On ranks `2/4/8`, `unique_exchange` is slightly slower than `overlap`, by about `0.9-1.0%`.
- The explicit `local_constitutive_comm` cost grows with rank count:
  - `1.264 s` at rank `1`
  - `1.290 s` at rank `2`
  - `1.817 s` at rank `4`
  - `2.160 s` at rank `8`
- On this machine and case, that communication overhead is not offset by enough saved constitutive work once `overlap` is already using the optimized `rows` tangent path.

## Comparison to the Previous P4 Step-2 Baseline

The old report in [report_p4_scaling_step2.md](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_p4_scaling_step2.md) is the historical baseline. Relative to that report:

| Ranks | Old runtime [s] | Current legacy runtime [s] | Delta vs old | Current rows runtime [s] | Delta vs old |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 748.878 | 702.030 | -6.26% | 644.763 | -13.90% |
| 2 | 443.372 | 436.841 | -1.47% | 400.058 | -9.77% |
| 4 | 330.044 | 327.511 | -0.77% | 308.130 | -6.64% |
| 8 | 273.377 | 272.450 | -0.34% | 264.676 | -3.18% |

And for `build_tangent_local`:

| Ranks | Old tangent [s] | Current legacy tangent [s] | Delta vs old | Current rows tangent [s] | Delta vs old |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 140.328 | 135.022 | -3.78% | 61.328 | -56.30% |
| 2 | 76.026 | 73.525 | -3.29% | 34.819 | -54.20% |
| 4 | 45.140 | 43.722 | -3.14% | 23.594 | -47.73% |
| 8 | 26.940 | 26.339 | -2.23% | 18.138 | -32.67% |

### Interpretation

- The current `legacy` run is already slightly faster than the older report, so the previous baseline is not a perfect apples-to-apples control for wall time.
- The controlled comparison for the optimization itself is the current phase-1 `legacy` vs `rows` run, and that result is unambiguous: `rows` wins at every tested rank.
- Against the historical baseline, `rows` still shows a clear reduction in end-to-end runtime and a very large reduction in local tangent build time.

## Recommendation

- Keep `tangent_kernel="rows"` as the production default.
- Keep `constitutive_mode="overlap"` as the production default on this case and machine.
- Keep `unique_exchange` available behind the flag for further work; it is correct, but it is not yet a throughput win on `2/4/8` ranks here.
- If `unique_exchange` is revisited, the next target should be reducing or restructuring the communication cost rather than the constitutive kernel itself.
