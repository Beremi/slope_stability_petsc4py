# P4 Rank-8 Recycle-Enabled Run: Guarded Failure

- Date: `2026-03-16`
- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- MPI ranks: `8`
- Element order: `P4`
- Tangent kernel: `rows`
- Constitutive mode: `overlap`
- `step_max = 100`
- `recycle_preconditioner = true`
- `max_deflation_basis_vectors = 16`
- Guard limit: `80 GiB`

## Artifacts

- Failed guard log: [memory_guard_p4_rows_overlap_recycle_guard80_failed.jsonl](/home/beremi/repos/slope_stability-1/artifacts/p2_p4_compare_rank8_final_memfix/memory_guard_p4_rows_overlap_recycle_guard80_failed.jsonl)
- Failed stdout: [p4_rows_overlap_recycle_guard80_failed.stdout.log](/home/beremi/repos/slope_stability-1/artifacts/p2_p4_compare_rank8_final_memfix/p4_rows_overlap_recycle_guard80_failed.stdout.log)
- Failed stderr: [p4_rows_overlap_recycle_guard80_failed.stderr.log](/home/beremi/repos/slope_stability-1/artifacts/p2_p4_compare_rank8_final_memfix/p4_rows_overlap_recycle_guard80_failed.stderr.log)
- Partial progress: [progress.jsonl](/home/beremi/repos/slope_stability-1/artifacts/p2_p4_compare_rank8_final_memfix/p4_rank8_step100_recycle_guard80_failed/data/progress.jsonl)
- Previous guarded comparison baseline: [report_p2_vs_p4_rank8_final_guarded80_v2.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/report_p2_vs_p4_rank8_final_guarded80_v2.md)

## Outcome

| Metric | Value |
| --- | ---: |
| Run completed | no |
| Guard triggered | yes |
| Peak RSS [GiB] | 80.162 |
| Last accepted state index | 8 |
| Last accepted lambda | 1.565494461 |
| Last accepted omega | 6527841.506036154 |
| Last accepted Umax | 2.336503046 |
| Wall time to last accepted state [s] | 2176.683 |

## Comparison to Old Guarded Baseline

| Metric | Old guarded baseline | New recycle-enabled run |
| --- | ---: | ---: |
| Peak RSS [GiB] | 45.806 | 80.162 |
| Completion | yes | no |
| Final accepted states | 12 | 8 before guard stop |

## Interpretation

- Removing `overlap_B` from the owned-row geometry path did not fix the rank-8 final-state memory problem for the recycle-enabled production configuration.
- The persistent-memory reduction is real on the owned-pattern side, but the full rank-8 run still grows far beyond the old guarded baseline.
- On this case, the dominant remaining pressure is not the removed sparse overlap strain matrix. The likely drivers are retained solver/preconditioner state and the still-large row-slot plus `dphi*` footprint.
- This failure is why the completed side-by-side comparison for this validation cycle uses the no-recycle fallback as a separate run.
