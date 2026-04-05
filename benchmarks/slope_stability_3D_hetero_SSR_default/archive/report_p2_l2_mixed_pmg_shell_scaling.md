# P2(L2) Mixed PMG-Shell Scaling

## Configuration

- Fine hierarchy: `P2(L2) -> P1(L2) -> P1(L1)`
- Backend: `pc_backend=pmg_shell`
- Fine/mid smoother: `chebyshev + jacobi`, `3` steps
- Coarse solve: `cg + hypre(boomeramg)`
- Coarse operator source: `direct_elastic_full_system`
- Frozen state source: `artifacts/l2_p2_hypre_step1_for_mixed_pmg/data/petsc_run.npz`
- All scaling tables use wall-time maxima across ranks, not rank-summed CPU time

## Summary

| Ranks | Outer Iters | Final Rel Residual | Runtime Max [s] | Setup+Solve Max [s] | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.000815 | 39.754 | 5.535 | 1.000x | 1.000 |
| 2 | 12 | 0.000815 | 32.518 | 4.142 | 1.223x | 0.611 |
| 4 | 12 | 0.000815 | 29.662 | 3.342 | 1.340x | 0.335 |
| 8 | 12 | 0.000815 | 26.719 | 2.575 | 1.488x | 0.186 |

## Timing Table

| Ranks | Problem Build Max [s] | Operator Build Max [s] | Setup Max [s] | Solve Max [s] | Other Max [s] | PC Setup Max [s] | PC Apply Max [s] | Applies | Coarse CG Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.737 | 0.424 | 1.190 | 4.345 | 0.057 | 1.190 | 4.190 | 12 | 48 |
| 2 | 27.972 | 3.641 | 1.086 | 3.056 | 0.041 | 1.086 | 2.934 | 12 | 48 |
| 4 | 25.960 | 4.018 | 0.919 | 2.424 | 0.040 | 0.919 | 2.290 | 12 | 48 |
| 8 | 23.796 | 3.292 | 0.723 | 1.852 | 0.041 | 0.723 | 1.739 | 12 | 48 |

## Speedup By Stage

| Ranks | Runtime | Problem Build | Operator Build | Setup+Solve | Setup | Solve | PC Setup | PC Apply |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x |
| 2 | 1.223x | 1.206x | 0.117x | 1.336x | 1.096x | 1.422x | 1.096x | 1.428x |
| 4 | 1.340x | 1.300x | 0.106x | 1.656x | 1.296x | 1.793x | 1.296x | 1.829x |
| 8 | 1.488x | 1.418x | 0.129x | 2.149x | 1.646x | 2.346x | 1.646x | 2.409x |

## Interpretation

- The gap between `Runtime Max` and `Setup+Solve Max` is mostly `Problem Build Max`, not hidden solver time.
- The stage columns are independent rank-max wall times, so they are not additive; the slowest rank in one stage need not be the slowest rank in another.
- `Other Max` is the leftover wall time after problem build, operator build, and linear setup+solve; it is tiny in these runs.
- `Operator Build Max` is a rank-max wall time. It can be larger than the local value on rank 0 because the slowest rank sets the column.
- The PC breakdown tables below are also rank-max wall times, not summed CPU seconds over all MPI ranks.

## Rank-Max PCApply Wall-Time Breakdown

| Ranks | fine_pre | fine_post | mid_pre | mid_post | fine_residual | mid_residual | restrict_f2m | restrict_m2c | prolong_c2m | prolong_m2f | vector_sum | coarse_hypre | unattributed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.229 | 0.311 | 0.017 | 0.018 | 0.074 | 0.004 | 0.004 | 0.002 | 0.002 | 0.008 | 0.002 | 3.517 | 0.003 |
| 2 | 0.179 | 0.241 | 0.016 | 0.012 | 0.066 | 0.004 | 0.008 | 0.002 | 0.002 | 0.007 | 0.001 | 2.399 | 0.000 |
| 4 | 0.197 | 0.245 | 0.014 | 0.010 | 0.066 | 0.003 | 0.020 | 0.002 | 0.001 | 0.005 | 0.001 | 1.741 | 0.000 |
| 8 | 0.209 | 0.251 | 0.011 | 0.009 | 0.064 | 0.002 | 0.019 | 0.001 | 0.001 | 0.006 | 0.001 | 1.185 | 0.000 |

## Rank-Max PCApply Wall-Time Breakdown Per Apply

| Ranks | fine_pre | fine_post | mid_pre | mid_post | fine_residual | mid_residual | restrict_f2m | restrict_m2c | prolong_c2m | prolong_m2f | vector_sum | coarse_hypre | unattributed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0191 | 0.0259 | 0.0014 | 0.0015 | 0.0062 | 0.0003 | 0.0003 | 0.0002 | 0.0001 | 0.0007 | 0.0001 | 0.2931 | 0.0002 |
| 2 | 0.0149 | 0.0201 | 0.0014 | 0.0010 | 0.0055 | 0.0003 | 0.0007 | 0.0002 | 0.0001 | 0.0006 | 0.0001 | 0.1999 | 0.0000 |
| 4 | 0.0164 | 0.0204 | 0.0012 | 0.0008 | 0.0055 | 0.0003 | 0.0016 | 0.0001 | 0.0001 | 0.0005 | 0.0001 | 0.1451 | 0.0000 |
| 8 | 0.0174 | 0.0209 | 0.0010 | 0.0007 | 0.0053 | 0.0002 | 0.0016 | 0.0001 | 0.0001 | 0.0005 | 0.0001 | 0.0988 | 0.0000 |

## Plots

[wall_time_scaling.png](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/wall_time_scaling.png)

![Wall Time Scaling](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/wall_time_scaling.png)

[pc_breakdown_sum.png](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/pc_breakdown_sum.png)

![PCApply Breakdown Sum](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/pc_breakdown_sum.png)

[pc_breakdown_per_apply.png](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/pc_breakdown_per_apply.png)

![PCApply Breakdown Per Apply](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/pc_breakdown_per_apply.png)

[iteration_counts.png](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/iteration_counts.png)

![Iteration Counts](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/plots/iteration_counts.png)

## Raw Artifacts

- rank 1: [run_info.json](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/rank1/data/run_info.json)
- rank 2: [run_info.json](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/rank2/data/run_info.json)
- rank 4: [run_info.json](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/rank4/data/run_info.json)
- rank 8: [run_info.json](../../../artifacts/l2_p2_mixed_pmg_shell_scaling/rank8/data/run_info.json)
