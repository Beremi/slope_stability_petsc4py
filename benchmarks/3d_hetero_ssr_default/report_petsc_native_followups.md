# PETSc Native Follow-Ups After Hypre Pmat Pivot

## Scope

This report records the next PETSc-native branches tried after the `Hypre + explicit Pmat` pivot:

- `PCHMG` with inner Hypre
- global `PCGAMG`

Branches explicitly not pursued here:

- `KSPFETIDP`, because PETSc builds it on top of an internal `PCBDDC`
- `PCMG + GDSW`, because the current frozen-state probe is matrix-only and does not attach a `DM`; PETSc documents that `PCMG` without a usable `DM` or explicit transfer operators degenerates into a meaningless single-level test

Implementation enabling these runs:

- the frozen-state probe now supports generic native PETSc preconditioners, not only native Hypre
- raw PETSc options are still passed through with the native KSP prefix

Relevant code:

- `benchmarks/3d_hetero_ssr_default/probe_hypre_frozen.py`

Baseline for contained hard-state comparison:

- `artifacts/hypre_pmat_pivot/hard_tangent_single_v1/data/run_info.json`

Baseline values:

- setup `20.776 s`
- solve `1.199 s`
- setup+solve `21.975 s`
- outer iterations `1`
- final relative residual `7.554e-2`

## HMG Follow-Up

Configuration:

- native PETSc outer `FGMRES`
- native `pc_type hmg`
- `pc_hmg_use_subspace_coarsening=true`
- `pc_hmg_reuse_interpolation=true`
- `pc_hmg_use_matmaij=true`
- inner coarsener `Hypre/BoomerAMG`
- `mg_levels_ksp_type=chebyshev`
- `mg_levels_ksp_max_it=2`
- `mg_levels_pc_type=jacobi`

Artifacts:

- `artifacts/hmg_pivot/hard_tangent_hmg_c0_v1/data/run_info.json`
- `artifacts/hmg_pivot/hard_tangent_hmg_c1_v1/data/run_info.json`
- `artifacts/hmg_pivot/hard_tangent_hmg_c2_v1/data/run_info.json`

Contained hard-state results:

| Coarsening component | Setup [s] | Solve [s] | Setup+Solve [s] | Iterations | Final relative residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 34.359 | 2.911 | 37.271 | 1 | 8.681e-2 |
| 1 | 34.387 | 2.882 | 37.269 | 1 | 8.681e-2 |
| 2 | 34.198 | 2.862 | 37.060 | 1 | 8.681e-2 |

Interpretation:

- all three coarsening components are effectively identical
- HMG is slower than the contained Hypre tangent baseline by about `15 s` in setup+solve
- HMG does not improve the outer iteration count
- HMG ends with a slightly worse final residual than Hypre at the same outer tolerance

Decision:

- `PCHMG` is not competitive enough to move forward on this path

## Global GAMG Follow-Up

Configuration:

- native PETSc outer `FGMRES`
- native `pc_type gamg`
- `pc_gamg_type=agg`
- `pc_gamg_agg_nsmooths=1`
- `pc_gamg_aggressive_coarsening=1`
- `pc_gamg_threshold=0.05`
- `pc_gamg_threshold_scale=0.0`
- `pc_gamg_esteig_ksp_max_it=10`
- `pc_gamg_reuse_interpolation=true`
- `pc_gamg_repartition=true`
- `pc_gamg_process_eq_limit=20`
- `pc_gamg_coarse_eq_limit=10`
- `mg_levels_ksp_type=chebyshev`
- `mg_levels_ksp_max_it=2`
- `mg_levels_pc_type=jacobi`

Contained hard-state artifact:

- `artifacts/gamg_pivot/hard_tangent_gamg_v1/data/run_info.json`

Contained hard-state result:

- setup `10.625 s`
- solve `0.479 s`
- setup+solve `11.104 s`
- outer iterations `2`
- final relative residual `6.064e-2`

Interpretation:

- global GAMG is the first PETSc-native alternative that clearly beats the contained Hypre tangent baseline on setup+solve
- it takes one extra outer iteration, but setup and apply are much cheaper than Hypre on this frozen-state gate

## GAMG Nonlinear Gate

Nonlinear artifact directory:

- `artifacts/gamg_pivot/p4_step1_gamg_tangent_v1`

Run status:

- started
- no `progress.jsonl` or `progress_latest.json` was produced before manual stop
- the run was stopped after it had already exceeded the reused Hypre tangent baseline’s init wall time

Reused Hypre tangent baseline from:

- `artifacts/bddc_short_runs/p4_step1_hypre_current_v1/data/run_info.json`

Baseline init wall context from the existing report:

- Hypre tangent `init wall = 329.926 s`

Observed GAMG nonlinear behavior:

- after more than `7 minutes`, the run still had not reached first continuation progress
- stack sampling showed PETSc inside:
  - `PCSetUp_GAMG`
  - `PCGAMGCreateLevel_GAMG`
  - `MatPtAPSymbolic_MPIAIJ_MPIAIJ`

Interpretation:

- global GAMG is promising on the frozen-state linear gate
- on the real nonlinear path, coarse-hierarchy setup is currently expensive enough to erase that advantage before first progress
- this makes `pc_gamg_reuse_interpolation=true` interesting in principle, but the current repo-side nonlinear driver still rebuilds enough structure that the first real gate is already behind Hypre

## Conclusion

Current decision:

- keep `Hypre + P=tangent` as the production-track solver path
- keep `Hypre + P=regularized` as a correct but non-improving reusable variant
- reject `PCHMG` for this path
- keep global `PCGAMG` as the only PETSc-native branch that still looks worth deeper investigation

Recommended next step if solver exploration continues:

1. do not spend more time on HMG
2. do not run matrix-only `PCMG + GDSW`
3. if continuing the PETSc-native search, focus on making the nonlinear `PCGAMG` setup cheaper or more reusable
4. otherwise stop the solver search here and pivot to a genuine multilevel surrogate / low-order hierarchy branch
