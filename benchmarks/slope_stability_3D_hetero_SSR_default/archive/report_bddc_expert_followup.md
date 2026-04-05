# BDDC Expert Follow-up on Rank-2 P4 Elastic Probe

## Scope

This note records the follow-up experiments run after the expert review that recommended:

1. one last adaptive-deluxe BDDC check with MUMPS-backed Schur work,
2. one last approximate-local alternative using Hypre inside BDDC,
3. and then a pivot if the contained elastic probe still stayed far behind plain Hypre.

All runs below use the same contained problem:

- mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- element type: `P4`
- ranks: `2`
- operator and preconditioner matrix: `A = P = K_elast`
- outer solver: native PETSc `CG`
- norm type: `unpreconditioned`
- target relative residual: `1e-5`

These follow-ups are downstream of the corrected V2 BDDC metadata state:

- node-wise coordinates
- global `MATIS` near-nullspace
- local `SeqAIJ` nullspace and near-nullspace
- no explicit primal vertices by default
- `pc_bddc_symmetric=true`
- `pc_bddc_use_faces=true`

The V2 baseline report remains:

- [report_bddc_param_sweep_v2.md](report_bddc_param_sweep_v2.md)

## Baseline for comparison

| Variant | Iter | Setup [s] | Solve [s] | Runtime [s] | Final rel. residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hypre control | 14 | 16.218 | 25.390 | 105.432 | 7.711e-06 |
| BDDC doc-base + local GAMG | 103 | 47.179 | 80.939 | 192.354 | 8.235e-06 |

Artifacts:

- Hypre control: `artifacts/bddc_param_sweep_v2/linear_screen/hypre_control_v2/rank2_p4_single/data/run_info.json`
- BDDC doc-base: `artifacts/bddc_param_sweep_v2/linear_screen/bddc_gamg_doc_base_v2/rank2_p4_single/data/run_info.json`

## Follow-up 1: adaptive-deluxe BDDC with MUMPS-backed Schur work

Configuration:

- local Dirichlet and Neumann: approximate, `GAMG`
- `pc_bddc_use_deluxe_scaling=true`
- `pc_bddc_adaptive_threshold=2.0`
- `pc_bddc_schur_layers=1`
- `sub_schurs_mat_solver_type=mumps`
- `sub_schurs_schur_mat_type=seqdense`

Result:

| Variant | Iter | Setup [s] | Solve [s] | Runtime [s] | Final rel. residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| BDDC adaptive + deluxe | 51 | 145.463 | 40.468 | 248.375 | 7.637e-06 |

What changed relative to the BDDC doc-base:

- iterations improved strongly: `103 -> 51`
- coarse problem size increased: `9 -> 38`
- local primal constraint count increased to `38`

What did **not** change:

- PETSc still reported `0` candidate faces on both subdomains
- PETSc still reported only `3` candidate edges
- total runtime became much worse because setup dominated the solve

Interpretation:

- adaptive constraints can strengthen the coarse correction even when face detection is absent
- but on this case the added setup cost overwhelms the iteration reduction
- this is not competitive with Hypre on the contained elastic problem

Artifacts:

- run info: `artifacts/bddc_param_sweep_v2_lastchance/rank2_p4_bddc_adaptive_deluxe/data/run_info.json`
- plot: `artifacts/bddc_param_sweep_v2_lastchance/plots/p4_linear_hypre_vs_bddc_adaptive_deluxe.png`

## Follow-up 2: local Hypre inside BDDC

Configuration:

- local Dirichlet and Neumann: approximate, `PCHYPRE`
- local Hypre type: `boomeramg`
- local BoomerAMG options:
  - `coarsen_type=HMIS`
  - `interp_type=ext+i`
  - `agg_nl=1`
  - `strong_threshold=0.75`
  - `no_CF=true`

Result:

| Variant | Iter | Setup [s] | Solve [s] | Runtime [s] | Final rel. residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| BDDC local Hypre | 54 | 19.193 | 74.319 | 155.939 | 7.926e-06 |

Relative to the BDDC doc-base + local GAMG:

- iterations improved: `103 -> 54`
- setup dropped sharply: `47.179 -> 19.193 s`
- runtime improved: `192.354 -> 155.939 s`

Relative to the plain Hypre control:

- iterations are still much higher: `54` vs `14`
- runtime is still worse: `155.939 s` vs `105.432 s`
- the contained elastic gap remains large enough that nonlinear reuse is still premature

Topology/coarse classification signal remained weak:

- candidate vertices: `0`
- candidate edges: `3`
- candidate faces: `0`
- coarse size: `9`

Interpretation:

- the approximate local solver choice matters
- local Hypre is the best BDDC branch tested so far on runtime
- but the main limitation still looks like weak interface/coarse correction, not just poor local smoothing

Artifacts:

- run info: `artifacts/bddc_param_sweep_v2_lastchance/rank2_p4_bddc_local_hypre/data/run_info.json`
- plot: `artifacts/bddc_param_sweep_v2_lastchance/plots/p4_linear_hypre_vs_bddc_local_hypre.png`

## Combined convergence view

- Hypre vs BDDC doc-base vs adaptive-deluxe vs local-Hypre:
  - `artifacts/bddc_param_sweep_v2_lastchance/plots/p4_linear_hypre_vs_bddc_followups.png`

This combined plot matches the scalar metrics:

- Hypre reaches the target in `14` iterations
- both improved BDDC follow-ups reduce the large `103`-iteration baseline substantially
- neither closes the gap enough to justify continuing directly into nonlinear production runs

## Open issues

Two probe-path issues remain visible in the follow-up runs:

1. `pc_bddc_coarse_redundant_pc_type=svd` still appears unused in PETSc stderr in this probe path.
2. `log_view_memory` still triggers `PetscLogHandler of type default has not been started` at finalize in the adaptive-deluxe run, so PETSc event timing collection is still not clean here.

These issues do **not** invalidate the main convergence/timing numbers above because the run summaries were written before finalize, but they do block clean PETSc event profiling.

## Conclusion

The expert review was directionally correct:

- adaptive-deluxe BDDC can strengthen the coarse correction on this problem
- local Hypre is a better approximate local solver than the local GAMG configurations tested so far
- but `Faces = 0` persists, and the contained elastic BDDC quality is still too weak to justify treating BDDC as the production-track path for this heterogeneous `3D + P4` elasticity case

The strongest current conclusion is:

> BDDC is no longer failing because of `MATIS` bring-up or obvious local-solver instability, but it is still not competitive enough on the contained elastic `P4` problem to justify nonlinear production work.

If the BDDC branch continues, the next experiment should be a **topology sanity check on an obviously geometric rank-2 decomposition** to determine whether `Faces = 0` is a real metadata/classification problem or just a property of the current distributed row/overlap presentation to PETSc.

If the goal is the best production path rather than more BDDC diagnosis, the next branch should be:

1. Hypre with explicit `Pmat` reuse experiments
2. Hypre with low-order elastic `Pmat`
3. and only then a separate `PCHPDDM`/GenEO investigation if a new domain-decomposition family is still needed
