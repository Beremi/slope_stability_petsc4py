# BDDC P4 Recovery Progress

## Scope

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Path under test: `rows + overlap + block_metis`
- BDDC target:
  - `pc_backend=bddc`
  - `preconditioner_matrix_source=elastic`
  - `pc_bddc_symmetric=true`
  - `pc_bddc_use_vertices=true`
  - `pc_bddc_use_edges=true`
  - `pc_bddc_use_faces=true`
  - `pc_bddc_use_change_of_basis=false`
  - `pc_bddc_use_change_on_faces=false`
  - `pc_bddc_switch_static=true`
  - `pc_bddc_dirichlet_approximate=true`
  - `pc_bddc_dirichlet_pc_type=gamg`
  - `pc_bddc_neumann_approximate=true`
  - `pc_bddc_neumann_pc_type=gamg`
  - `pc_bddc_coarse_pc_type=lu`
- Current-production control: Hypre on the same mesh and solver runner

## Repo-Side Fixes Implemented

The distributed `P4` BDDC branch is no longer failing in the same way as before. These fixes are now on the branch:

- explicit primal vertices are disabled by default in the BDDC subdomain pattern
- BDDC coordinate metadata is node-wise at the metadata layer instead of dof-wise
- `pc_bddc_switch_static` is plumbed through config, CLI, and probe paths
- a rigid-body near-nullspace is attached to the global `MATIS` `Pmat`
- local nullspace / near-nullspace remain attached to the local `SeqAIJ`
- the native elastic probe uses the direct local elastic-value assembly path instead of the old SciPy-heavy path
- the earlier stale-process contamination during P4 timing was removed from the benchmarking workflow

Focused validation after these changes:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_petsc_matis_bddc_helpers.py \
  tests/test_probe_bddc_elastic.py \
  tests/test_compare_preconditioners.py \
  tests/test_solver_preconditioner_policies.py \
  tests/test_preconditioner_mpi.py
```

Result: `24 passed in 7.95s`

## What Now Works

### 1. Distributed P4 elastic-only BDDC solve

This is the contained PETSc-first probe that was the immediate recovery target:

- artifact:
  - `artifacts/bddc_elastic_probe/rank2_p4_gamg_switch_static_v1`
- result:
  - completed
  - runtime: `259.031 s`
  - setup: `60.899 s`
  - solve: `107.859 s`
  - iterations: `129`
  - relative residual: `3.94e-07`
  - explicit primal vertices used: `0`

Matching Hypre elastic control:

- artifact:
  - `artifacts/bddc_elastic_probe/rank2_p4_hypre_single_v4`
- result:
  - completed
  - runtime: `145.874 s`
  - setup: `18.741 s`
  - solve: `37.892 s`
  - iterations: `18`
  - relative residual: `2.94e-07`

Interpretation:

- the distributed `P4` BDDC elastic solve is now functionally working
- the old failure mode inside BDDC local-solver setup is resolved for the GAMG-based approximate-local path
- BDDC is still much more expensive than Hypre on this contained elastic solve

### 2. P2 short nonlinear continuation with elastic BDDC `P`

- BDDC artifact:
  - `artifacts/bddc_short_runs/p2_step10_bddc_gamg_elastic_v1`
- Hypre control:
  - `artifacts/bddc_short_runs/p2_step10_hypre_current_v2`

Results:

| Metric | BDDC GAMG elastic `P` | Hypre current |
| --- | ---: | ---: |
| accepted steps | 10 | 10 |
| runtime [s] | 113.828 | 107.708 |
| final lambda | 1.638606285 | 1.638606206 |
| final omega | 6872366.436 | 6872377.551 |
| init wall [s] | 10.903 | 11.916 |
| last step wall [s] | 15.782 | 24.762 |
| last step linear solve [s] | 12.995 | 9.624 |
| last step linear preconditioner [s] | 0.013 | 10.149 |
| last step linear iterations | 255 | 147 |

Interpretation:

- the corrected BDDC elastic-`P` continuation path is functionally valid on `P2`
- elastic `P` reuse is working
- BDDC is competitive on `P2`, although not clearly better overall

### 3. P4 short nonlinear continuation, first real success

The first clean nonlinear `P4` short run on the corrected branch is:

- artifact:
  - `artifacts/bddc_short_runs/p4_step1_bddc_gamg_elastic_clean_v1`

Matching Hypre control:

- artifact:
  - `artifacts/bddc_short_runs/p4_step1_hypre_current_v1`

Results:

| Metric | BDDC GAMG elastic `P` | Hypre current |
| --- | ---: | ---: |
| accepted steps | 3 | 3 |
| runtime [s] | 1004.189 | 514.318 |
| final lambda | 1.160357735 | 1.160360965 |
| final omega | 6244976.157 | 6244975.722 |
| init wall [s] | 525.269 | 329.926 |
| init linear solve [s] | 424.011 | 87.978 |
| init linear preconditioner [s] | 39.971 | 193.779 |
| step-3 wall [s] | 478.904 | 184.376 |
| step-3 linear solve [s] | 406.560 | 70.080 |
| step-3 linear preconditioner [s] | 39.936 | 90.180 |
| step-3 linear iterations | 453 | 66 |

Interpretation:

- this is the first time the distributed `P4` nonlinear BDDC path completed a short continuation gate instead of failing before first progress
- the branch is now functionally viable on nonlinear `P4`
- it is not yet competitive with Hypre on runtime

## Long P4 Short-Run Prototype

I started the rank-2 `P4 step_max=10` BDDC short-run gate and stopped it once it had already demonstrated stable continuation beyond initialization. The goal of this run was correctness/stability signal, not a final timing number.

- artifact:
  - `artifacts/bddc_short_runs/p4_step10_bddc_gamg_elastic_clean_v1`
- state when stopped:
  - accepted steps reached: `5`
  - no failed attempts recorded
  - no setup crash
  - no startup stall

Partial progression:

| Accepted step | wall [s] | linear solve [s] | linear preconditioner [s] | linear iterations |
| --- | ---: | ---: | ---: | ---: |
| init -> 2 steps | 526.247 | 424.763 | 39.988 | 479 |
| 3 | 478.630 | 406.245 | 39.787 | 453 |
| 4 | 509.809 | 472.918 | 0.220 | 526 |
| 5 | 572.383 | 529.675 | 0.238 | 582 |

Interpretation:

- the continuation path remains numerically stable through multiple accepted `P4` steps
- once the elastic `P` is built, preconditioner setup cost becomes almost irrelevant on later steps
- the dominant remaining cost is linear solve time and rising linear-iteration counts
- this is now a tuning problem, not a bring-up problem

## Secondary Findings

### 1. The old diagnostic-heavy run was artificially worse

An earlier `P4 step_max=1` attempt was polluted by two stale rank-2 `P4` Hypre probe processes still consuming full CPU and memory. Those stale processes were removed before the clean short-run comparison was rerun.

### 2. Two PETSc-outer-FGMRES branches still need explicit-`P` plumbing

During investigation I also tried to move the nonlinear P4 path off the fully Python-driven DFGMRES outer loop.

Findings:

- `KSPFGMRES_GAMG` with `pc_backend=bddc` is not wired for the owned-row local solve path
  - it rejects `local_rhs`
  - then falls back to a full-system RHS preparation path that mismatches the local vector length and `q_mask`
- `KSPFGMRES_MATLAB_GAMG` reaches BDDC setup but still feeds `PCBDDC` an `AIJ` operator matrix instead of the explicit `MATIS` `Pmat`
  - PETSc then aborts with:
    - `PCBDDC preconditioner requires matrix of type MATIS`

These are real repo bugs in the PETSc-outer-FGMRES branches, but they are not the reason the current `PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE` path now works on nonlinear `P4`.

## Current Conclusion

The distributed `P4` BDDC branch is now in a materially better state than before:

- distributed `P4` elastic-only BDDC works
- nonlinear `P4 step_max=1` BDDC works
- nonlinear `P4` remains stable through at least five accepted steps on `step_max=10`

So the original recovery target is achieved in the narrow sense:

- the branch is no longer failing in BDDC setup
- the continuation path is no longer blocked before first progress

The remaining blocker is different:

- BDDC with local GAMG is still too expensive on nonlinear `P4`
- linear iterations are high and increase with continuation steps
- the next work should be BDDC tuning, not more MATIS/metadata bring-up

## Next Recommended Experiments

The next contained tuning candidates are:

1. try `pc_bddc_use_deluxe_scaling=true` now that the branch is stable on heterogeneous `P4`
2. try PETSc-documented local-GAMG smoother tuning under the BDDC Dirichlet/Neumann prefixes
3. fix explicit-`P` support in the `KSPFGMRES_*` PETSc-outer branches so the same BDDC `MATIS` preconditioner can be tested without the fully Python-driven outer DFGMRES path

