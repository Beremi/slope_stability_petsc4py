# BDDC P4 Expert Status

## Purpose

This note is the current checkpoint for the distributed `P4` BDDC recovery work in this repo. The branch is no longer in the original failure state. The immediate ask for review is:

- why the now-working distributed `P4` BDDC path still has very high nonlinear solve cost
- what the next PETSc/BDDC tuning steps should be so this becomes viable against the current Hypre path

## Current State in One Sentence

Distributed `P4` BDDC with `P = K_elast` now works on the intended `MATIS + PCBDDC + local GAMG + switch_static` path, and nonlinear `P4` continuation now runs past initialization and multiple accepted steps, but it is still much slower than Hypre because linear iterations are too high.

## Repo Changes Already Made

These fixes are already implemented on the branch:

- default explicit primal vertices removed from the BDDC subdomain pattern
- BDDC coordinate metadata corrected to node-wise at the metadata layer
- `pc_bddc_switch_static` plumbed through config, CLI, solver, and elastic probe paths
- rigid-body near-nullspace attached to the global `MATIS` `Pmat`
- local nullspace and local near-nullspace kept on the local `SeqAIJ`
- elastic probe switched to direct local elastic-value assembly instead of the old SciPy-heavy local CSR path
- stale benchmark-process contamination was removed from the measurement workflow

Main code paths touched:

- `src/slope_stability/fem/distributed_tangent.py`
- `src/slope_stability/constitutive/problem.py`
- `src/slope_stability/utils.py`
- `src/slope_stability/linear/solver.py`
- `src/slope_stability/cli/run_3D_hetero_SSR_capture.py`
- `benchmarks/3d_hetero_ssr_default/archive/probe_bddc_elastic.py`
- `benchmarks/3d_hetero_ssr_default/archive/compare_preconditioners.py`

Validation after these changes:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_petsc_matis_bddc_helpers.py \
  tests/test_probe_bddc_elastic.py \
  tests/test_compare_preconditioners.py \
  tests/test_solver_preconditioner_policies.py \
  tests/test_preconditioner_mpi.py
```

Result: `24 passed in 7.95s`

## What Works Now

### 1. Distributed P4 elastic-only BDDC solve

Contained probe:

- artifact:
  - `artifacts/bddc_elastic_probe/rank2_p4_gamg_switch_static_v1`
- configuration:
  - outer solver: native PETSc `CG`
  - `pc_backend=bddc`
  - `preconditioner_matrix_source=elastic`
  - local Dirichlet/Neumann approximate solves: `preonly + gamg`
  - coarse solve: `preonly + lu`
  - `pc_bddc_switch_static=true`
  - no explicit primal vertices
- result:
  - completed
  - runtime: `259.031 s`
  - setup: `60.899 s`
  - solve: `107.859 s`
  - iterations: `129`
  - relative residual: `3.94e-07`
  - explicit primal vertices used: `0`

Control:

- artifact:
  - `artifacts/bddc_elastic_probe/rank2_p4_hypre_single_v4`
- result:
  - completed
  - runtime: `145.874 s`
  - setup: `18.741 s`
  - solve: `37.892 s`
  - iterations: `18`

Interpretation:

- the original distributed `P4` BDDC setup failure is fixed for the GAMG-based approximate-local path
- BDDC is still much more expensive than Hypre on this contained elastic solve

### 2. Nonlinear P2 with elastic BDDC `P`

- BDDC:
  - `artifacts/bddc_short_runs/p2_step10_bddc_gamg_elastic_v1`
- Hypre control:
  - `artifacts/bddc_short_runs/p2_step10_hypre_current_v2`

Short result:

- both reached `10` accepted steps
- BDDC runtime: `113.828 s`
- Hypre runtime: `107.708 s`
- final state matched closely

Interpretation:

- the corrected BDDC elastic-`P` continuation path is functionally sound
- elastic `P` reuse is working correctly
- `P2` is no longer the problem case

### 3. Nonlinear P4 short gate, step_max=1

- BDDC:
  - `artifacts/bddc_short_runs/p4_step1_bddc_gamg_elastic_clean_v1`
- Hypre control:
  - `artifacts/bddc_short_runs/p4_step1_hypre_current_v1`

Result:

| Metric | BDDC | Hypre |
| --- | ---: | ---: |
| accepted steps | 3 | 3 |
| runtime [s] | 1004.189 | 514.318 |
| final lambda | 1.160357735 | 1.160360965 |
| final omega | 6244976.157 | 6244975.722 |
| init linear iterations | 479 | 66 |
| step-3 linear iterations | 453 | 66 |

Interpretation:

- this is the first successful nonlinear distributed `P4` BDDC short run on the branch
- the branch is no longer blocked before first progress
- it is still roughly `1.95x` slower than Hypre on this gate

### 4. Nonlinear P4 longer short-run prototype

- artifact:
  - `artifacts/bddc_short_runs/p4_step10_bddc_gamg_elastic_clean_v1`
- status when stopped intentionally:
  - stable through `5` accepted steps
  - no failed attempts
  - no setup crash
  - no startup stall

Partial progression:

| Accepted step | wall [s] | linear solve [s] | linear prec [s] | linear iterations |
| --- | ---: | ---: | ---: | ---: |
| init -> 2 steps | 526.247 | 424.763 | 39.988 | 479 |
| 3 | 478.630 | 406.245 | 39.787 | 453 |
| 4 | 509.809 | 472.918 | 0.220 | 526 |
| 5 | 572.383 | 529.675 | 0.238 | 582 |

Interpretation:

- the branch is stable over multiple nonlinear `P4` steps
- once the elastic `P` is built, later-step preconditioner setup cost becomes negligible
- the dominant problem is rising linear iteration counts and solve time

## What Still Fails or Is Incomplete

### 1. Exact local LU is still not viable on P4

Exact-LU BDDC local solves were kept as a correctness fallback but remain impractical:

- serial `P4` exact LU OOMed in `PCBDDCSetUpLocalSolvers()`
- distributed rank-2 exact LU no longer showed the original crash, but remained too slow to use

### 2. PETSc-outer-FGMRES branches are not yet usable for this explicit-`P` BDDC path

I tried to move off the fully Python-driven outer DFGMRES branch and found two repo bugs:

- `KSPFGMRES_GAMG` with `pc_backend=bddc`
  - not wired for the owned-row local solve API
  - rejects `local_rhs`
  - falls back to a full-system RHS preparation path that mismatches local vector length and `q_mask`
- `KSPFGMRES_MATLAB_GAMG` with `pc_backend=bddc`
  - reaches BDDC setup
  - still feeds `PCBDDC` an `AIJ` operator instead of the explicit `MATIS` `Pmat`
  - PETSc aborts with:
    - `PCBDDC preconditioner requires matrix of type MATIS`

These are real follow-up bugs, but they are not blocking the current working `PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE` path.

## Best Current Diagnosis

The branch is no longer failing because of MATIS construction or BDDC startup. The main remaining issue is now solver quality:

- local GAMG inside BDDC is stable enough to run
- the reused elastic `P` does what it should
- but the outer nonlinear solves still require very high linear iteration counts on `P4`

The strongest evidence for that is:

- elastic-only probe:
  - `129` iterations for BDDC vs `18` for Hypre
- nonlinear `P4 step_max=1`:
  - `479` init linear iterations for BDDC vs `66` for Hypre
  - `453` step-3 linear iterations for BDDC vs `66` for Hypre
- nonlinear `P4 step_max=10` partial:
  - iteration counts rise from `453` to `582` while the branch remains stable

So the current branch is in a “working but not viable yet” state.

## Expert Review Questions

These are the concrete next-step questions where guidance would be most useful:

1. Is the current BDDC constraint selection still too weak for high-order `P4`?
   - current successful path uses no explicit primal vertices
   - PETSc derives `3` candidate edges and a coarse problem size of `9`
   - should we now add corner-only explicit primals or topology-derived adjacency to improve coarse correction quality?

2. What is the best first local-GAMG tuning for BDDC Dirichlet/Neumann solves here?
   - current path uses `preonly + gamg` with default internal settings
   - should we explicitly set GAMG thresholds, smoother type, cycle type, or MG level KSP/PC under the `pc_bddc_dirichlet_` and `pc_bddc_neumann_` prefixes?

3. Is `pc_bddc_use_deluxe_scaling=true` the most promising next contained experiment on this heterogeneous `P4` case?
   - it is currently off
   - the branch is stable enough now that this is a genuine tuning experiment rather than a bring-up fix

4. Should we keep `P = K_elast` for the next cycle, or is the next meaningful jump to make the PETSc-outer-FGMRES branches correctly support explicit `MATIS` `Pmat` so we can test the same BDDC preconditioner path without the Python outer DFGMRES loop?

5. If the current coarse space is too small, what PETSc-side customization would you try first for nodal `H1` elasticity at `P4`?
   - corner-only primals
   - topology-based adjacency
   - both together

## Recommended Next Experiments

If continuing from the current branch, the next contained sequence I would run is:

1. `P4 step_max=1`, same working branch, but with `pc_bddc_use_deluxe_scaling=true`
2. `P4` elastic-only probe with tuned local-GAMG options under Dirichlet/Neumann prefixes
3. if still too many iterations, add an experiment-only corner-only primal path
4. separately, fix explicit-`P` support in `KSPFGMRES_GAMG` and `KSPFGMRES_MATLAB_GAMG`

I would not go to rank-8 or full-trajectory `P4` with BDDC yet. The branch is finally functioning, but it is not yet competitive enough.

