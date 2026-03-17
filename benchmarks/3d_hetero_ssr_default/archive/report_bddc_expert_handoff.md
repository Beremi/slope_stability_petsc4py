# BDDC Expert Handoff

## Goal

This note summarizes the current BDDC work on the production 3D path:

- mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- assembly path: `rows + overlap + block_metis`
- target hard case: 3D displacement `P4`
- current production baseline to beat: Hypre-based path summarized in [report_p2_vs_p4_rank8_final_memfix.md](./report_p2_vs_p4_rank8_final_memfix.md)

The intent is to hand one self-contained technical summary to an expert so they can review:

- what was implemented
- what was tested
- what was fixed
- what still fails
- what is now ruled out
- what the next PETSc-side experiment should be

## Current Status

The BDDC branch made real progress, but it is not production-ready for distributed `P4`.

What works now:

- `P2` elastic-only `MATIS + PCBDDC` probes work.
- `P2` short nonlinear run with `A = K_tangent`, `P = K_elast` works.
- the old solver lifecycle bug that rebuilt the same elastic `P` every Newton solve is fixed.
- the old local `P4` elastic assembly path that stalled in SciPy `csr_sort_indices` was replaced for the BDDC subdomain path.
- rank-1 `P4` native PETSc `CG + PCBDDC` elastic probe now completes successfully.

What still fails:

- rank-2 `P4` with approximate ILU local BDDC solves crashes inside PETSc `PCBDDCSetUpCorrection()`.
- rank-2 `P4` with exact LU local BDDC solves appears stable, but setup remains trapped in local numeric LU long enough to be impractical.
- no distributed `P4` BDDC run has yet cleared the contained elastic gate, so no justified `P4` short nonlinear or full-trajectory BDDC comparison exists.

## Relevant Code Paths

Main source changes relevant to this handoff:

- [solver.py](../../src/slope_stability/linear/solver.py)
- [problem.py](../../src/slope_stability/constitutive/problem.py)
- [distributed_tangent.py](../../src/slope_stability/fem/distributed_tangent.py)
- [utils.py](../../src/slope_stability/utils.py)
- [run_3D_hetero_SSR_capture.py](../../src/slope_stability/cli/run_3D_hetero_SSR_capture.py)
- [run_case_from_config.py](../../src/slope_stability/cli/run_case_from_config.py)
- [config.py](../../src/slope_stability/core/config.py)
- [run_config.py](../../src/slope_stability/core/run_config.py)

Benchmark and probe scripts:

- [compare_preconditioners.py](./compare_preconditioners.py)
- [probe_bddc_elastic.py](./probe_bddc_elastic.py)

Relevant tests:

- [test_solver_preconditioner_policies.py](../../tests/test_solver_preconditioner_policies.py)
- [test_petsc_matis_bddc_helpers.py](../../tests/test_petsc_matis_bddc_helpers.py)
- [test_preconditioner_mpi.py](../../tests/test_preconditioner_mpi.py)
- [test_probe_bddc_elastic.py](../../tests/test_probe_bddc_elastic.py)
- [test_compare_preconditioners.py](../../tests/test_compare_preconditioners.py)
- [mpi_bddc_overlap_check.py](../../tests/mpi_bddc_overlap_check.py)
- [mpi_preconditioner_linear_check.py](../../tests/mpi_preconditioner_linear_check.py)

## Important Fixes Already Applied

### 1. Solver reuse bug on elastic preconditioner

The first nonlinear BDDC prototype used `preconditioner_matrix_source=elastic`, but still rebuilt the same elastic `P` every Newton solve.

That caused:

- `75` rebuilds
- `0` reuses
- about `325.291 s` of preconditioner setup
- about `393.264 s` total runtime on the rank-2 `P2`, `step_max=10` short run

After treating elastic `P` as static after first build:

- rebuild count dropped to `2`
- reuse count rose to `73`
- setup dropped to about `8.803 s`
- total runtime dropped to about `77.764 s`

This fix matters because it proved the first short nonlinear BDDC bottleneck was lifecycle/reuse logic, not just PETSc options.

Primary summary:

- [report_bddc_short_runs.md](./report_bddc_short_runs.md)

### 2. Local BDDC elastic layout mismatch

The BDDC `MATIS` path had a local-size / coordinate mismatch between owned rows and local overlap-subdomain data. That was corrected in the BDDC helper path and covered by MPI regression tests.

This was necessary before any multi-rank BDDC run could be trusted.

### 3. P4 local elastic assembly path for BDDC

The original `P4` BDDC elastic path built a full local `K_elast` CSR and then projected it onto the local square pattern. In practice, that got stuck in SciPy sparse CSR handling before PETSc BDDC setup became the dominant issue.

That path was replaced in the BDDC subdomain builder:

- `prepare_bddc_subdomain_pattern(...)` now uses strain geometry directly
- local elastic values are assembled directly on the fixed local square pattern
- the old full local `K_elast_local` build/projection path is no longer used for the BDDC subdomain pattern

This was the key change that allowed `P4` to progress past the old Python/SciPy bottleneck.

### 4. Native PETSc outer-KSP probe path

The elastic probe now supports a native PETSc outer solver path:

- `--outer_solver_family native_petsc`
- `--native_ksp_type cg|fgmres|gmres`

That made it possible to separate:

- repo-side matrix/pattern construction issues
- PETSc `PCBDDC` setup/solve behavior

## What Was Tested

## A. P2 elastic-only BDDC probes

These are contained `K_elast` experiments using the production mesh/order/partitioning path but stopping before constitutive/Newton logic.

Primary report:

- [report_bddc_elastic_probe.md](./report_bddc_elastic_probe.md)

### Completed cases

| Case | Status | Runtime [s] | Setup [s] | Solve [s] | Relative residual | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| rank-1 `P2` `bddc_ilu_elastic` single | completed | `25.707` | `0.648` | `2.410` | first artifact did not store it | first contained success |
| rank-2 `P2` `bddc_ilu_elastic` single | completed | `24.522` | `9.959` | `1.710` | `8.69e-09` | working multi-rank elastic BDDC |
| rank-2 `P2` `bddc_ilu_elastic` repeat | completed | `27.974` | `9.994` | `1.743`, `1.686`, `1.683` | `8.69e-09` each | repeated solves stable |
| rank-2 `P2` `bddc_ilu_elastic_deluxe` single | completed | `26.223` | `11.513` | `1.714` | `8.69e-09` | deluxe slower than no-deluxe |
| rank-2 `P2` Hypre control | completed | `15.417` | `1.159` | `1.339` | `4.08e-09` | control |

### Interpretation

- contained elastic BDDC works on `P2`
- repeated elastic solves are stable
- deluxe scaling did not help on `P2`
- exact local LU is structurally valid but already looked expensive at this stage

## B. P2 short nonlinear run with elastic `P`

These use:

- operator `A = current tangent`
- preconditioner `P = K_elast`
- BDDC branch with elastic preconditioner reuse

Primary report:

- [report_bddc_short_runs.md](./report_bddc_short_runs.md)

### Final measured comparison

| Metric | Hypre current | BDDC elastic reuse | BDDC / Hypre |
| --- | ---: | ---: | ---: |
| Runtime [s] | `89.159` | `77.764` | `0.872x` |
| First progress [s] | `20.038` | `30.062` | `1.500x` |
| Final accepted states | `10` | `10` | match |
| Final lambda | `1.638606206` | `1.638603589` | close |
| Final omega | `6872377.551148129` | `6872365.494565467` | close |
| Final Umax | `8.437590906` | `8.448082929` | close |
| Linear total [s] | `63.505` | `50.993` | `0.803x` |
| Attempt preconditioner [s] | `37.652` | `4.464` | `0.119x` |
| Attempt solve [s] | `25.853` | `46.530` | `1.800x` |
| Preconditioner setup total [s] | `43.876` | `8.803` | `0.201x` |
| Peak RSS [GiB] | `2.481` | `5.665` | `2.283x` |
| Rebuild count | `69` | `2` | much lower |
| Reuse count | `0` | `73` | much higher |

### Interpretation

- BDDC is functionally valid on this short nonlinear `P2` case
- BDDC beat Hypre on total runtime and on total linear time
- BDDC still failed the strict memory gate on `P2`

## C. P4 before the direct local elastic-value assembly fix

Historical failure mode:

- rank-1 `P4` elastic BDDC probe stalled before PETSc setup became dominant
- backtrace landed in SciPy sparse CSR sorting:
  - `artifacts/bddc_elastic_probe/rank1_p4_ilu_single.startup.bt.txt`

Interpretation at that stage:

- the old local `P4` elastic assembly / CSR path was too heavy before PETSc BDDC could be evaluated meaningfully

This historical failure is important because it is **not** the current primary blocker anymore.

## D. P4 after the direct local elastic-value assembly fix

Primary report:

- [report_bddc_p4_native_probe.md](./report_bddc_p4_native_probe.md)

### D1. Rank-1 `P4`, native PETSc `CG + PCBDDC`, elastic-only

Artifact paths:

- local run info: `artifacts/bddc_elastic_probe/rank1_p4_native_cg/data/run_info.json`
- local progress log: `artifacts/bddc_elastic_probe/rank1_p4_native_cg/data/progress.jsonl`

Measured result:

- status: completed
- build problem: about `109.5 s`
- elastic `MATIS` build: about `0.8 s`
- `KSP/PC` setup: `10.935 s`
- solve time: `92.827 s`
- total runtime: `214.142 s`
- iterations: `144`
- converged reason: `2`
- relative residual: `8.85e-08`
- local BDDC bytes: about `3.84 GiB`

Interpretation:

- this is the first clean rank-1 `P4` elastic-only BDDC success in this repo
- repo-side `P4` elastic assembly is now good enough to reach and complete PETSc solve in serial

### D2. Rank-2 `P4`, native PETSc `CG + PCBDDC`, approximate ILU local solves

Artifact paths:

- local progress log: `artifacts/bddc_elastic_probe/rank2_p4_native_cg_dbg/data/progress.jsonl`
- local backtrace: `artifacts/bddc_elastic_probe/rank2_p4_native_cg_dbg/rank0_setup.bt.txt`

Observed behavior:

- reaches `elastic_problem_built`
- reaches `elastic_pmat_built`
- then crashes during `KSPSetUp()`

Backtrace top:

- `PCBDDCSetUpCorrection()`
- `KSPSolve_PREONLY()`
- `PCApply_ILU()`
- `MatSolve_SeqAIJ_Inode()`

Interpretation:

- distributed `P4` no longer dies in Python/SciPy first
- the failing branch is now PETSc BDDC setup with approximate ILU local solves

### D3. Rank-2 `P4`, native PETSc `CG + PCBDDC`, exact LU local solves

Artifact paths:

- local progress log: `artifacts/bddc_elastic_probe/rank2_p4_native_cg_exact/data/progress.jsonl`
- local backtrace: `artifacts/bddc_elastic_probe/rank2_p4_native_cg_exact/rank0_setup.bt.txt`

Observed behavior:

- reaches `elastic_problem_built`
- reaches `elastic_pmat_built`
- remains in `KSPSetUp()` for several minutes
- manually stopped after diagnosis

Backtrace top:

- `PCBDDCSetUpLocalSolvers()`
- `PCSetUp_LU()`
- `MatLUFactorNumeric_SeqAIJ_Inode()`

Interpretation:

- exact local solves appear stable
- but current numeric LU setup is too expensive to be a practical distributed baseline

## What Is Now Ruled Out

These were genuine blockers earlier, but current evidence says they are no longer the main issue:

1. `P2` BDDC startup was not fundamentally blocked by PETSc choice alone.
   The major runtime pathology there was repeated rebuilding of a static elastic `P`, and that is fixed.

2. Distributed `P4` is no longer primarily blocked by the old Python/SciPy full local elastic CSR construction path.
   The direct local elastic-value assembly change moved the failure point forward into PETSc BDDC setup.

3. Rank-1 `P4` elastic-only BDDC is not fundamentally broken.
   It now completes with native PETSc `CG + PCBDDC`.

## What Still Looks Wrong

Based on the current evidence, the distributed `P4` problem is now centered on PETSc-side local BDDC solver choice and setup cost:

- approximate ILU local solves: unstable on rank-2 `P4`
- exact LU local solves: apparently stable but too expensive in setup

This means the open problem is not just “BDDC does not work”. It is narrower:

- serial `P4` elastic probe works
- distributed `P4` gets through local matrix build
- the remaining failure is in PETSc local BDDC subsolvers and/or their required metadata/option combination

## Validation That Currently Passes

Latest focused validation command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_petsc_matis_bddc_helpers.py \
  tests/test_probe_bddc_elastic.py \
  tests/test_compare_preconditioners.py
```

This passed with:

- `16 passed`

Additional BDDC-related tests also exist in the tree and were used earlier in the cycle:

- [test_preconditioner_mpi.py](../../tests/test_preconditioner_mpi.py)
- [test_solver_preconditioner_policies.py](../../tests/test_solver_preconditioner_policies.py)

## Recommended Next Experiment For The Expert

The next contained experiment should stay as close as possible to the current successful serial `P4` probe while changing only the local BDDC solver strategy.

Recommended setup:

- keep `outer_solver_family=native_petsc`
- keep `native_ksp_type=cg`
- keep elastic-only probe first with `A = P = K_elast`
- keep `MATIS` preconditioner path
- keep local coordinates / nullspace / near-nullspace metadata
- do **not** go back to the old full local `K_elast` CSR build/projection path
- do **not** start with ILU again as the distributed approximate-local baseline

Most defensible PETSc-side next step:

- use PETSc’s documented approximate-local path next:
  - `pc_bddc_dirichlet_approximate`
  - `pc_bddc_neumann_approximate`
  - local `pc_type=gamg`
  - `pc_bddc_switch_static`

Reason:

- current repo-side `P4` matrix/pattern build is now far enough along to run real serial `P4`
- current distributed failure location is inside PETSc local BDDC setup, not upstream assembly
- ILU already demonstrated instability on distributed `P4`
- exact LU already demonstrated impractically high setup cost on distributed `P4`

## Existing Reports For More Detail

- [report_bddc_elastic_probe.md](./report_bddc_elastic_probe.md)
- [report_bddc_short_runs.md](./report_bddc_short_runs.md)
- [report_bddc_p4_native_probe.md](./report_bddc_p4_native_probe.md)
- [report_bddc_vs_current_full_trajectory.md](./report_bddc_vs_current_full_trajectory.md)
- [report_p4_bddc_prototype_status.md](./report_p4_bddc_prototype_status.md)
- [report_p4_preconditioner_screen_partial.md](./report_p4_preconditioner_screen_partial.md)

## Bottom Line

The branch is worth expert review because it is no longer “BDDC never starts”.

It is now at this more precise state:

- `P2`: working elastic probe, working short nonlinear run, competitive runtime, still high memory
- `P4` serial: working elastic-only native PETSc BDDC solve
- `P4` distributed: local matrix path now gets through build, but PETSc BDDC local-solver choice is still the blocker

That is a much better debugging position than the starting point, but it is still a failing distributed `P4` branch today.
