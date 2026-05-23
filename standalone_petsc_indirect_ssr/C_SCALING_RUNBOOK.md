# Pure C Indirect SSR Scaling Runbook

This note records the maintained pure C/PETSc continuation path and the PMG
settings that made the refined L1 indirect SSR run scale on Karolina. It is
intended to be operational: use it to reproduce the current best C run, avoid
known option traps, and tune the next scaling sweep without mixing unrelated
solver variants.

## Maintained Executable

Use:

```bash
standalone_petsc_indirect_ssr/p4_indirect_ssr
```

Do not use the petsc4py runtime in this solver path. The C executable is
self-contained and uses only PETSc plus the local standalone C sources:

- `p4_indirect_ssr.c`
- `assembly.c`, `assembly.h`
- `material_mc.c`, `material_mc.h`, `material_mc_kernel.h`
- `p4_basis.c`, `p4_basis.h`
- `data/adaptive_family_a_l1.msh`

Build on the local workstation:

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  make -C standalone_petsc_indirect_ssr
```

Build on Karolina from the synced repository:

```bash
cd /home/ber0061/slope_stability_petsc4py
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  make -C standalone_petsc_indirect_ssr
```

## Canonical Problem

The refined benchmark problem is:

```text
mesh              data/adaptive_family_a_l1.msh
refine_levels     1
partitioner       parmetis
boundary mode     rollers
omega_max         7e6
lambda_init       1.0
d_lambda_init     0.1
linear_rtol       1e-1
ksp_max_it        200
deflation         true
```

The refined L1 mesh has:

```text
P4 global displacement dofs  4,823,254
P2 global displacement dofs    616,322
P1 global displacement dofs     80,362
```

The solver prints these as parseable diagnostics:

```text
PMG_LEVEL_DOF level=2 degree=4 ...
PMG_LEVEL_DOF level=1 degree=2 ...
PMG_LEVEL_DOF level=0 degree=1 ...
```

## DMPlex And Boundary Path

The mesh is loaded and refined in `CreateMesh()` in `p4_indirect_ssr.c`:

1. `DMPlexCreateFromFile()` reads `data/adaptive_family_a_l1.msh`.
2. `DMPlexSetRefinementUniform()` and `DMRefine()` apply
   `-refine_levels 1`.
3. PETSc partitions the DMPlex, normally with
   `-petscpartitioner_type parmetis`.
4. `DMCreateDS()` installs the P4 displacement discretization.
5. Boundary labels are completed and the standalone boundary marker is built.

The canonical boundary condition is:

```text
-mesh_bc_mode rollers
```

This matches the canonical petsc4py `3d_hetero_slope` setup:

- x-lock side faces constrain `u_x`
- base faces constrain `u_y`
- z-lock side faces constrain `u_z`

The executable reports the selected mode in the startup line:

```text
bc=rollers
```

## Continuation Algorithm Code Path

The high-level execution path is:

```text
main
  ProcessOptions
  CreateMesh
  AssemblyCtxCreate
  AssembleElasticProblem
  LinearSolverInit
  SSRContinuationSolve
```

The continuation driver is `SSRContinuationSolve()`:

1. Seed solve at `lambda_init`.
2. If the seed solve fails, halve `lambda_init` and `d_lambda_init` until it
   succeeds or the minimum step guard is hit.
3. Advance solve at `lambda_init + d_lambda_init`, halving `d_lambda_init` on
   failure.
4. Use the two accepted seed states to start the secant continuation curve.
5. For each continuation step, predict `(u, lambda)` at the next target
   `omega`.
6. Call `IndirectNewtonSolve()` to solve the coupled indirect nonlinear
   problem for displacement and `lambda`.
7. On success, accept the point, append a CSV row, and continue until
   `omega_max`.
8. On failure, restore the deflation basis snapshot, halve `d_omega`, and retry.

The two initialization solves use `FixedLambdaNewtonSolve()`:

```text
R(u, lambda) = F_int(u, lambda) - f_ext
Ktangent du = -R
```

Each fixed-lambda Newton correction is damped by
`FixedLambdaDirectionalDamping()` and then appended to the raw deflation pool
when deflation is enabled.

The continuation Newton solve uses `IndirectNewtonSolve()`:

```text
R(u, lambda) = F_int(u, lambda) - f_ext
omega        = f_ext^T u
K_r          = r*K_elastic + (1-r)*K_tangent
G            = dF_int/dlambda, finite-difference derivative

K_r dW = -G
K_r dV = f_ext - F_int
d_lambda = -(f_ext^T dV) / (f_ext^T dW)
dU       = dV + d_lambda*dW
```

The accepted update is line-searched by `IndirectNewtonLineSearch()` and then
rescaled to satisfy the target `omega = f_ext^T u`.

Important continuation diagnostics:

```text
SSR_INIT
SSR_INIT_NEWTON
SSR_INIT_NEWTON_ACCEPT
SSR_INIT_NEWTON_SUMMARY
SSR_ATTEMPT
SSR_NEWTON
SSR_NEWTON_ACCEPT
SSR_NEWTON_SUMMARY
SSR_STEP
SSR_RESULT
RESULT
```

The rank-0 CSV curve contains:

```text
step,phase,omega,lambda,d_omega,d_lambda,u_max,attempts,
newton_iterations,linear_iterations,line_search_iterations,
rel_residual,rel_correction,step_wall_time,stop_reason
```

## Linear Solver And Deflation Path

Use the maintained explicit deflated FGMRES path:

```text
-deflation true
-deflation_solver fgmres
-deflation_basis_tol 1e-3
```

The default C path is `fgmres`. The `matlab_dfgmres` mode exists for
petsc4py replay parity, not for the maintained scalable Karolina runs.

The important linear code path is:

```text
SolveLinearSystem
  LinearSolverAOrthogonalizeBasis
  DeflationCoarseInitialGuess
  DeflatedFGMRESSolve
    DeflationApplyProjectedPC
      PCApply
      MatMult/projector correction
```

The timer named `deflation_pc_apply_time` is the time spent in `PCApply()` from
inside the projected preconditioner. In the maintained run this is essentially
the PMG shell V-cycle apply time. The projector work after PC apply is timed
separately as `deflation_projector_time`.

Important deflation diagnostics:

```text
DEFLATION_BASIS_ADD
DEFLATION_ORTHO
DEFLATION_CACHE_CONFIG
DEFLATION_COARSE_INITIAL
DEFLATED_FGMRES_INITIAL
DEFLATED_FGMRES_SUMMARY
DEFLATION_TIMING
```

Do not enable the abandoned experimental flags for production scaling runs:

```text
-deflation_projector biorthogonal
-deflation_krylov_persistent true
-deflation_intra_newton_recycle true
-indirect_newton_pair_freeze_matrix true
```

Those were investigation paths and are not part of the maintained baseline.

## PMG Shell V-Cycle Code Path

Use the shell V-cycle PMG backend:

```text
-pc_variant pmg
-pmg_apply_backend shell_vcycle
```

The shell backend is configured by `ConfigurePMGShellVCycle()` and PETSc calls:

```text
PMGShellVCycleSetUp
  PMGShellCreateHierarchy
  PMGShellUpdateOperators

PMGShellVCycleApply
```

The hierarchy is same-mesh p-multigrid:

```text
P4 -> P2 -> P1
```

All levels use DMPlex DMs on the same mesh topology:

- P4 is the fine displacement DM from `CreateMesh()`.
- P2 and P1 DMs are built by `CreateSameMeshLevelDM()`.
- Transfers are explicit interpolation matrices from
  `BuildInterpolationMatrixWithLayouts()`.
- Operators are Galerkin:

```text
A2 = P42^T A4 P42
A1 = P21^T A2 P21
```

The shell backend then redistributes P2 and P1 matrices/vectors onto active
coarse layouts:

```text
PMGShellRedistributeActiveMatrix
```

The V-cycle stages are:

1. Fine pre-smooth on P4.
2. Fine residual.
3. Restrict P4 residual to active-layout P2.
4. P2 smooth.
5. Restrict P2 residual to active-layout P1.
6. P1 coarse solve.
7. Prolong/add P1 correction to P2.
8. P2 post-smooth.
9. Prolong/add P2 correction to P4.
10. Fine post-smooth.

Important shell diagnostics:

```text
PMG_BACKEND backend=shell_vcycle
PMG_SHELL_LEVEL level=2 degree=4 active_ranks=...
PMG_SHELL_LEVEL level=1 degree=2 active_ranks=...
PMG_SHELL_LEVEL level=0 degree=1 active_ranks=...
PMG_LEVEL_SOLVER level=2 ...
PMG_LEVEL_SOLVER level=1 ...
PMG_LEVEL_SOLVER level=0 ...
PMG_SHELL_OPERATOR_UPDATE level=1 ...
PMG_SHELL_OPERATOR_UPDATE level=0 ...
PMG_SHELL_APPLY_SUMMARY ...
```

PETSc stages to inspect in `-log_view`:

```text
pmg_shell_fine_smooth
pmg_shell_residual
pmg_shell_transfer
pmg_shell_p2
pmg_shell_p1
```

## Baseline Option File

The maintained option file is:

```text
standalone_petsc_indirect_ssr/options/pmg_shell_split_smoother.opts
```

Current contents that matter:

```text
-pc_variant pmg
-pmg_apply_backend shell_vcycle
-ksp_type fgmres
-ksp_norm_type unpreconditioned
-dm_plex_partition_balance true

-pmg_shell_p2_active_ranks 64
-pmg_shell_p1_active_ranks 32
-pmg_shell_subcomm_type interlaced

-pmg_smoother_ksp_type chebyshev
-pmg_smoother_pc_type jacobi
-pmg_smoother_max_it 2
-pmg_shell_fine_ksp_max_it 5
-pmg_shell_p2_ksp_max_it 10

-pmg_coarse_telescope_ksp_type fgmres
-pmg_coarse_telescope_ksp_rtol 1e-3
-pmg_coarse_telescope_ksp_max_it 5
-pmg_coarse_telescope_pc_type gamg

-deflation true
-linear_rtol 1.0e-1
-ksp_max_it 200
```

The option file default `P2=64, P1=32` is good for 1 and 2 full Karolina
nodes. For 4 nodes it should be overridden to `P2=128, P1=64`.

## Shell P1 Coarse Solver Footgun

This is the most important option trap.

For the shell V-cycle path, this option is not the P1 coarse-solve control:

```text
-pmg_coarse_redundant_group_size
```

That older option belongs to the PCMG path. It does not make the shell V-cycle
P1 solve redundant.

For shell V-cycle P1 redundancy, use the `pmg_shell_p1_` prefix:

```text
-pmg_shell_p1_pc_type redundant
-pmg_shell_p1_pc_redundant_number 1
-pmg_shell_p1_redundant_ksp_type fgmres
-pmg_shell_p1_redundant_ksp_rtol 1e-3
-pmg_shell_p1_redundant_ksp_max_it 5
-pmg_shell_p1_redundant_pc_type gamg
```

The run log must confirm:

```text
PMG_LEVEL_SOLVER level=0 ksp=fgmres pc=redundant ... max_it=5
```

If it prints `pc=gamg`, then the shell P1 redundant wrapper is not active.

## Active-Rank Tuning Rule

The most important scaling lesson is that P2 and P1 active ranks must not stay
fixed as the fine rank count grows.

Bad 4-node run:

```text
fine ranks 512
P2 active  64
P1 active  32
```

This made the transfer stage worse because the V-cycle moved data from more
fine ranks into the same small P2/P1 active layouts.

Better 4-node run:

```text
fine ranks 512
P2 active  128
P1 active  64
```

This reduced transfer cost, P2 smoothing cost, total PMG apply time, and total
linear iterations.

Practical starting table:

| Karolina layout | ranks | P2 active | P1 active | status |
|---|---:|---:|---:|---|
| 1x128 | 128 | 64 | 32 | tested with `gamg` P1 |
| 2x128 | 256 | 64 | 32 | tested with `gamg` P1 |
| 4x128 | 512 | 128 | 64 | tested, best 4-node result |
| 8x128 | 1024 | 256 | 128 | recommended next test |
| 16x128 | 2048 | 256 | 128 | conservative next test |
| 16x128 | 2048 | 512 | 256 | aggressive next test |

General rule for full-node scaling experiments:

```text
P2 active = max(64, ranks / 4)
P1 active = max(32, ranks / 8)
```

Treat this as a starting rule, not a theorem. At high node counts the P1 active
rank count may need a cap because the P1 problem has only about 80k dofs.

## Exact Best 4-Node Command

This is the current best 4-node C run command shape on Karolina:

```bash
cd /home/ber0061/slope_stability_petsc4py/standalone_petsc_indirect_ssr/karolina

PARTITION=qcpu \
TIME_LIMIT=00:20:00 \
LAYOUTS="4:128" \
ENGINES=c \
PROFILES=split \
REFINE_LEVELS=1 \
OMEGA_MAX=7e6 \
LINEAR_RTOL=1e-1 \
KSP_MAX_IT=200 \
PMG_COARSE_MAX_IT=5 \
RUN_ROOT=/mnt/proj1/fta-26-40/slope_stability_petsc4py_ssr_split_cf17478/ssr_4node_scaled_active_YYYYMMDD_HHMMSS \
EXTRA_PETSC_OPTIONS="-pmg_shell_p2_active_ranks 128 -pmg_shell_p1_active_ranks 64 -pmg_shell_p1_pc_type redundant -pmg_shell_p1_pc_redundant_number 1 -pmg_shell_p1_redundant_ksp_type fgmres -pmg_shell_p1_redundant_ksp_rtol 1e-3 -pmg_shell_p1_redundant_ksp_max_it 5 -pmg_shell_p1_redundant_pc_type gamg" \
./submit_omega7_grid.sh
```

The submitted `command.sh` should contain:

```text
srun -n 512 --cpu-bind=cores --distribution=block:block --mpi=pmix_v4
```

For one-node C runs the harness should use OpenMPI `mpiexec`. For multi-node C
runs it should use:

```text
srun --mpi=pmix_v4
```

Plain `srun` without the PMIx option previously launched incorrectly and must
not be used.

## Known Karolina Results

All rows are refined L1, `omega_max=7e6`, `linear_rtol=1e-1`, split shell
profile, deflation on.

| nodes | ranks | P2/P1 active | P1 coarse | wall | continuation | Newton its | linear its | note |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 128 | 64/32 | `gamg` | 1654.84s | 1646.17s | 137 | 1363 | earlier baseline |
| 2 | 256 | 64/32 | `gamg` | 873.90s | 861.77s | 110 | 1302 | earlier baseline |
| 4 | 512 | 64/32 | `redundant(gamg)` | 649.55s | 635.18s | 110 | 1346 | fixed active ranks |
| 4 | 512 | 128/64 | `redundant(gamg)` | 575.57s | 560.54s | 111 | 1144 | best 4-node result |
| 8 | 1024 | 64/32 | `redundant(gamg)` | 491.21s | 472.82s | 111 | 1285 | fixed active ranks |
| 16 | 2048 | 64/32 | `redundant(gamg)` | 601.52s | 574.99s | 118 | 1301 | fixed active ranks regressed |

The 1-node and 2-node rows used `pc=gamg` on the shell P1 level. The 4-node
and larger rows used `pc=redundant`. Do not use these rows as a perfect
solver-identical scaling curve; they are still useful for showing what changed
and where the active-rank tuning helped.

Best 4-node artifact:

```text
/mnt/proj1/fta-26-40/slope_stability_petsc4py_ssr_split_cf17478/ssr_4node_scaled_active_20260523_201405
job 4375096
```

Previous fixed-active 4-node artifact:

```text
/mnt/proj1/fta-26-40/slope_stability_petsc4py_ssr_split_cf17478/ssr_fullnode_4x8x_qcpu_rednode_20260523_190403
job 4374987
```

Earlier 1-node and 2-node artifacts:

```text
/mnt/proj1/fta-26-40/slope_stability_petsc4py_ssr_split_7a50af4/ssr_refined_p1max5_mpiexec_20260523_175526
/mnt/proj1/fta-26-40/slope_stability_petsc4py_ssr_split_7a50af4/ssr_refined_p1max5_2n_pmix_20260523_175644
```

## 4-Node Improvement From Active-Rank Scaling

Changing only P2/P1 active ranks from `64/32` to `128/64` on 4 nodes produced:

| metric | old 4-node 64/32 | new 4-node 128/64 | change |
|---|---:|---:|---:|
| wall | 649.55s | 575.57s | -73.98s |
| continuation wall | 635.18s | 560.54s | -74.64s |
| total Newton iterations | 110 | 111 | +1 |
| total linear iterations | 1346 | 1144 | -202 |
| deflation PC apply | 237.58s | 187.52s | -50.06s |
| deflation orthogonalization | 108.69s | 95.81s | -12.88s |
| deflation projector | 31.36s | 28.15s | -3.21s |
| PMG fine smooth | 123.85s | 104.09s | -19.76s |
| PMG P2 smooth | 33.60s | 19.77s | -13.83s |
| PMG restrict | 8.33s | 6.25s | -2.08s |
| PMG prolong | 1.14s | 0.43s | -0.71s |
| PMG coarse solve | 55.81s | 46.03s | -9.77s |
| PMG residual | 15.65s | 12.66s | -2.99s |
| PMG operator update | 58.05s | 45.25s | -12.80s |
| PETSc `pmg_shell_transfer` stage | 91.46s | 61.44s | -30.02s |

This confirms that the fixed active coarse layouts were a scaling limiter.
Increasing active ranks reduced the many-to-few transfer pressure and improved
both runtime and iteration count.

## What To Tune Next

Use one change at a time and always collect `RESULT`, `DEFLATION_TIMING`,
`PMG_SHELL_APPLY_SUMMARY`, PETSc events, and Slurm memory.

Recommended next sweeps:

1. Rerun 1x128 and 2x128 with the same shell P1 redundant options used on
   4-node runs. This gives a solver-identical baseline curve.
2. Run 8x128 with `P2=256, P1=128`.
3. Run 16x128 with `P2=256, P1=128` first.
4. Only then try 16x128 with `P2=512, P1=256`.

Do not tune all of these at once:

- active ranks
- P1 redundant wrapper
- P1 max iterations
- smoother iterations
- deflation solver mode

The current split smoother values are:

```text
fine P4 Chebyshev/Jacobi max_it 5
P2 Chebyshev/Jacobi max_it 10
P1 FGMRES/GAMG max_it 5
```

Earlier local tests showed P1 max_it 10 was not materially better for the
refined continuation, so keep max_it 5 unless a specific hard-step replay says
otherwise.

## Reading Scaling Bottlenecks

Use this interpretation of timers:

- `deflation_pc_apply_time`: total `PCApply()` time inside projected FGMRES.
  In the maintained path this means PMG shell V-cycle apply time.
- `deflation_projector_time`: extra MatMult/dot/AXPY work to project the PC
  result against the A-orthonormal deflation basis.
- `pmg_shell_transfer`: PETSc stage covering transfer-heavy work. This was the
  clearest non-scaling signal when active P2/P1 ranks were too small.
- `PMG_SHELL_APPLY_SUMMARY coarse_solve`: explicit shell timer for the P1
  coarse solve. Prefer this over trying to infer coarse cost from raw PETSc
  events.
- `PMG_SHELL_APPLY_SUMMARY operator_update`: Galerkin coarse operator update
  and redistribution cost.

For 2-to-4 node analysis, compare each component against `time_2nodes / 2`.
The fixed-active 4-node run had large unsaved time in transfer and operator
update. The `128/64` active-rank run reduced that substantially.

## Collection Commands

After a Karolina campaign:

```bash
cd /home/ber0061/slope_stability_petsc4py/standalone_petsc_indirect_ssr/karolina
/home/ber0061/slope_stability_petsc4py/.venv/bin/python \
  ./collect_omega7_results.py /path/to/run_root
```

The collector writes:

```text
ssr_omega7_results.csv
ssr_omega7_steps.csv
ssr_omega7_petsc_events.csv
```

Also keep these per-case files:

```text
command.sh
job.env
run.log
time.txt
result_line.txt
ssr_result_line.txt
deflation_timing.txt
pmg_shell_summary.txt
diagnostics.txt
sacct.txt
continuation_curve.csv
```

Do not commit generated run directories, logs, CSV curves, object files, or
executables.
