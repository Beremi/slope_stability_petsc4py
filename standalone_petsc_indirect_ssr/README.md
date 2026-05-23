# Standalone Pure PETSc Indirect SSR

`p4_indirect_ssr` is a pure C/PETSc standalone port of the classic indirect SSR continuation path. It reuses the P4 tetrahedral mesh, Mohr-Coulomb material assembly, PMG shell V-cycle baseline, and explicit deflation machinery from `standalone_petsc_p4_plasticity`, then solves the indirect nonlinear system for both displacement and strength reduction factor `lambda`.

For the maintained C continuation/scaling path, Karolina command templates,
active-rank PMG tuning rules, and the known 1/2/4/8/16-node results, see
[`C_SCALING_RUNBOOK.md`](C_SCALING_RUNBOOK.md).

For the local C timing target used by the petsc4py DMPlex rewrite, plus the
memory-sampling benchmark harness, see [`benchmarks/`](benchmarks/).

Build:

```bash
PETSC_DIR=$PWD/../.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make
```

Tiny smoke:

```bash
OMP_NUM_THREADS=1 mpiexec -n 1 ./p4_indirect_ssr \
  -mesh data/tiny_box.msh \
  -pc_variant none \
  -ksp_type preonly \
  -pc_type lu \
  -omega_max 10 \
  -continuation_step_max 3 \
  -curve_csv ../.local/tmp/indirect_ssr_tiny_curve.csv
```

Baseline PMG smoke:

```bash
OMP_NUM_THREADS=1 mpiexec -n 4 ./p4_indirect_ssr \
  -options_file options/pmg_shell_vcycle.opts \
  -refine_levels 0 \
  -omega_max 1e5 \
  -continuation_step_max 5 \
  -ksp_converged_reason
```

Linear replay smoke:

```bash
./replay/export_petsc4py_linear_state.sh
PETSC_DIR=$PWD/../.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make
./replay/run_c_replay.sh ../.local/tmp/ssr_linear_replay_state/sample_0000 petsc4py
./replay/run_c_replay.sh ../.local/tmp/ssr_linear_replay_state/sample_0000 baseline
./replay/compare_replay.py ../.local/tmp/ssr_linear_replay_state/sample_0000 \
  ../.local/tmp/c_linear_replay_petsc4py.log \
  ../.local/tmp/c_linear_replay_baseline.log
```

The replay path exports a petsc4py indirect-Newton linear state in free-DOF
layout, including `A_free.mat`, RHS vectors, solutions, deflation basis, DOF
maps, and petsc4py DFGMRES residual histories. The C replay then builds the
same standalone problem for mesh metadata, loads the exported linear system,
and solves it with either the petsc4py-like PMG profile or the maintained C
baseline PMG profile.

Karolina omega=7e6 scaling harness:

```bash
cd karolina
DRY_RUN=1 ./submit_omega7_grid.sh
./submit_omega7_grid.sh
./collect_omega7_results.py runs/ssr_omega7_grid_YYYYMMDD_HHMMSS
```

The default Karolina grid runs C and petsc4py, baseline and petsc4py-like PMG,
on `1:64`, `1:128`, and `2:128` layouts, with per-step CSV/progress data,
PETSc log events when available, `/usr/bin/time -v`, and Slurm `sacct` memory.

The CSV is written by rank 0 with columns:

```text
step,phase,omega,lambda,d_omega,d_lambda,u_max,attempts,newton_iterations,linear_iterations,line_search_iterations,rel_residual,rel_correction,step_wall_time,stop_reason
```

Implemented v1 scope: initialization backoff, fixed-lambda seed solves, secant predictor, legacy omega step controller, indirect Newton with `K_r = r*K_elastic + (1-r)*K_tangent`, `omega_max` stop, and continuation CSV output. The default `rollers` boundary mode matches the canonical `3d_hetero_slope` petsc4py asset: `u_x=0` on the x-lock sides, `u_y=0` on the base, and `u_z=0` on the z-lock sides. Advanced Python-only predictors/controllers and history warm-start modes are intentionally left out of this first standalone port.
