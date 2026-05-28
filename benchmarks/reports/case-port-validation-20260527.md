# Root Port Validation, 2026-05-27

Validation used the self-contained `standalone_petsc_ssr` PETSc/DMPlex
path from the repository root. Generated logs, curves, VTK, PETSc binaries, and
memory samples remain under `.local/tmp`.

## Fixes Validated

- The 2D mechanics generalization keeps 3D gravity on displacement component
  `y` (`component=1`). A previous local run accidentally used `dim-1`, which
  moved 3D gravity to `z` and broke the first fixed-lambda solve.
- Root mechanics node-set constraints are now exported from the root asset
  `q_mask` into `data/mechanics_bc_nodes.csv` and consumed by the C engine as
  algebraic constraints. This is required for meshes such as
  `3d_hetero_seepage_transition`, where `x_lock` and `z_lock` are Gmsh node
  sets rather than face labels.
- Seepage-coupled SSR now runs as a C/PETSc hydro prepass followed by C/PETSc
  mechanics with pressure-gradient body forces and saturated/unsaturated unit
  weight selection.
- Direct SSR continuation is implemented as an opt-in C path
  (`-continuation_method direct`) using the same fixed-lambda Newton,
  assembly, PMG, deflation, and damping machinery as the indirect
  initialization solves.
- Final mechanics states are copied back to the runner-owned PETSc vector before
  output. The C monolithic SSR path and Python-loop limit-load path both write
  `final_displacement.petscbin`, `final_displacement_points.csv`, and
  `final_solution.vtu`.
- Root-compatible `petsc_run.npz` now contains nonzero final displacement in
  root mesh node order when a root output mesh is available. P4 high-order VTU
  output is linearized to vertex tetra/triangle cells for meshio compatibility
  while retaining high-order point displacement and strain fields.
- Limit-load continuation no longer carries the elastic predictor as an active
  deflation vector into the first continuation Newton. This matches the root
  behavior more closely on `SIOPT_LL`; otherwise the first dW/dV solves stopped
  at zero Krylov iterations and the line search stalled at `t=0`.
- The default engine PMG profile is the recorded robust local baseline:
  shell V-cycle, P2/P1 active layouts `64/32`, interlaced, split smoothers
  `P4=5`, `P2=10`, and P1 `gamg`. P1 redundant coarse solve remains an
  explicit experiment rather than the default because it changed the first
  Newton update on local L1.

## Local Mechanics Scaling

Command shape:

```bash
RANKS=<ranks> OUT=$PWD/standalone_petsc_ssr/.local/tmp/l1_regression_r<ranks>_fixed \
  SKIP_LOAD_CHECK=1 OMP_NUM_THREADS=1 \
  standalone_petsc_ssr/tools/run_local_l1_benchmark.sh
```

For 64 ranks the local launcher required:

```bash
MPIEXEC_FLAGS="--map-by :OVERSUBSCRIBE"
```

| case | ranks | wall | continuation | Newton | linear | final lambda | final residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 P4 indirect SSR | 16 | 309.19s | 307.11s | 85 | 803 | 1.569721 | 1.47e-02 |
| L1 P4 indirect SSR | 32 | 179.97s | 178.50s | 92 | 864 | 1.569160 | 3.31e-03 |
| L1 P4 indirect SSR | 64 | 149.52s | 147.60s | 74 | 695 | 1.570704 | 3.50e-02 |

Current local 16 -> 32 rank speedup is `1.72x` on wall time. The 32-rank
wall/linear cost is `0.208s`; the 16-rank wall/linear cost is `0.385s`.
The 64-rank row is the earlier same-machine target retained for context, not
rerun in this pass.

The first two initialization solves match the recorded baseline again:

| ranks | lambda | Newton | linear | final residual |
|---:|---:|---:|---:|---:|
| 16 | 1.0 | 6 | 20 | 2.4764e-03 |
| 16 | 1.1 | 4 | 16 | 5.2748e-03 |
| 32 | 1.0 | 5 | 14 | 1.1083e-02 |
| 32 | 1.1 | 4 | 16 | 5.4692e-03 |
| 64 | 1.0 | 6 | 21 | 2.2574e-03 |
| 64 | 1.1 | 4 | 16 | 5.4389e-03 |

The 32-rank late continuation branch is still more iteration-heavy than the
best stored 2026-05-24 run (`742` linear iterations), but it is close to the
recent 2026-05-27 pre-fix comparison run (`864` linear iterations). The 64-rank
run is faster than the stored 64-rank target on this workstation.

## Root Benchmark Smokes

Command:

```bash
OUT_ROOT=$PWD/standalone_petsc_ssr/.local/tmp/root_smoke_suite_full_port_llfix_retry_20260527 \
  RANKS=4 TIMEOUT_SECONDS=240 CONTINUATION_STEP_MAX=3 \
  LINEAR_RTOL=1e-1 KSP_MAX_IT=100 OMP_NUM_THREADS=1 \
  standalone_petsc_ssr/tools/run_root_smoke_suite.sh
```

All 18 root `benchmarks/*/case.toml` cases passed. Representative rows:

| root case | ranks | result |
|---|---:|---|
| `SIOPT_LL` | 4 | limit-load converged to step max, 13 linear, lambda `0.396512` |
| `run_2D_sloan2013_seepage_capture` | 4 | seepage converged, 2423 linear, final criterion `1.95e-14` |
| `run_3D_hetero_seepage_capture` | 4 | seepage converged, 142 linear, final criterion `8.61e-17` |
| `run_3D_hetero_seepage_SSR_comsol_capture` | 4 | coupled SSR converged to step max, 44 linear, lambda `1.157680` |
| `petsc_ssr_2D_homo_LL` | 4 | limit-load converged to step max, 31 linear, lambda `0.400859` |
| `petsc_ssr_3D_hetero_LL` | 4 | limit-load converged to step max, 12 linear, lambda `0.986395` |
| `petsc_ssr_2D_Franz_dam_SSR` | 4 | coupled SSR converged to step max, 99 linear, lambda `0.731513` |
| `petsc_ssr_2D_Luzec_SSR` | 4 | coupled SSR converged to step max, 20 linear, lambda `1.265275` |
| `petsc_ssr_3D_hetero_SSR_default` | 4 | P4 mechanics SSR converged to step max, 62 linear, lambda `1.160392` |
| `petsc_ssr_3D_homo_seepage_SSR_concave` | 4 | coupled SSR converged to step max, 44 linear, lambda `1.157680` |

The full pass/fail table is generated under
`.local/tmp/root_smoke_suite_full_port_llfix_retry_20260527/summary.tsv`.

Direct continuation smoke:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=standalone_petsc_ssr/src:src \
  mpiexec -n 4 .venv/bin/python -m petsc_ssr.runners.run_case_from_config \
  benchmarks/cases/run_2D_homo_SSR_capture/case.toml \
  --output-dir standalone_petsc_ssr/.local/tmp/direct_2d_homo_smoke_r4 \
  --continuation-method direct \
  --continuation-step-max 3 --linear-rtol 1e-1 --ksp-max-it 100
```

| case | method | ranks | accepted | Newton | linear | lambda | omega | wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_2D_homo_SSR_capture` | direct | 4 | 3 | 22 | 39 | 1.100000 | 425.765 | 0.51s |

## Remaining Blockers

- Full production-length similarity runs for every root benchmark are still a
  follow-up; this pass validates translation, root-style outputs, and short
  convergence on all cases.
- Direct continuation has only been smoke-tested locally; production-length
  direct SSR parity against the root implementation remains a follow-up.
- The P1 redundant coarse-solve option is available for cluster experiments,
  but it is not the default local parity profile because it worsened the
  first fixed-lambda Newton path.
