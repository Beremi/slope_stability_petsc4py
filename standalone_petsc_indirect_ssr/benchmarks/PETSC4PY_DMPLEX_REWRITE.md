# petsc4py DMPlex Rewrite Target

The rewrite target is the maintained pure C indirect SSR path documented in
`C_SCALING_RUNBOOK.md` and captured in `c_l1_unrefined_20260523.*`.  The first
acceptance problem is unrefined L1, `omega_max=7e6`, on 32 and 64 local ranks.

The C free-DOF target is P4 `616322`, P2 `80362`, and P1 `10859`.  A raw
unconstrained one-rank DMPlex Lagrange probe currently reports P4 `625647`,
P2 `82815`, and P1 `11535`; the difference is the boundary-constrained section
that the C path applies with DMPlex `DMAddBoundary`.  The petsc4py rewrite is
not considered DMPlex-parity complete until the constrained/free layouts match
the C counts.

## Opt-In petsc4py Path

Config-driven runs can select:

```toml
[execution]
mechanics_backend = "dmplex_c_compatible"

[linear_solver]
solver_type = "KSPFGMRES"
pc_backend = "pmg_shell"
pmg_profile = "c_split_smoother"
pmg_shell_p2_active_ranks = 64
pmg_shell_p1_active_ranks = 32
pmg_shell_subcomm_type = "interlaced"
pmg_shell_fine_ksp_max_it = 5
pmg_shell_p2_ksp_max_it = 10
pmg_shell_p1_pc_type = "redundant"
pmg_shell_p1_redundant_number = 1
pmg_shell_p1_redundant_ksp_type = "fgmres"
pmg_shell_p1_redundant_ksp_rtol = 1e-3
pmg_shell_p1_redundant_ksp_max_it = 5
pmg_shell_p1_redundant_pc_type = "gamg"
```

This currently installs the same solver/preconditioner knobs and records a
non-fatal DMPlex Lagrange layout probe in `run_info.json`.  The existing
array/CSR assembly and Python manual PMG apply remain the execution path until
the next implementation step makes DMPlex DMs and active-rank subcommunicators
the source of truth.

Current memory status: a one-rank full-L1 smoke of this opt-in path reached
about 18 GiB sampled RSS before it was stopped, because the existing petsc4py
array/CSR path is still doing the heavy lifting.  That is intentionally worse
than the C target and is the main reason the next rewrite step must replace
the mesh/assembly ownership with PETSc DMPlex DMs, or wrap the C hot path.

## Remaining Rewrite Work

The major remaining performance gap is the C shell V-cycle's active-layout
redistribution:

- C redistributes P2/P1 Galerkin operators onto active ranks and runs P2/P1
  KSPs on reduced communicators.
- The current petsc4py manual PMG accepts the same active-rank knobs for
  bookkeeping, but still reports `manualmg_active_layout_status =
  not_yet_redistributed_in_petsc4py`.
- The opt-in `dmplex_c_compatible` backend currently probes DMPlex Lagrange
  layouts and records DOF counts; it is not yet the assembled operator source.
- Matching the C memory and time-per-linear target requires moving this layout
  ownership into petsc4py, or wrapping the C shell V-cycle as a Cython/PETSc
  PC implementation while keeping continuation orchestration in Python.

Use `run_local_ssr_benchmark.py --engines c py --ranks 32 64` after each
implementation step.  The acceptance gate is:

- petsc4py total linear iterations within 10% of C,
- petsc4py wall/linear and continuation/linear within 10% of C,
- petsc4py max/average RSS per rank within 20% of C at first, then tighten.
