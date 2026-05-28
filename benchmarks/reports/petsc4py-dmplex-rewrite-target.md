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

## Opt-In petsc4py Paths

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

The performance-matching path is:

```toml
[execution]
mechanics_backend = "dmplex_c_hotpath"
```

This routes the config-driven petsc4py runner into the pure C DMPlex SSR
solver through the `petsc_ssr._petsc_ssr` Cython bridge.  Python still
owns config parsing, benchmark orchestration, memory sampling, and the
standard `run_info.json`/NPZ artifact shape; the mesh, P4 assembly, shell
V-cycle PMG, deflation, and Krylov solves stay in C/PETSc.  The old
`dmplex_c_compatible` backend remains available for the legacy petsc4py
array/CSR comparison.

Local unrefined L1 `omega_max=7e6` smoke after the C-hotpath bridge:

| backend | ranks | wall | continuation | Newton | linear | wall/linear | max RSS/rank | total sampled RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dmplex_c_hotpath` | 32 | 161.52s | 160.09s | 84 | 771 | 0.209s | 653 MiB | 18.29 GiB |
| `dmplex_c_hotpath` | 64 | 203.28s | 201.32s | 89 | 820 | 0.248s | 485 MiB | 27.54 GiB |

These runs were written under
`.local/tmp/ssr_local_benchmark_hotpath_20260524`.  The 64-rank run is a local
oversubscribed check, matching the earlier C benchmark protocol.

## Remaining Rewrite Work

The legacy petsc4py path still has the C shell V-cycle's active-layout
redistribution gap:

- C redistributes P2/P1 Galerkin operators onto active ranks and runs P2/P1
  KSPs on reduced communicators.
- The current petsc4py manual PMG accepts the same active-rank knobs for
  bookkeeping, but still reports `manualmg_active_layout_status =
  not_yet_redistributed_in_petsc4py`.
- The opt-in `dmplex_c_compatible` backend currently probes DMPlex Lagrange
  layouts and records DOF counts; it is not yet the assembled operator source.
- The new `dmplex_c_hotpath` backend closes the performance gap by wrapping
  the complete C hot path.  A future deeper rewrite could expose individual C
  DMPlex/PMG/deflation objects to Python, but the current maintained matching
  backend deliberately keeps those objects opaque for memory parity.

Use `run_local_ssr_benchmark.py --engines c py --ranks 32 64` after each
implementation step.  The acceptance gate is:

- petsc4py total linear iterations within 10% of C,
- petsc4py wall/linear and continuation/linear within 10% of C,
- petsc4py max/average RSS per rank within 20% of C at first, then tighten.
