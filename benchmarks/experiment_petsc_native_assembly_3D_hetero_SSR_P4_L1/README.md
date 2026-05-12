# PETSc-native tangent assembly P4(L1) experiment

This directory is for the first PETSc-native tangent assembly screen on the
3D heterogeneous SSR P4(L1) case up to `omega_max = 7e6`.

The key switch is:

```toml
[execution]
tangent_matrix_backend = "petsc_aij_element"
```

In this branch `petsc_aij_element` is accepted as the production-facing name,
but the implemented v1 path is a direct PETSc `MPIAIJ` CSR-buffer path. It
avoids SciPy CSR materialization during Newton tangent updates and reuses the
same PETSc matrix handle while the compiled row tangent kernel refreshes only
the numeric values. A separate experimental `petsc_coo` alias remains in code,
but it is not the production-facing path for MPI runs on the local PETSc 3.24
CPU build.

Recommended local comparison:

```bash
RANKS=32 ./benchmarks/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/run_preconditioner_screen.sh
./.venv/bin/python benchmarks/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/summarize_runs.py \
  artifacts/cases/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/latest
```

Start with `owned_csr_pmg_shell_32.toml` and `petsc_aij_pmg_shell_32.toml`.
The other configs are preconditioner probes over the same PETSc-native AIJ
matrix path.
