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

## Karolina communication/preconditioner screen

The timing diagnosis from the first Karolina scaling runs is in
`karolina_scaling_diagnosis.md`. The short version: tangent assembly keeps
scaling; the bad 128 -> 256 behavior comes mainly from PMG-shell coarse hypre
and preconditioner apply/setup when ranks cross nodes.

Generate one qexp job per experiment:

```bash
./.venv/bin/python benchmarks/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/prepare_karolina_pc_screen.py \
  --layout 1x128 --layout 2x64 --layout 2x128
```

Submit directly on Karolina by adding `--submit`. The default screen includes:

- `pmg_shell_hypre`
- `pmg_shell_redundant_nodes`
- `hypre_lagged_pmis`
- `gamg_lagged_lowcomm`
- `bddc_ilu`
- `bddc_gamg`

Summarize after jobs finish:

```bash
./.venv/bin/python benchmarks/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/summarize_karolina_pc_screen.py \
  artifacts/cases/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/karolina_pc_screen_YYYYMMDD_HHMMSS \
  --output summary.md
```

For a smaller targeted test, pass `--variant` repeatedly, for example:

```bash
./.venv/bin/python benchmarks/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/prepare_karolina_pc_screen.py \
  --layout 2x64 \
  --variant pmg_shell_hypre \
  --variant pmg_shell_redundant_nodes \
  --submit
```

Use `--step-max N` only for smoke tests; omit it for full `omega_max = 7e6`
production screens.
