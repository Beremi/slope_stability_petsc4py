# Indirect SSR Benchmark Targets

This directory stores curated benchmark targets and comparison harnesses for
the petsc4py DMPlex rewrite. Generated logs, PETSc output, continuation CSVs,
and memory samples belong under `.local/tmp` and should not be committed.

Current target:

- [`c_l1_unrefined_20260523.md`](c_l1_unrefined_20260523.md)
- [`c_l1_unrefined_20260523.json`](c_l1_unrefined_20260523.json)
- [`c_l1_unrefined_20260523.csv`](c_l1_unrefined_20260523.csv)
- [`PETSC4PY_DMPLEX_REWRITE.md`](PETSC4PY_DMPLEX_REWRITE.md)
- [`C_VS_PETSC4PY_CURRENT_20260524.md`](C_VS_PETSC4PY_CURRENT_20260524.md)

Refresh the local C target, including memory samples:

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  ./.venv/bin/python standalone_petsc_indirect_ssr/benchmarks/run_local_ssr_benchmark.py \
  --engines c \
  --ranks 32 \
  --out-root .local/tmp/ssr_benchmark_refresh_32

PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  ./.venv/bin/python standalone_petsc_indirect_ssr/benchmarks/run_local_ssr_benchmark.py \
  --engines c \
  --ranks 64 \
  --oversubscribe \
  --out-root .local/tmp/ssr_benchmark_refresh
```

Run C and the petsc4py C-compatible path side by side:

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  ./.venv/bin/python standalone_petsc_indirect_ssr/benchmarks/run_local_ssr_benchmark.py \
  --engines c py \
  --ranks 32 64
```

Add `--oversubscribe` when local MPI slots are fewer than the requested ranks.
