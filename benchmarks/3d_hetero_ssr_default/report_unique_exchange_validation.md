# Unique Exchange Validation

- Date: `2026-03-15`
- Scope: phase-2 communication redesign for owned-row constitutive evaluation
- Tangent kernel default: `rows`
- Constitutive default remains: `overlap`

## What Changed

- Added `constitutive_mode="unique_exchange"` to the owned-row constitutive path in [problem.py](/home/beremi/repos/slope_stability-1/src/slope_stability/constitutive/problem.py).
- Added precomputed exchange metadata to [distributed_tangent.py](/home/beremi/repos/slope_stability-1/src/slope_stability/fem/distributed_tangent.py):
  - `local_overlap_owner_mask`
  - `local_overlap_to_unique_pos`
  - `recv_neighbor_ranks`, `recv_ptr`, `recv_overlap_pos`, `recv_global_ip`
  - `send_neighbor_ranks`, `send_ptr`, `send_unique_pos`, `send_global_ip`
- Runtime exchange uses packed `mpi4py` point-to-point sends/receives. The old `unique_gather` path remains as a fallback/reference.
- Extended [scale_p4.py](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/scale_p4.py) so it can benchmark:
  - tangent kernels: `--kernels legacy rows`
  - constitutive modes: `--constitutive-modes overlap unique_exchange`
  - and report `local_constitutive_comm` separately.

## Validation Run Here

### Automated tests

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_distributed_tangent_rows.py tests/test_tangent_microbench_smoke.py tests/test_owned_constitutive_exchange.py tests/test_owned_constitutive_exchange_mpi.py`
- Result: `12 passed`

### Rank-1 constitutive equivalence

- Test file: [test_owned_constitutive_exchange.py](/home/beremi/repos/slope_stability-1/tests/test_owned_constitutive_exchange.py)
- Compared:
  - `overlap`
  - `unique_gather`
  - `unique_exchange`
- Checked equality of:
  - `_owned_local_S`
  - `_owned_local_DS`
  - `build_F_local()`
  - owned tangent values from `assemble_owned_tangent_values(..., kernel="rows")`
- Tolerance: `rtol=1e-11`, `atol=1e-11`

### MPI constitutive equivalence

- Helper: [mpi_owned_constitutive_exchange_check.py](/home/beremi/repos/slope_stability-1/tests/mpi_owned_constitutive_exchange_check.py)
- Wrapper test: [test_owned_constitutive_exchange_mpi.py](/home/beremi/repos/slope_stability-1/tests/test_owned_constitutive_exchange_mpi.py)
- Mesh: synthetic 3D chain, `P2`, with ownership ranges chosen from element-owner nodes so remote overlap points are guaranteed.

#### 2 ranks

- Owned ranges: `[0, 3)`, `[3, 30)`
- Remote overlap integration points exercised: `33`
- Result:
  - `unique_exchange`: all checks passed
  - `unique_gather`: all checks passed
- Worst reported discrepancy:
  - `S_local_max_abs = 2.3283064365386963e-10`
  - all other reported maxima were `0.0`

#### 4 ranks

- Owned ranges: `[0, 3)`, `[3, 18)`, `[18, 30)`, `[30, 54)`
- Remote overlap integration points exercised: `99`
- Result:
  - `unique_exchange`: all checks passed
  - `unique_gather`: all checks passed
- Worst reported discrepancy:
  - all reported maxima were `0.0`

## Comparison To The Previous Version

- The previous production optimization report was [report_tangent_kernel_validation_vs_previous.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/report_tangent_kernel_validation_vs_previous.md). That report covered the `legacy` vs `rows` tangent-kernel change and showed real local owned-row assembly speedups.
- The previous end-to-end runtime baselines used for context remain:
  - [report_p4_scaling_step2.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/report_p4_scaling_step2.md)
  - [report_p2_vs_p4_rank8_final_guarded80_v2.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/report_p2_vs_p4_rank8_final_guarded80_v2.md)
- Relative to that previous version, the new change is specifically:
  - the unique constitutive path no longer needs runtime global reconstruction of overlap-local constitutive fields via `allgather`
  - it now has an exchange-ready owned-row metadata layer and a `unique_exchange` runtime mode
  - but the solver default stays `overlap` until PETSc scaling confirms the new mode is an end-to-end win

## PETSc Scaling Status

Full phase-1 and phase-2 acceptance runs were not executable in this environment because `petsc4py` is not installed for the available Python interpreters.

What is ready to run on a PETSc-capable machine:

```bash
PYTHONPATH=src python benchmarks/3d_hetero_ssr_default/scale_p4.py \
  --kernels legacy rows \
  --constitutive-modes overlap \
  --ranks 1 2 4 8 \
  --step-max 2
```

```bash
PYTHONPATH=src python benchmarks/3d_hetero_ssr_default/scale_p4.py \
  --kernels rows \
  --constitutive-modes overlap unique_exchange \
  --ranks 1 2 4 8 \
  --step-max 2
```

Expected report outputs from the updated script:

- per-variant summaries under `mode_<constitutive_mode>/kernel_<kernel>/`
- kernel comparison report for a fixed constitutive mode
- constitutive comparison report for a fixed tangent kernel

## Current Recommendation

- Keep `tangent_kernel="rows"` as the production tangent path.
- Keep `constitutive_mode="overlap"` as the default until the PETSc `1/2/4/8` study shows `unique_exchange` is numerically stable and faster end-to-end.
- Keep `unique_gather` and `legacy` only as validation fallbacks for one cycle.
