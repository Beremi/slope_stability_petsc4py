# Memory Mitigations

- Date: `2026-03-15`
- Scope: owned-row P4 memory pressure in the PETSc path

## Implemented

### 1. Early-convergence PETSc cleanup

- Fixed the Newton early-exit path in [newton.py](/home/beremi/repos/slope_stability-1/src/slope_stability/nonlinear/newton.py).
- Before this change, the plain and nested Newton loops could break on the convergence check after building `K_tangent` or `K_r` but before entering the later linear-solve `finally` block.
- The fix explicitly cleans pre-solve iteration matrices on those early-break paths.

### 2. Remove duplicated owned elastic rebuild in the lightweight MPI startup path

- [prepare_owned_tangent_pattern](/home/beremi/repos/slope_stability-1/src/slope_stability/fem/distributed_tangent.py) now accepts prebuilt `elastic_rows`.
- The PETSc runners now pass the already-built overlap elastic rows into pattern preparation instead of recomputing them:
  - [run_3D_hetero_SSR_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_3D_hetero_SSR_capture.py)
  - [run_3D_hetero_seepage_SSR_comsol_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_3D_hetero_seepage_SSR_comsol_capture.py)
  - [run_2D_homo_SSR_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_2D_homo_SSR_capture.py)
  - [run_2D_textmesh_case_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_2D_textmesh_case_capture.py)
- This removes one full extra `assemble_owned_elastic_rows(...)` pass from the startup path.

### 3. Stop allocating `scatter_map` on the default `rows` path

- [prepare_owned_tangent_pattern](/home/beremi/repos/slope_stability-1/src/slope_stability/fem/distributed_tangent.py) now has `include_legacy_scatter`.
- Production runners set `include_legacy_scatter=(tangent_kernel == "legacy")`.
- For the default `rows` kernel, the dense per-element `scatter_map` is now skipped entirely.
- The legacy kernel still works when explicitly requested and now raises clearly if the pattern was prepared without scatter metadata.

### 4. Add owned-pattern memory accounting

- `OwnedTangentPattern.stats` now records:
  - `scatter_bytes`
  - `row_slot_bytes`
  - `overlap_B_bytes`
  - `unique_B_bytes`
  - `dphi_bytes`
  - `legacy_scatter_enabled`
- The 3D hetero SSR runner writes max and sum reductions of these stats into `run_info.json` under `owned_tangent_pattern`.

## Validation

- Added tests in:
  - [test_distributed_tangent_rows.py](/home/beremi/repos/slope_stability-1/tests/test_distributed_tangent_rows.py)
  - [test_newton_cleanup.py](/home/beremi/repos/slope_stability-1/tests/test_newton_cleanup.py)
- Current result:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_distributed_tangent_rows.py \
  tests/test_owned_constitutive_exchange.py \
  tests/test_owned_constitutive_exchange_mpi.py \
  tests/test_newton_cleanup.py
```

- Result: `14 passed`

### Post-patch real-case smoke

- Command:

```bash
PYTHONPATH=src mpiexec -n 1 .venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture \
  --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh \
  --elem_type P4 \
  --step_max 1 \
  --tangent_kernel rows \
  --constitutive_mode overlap \
  --out_dir artifacts/petsc_smoke_rows_overlap_memfix_step1
```

- Output: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/petsc_smoke_rows_overlap_memfix_step1/data/run_info.json)
- Runtime: `627.20 s`
- `build_tangent_local`: `59.58 s`
- `owned_tangent_pattern.stats_max` on rank `1`:
  - `scatter_bytes = 0`
  - `overlap_B_bytes = 1,681,224,196`
  - `row_slot_bytes = 814,743,042`
  - `dphi_bytes = 371,327,040`
  - `legacy_scatter_enabled = 0`
  - `elastic_pattern_reused = 1`

This confirms the new default `rows` path is no longer carrying the dense legacy scatter map. The remaining Python-side footprint is dominated by `overlap_B`, row-slot metadata, and `dphi*`.

## Not Yet Fixed

- The default owned-row path still keeps both:
  - sparse `overlap_B`
  - dense `dphi1/dphi2/dphi3`
- `_assemble_3d` in [assembly.py](/home/beremi/repos/slope_stability-1/src/slope_stability/fem/assembly.py) is still temporary-heavy for P4 overlap assembly.
- `unique_exchange` is now correct, but the actual PETSc scaling run in [report_p4_constitutive_actual.md](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_p4_constitutive_actual.md) showed it is not yet faster than `overlap` on `2/4/8` ranks.

## Practical Effect

- The biggest immediate memory reduction from this patch is that `rows` runs no longer allocate `scatter_map`.
- The next useful measurement is to compare the new `owned_tangent_pattern.stats_*` section in `run_info.json` against the old peak-RSS observations to confirm whether the Python-side overlap metadata or the PETSc hierarchy is dominating the remaining memory peak.
