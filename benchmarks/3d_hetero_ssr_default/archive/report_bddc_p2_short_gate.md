# BDDC P2 Short-Run Gate

## Scope

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Case: `P2`, `rows`, `overlap`, `block_metis`
- MPI ranks: `2`
- Continuation limit: `step_max = 10`
- Current control: `hypre_current`
- BDDC candidate tested: `bddc_local_ilu_coarse_gamg_lowsetup`

## What Was Fixed First

The original production BDDC path had a real matrix/vector-layout bug.

- `MatIS` preconditioners were being created with local vector sizes equal to the overlap subdomain size instead of the owned-row vector size.
- That made the explicit BDDC preconditioner incompatible with the outer owned-row solver vectors and caused the earlier `PCApply()` local-size mismatch.
- The fix was to separate:
  - local `MatMult` vector size = owned-row local size
  - local `LGMap` size = overlap subdomain dof count
- The fix is in `src/slope_stability/utils.py` and the BDDC matrix builders in `src/slope_stability/constitutive/problem.py`.

Additional harness/reporting fixes made during this pass:

- `compare_preconditioners.py` now handles `bddc_subdomain_pattern = null` correctly on non-BDDC runs.
- `compare_preconditioners.py` now loads sibling `*.memory_guard.jsonl` files correctly on `reuse_existing=True`.
- Added the overlap-layout MPI regression in `tests/mpi_bddc_overlap_check.py`.

Validation after these fixes:

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_petsc_matis_bddc_helpers.py tests/test_preconditioner_mpi.py tests/test_compare_preconditioners.py tests/test_solver_preconditioner_policies.py`
- Result: `17 passed`

## Short-Run Result

| Variant | Status | Accepted states | Runtime / timeout [s] | First progress | Peak RSS [GiB] | Linear preconditioner [s] | Linear solve [s] |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `hypre_current` | completed | 10 | 90.743 | `init_complete` at `9.812 s` in `progress.jsonl`; progress file observed by wrapper at `25.058 s` | 2.487 | 44.288 | 28.113 |
| `bddc_local_ilu_coarse_gamg_lowsetup` | startup stall | 0 | 60.116 timeout | no `progress.jsonl` before timeout | 5.093 | not reached | not reached |

Artifacts:

- Hypre short-run case: `artifacts/hypre_single_p2_step10_current`
- BDDC timeout case: `artifacts/bddc_single_p2_step10_coarse_gamg_lowsetup_timeout60`
- BDDC backtrace: `artifacts/bddc_single_p2_step10_coarse_gamg_lowsetup_timeout60.startup_stall.bt.txt`

## Diagnosis

The fixed BDDC path is no longer failing on matrix compatibility. The remaining blocker is setup cost inside PETSc BDDC itself.

The timeout-gated BDDC backtrace shows the active rank in:

- `PCSetUp_BDDC`
- `PCBDDCSetUpSolvers`
- `PCBDDCSetUpCorrection`
- `KSPSolve_PREONLY`
- `PCApply_ILU`
- `MatSolve_SeqAIJ_Inode`

So the current bottleneck is BDDC correction/local-solve setup, not the earlier MATIS ownership bug.

## Decision

`bddc_local_ilu_coarse_gamg_lowsetup` fails the P2 gate.

Reasons:

- no continuation progress file by `60.116 s`
- peak RSS already about `2.05x` the Hypre control at the same gate (`5.093 GiB` vs `2.487 GiB`)
- live backtrace confirms the run is still inside BDDC correction setup rather than advancing continuation

Because BDDC did not clear the P2 gate, it was **not promoted to P4 short runs or any full-trajectory run**.

## Practical Conclusion

The important bug is fixed: BDDC now uses the correct owned-row vector layout and the production path no longer crashes with local-size mismatch.

But that was not the last blocker. On the real rank-2 `P2 step_max=10` gate, the BDDC branch is still too expensive in `PCBDDCSetUpCorrection()` to be competitive with current Hypre. The next BDDC work should target PETSc BDDC correction/coarse-space setup cost, not more solver-plumbing changes.
