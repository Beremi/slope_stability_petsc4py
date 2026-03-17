# BDDC Elastic Probe

## Scope

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Path: `rows + overlap + block_metis`
- Preconditioner matrix source: `elastic`
- Goal: prove the `MATIS + PCBDDC` branch on `K_elast` first, then only expand if the contained elastic probe is correct and bounded

## Structural Fixes Applied

- Fixed the BDDC local-size/coordinate mismatch for multi-rank runs by separating owned-vector coordinates from local subdomain coordinates.
- Attached local nullspace and near-nullspace data to the local `SeqAIJ` matrices used inside `MATIS`.
- Preserved a static elastic `P` across solves instead of rebuilding the same BDDC hierarchy every Newton call.

## P2 Elastic Probe Results

| Case | Status | Runtime [s] | Setup [s] | Solve [s] | Relative residual | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Rank-1 `bddc_ilu_elastic` single | completed | 25.707 | 0.648 | 2.410 | not recorded in first artifact | First contained success on production mesh path |
| Rank-2 `bddc_ilu_elastic` single | completed | 24.522 | 9.959 | 1.710 | `8.69e-09` | Working multi-rank elastic BDDC |
| Rank-2 `bddc_ilu_elastic` repeat | completed | 27.974 | 9.994 | `1.743, 1.686, 1.683` | `8.69e-09` each | One setup, repeated solves stable |
| Rank-2 `bddc_ilu_elastic_deluxe` single | completed | 26.223 | 11.513 | 1.714 | `8.69e-09` | Deluxe scaling slower than no-deluxe |
| Rank-2 `hypre_current` single | completed | 15.417 | 1.159 | 1.339 | `4.08e-09` | Control |

Contained-probe interpretation:

- `bddc_ilu_elastic` is correct and reusable on `P2`.
- `bddc_ilu_elastic_deluxe` is strictly worse here: same iteration count and residual, higher setup time.
- Exact local LU is structurally valid but not practical. The rank-2 exact-control backtrace shows rank-local LU setup in PETSc:
  - `artifacts/bddc_elastic_probe/rank2_p2_exact_single_v3.startup.bt.txt`

## P4 Elastic Probe Results

### Rank-2 `bddc_ilu_elastic` single

- No `progress.jsonl` before failure.
- RSS reached about `13.3 GiB` on rank 0 and `13.0 GiB` on rank 1 before abort.
- PETSc ended with `SIGSEGV` on rank 0 before the first probe event.

### Rank-1 `bddc_ilu_elastic` single

- No `progress.jsonl` after about one minute.
- RSS reached about `13.7 GiB`.
- The captured backtrace is not in PETSc BDDC setup. It is in SciPy sparse CSR sorting:
  - `artifacts/bddc_elastic_probe/rank1_p4_ilu_single.startup.bt.txt`

Root-cause interpretation:

- The current `P4` blocker is upstream of PETSc BDDC.
- The local high-order elastic assembly / CSR sorting path is already too heavy before PETSc gets a usable matrix.
- The next `P4` fix is not another BDDC option tweak. It is reducing the Python/SciPy `P4` elastic matrix construction footprint.

## Conclusion

- `P2` elastic BDDC now works correctly and repeatably.
- `P4` elastic BDDC is blocked by local elastic matrix construction cost and memory, not by the BDDC option set.
- Keep `bddc_ilu_elastic` as the working contained prototype and use it only on `P2` until the `P4` elastic assembly path is reduced.
