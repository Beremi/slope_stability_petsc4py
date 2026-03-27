# Row-Kernel Validation Against Previous P4 Baselines

- Date: `2026-03-15`
- Repo: `slope_stability-1`
- Focus: owned-row tangent assembly kernel only

## Scope

This validation does **not** rerun the full PETSc SSR workflow, because `petsc4py` is not installed in the current environment. What was validated here is the new owned-row tangent kernel on a realistic `P4` local assembly problem built from the production L1 mesh.

The full-solver comparison below therefore has two layers:

1. **Directly measured here**: `legacy` vs `rows` local tangent assembly cost on a realistic rank-like `P4` local pattern.
2. **Projected from previous reports**: what the old full-run `build_tangent_local` totals would become if the measured local speedup carried over to the full continuation run.

## Previous Baselines

Two existing reports in the repo provide the old-runtime context:

- [report_p4_scaling_step2.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/report_p4_scaling_step2.md)
  - `P4`, `step_max = 2`
  - `8` ranks runtime: `273.377 s`
  - `8` ranks `build_tangent_local`: `26.940 s`
- [report_p2_vs_p4_rank8_final_guarded80_v2.md](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/report_p2_vs_p4_rank8_final_guarded80_v2.md)
  - `P4`, `8` ranks, final-state run
  - runtime: `5908.031 s`
  - `build_tangent_local`: `377.725 s`

Those imply that tangent assembly was only part of end-to-end runtime:

- Step-2 rank-8 baseline: `26.940 / 273.377 = 9.85%`
- Final-state rank-8 baseline: `377.725 / 5908.031 = 6.39%`

That matters: even a large tangent-kernel speedup cannot produce the same large full-run speedup by itself.

## Validation Setup

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- Benchmark harness: [bench_tangent_kernels.py](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/bench_tangent_kernels.py)
- Mode: `virtual-rank`
- Node ordering used for this validation: `xyz`
  - `block_metis` was not available because `pymetis` is not installed here.
- Virtual partition count: `8`
- Selected block: heaviest overlap block (`selector = max_overlap`)
  - selected virtual rank: `3`
  - owned nodes: `26069`
  - overlap nodes: `39369`
  - overlap elements: `3312`
  - owned local rows: `78207`
  - owned CSR nnz: `17079016`

Pattern metadata for the selected block:

- `avg_active_rows_per_overlap_element = 74.41`
- `max_active_rows_per_overlap_element = 105`
- legacy scatter metadata bytes: `146059200`
- row-slot metadata bytes: `105054082`

Measured artifacts:

- `OMP_NUM_THREADS=1`: `artifacts/tangent_virtual_rank8_t1/summary.json`
- `OMP_NUM_THREADS=2`: `artifacts/tangent_virtual_rank8_t2/summary.json`
- `OMP_NUM_THREADS=4`: `artifacts/tangent_virtual_rank8_t4/summary.json`
- `OMP_NUM_THREADS=8`: `artifacts/tangent_virtual_rank8_t8/summary.json`

## Direct Measurements

Median local tangent assembly time on the selected realistic rank-8 block:

| OMP threads | Legacy median [s] | Rows median [s] | Rows speedup vs legacy |
| ---: | ---: | ---: | ---: |
| 1 | 0.967075 | 0.293278 | 3.297x |
| 2 | 0.502421 | 0.166988 | 3.009x |
| 4 | 0.257698 | 0.100206 | 2.572x |
| 8 | 0.145038 | 0.067905 | 2.136x |

Observed OpenMP scaling on the same local problem:

| Kernel | 1T median [s] | 2T speedup | 2T efficiency | 4T speedup | 4T efficiency | 8T speedup | 8T efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 0.967075 | 1.925x | 0.962 | 3.753x | 0.938 | 6.668x | 0.833 |
| Rows | 0.293278 | 1.756x | 0.878 | 2.927x | 0.732 | 4.319x | 0.540 |

Interpretation:

- The new row kernel is faster at every tested thread count.
- The absolute gain on this realistic local pattern is about `2.14x` to `3.30x`.
- Legacy still scales reasonably with threads because it was already OpenMP-parallel over elements.
- Rows gives lower thread-scaling efficiency than legacy here, but still wins on absolute time because it does materially less work per owned row.

## Comparison To Previous Full-Run Baselines

These are **projections**, not direct full-run reruns.

Projected `build_tangent_local` if the old rank-8 P4 runs saw the same local kernel speedup:

| Assumed matching local speedup | Projected step-2 `build_tangent_local` [s] | Projected final-state `build_tangent_local` [s] |
| ---: | ---: | ---: |
| `3.297x` (1T benchmark) | 8.170 | 114.550 |
| `3.009x` (2T benchmark) | 8.954 | 125.544 |
| `2.572x` (4T benchmark) | 10.476 | 146.879 |
| `2.136x` (8T benchmark) | 12.613 | 176.847 |

Projected total runtime if **only** `build_tangent_local` improved and every other cost stayed unchanged:

| Assumed matching local speedup | Projected step-2 rank-8 runtime [s] | Projected final-state rank-8 runtime [s] |
| ---: | ---: | ---: |
| `3.297x` (1T benchmark) | 254.607 | 5644.856 |
| `3.009x` (2T benchmark) | 255.391 | 5655.850 |
| `2.572x` (4T benchmark) | 256.913 | 5677.185 |
| `2.136x` (8T benchmark) | 259.050 | 5707.153 |

That translates to:

- Step-2 rank-8 total runtime improvement of about `5.2%` to `6.9%`
- Final-state rank-8 total runtime improvement of about `3.4%` to `4.5%`

## Conclusions

1. The row-oriented kernel is validated on a realistic `P4` local assembly problem and is consistently faster than the legacy kernel.
2. The directly measured local speedup is significant: roughly `2.1x` to `3.3x`.
3. End-to-end full-run improvement is likely much smaller than the local kernel speedup, because previous reports show `build_tangent_local` was only `6%` to `10%` of total rank-8 runtime.
4. If the goal is major full-solver scaling improvement beyond a few percent, the next bottlenecks are still outside the kernel:
   - Krylov/preconditioner time
   - constitutive communication path
   - other continuation/Newton overheads

## Recommendation

The kernel change is worth keeping: it removes dense local `ke`, removes atomic scatter, reduces metadata, and gives a clear local assembly win on realistic `P4` work.

For the next optimization cycle, the highest-value validation is a real PETSc rerun of:

- `benchmarks/3d_hetero_ssr_default/archive/scale_p4.py --kernels legacy rows`

in an environment with `petsc4py`, so the projected `build_tangent_local` reduction can be confirmed against true rank-`1/2/4/8` continuation timing.
