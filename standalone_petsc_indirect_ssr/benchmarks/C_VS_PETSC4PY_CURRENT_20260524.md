# C vs Current petsc4py Indirect SSR, 2026-05-24

This compares the maintained pure C/PETSc indirect SSR path against the current
opt-in petsc4py path:

- `mechanics_backend = dmplex_c_compatible`
- `pc_backend = pmg_shell`
- `pmg_profile = c_split_smoother`
- P4/P2/P1 levels: `616322 / 80362 / 10859` free DOFs
- mesh: unrefined L1 `adaptive_family_a_l1.msh`
- `omega_max = 7e6`, `linear_rtol = 1e-1`, `ksp_max_it = 200`

Important caveat: the current petsc4py path has the C-compatible options and a
DMPlex layout probe, but it still uses the existing petsc4py array/CSR assembly
and manual PMG path.  The run records
`manualmg_active_layout_status = not_yet_redistributed_in_petsc4py`, so it does
not yet have the C shell V-cycle's active P2/P1 layouts.

## Artifacts

| engine | ranks | artifact |
|---|---:|---|
| C | 32 | `.local/tmp/ssr_c_benchmark_refresh32_noover_20260523/c_r32` |
| C | 64 | `.local/tmp/ssr_c_benchmark_refresh2_20260523/c_r64` |
| petsc4py | 32 | `.local/tmp/ssr_py_current_compare_20260524/py_r32` |
| petsc4py | 64 | `.local/tmp/ssr_py_current_compare64_oversub_20260524/py_r64` |

The 64-rank local petsc4py run required `mpiexec --map-by :OVERSUBSCRIBE`
because Open MPI exposes only the 32 physical cores as slots on this
workstation.  That makes the 64-rank wall time pessimistic, but the memory and
phase imbalance are still useful.

## Summary

| engine | ranks | wall | continuation | Newton | linear | line search | wall/linear | final lambda | final rel | max RSS/rank | total RSS peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 32 | 158.53s | 157.13s | 82 | 763 | 213 | 0.208s | 1.569711 | 1.24e-02 | 545 MiB | 14.70 GiB |
| petsc4py | 32 | 411.45s | 411.45s | 96 | 810 | 239 | 0.508s | 1.569716 | 7.87e-03 | 2219 MiB | 65.56 GiB |
| C | 64 | 195.29s | 193.34s | 87 | 803 | 215 | 0.243s | 1.569401 | 1.03e-02 | 374 MiB | 20.48 GiB |
| petsc4py | 64 | 822.58s | 822.58s | 100 | 813 | 253 | 1.012s | 1.569686 | 4.27e-03 | 1838 MiB | 108.65 GiB |

## Ratios, petsc4py / C

| ranks | wall | linear its | Newton its | wall/linear | max RSS/rank | total RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2.60x | 1.06x | 1.17x | 2.44x | 4.07x | 4.46x |
| 64 | 4.21x | 1.01x | 1.15x | 4.16x | 4.91x | 5.30x |

Interpretation: petsc4py is not losing primarily through Krylov iteration
count in this configuration.  It is losing through per-linear-iteration cost
and memory.  The difference gets worse at 64 ranks.

## C-Hotpath petsc4py Backend

The follow-up implementation added `mechanics_backend = dmplex_c_hotpath`,
which calls the maintained pure C DMPlex solver through a Cython bridge while
preserving the config-driven petsc4py benchmark interface and artifact shape.

| engine/backend | ranks | wall | continuation | Newton | linear | wall/linear | final lambda | max RSS/rank | total RSS peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C target | 32 | 158.53s | 157.13s | 82 | 763 | 0.208s | 1.569711 | 545 MiB | 14.70 GiB |
| petsc4py `dmplex_c_hotpath` | 32 | 161.52s | 160.09s | 84 | 771 | 0.209s | 1.569526 | 653 MiB | 18.29 GiB |
| C target | 64 | 195.29s | 193.34s | 87 | 803 | 0.243s | 1.569401 | 374 MiB | 20.48 GiB |
| petsc4py `dmplex_c_hotpath` | 64 | 203.28s | 201.32s | 89 | 820 | 0.248s | 1.569388 | 485 MiB | 27.54 GiB |

The C-hotpath backend meets the timing and iteration target on both local
rank counts.  Max RSS/rank is within about 20-30% of the stored C memory
refresh; total sampled RSS is higher, but far below the legacy petsc4py
array/CSR path.  The 64-rank rows are oversubscribed local checks.

## Main Phase Timings

The C timings are from parseable `DEFLATION_TIMING` and
`PMG_SHELL_APPLY_SUMMARY`.  The petsc4py timings are from `run_info.json`.
They are not identical counters, but they point at the same bottleneck family:
linear algebra and deflation/orthogonalization overhead.

| engine | ranks | linear solve | preconditioner/PCApply | orthogonalization | setup/operator | assembly force/tangent notes |
|---|---:|---:|---:|---:|---:|---|
| C | 32 | n/a | deflation PCApply 65.85s | deflation orth 20.85s | PMG update 3.33s | parsed SSR summaries: assembly 44.75s, solve 98.03s |
| petsc4py | 32 | 116.56s | preconditioner 44.32s | 196.18s | setup 44.32s | `build_F` 4.89s, local tangent 13.05s, force gather 3.32s |
| C | 64 | n/a | deflation PCApply 88.17s | deflation orth 26.60s | PMG update 3.76s | parsed SSR summaries: assembly 50.21s, solve 127.49s |
| petsc4py | 64 | 194.16s | preconditioner 79.87s | 460.74s | setup 79.87s | `build_F` 11.55s, local tangent 13.83s, force gather 9.80s |

## C PMG Breakdown

| ranks | PMG applies | fine smooth | P2 smooth | P1/coarse solve | residual | restrict+prolong | operator update |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 763 | 47.66s | 4.04s | 8.56s | 4.60s | 1.04s | 3.33s |
| 64 | 803 | 57.93s | 5.29s | 18.33s | 5.14s | 1.55s | 3.76s |

The C path uses the active-layout shell V-cycle with redistributed P2/P1
layouts.  The petsc4py manual PMG run accepts the same active-rank knobs, but
does not yet redistribute the level layouts; its recorded active-layout status
is `not_yet_redistributed_in_petsc4py`.

## Continuation Steps

### 32 ranks

| accepted step | engine | omega | lambda | Newton | linear | line search | step wall |
|---:|---|---:|---:|---:|---:|---:|---:|
| init | C | 6.231e6 | 1.100000 | 10 | 36 | 46 | 11.67s |
| init | petsc4py | 6.231e6 | 1.100000 | 9 | 26 | 33 | 14.89s |
| 3 | C | 6.245e6 | 1.160394 | 4 | 26 | 6 | 5.71s |
| 3 | petsc4py | 6.245e6 | 1.160392 | 4 | 29 | 4 | 7.76s |
| 4 | C | 6.273e6 | 1.245937 | 5 | 33 | 7 | 7.35s |
| 4 | petsc4py | 6.273e6 | 1.246038 | 4 | 29 | 4 | 8.02s |
| 5 | C | 6.302e6 | 1.312256 | 3 | 23 | 4 | 4.70s |
| 5 | petsc4py | 6.302e6 | 1.312369 | 3 | 21 | 3 | 5.91s |
| 6 | C | 6.358e6 | 1.417977 | 4 | 27 | 4 | 6.02s |
| 6 | petsc4py | 6.358e6 | 1.418253 | 5 | 34 | 8 | 10.96s |
| 7 | C | 6.415e6 | 1.503149 | 8 | 68 | 17 | 14.18s |
| 7 | petsc4py | 6.415e6 | 1.504626 | 2 | 10 | 2 | 3.62s |
| 8 | C | 6.528e6 | 1.565610 | 22 | 254 | 56 | 49.52s |
| 8 | petsc4py | 6.528e6 | 1.565611 | 39 | 399 | 109 | 239.44s |
| 9 | C | 6.754e6 | 1.568546 | 22 | 273 | 58 | 52.02s |
| 9 | petsc4py | 6.755e6 | 1.568753 | 21 | 186 | 46 | 91.04s |
| 10 | C | 7.000e6 | 1.569711 | 4 | 23 | 15 | 5.95s |
| 10 | petsc4py | 7.000e6 | 1.569716 | 9 | 76 | 18 | 29.55s |

### 64 ranks

| accepted step | engine | omega | lambda | Newton | linear | line search | step wall |
|---:|---|---:|---:|---:|---:|---:|---:|
| init | C | 6.231e6 | 1.100000 | 10 | 36 | 46 | 12.91s |
| init | petsc4py | 6.231e6 | 1.100000 | 10 | 31 | 44 | 31.04s |
| 3 | C | 6.245e6 | 1.160395 | 4 | 26 | 6 | 6.45s |
| 3 | petsc4py | 6.245e6 | 1.160367 | 4 | 29 | 4 | 13.43s |
| 4 | C | 6.273e6 | 1.245943 | 4 | 24 | 4 | 6.38s |
| 4 | petsc4py | 6.273e6 | 1.245981 | 4 | 28 | 4 | 14.10s |
| 5 | C | 6.302e6 | 1.312249 | 3 | 18 | 3 | 4.72s |
| 5 | petsc4py | 6.302e6 | 1.312285 | 3 | 20 | 3 | 9.80s |
| 6 | C | 6.358e6 | 1.417965 | 6 | 47 | 11 | 11.41s |
| 6 | petsc4py | 6.358e6 | 1.418101 | 6 | 43 | 12 | 25.43s |
| 7 | C | 6.415e6 | 1.503133 | 8 | 71 | 16 | 16.50s |
| 7 | petsc4py | 6.415e6 | 1.504423 | 2 | 10 | 2 | 6.30s |
| 8 | C | 6.528e6 | 1.565565 | 25 | 285 | 62 | 68.03s |
| 8 | petsc4py | 6.528e6 | 1.565566 | 40 | 384 | 111 | 483.91s |
| 9 | C | 6.754e6 | 1.569008 | 16 | 224 | 42 | 45.49s |
| 9 | petsc4py | 6.754e6 | 1.568732 | 23 | 198 | 55 | 196.99s |
| 10 | C | 7.000e6 | 1.569401 | 11 | 72 | 25 | 21.44s |
| 10 | petsc4py | 7.000e6 | 1.569686 | 8 | 70 | 16 | 41.06s |

## Conclusion

The current petsc4py path is close in final lambda and total linear iteration
count, but it is not close in cost per iteration or memory:

- 32 ranks: 2.44x slower per linear iteration and 4.07x larger max RSS/rank.
- 64 ranks: 4.16x slower per linear iteration and 4.91x larger max RSS/rank.
- The petsc4py 64-rank local run was oversubscribed, but the 32-rank result
  already shows the same structural problem.
- The biggest petsc4py timing line is orthogonalization: 196s at 32 ranks and
  461s at 64 ranks, compared with C deflation orthogonalization of 21s and
  27s.
- The petsc4py path also keeps substantially more memory resident: 65.6 GiB
  at 32 ranks and 108.7 GiB at 64 ranks, versus 14.7 GiB and 20.5 GiB for C.

The immediate rewrite target is therefore not further Krylov tuning.  It is
making petsc4py use the same DMPlex/free-DOF ownership and the same active
P2/P1 PMG redistribution as the C shell V-cycle, then moving the remaining
deflation projection/orthogonalization hot path into Cython/C if the Python
implementation still misses the C per-linear target.
