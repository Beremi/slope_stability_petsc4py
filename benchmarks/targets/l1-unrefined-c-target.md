# C Indirect SSR Unrefined L1 Target, 2026-05-23

This is the local pure C/PETSc target for the petsc4py DMPlex rewrite.  The
most important target metric is time per linear iteration, because the
continuation path can shift Newton and Krylov counts slightly between runs.

## Command Shape

```bash
RANKS=32 OMEGA_MAX=7000000 ./tools/run_local_l1_benchmark.sh
```

## Targets

| ranks | P4 dofs | wall | continuation | Newton | linear | wall/linear | continuation/linear | final lambda |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 616322 | 146.56s | 145.13s | 77 | 719 | 0.2038s | 0.2019s | 1.570195 |
| 64 | 616322 | 196.66s | 195.04s | 87 | 803 | 0.2449s | 0.2429s | 1.569401 |

## Phase Targets

| ranks | deflation PCApply | deflation orthogonalize | deflation projector | PMG fine smooth | PMG P2 smooth | PMG coarse solve | PMG residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 62.01s | 17.67s | 8.18s | 44.23s | 4.11s | 8.27s | 4.19s |
| 64 | 88.36s | 27.06s | 9.00s | 58.22s | 4.89s | 18.21s | 5.35s |

## Memory

Memory was sampled with `run_local_ssr_benchmark.py` from rank command lines
and `/proc/<pid>/status`.  The 32-rank timing row above remains the target;
the memory refresh followed the same profile but took a nearby continuation
path with 763 linear iterations and 158.53s wall time.

| ranks | max RSS/rank | avg HWM/rank | total sampled RSS peak |
|---:|---:|---:|---:|
| 32 | 557940 KiB (544.86 MiB) | 486115 KiB (474.72 MiB) | 15411012 KiB (14.70 GiB) |
| 64 | 383116 KiB (374.14 MiB) | 337442 KiB (329.53 MiB) | 21479660 KiB (20.48 GiB) |

For local 64-rank oversubscription, pass `--oversubscribe` to the benchmark
runner; the 64-rank memory refresh matched the target iteration count.
