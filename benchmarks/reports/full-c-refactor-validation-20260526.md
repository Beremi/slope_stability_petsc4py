# Full-C Refactor Validation, 2026-05-26

This record validates the modular C-source refactor and the switch back to the
full-C continuation/Newton path as the default `standalone_petsc_ssr`
runner. Runs used unrefined L1, `omega_max=7e6`, `linear_rtol=1e-1`,
`ksp_max_it=200`, ParMETIS, and `configs/petsc/pmg_shell_baseline.opts`.

## Local Results

| ranks | wall | continuation | Newton | linear | wall/linear | final lambda | max RSS/rank | avg peak RSS/rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 184.79s | 182.98s | 78 | 732 | 0.2524s | 1.569682 | 550.3 MiB | 505.0 MiB |
| 64 | 236.13s | 234.00s | 87 | 803 | 0.2941s | 1.569243 | 408.8 MiB | 361.6 MiB |

## Stored C Target Comparison

| ranks | target wall | refactor wall | target linear | refactor linear | target max RSS/rank | refactor max RSS/rank |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 146.56s | 184.79s | 719 | 732 | ~544.9 MiB | 550.3 MiB |
| 64 | 196.66s | 236.13s | 803 | 803 | ~374.1 MiB | 408.8 MiB |

The refactor keeps the C iteration profile and memory shape, but this local
validation was slower than the stored target. Since the code split compiles the
same C implementation as one translation unit and the 64-rank linear count
matches the target exactly, treat the time delta as a benchmark warning rather
than a numerical regression until rerun on a quiet machine or Karolina.

## Phase Timings

| ranks | deflation PCApply | deflation orthogonalize | deflation projector | PMG fine smooth | PMG P2 smooth | PMG coarse solve |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 62.50s | 21.24s | 10.82s | 46.14s | 6.03s | 3.59s |
| 64 | 90.27s | 33.75s | 14.77s | 61.98s | 9.44s | 10.12s |

Raw logs and memory samples were generated under `.local/tmp` and are not
committed.
