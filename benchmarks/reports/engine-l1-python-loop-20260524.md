# Python-Loop L1 Verification, 2026-05-24

This record verifies the requested ownership split for
`standalone_petsc_ssr`: the continuation loop and Newton iteration loop
run in Python, while PETSc-owned DMPlex setup, assembly, trial residuals, vector
updates, deflation, PMG, and Krylov solves remain in C/Cython.

Case: unrefined L1, `omega_max=7e6`, maintained shell PMG baseline,
deflation enabled.

| ranks | wall | continuation | Newton its | linear its | wall/linear | final lambda | max RSS/rank | avg peak RSS/rank | peak total RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 175.99s | 174.57s | 92 | 864 | 0.2037s | 1.569160 | 561.97 MiB | 507.13 MiB | 15.81 GiB |
| 64 | 185.53s | 183.64s | 83 | 778 | 0.2385s | 1.569736 | 408.27 MiB | 364.33 MiB | 22.67 GiB |

For comparison, the memory-sampled monolithic C-loop target on the same local
machine was:

| ranks | wall | continuation | Newton its | linear its | wall/linear | final lambda | max RSS/rank | avg peak RSS/rank | peak total RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 146.69s | 145.28s | 77 | 719 | 0.2040s | 1.569749 | 548.34 MiB | 508.59 MiB | 15.86 GiB |
| 64 | 193.90s | 192.16s | 87 | 803 | 0.2415s | 1.569243 | 408.39 MiB | 366.38 MiB | 22.73 GiB |

The 32-rank Python-loop run followed a more expensive late damping path from
continuation step 7 onward. The first six continuation steps match the C-loop
run in iteration counts, then small state differences flip damping choices in
the sensitive near-limit steps. The per-linear-iteration time and memory remain
in the C target band, so the Python loop itself is not a memory problem; the
remaining gap is nonlinear path sensitivity.

The 64-rank oversubscribed run stayed within the C target band and was slightly
faster in total iterations and wall time than the stored C target.

Source runs:

- `.local/tmp/l1_r32_python_loop_20260524_151441`
- `.local/tmp/l1_r64_python_loop_20260524_152116`
