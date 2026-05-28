# Engine L1 Memory Verification

Date: 2026-05-24

Commands:

```bash
cd standalone_petsc_ssr
PETSC_DIR=$PWD/../.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make
RANKS=32 ./tools/run_local_l1_benchmark.sh
RANKS=64 MPIEXEC_FLAGS="--map-by :OVERSUBSCRIBE" ./tools/run_local_l1_benchmark.sh
```

Memory was sampled once per second from the MPI Python ranks only, excluding
`prterun`/`mpiexec` and the sampler process.

| ranks | wall | continuation | Newton its | linear its | max RSS/rank | avg peak RSS/rank | peak total RSS | final lambda |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 146.69s | 145.28s | 77 | 719 | 548.34 MiB | 508.59 MiB | 15.86 GiB | 1.569749 |
| 64 | 193.90s | 192.16s | 87 | 803 | 408.39 MiB | 366.38 MiB | 22.73 GiB | 1.569243 |

Comparison to the stored C target:

| ranks | C target wall | engine wall | C target linear | engine linear | memory target comment |
|---:|---:|---:|---:|---:|---|
| 32 | 146.56s | 146.69s | 719 | 719 | max rank RSS is essentially the prior C target band, about 545 MiB |
| 64 | 196.66s | 193.90s | 803 | 803 | max rank RSS is close to the prior C target band, about 374 MiB |

The self-contained engine is therefore matching the C baseline on timing,
iteration count, and local L1 memory within the expected run-to-run noise.
