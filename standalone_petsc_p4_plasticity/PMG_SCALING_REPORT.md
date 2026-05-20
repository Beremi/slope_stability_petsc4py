# PMG Scaling Report

This note records the PMG scaling work for the standalone pure C/PETSc P4
plasticity case. The maintained benchmark state is now:

```text
backend=shell_vcycle
P2 active ranks=64
P1 active ranks=32
subcomm=interlaced
deflation=true
linear_rtol=1e-1
ksp_max_it=200
```

The raw executable defaults remain conservative. The maintained benchmark
profile and the Karolina harness defaults point to the shell V-cycle baseline.

## Baseline Commands

Local smoke:

```bash
cd standalone_petsc_p4_plasticity
OMP_NUM_THREADS=1 mpiexec -n 4 ./p4_plasticity \
  -options_file options/pmg_shell_vcycle.opts \
  -refine_levels 0 \
  -linear_rtol 1e-1 \
  -ksp_max_it 200 \
  -deflation true \
  -ksp_converged_reason
```

Karolina baseline submission:

```bash
cd standalone_petsc_p4_plasticity/karolina
PARTITION=qcpu_exp TIME_LIMIT=00:45:00 ./submit_pmg_scaling.sh
```

That default submits only the `1x128` and `2x128` shell V-cycle baseline jobs.

## Canonical Reference

Previous best maintained campaign:

```text
/mnt/proj1/fta-26-40/slope_stability_petsc4py_p4plasticity_scaling/.../pmg_deflcache_refined_20260519_121919
```

Best maintained result from that campaign:

| profile | nodes | ranks | Newton its | linear its | Newton assembly | Newton solve | wall | final rel | VecScatterEnd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shell P2=64 P1=32 interlaced + deflation | 2x128 | 256 | 17 | 355 | 18.06s | 67.51s | 105.29s | 9.85e-05 | 41.15s |

The `VecScatterEnd` value above is the staged PETSc-log total used in the
manual scaling breakdown. The global PETSc event row is still collected in
`pmg_results.csv`.

## Main Comparisons

| campaign/profile | nodes | linear its | Newton solve | wall | VecScatterEnd | outcome |
|---|---:|---:|---:|---:|---:|---|
| Previous best shell P2=64 P1=32 interlaced + deflation | 2x128 | 355 | 67.51s | 105.29s | 41.15s | Maintained baseline |
| Repartitioned-DM shell comparison | 2x128 | 357 | 71.39s | 117.19s | 41.79s | Valid, slower |
| Active-layout rerun from same comparison batch | 2x128 | 355 | 72.48s | 112.01s | 40.56s | Same iteration count, slower than previous best |

Current repartitioned comparison campaign:

```text
/mnt/proj1/fta-26-40/slope_stability_petsc4py_pmg_repart_48a7223/standalone_petsc_p4_plasticity/karolina/runs/pmg_plasticity_20260520_092754
```

Expanded current-batch rows:

| profile | nodes | Newton its | linear its | Newton solve | wall | final rel | VecScatterEnd |
|---|---:|---:|---:|---:|---:|---:|---:|
| shell active-layout P2=64 P1=32 interlaced + deflation | 1x128 | 18 | 416 | 128.61s | 182.82s | 8.84e-05 | 33.82s |
| shell active-layout P2=64 P1=32 interlaced + deflation | 2x128 | 17 | 355 | 72.48s | 112.01s | 9.85e-05 | 40.56s |
| shell repartitioned-DM P2=64 P1=32 interlaced + deflation | 1x128 | 17 | 355 | 111.78s | 164.53s | 9.96e-05 | 30.09s |
| shell repartitioned-DM P2=64 P1=32 interlaced + deflation | 2x128 | 17 | 357 | 71.39s | 117.19s | 9.80e-05 | 41.79s |

## What Was Tested

| commit | experiment | result |
|---|---|---|
| [`cf12e9f`](https://github.com/Beremi/slope_stability_petsc4py/commit/cf12e9f) | PETSc coarse-DM telescope probe for the PCMG P1 solve | Useful as an investigation, but not kept as a maintained path. The coarse-DM route did not remove the main transfer scatter limit cleanly. |
| [`9326ff2`](https://github.com/Beremi/slope_stability_petsc4py/commit/9326ff2) | Coarse-DM shell reuse fix | Stabilized that experiment enough to compare, but the path was superseded by the explicit shell V-cycle backend. |
| [`bf70bd1`](https://github.com/Beremi/slope_stability_petsc4py/commit/bf70bd1) | Shell V-cycle with active-layout P2/P1 redistribution | The first large improvement over PCMG/P1 telescope; became the baseline family. |
| [`d6e5681`](https://github.com/Beremi/slope_stability_petsc4py/commit/d6e5681) | Deflation cache plus PMG stage diagnostics | Removed redundant deflation `MatMult`s and produced the best maintained `2x128` run: 105.29s wall, 67.51s Newton solve, 355 linear iterations. |
| [`48a7223`](https://github.com/Beremi/slope_stability_petsc4py/commit/48a7223) | Repartitioned-DM/operator shell layout | Numerically valid, but did not improve two-node scaling over the simpler active-layout shell baseline. Removed from maintained code. |

Other outcomes:

- PCMG/P1 telescope remained reliable and is still available as a legacy
  comparison profile, but it is slower than the shell V-cycle baseline.
- Pipe FGMRES did not improve the maintained baseline.
- The native/coarse-DM telescope modes were useful probes but not worth keeping
  in the production solver.
- The `repartitioned_dm` shell layout was valid, but it did not beat the
  active-layout shell V-cycle on the target `2x128` refined run.

## Current Code State

Kept:

- `GAMG`, `BDDC`, `FETI-DP`, and PETSc `PCMG` PMG paths.
- Shell V-cycle backend with active-layout P2/P1 redistribution.
- Deflation cache and PMG shell stage diagnostics.
- Karolina collector parsing for `PCApply`, `KSPSolve`, `MatMult`,
  `VecScatterEnd`, `VecMDot`, `KSPGMRESOrthog`, `MatPtAPNumeric`,
  `MatPtAPSymbolic`, and `PCSetUp`.

Removed from maintained code:

- `-pmg_coarse_telescope_mode native_coarse_dm|custom_shell|coarse_dm`.
- Custom PCMG coarse-DM shell fallback.
- `-pmg_shell_coarse_layout repartitioned_dm`.
- Repartitioned-DM/operator construction and diagnostics.
- Temporary option files for the removed experiments.

Those experiments remain recoverable through the commit links above.

## Remaining Limit

The largest unresolved scaling limit is still communication in transfer-related
scatter. The best maintained shell backend greatly improved total solve time,
but the staged `VecScatterEnd` total is still roughly flat or worse from
`1x128` to `2x128`. The repartitioned-DM experiment did not change that enough
to justify its complexity.

Future work should focus on either a different transfer implementation or a
more fundamental communication restructuring. More telescope tuning inside the
current PCMG shape is unlikely to move the main `VecScatterEnd` limit by itself.
