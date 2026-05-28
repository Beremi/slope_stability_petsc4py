# Root Benchmark Compatibility Check

This record is generated from the root `benchmarks/*/case.toml` files using:

```bash
PYTHONPATH=standalone_petsc_ssr/src:src \
  .venv/bin/python -m petsc_ssr.runners.run_case_from_config \
  --list-benchmarks \
  --capability-report standalone_petsc_ssr/benchmarks/cases/case_benchmark_capability.csv
```

Current supported C engine path:

- 2D and 3D mechanics SSR
- 2D and 3D mechanics limit-load continuation
- 2D and 3D seepage-only root assets
- 2D and 3D seepage-coupled SSR via a PETSc hydro prepass and C mechanics load coupling
- indirect continuation
- direct continuation as an opt-in C path
- triangular 2D and tetrahedral 3D P1/P2/P4 DMPlex finite elements
- full C assembly/Newton/PMG/deflation/Krylov hot path
- shared continuation/Newton implementation for 2D and 3D
- root TOML translation for mesh path, material regions, boundary/node-set constraints, continuation, Newton, seepage, and linear solver settings
- root-style mechanics outputs: `data/run_info.json`, `data/petsc_run.npz`, `data/final_displacement.petscbin`, `data/final_displacement_points.csv`, `exports/resolved_config.toml`, `exports/final_solution.vtu`

Current limitations:

- direct continuation has local smoke coverage only; production-length root parity is still pending
- advanced root predictors/controllers outside the classic indirect SSR path

| case | supported | reason |
|---|---:|---|
| `SIOPT_LL` | yes | supported 3D indirect LL, P2 |
| `SIOPT_SSR` | yes | supported 3D indirect SSR, P2 |
| `run_2D_homo_SSR_capture` | yes | supported 2D indirect SSR, P2 |
| `run_2D_sloan2013_seepage_capture` | yes | supported 2D seepage PETSc/DMPlex solve, P1 |
| `run_3D_hetero_SSR_capture` | yes | supported 3D indirect SSR, P2 |
| `run_3D_hetero_seepage_SSR_comsol_capture` | yes | supported 3D indirect SSR with seepage coupling, P2 |
| `run_3D_hetero_seepage_capture` | yes | supported 3D seepage PETSc/DMPlex solve, P2 |
| `petsc_ssr_2D_Franz_dam_SSR` | yes | supported 2D indirect SSR with seepage coupling, P2 |
| `petsc_ssr_2D_Kozinec_LL` | yes | supported 2D indirect LL, P2 |
| `petsc_ssr_2D_Kozinec_SSR` | yes | supported 2D indirect SSR, P2 |
| `petsc_ssr_2D_Luzec_SSR` | yes | supported 2D indirect SSR with seepage coupling, P2 |
| `petsc_ssr_2D_homo_LL` | yes | supported 2D indirect LL, P2 |
| `petsc_ssr_3D_hetero_LL` | yes | supported 3D indirect LL, P2 |
| `petsc_ssr_3D_hetero_SSR_default` | yes | supported 3D indirect SSR, P4 |
| `petsc_ssr_3D_homo_LL` | yes | supported 3D indirect LL, P2 |
| `petsc_ssr_3D_homo_SSR` | yes | supported 3D indirect SSR, P2 |
| `petsc_ssr_3D_homo_SSR_default` | yes | supported 3D indirect SSR, P2 |
| `petsc_ssr_3D_homo_seepage_SSR_concave` | yes | supported 3D indirect SSR with seepage coupling, P2 |

The smoke-suite runner used for the current validation is:

```bash
OUT_ROOT=$PWD/standalone_petsc_ssr/.local/tmp/root_smoke_suite_full_port_llfix_retry_20260527 \
  RANKS=4 TIMEOUT_SECONDS=240 CONTINUATION_STEP_MAX=3 \
  LINEAR_RTOL=1e-1 KSP_MAX_IT=100 OMP_NUM_THREADS=1 \
  standalone_petsc_ssr/tools/run_root_smoke_suite.sh
```

All listed root cases passed this smoke suite on 4 ranks.

Direct SSR smoke was also validated on
`benchmarks/cases/run_2D_homo_SSR_capture/case.toml` with
`--continuation-method direct --continuation-step-max 3`: 3 accepted points,
22 Newton iterations, 39 linear iterations, final `lambda=1.1`, and wall
`0.51s` on 4 ranks.
