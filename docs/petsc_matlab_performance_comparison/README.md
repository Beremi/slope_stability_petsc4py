# PETSc vs MATLAB 3D SSR Performance Comparison

This study package runs a controlled PETSc vs MATLAB benchmark for 3D indirect SSR using P2 elements only.

The study covers:

- homogeneous 3D SSR
- heterogeneous 3D SSR
- heterogeneous 3D seepage SSR on the concave water-level ladder

The workflow is intentionally split into validation, execution, normalization, figure generation, and PDF build.

## Files

- `study.toml`: committed study definition, solver settings, case ordering, and asset mesh variants
- `mesh_manifest.example.toml`: example mapping from logical MATLAB H5 keys to local absolute paths
- `mesh_manifest.local.toml`: local-only mesh mapping used by the runner
- `data/*.csv`: normalized source tables used by the report and figure scripts
- `scripts/`: validators, sequential runner, collectors, and figure/table generation
- `petsc_vs_matlab_3d_ssr_performance_comparison.tex`: LaTeX report source
- `petsc_vs_matlab_3d_ssr_performance_comparison.pdf`: built report PDF tracked in the repository

## Workflow

1. Copy `mesh_manifest.example.toml` to `mesh_manifest.local.toml` and fill in the absolute MATLAB `.h5` paths.
2. Validate the setup:

```bash
./.venv/bin/python docs/petsc_matlab_performance_comparison/scripts/validate_study.py
```

3. Run smoke horizons:

```bash
./.venv/bin/python docs/petsc_matlab_performance_comparison/scripts/run_study.py --phase smoke
```

4. Run the full sequential benchmark:

```bash
./.venv/bin/python docs/petsc_matlab_performance_comparison/scripts/run_study.py --phase main --resume
```

5. Normalize raw outputs into CSV:

```bash
./.venv/bin/python docs/petsc_matlab_performance_comparison/scripts/collect_results.py
```

6. Generate figures and tables, then build the PDF:

```bash
make -C docs/petsc_matlab_performance_comparison pdf
```

## Notes

- PETSc runs use `mpirun -n 8` with `OMP_NUM_THREADS=1`.
- The seepage `water_unit_weight` and `conductivity` entries in `study.toml`
  are MATLAB harness inputs only. PETSc study configs are generated through
  `run_case_from_config` and take hydraulics from `meshes/<asset>/definition.py`.
- MATLAB runs use `OMP_NUM_THREADS=8` and BoomerAMG threads `8`.
- Runs are executed one by one and recorded under `artifacts/petsc_matlab_performance_comparison`.
- The runner stops adding finer levels for a case once the PETSc main runtime exceeds `1000 s`, while still finishing the matching MATLAB run and the hetero appendix run on that level.
- The normalized seepage rows use the verified benchmark-suite
  `run_3D_hetero_seepage_SSR_comsol_capture` artifacts under
  `artifacts/benchmarks/mpi8`. The waterlevels/concave study rows stay out of
  the report until their separate seepage-field parity issue is resolved.

## Seepage Report Source

The seepage figure and table use the benchmarked COMSOL-transition SSR case:

- PETSc: `artifacts/benchmarks/mpi8/run_3D_hetero_seepage_SSR_comsol_capture/petsc`
- MATLAB: `artifacts/benchmarks/mpi8/run_3D_hetero_seepage_SSR_comsol_capture/matlab`

Those histories use `lambda_init = 1.0`, `d_lambda_init = 0.1`,
`d_lambda_min = 1e-5`, `d_lambda_diff_scaled_min = 0.005`, and `tol = 1e-4`.
The previous waterlevels `concave_L2` row is intentionally omitted because the
PETSc and MATLAB seepage pressure fields do not yet match closely enough for a
performance comparison.
