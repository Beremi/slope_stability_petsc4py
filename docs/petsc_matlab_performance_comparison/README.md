# PETSc vs MATLAB 3D SSR Performance Comparison

This study package runs a controlled PETSc vs MATLAB benchmark for 3D indirect SSR using P2 elements only.

The study covers:

- homogeneous 3D SSR
- heterogeneous 3D SSR
- heterogeneous 3D seepage SSR on the concave water-level ladder

The workflow is intentionally split into validation, execution, normalization, figure generation, and PDF build.

## Files

- `study.toml`: committed study definition, solver settings, case ordering, and mesh ladders
- `mesh_manifest.example.toml`: example mapping from logical MATLAB H5 keys to local absolute paths, with optional PETSc mesh overrides
- `mesh_manifest.local.toml`: local-only mesh mapping used by the runner
- `data/*.csv`: normalized source tables used by the report and figure scripts
- `scripts/`: validators, sequential runner, collectors, and figure/table generation
- `petsc_vs_matlab_3d_ssr_performance_comparison.tex`: LaTeX report source
- `petsc_vs_matlab_3d_ssr_performance_comparison.pdf`: built report PDF tracked in the repository

## Workflow

1. Copy `mesh_manifest.example.toml` to `mesh_manifest.local.toml` and fill in the absolute MATLAB `.h5` paths.
2. If needed, add `petsc_mesh_override` entries for levels that should run on external meshes instead of the committed PETSc mesh paths.
2. Validate the setup:

```bash
python docs/petsc_matlab_performance_comparison/scripts/validate_study.py
```

3. Run smoke horizons:

```bash
python docs/petsc_matlab_performance_comparison/scripts/run_study.py --phase smoke
```

4. Run the full sequential benchmark:

```bash
python docs/petsc_matlab_performance_comparison/scripts/run_study.py --phase main --resume
```

5. Normalize raw outputs into CSV:

```bash
python docs/petsc_matlab_performance_comparison/scripts/collect_results.py
```

6. Generate figures and tables, then build the PDF:

```bash
make -C docs/petsc_matlab_performance_comparison pdf
```

## Notes

- PETSc runs use `mpirun -n 8` with `OMP_NUM_THREADS=1`.
- MATLAB runs use `OMP_NUM_THREADS=8` and BoomerAMG threads `8`.
- Runs are executed one by one and recorded under `artifacts/petsc_matlab_performance_comparison`.
- The runner stops adding finer levels for a case once the PETSc main runtime exceeds `1000 s`, while still finishing the matching MATLAB run and the hetero appendix run on that level.
