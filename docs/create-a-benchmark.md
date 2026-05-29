# Create A Benchmark

This is the five-step path for adding a benchmark without weakening the public
model.

1. Choose or create a mesh asset under `meshes/<asset>/`.

   The asset owns geometry, variants, physical names, materials, region
   assignments, boundary supports, Dirichlet and Neumann declarations, seepage
   supports, and curved-boundary metadata.

2. Validate the asset.

   ```bash
   petsc-ssr asset validate 3d_hetero_slope
   ```

3. Generate the benchmark skeleton.

   ```bash
   petsc-ssr benchmark init 3d-new-slope-ssr-p4 \
     --asset 3d_hetero_slope \
     --element P4 \
     --analysis ssr
   ```

   The generator writes `case.toml`, `notebook.toml`, `README.md`, `run.sh`,
   and notebooks unless `--no-notebooks` is passed. Notebook display policy
   belongs in `notebook.toml`, not in the case.

4. Validate and dry-run the case.

   ```bash
   petsc-ssr case validate benchmarks/cases/3d-new-slope-ssr-p4/case.toml
   petsc-ssr case explain benchmarks/cases/3d-new-slope-ssr-p4/case.toml
   petsc-ssr mesh-only benchmarks/cases/3d-new-slope-ssr-p4/case.toml
   petsc-ssr case dry-run benchmarks/cases/3d-new-slope-ssr-p4/case.toml \
     --output .local/tmp/3d-new-slope-ssr-p4-dry-run
   ```

   Keep `case.toml` limited to mathematical choices: problem identity, asset
   selection, element order, physics model, selected profiles, and mathematical
   caps. MPI ranks, wall time, generated paths, solver tuning, partitioning, and
   notebook fields belong to suites, profiles, launchers, or debug flags.

5. Add the case to a suite and target set only when it should be part of a
   repeated benchmark sweep.

   ```bash
   petsc-ssr suite expand benchmarks/suites/local-32-smoke.toml \
     --output .local/runs/local-32-smoke/manifest.json
   petsc-ssr targets compare .local/runs/local-32-smoke benchmarks/targets/local-32
   ```

Check generated benchmark scaffolding before committing changes:

```bash
petsc-ssr benchmark init --check
```

Use `--no-notebooks` in minimal environments that intentionally skip notebook
extras.
