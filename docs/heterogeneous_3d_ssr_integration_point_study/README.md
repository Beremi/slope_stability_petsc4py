# Heterogeneous 3D SSR Integration-Point Study

This study compares tetrahedral integration-point counts on the standard heterogeneous 3D SSR benchmark `SSR_hetero_ada_L1`.

The matrix is:

- element order: `P1`, `P2`, `P4`
- `P1`: `q1`, `q4`, `q11`, `q24`, `q45`
- `P2`: `q4`, `q11`, `q24`, `q45`
- `P4`: `q24`, `q45`
- shared benchmark horizon: `omega_final = 6.7e6`
- Newton stop: `absolute_delta_lambda = 1e-4`
- continuation: `d_lambda_diff_scaled_min = 1e-4`
- execution: `mpirun -n 8`, `OMP_NUM_THREADS=1`
- wall-clock guard: `1200 s` for `P1` and `P2`, no wall-clock timeout for `P4`

## Important benchmark note

The standard `SSR_hetero_ada_L1` benchmark reaches `omega` on the order of `6.7e6` only on the indirect continuation path. The earlier direct-study scaffold was therefore not the same problem. This package uses the standard indirect heterogeneous 3D SSR capture path so the continuation curves live on the correct omega scale.

## Files

- `study.toml`: committed study definition and solver settings
- `scripts/run_study.py`: sequential MPI runner for the full element/quadrature matrix
- `scripts/build_report_assets.py`: figure and table generation from CSV only
- `data/*.csv`: normalized study outputs
- `figures/*.pdf`: paper-ready vector plots
- `generated/*.tex`: LaTeX fragments generated from CSV data
- `heterogeneous_3d_ssr_integration_point_study.tex`: report source

## Usage

```bash
PYTHONPATH=src .venv/bin/python docs/heterogeneous_3d_ssr_integration_point_study/scripts/run_study.py
PYTHONPATH=src .venv/bin/python docs/heterogeneous_3d_ssr_integration_point_study/scripts/build_report_assets.py
make -C docs/heterogeneous_3d_ssr_integration_point_study pdf
```

Use `--resume` on the study runner to skip runs that already have a normalized `record.json`.
With the corrected runner, `--resume` only reuses completed matching runs; incomplete or failed artifacts are rerun from clean directories.

## Raw artifacts

Raw logs and capture outputs are written under `artifacts/heterogeneous_3d_ssr_integration_point_study`.
