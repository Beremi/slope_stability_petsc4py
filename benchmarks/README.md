# MATLAB Original Script Benchmarks

Unified case registry.

Each case folder contains at least:
- `case.toml`
- `run.sh`
- `README.md`

Per-case `README.md` files are descriptive only. Generated MATLAB-vs-PETSc comparison output for parity cases
is kept in `archive/report.md`.

The canonical MATLAB-parity benchmark suite is the subset with `[benchmark].suite = true` in `case.toml`.

Run the full parity suite:

```bash
./.venv/bin/python -m slope_stability.cli.run_benchmark_suite
```

Run any single case from its folder with `./run.sh`.

## MATLAB-Parity Benchmarks

| Case | Title | Kind | Status | MATLAB [s] | PETSc [s] | Parity summary | README | Report | Run |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `run_2D_homo_SSR_capture` | 2D homogeneous SSR | continuation | done | 9.270 | 8.286 | `steps 14/14`, `lambda 6.74e-06`, `omega 3.05e-05` | [README](run_2D_homo_SSR_capture/README.md) | [report](run_2D_homo_SSR_capture/archive/report.md) | [run.sh](run_2D_homo_SSR_capture/run.sh) |
| `run_2D_sloan2013_seepage_capture` | 2D Sloan2013 seepage | seepage | done | 0.407 | 1.213 | `pw 3.33e-15`, `grad 1.82e-14`, `sat 0` | [README](run_2D_sloan2013_seepage_capture/README.md) | [report](run_2D_sloan2013_seepage_capture/archive/report.md) | [run.sh](run_2D_sloan2013_seepage_capture/run.sh) |
| `run_3D_hetero_SSR_capture` | 3D heterogeneous SSR | continuation | done | 264.817 | 135.706 | `steps 14/14`, `lambda 1.29e-05`, `omega 5.79e-06` | [README](run_3D_hetero_SSR_capture/README.md) | [report](run_3D_hetero_SSR_capture/archive/report.md) | [run.sh](run_3D_hetero_SSR_capture/run.sh) |
| `run_3D_hetero_seepage_SSR_comsol_capture` | 3D heterogeneous seepage SSR COMSOL | continuation | done | 2300.903 | 1217.168 | `steps 34/35`, `lambda 2.14e-05`, `omega 2.89e-04` | [README](run_3D_hetero_seepage_SSR_comsol_capture/README.md) | [report](run_3D_hetero_seepage_SSR_comsol_capture/archive/report.md) | [run.sh](run_3D_hetero_seepage_SSR_comsol_capture/run.sh) |
| `run_3D_hetero_seepage_capture` | 3D heterogeneous seepage | seepage | done | 15.100 | 16.017 | `pw 1.69e-16`, `grad 4.20e-15`, `sat 0` | [README](run_3D_hetero_seepage_capture/README.md) | [report](run_3D_hetero_seepage_capture/archive/report.md) | [run.sh](run_3D_hetero_seepage_capture/run.sh) |

## Additional Runnable Cases

These folders are part of the unified case registry, but they are not included in the canonical MATLAB-parity suite.

| Folder | Problem case | Analysis | Dimension | Element | README | Run | Config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SIOPT_LL` | 3d_siopt_ssr | ll | 3D | P2 | [README](SIOPT_LL/README.md) | [run.sh](SIOPT_LL/run.sh) | [case.toml](SIOPT_LL/case.toml) |
| `SIOPT_SSR` | 3d_siopt_ssr | ssr | 3D | P2 | [README](SIOPT_SSR/README.md) | [run.sh](SIOPT_SSR/run.sh) | [case.toml](SIOPT_SSR/case.toml) |
| `slope_stability_2D_Franz_dam_SSR` | 2d_franz_dam_ssr | ssr | 2D | P2 | [README](slope_stability_2D_Franz_dam_SSR/README.md) | [run.sh](slope_stability_2D_Franz_dam_SSR/run.sh) | [case.toml](slope_stability_2D_Franz_dam_SSR/case.toml) |
| `slope_stability_2D_Kozinec_LL` | 2d_kozinec_ll | ll | 2D | P2 | [README](slope_stability_2D_Kozinec_LL/README.md) | [run.sh](slope_stability_2D_Kozinec_LL/run.sh) | [case.toml](slope_stability_2D_Kozinec_LL/case.toml) |
| `slope_stability_2D_Kozinec_SSR` | 2d_kozinec_ssr | ssr | 2D | P2 | [README](slope_stability_2D_Kozinec_SSR/README.md) | [run.sh](slope_stability_2D_Kozinec_SSR/run.sh) | [case.toml](slope_stability_2D_Kozinec_SSR/case.toml) |
| `slope_stability_2D_Luzec_SSR` | 2d_luzec_ssr | ssr | 2D | P2 | [README](slope_stability_2D_Luzec_SSR/README.md) | [run.sh](slope_stability_2D_Luzec_SSR/run.sh) | [case.toml](slope_stability_2D_Luzec_SSR/case.toml) |
| `slope_stability_2D_homo_LL` | 2d_homo_ssr | ll | 2D | P2 | [README](slope_stability_2D_homo_LL/README.md) | [run.sh](slope_stability_2D_homo_LL/run.sh) | [case.toml](slope_stability_2D_homo_LL/case.toml) |
| `slope_stability_3D_hetero_LL` | 3d_hetero_ssr | ll | 3D | P2 | [README](slope_stability_3D_hetero_LL/README.md) | [run.sh](slope_stability_3D_hetero_LL/run.sh) | [case.toml](slope_stability_3D_hetero_LL/case.toml) |
| `slope_stability_3D_hetero_SSR_default` | 3d_hetero_ssr | ssr | 3D | P4 | [README](slope_stability_3D_hetero_SSR_default/README.md) | [run.sh](slope_stability_3D_hetero_SSR_default/run.sh) | [case.toml](slope_stability_3D_hetero_SSR_default/case.toml) |
| `slope_stability_3D_homo_LL` | 3d_homo_ssr | ll | 3D | P2 | [README](slope_stability_3D_homo_LL/README.md) | [run.sh](slope_stability_3D_homo_LL/run.sh) | [case.toml](slope_stability_3D_homo_LL/case.toml) |
| `slope_stability_3D_homo_SSR` | 3d_homo_ssr | ssr | 3D | P2 | [README](slope_stability_3D_homo_SSR/README.md) | [run.sh](slope_stability_3D_homo_SSR/run.sh) | [case.toml](slope_stability_3D_homo_SSR/case.toml) |
| `slope_stability_3D_homo_SSR_default` | 3d_homo_ssr | ssr | 3D | P2 | [README](slope_stability_3D_homo_SSR_default/README.md) | [run.sh](slope_stability_3D_homo_SSR_default/run.sh) | [case.toml](slope_stability_3D_homo_SSR_default/case.toml) |
| `slope_stability_3D_homo_seepage_SSR_concave` | 3d_concave_seepage_ssr | ssr | 3D | P2 | [README](slope_stability_3D_homo_seepage_SSR_concave/README.md) | [run.sh](slope_stability_3D_homo_seepage_SSR_concave/run.sh) | [case.toml](slope_stability_3D_homo_seepage_SSR_concave/case.toml) |

## Notes

### `run_2D_homo_SSR_capture`

- Linear iterations total: MATLAB `0`, PETSc `4240`

### `run_2D_sloan2013_seepage_capture`

- Mesh: `4160` nodes, `7996` elements
- PETSc MPI mode: `root_only`

### `run_3D_hetero_SSR_capture`

- Linear iterations total: MATLAB `2884`, PETSc `2657`

### `run_3D_hetero_seepage_SSR_comsol_capture`

- Linear iterations total: MATLAB `2781`, PETSc `5658`

### `run_3D_hetero_seepage_capture`

- Mesh: `69733` nodes, `48205` elements
- PETSc MPI mode: `root_only`
