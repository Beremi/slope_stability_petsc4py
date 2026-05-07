# Config Case Matrix

This is the current config-driven entrypoint coverage for the MATLAB drivers in
`slope_stability_matlab/` and the extra runnable PETSc cases.

Use:

```bash
./.venv/bin/python -m slope_stability.cli.run_case_from_config <benchmarks/.../case.toml> --out_dir <dir>
```

Every case is asset-first: `case.toml` selects `asset`, `mesh_variant`, optional `profile`,
analysis, element order, and numerical settings. Problem geometry, materials, hydraulics,
and BCs are defined in `meshes/<asset>/definition.py`.

For adding a new geometry, see
[new-benchmark-new-geometry-guide.md](new-benchmark-new-geometry-guide.md).

## Supported Config-Driven Cases

| MATLAB script / case | Config | Asset | Mesh variant | Profile | Analysis | Element | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `slope_stability_2D_homo_SSR.m` | [run_2D_homo_SSR_capture/case.toml](../benchmarks/run_2D_homo_SSR_capture/case.toml) | `2d_homo_slope` | `h1.0.msh` | `default` | `ssr` | `P2` | runnable |
| `slope_stability_2D_homo_LL.m` | [slope_stability_2D_homo_LL/case.toml](../benchmarks/slope_stability_2D_homo_LL/case.toml) | `2d_homo_slope` | `h0.5.msh` | `default` | `ll` | `P2` | runnable |
| `slope_stability_2D_Kozinec_SSR.m` | [slope_stability_2D_Kozinec_SSR/case.toml](../benchmarks/slope_stability_2D_Kozinec_SSR/case.toml) | `2d_kozinec` | `default.msh` | `default` | `ssr` | `P2` | runnable |
| `slope_stability_2D_Kozinec_LL.m` | [slope_stability_2D_Kozinec_LL/case.toml](../benchmarks/slope_stability_2D_Kozinec_LL/case.toml) | `2d_kozinec` | `default.msh` | `default` | `ll` | `P2` | runnable |
| `slope_stability_2D_Luzec_SSR.m` | [slope_stability_2D_Luzec_SSR/case.toml](../benchmarks/slope_stability_2D_Luzec_SSR/case.toml) | `2d_luzec` | `default.msh` | `default` | `ssr` | `P2` | runnable |
| `slope_stability_2D_Franz_dam_SSR.m` | [slope_stability_2D_Franz_dam_SSR/case.toml](../benchmarks/slope_stability_2D_Franz_dam_SSR/case.toml) | `2d_franz_dam` | `default.msh` | `default` | `ssr` | `P2` | runnable |
| `slope_stability_3D_homo_SSR.m` | [slope_stability_3D_homo_SSR/case.toml](../benchmarks/slope_stability_3D_homo_SSR/case.toml) | `3d_homo_slope` | `adaptive_family_a_l1.msh` | `default` | `ssr` | `P2` | runnable |
| `slope_stability_3D_homo_LL.m` | [slope_stability_3D_homo_LL/case.toml](../benchmarks/slope_stability_3D_homo_LL/case.toml) | `3d_homo_slope` | `adaptive_family_b_l1.msh` | `default` | `ll` | `P2` | runnable |
| `slope_stability_3D_hetero_SSR.m` | [run_3D_hetero_SSR_capture/case.toml](../benchmarks/run_3D_hetero_SSR_capture/case.toml) | `3d_hetero_slope` | `adaptive_family_a_l1.msh` | `default` | `ssr` | `P2` | runnable |
| `slope_stability_3D_hetero_LL.m` | [slope_stability_3D_hetero_LL/case.toml](../benchmarks/slope_stability_3D_hetero_LL/case.toml) | `3d_hetero_slope` | `adaptive_family_b_l1.msh` | `default` | `ll` | `P2` | runnable |
| `SIOPT_SSR.m` | [SIOPT_SSR/case.toml](../benchmarks/SIOPT_SSR/case.toml) | `3d_siopt` | `reference_l0.msh` | `fixed_base` | `ssr` | `P2` | runnable |
| `SIOPT_LL.m` | [SIOPT_LL/case.toml](../benchmarks/SIOPT_LL/case.toml) | `3d_siopt` | `reference_l0.msh` | `fixed_base` | `ll` | `P2` | runnable |
| `slope_stability_2D_Sloan2013_SSR.m` seepage subproblem | [run_2D_sloan2013_seepage_capture/case.toml](../benchmarks/run_2D_sloan2013_seepage_capture/case.toml) | `2d_sloan2013` | `default.msh` | `default` | `seepage` | `P1` | runnable |
| `slope_stability_3D_hetero_seepage_SSR.m` seepage subproblem | [run_3D_hetero_seepage_capture/case.toml](../benchmarks/run_3D_hetero_seepage_capture/case.toml) | `3d_hetero_seepage` | `concave_family_b.msh` | `default` | `seepage` | `P2` | runnable |
| `slope_stability_3D_hetero_seepage_SSR_comsol.m` | [run_3D_hetero_seepage_SSR_comsol_capture/case.toml](../benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml) | `3d_hetero_seepage_transition` | `transition_default.msh` | `fixed_base` | `ssr` | `P2` | runnable |
| `slope_stability_3D_homo_seepage_SSR.m` | [slope_stability_3D_homo_seepage_SSR_concave/case.toml](../benchmarks/slope_stability_3D_homo_seepage_SSR_concave/case.toml) | `3d_hetero_seepage_transition` | `transition_default.msh` | `fixed_base` | `ssr` | `P2` | runnable concave seepage+SSR alias |
| default 3D heterogeneous SSR config | [slope_stability_3D_hetero_SSR_default/case.toml](../benchmarks/slope_stability_3D_hetero_SSR_default/case.toml) | `3d_hetero_slope` | `adaptive_family_a_l1.msh` | `default` | `ssr` | `P4` | runnable |
| default 3D homogeneous SSR config | [slope_stability_3D_homo_SSR_default/case.toml](../benchmarks/slope_stability_3D_homo_SSR_default/case.toml) | `3d_homo_slope` | `adaptive_family_a_l1.msh` | `default` | `ssr` | `P2` | runnable |

## Outputs

Every config-driven run can write:

- `exports/run_debug.h5`
- `exports/continuation_history.json`
- `exports/final_solution.vtu`
- `exports/resolved_config.toml`

## Element-Order Contract

- 2D configs accept `P1`, `P2`, `P4`
- 3D configs accept `P1`, `P2`, `P3`, `P4`

Mesh promotion is generic. Numerical availability still depends on the selected analysis and
solver path.

## Removed Config Inputs

Committed configs must not define raw mesh paths, boundary types, material rows, water unit
weight, or conductivity. Those are asset-owned in `meshes/<asset>/definition.py`.
