# Config Case Matrix

This is the current config-driven entrypoint coverage for the MATLAB drivers in `slope_stability/`.

Use:

```bash
python -m slope_stability.cli.run_case_from_config <benchmarks/.../case.toml> --out_dir <dir>
```

Supported config-driven cases in this checkpoint:

| MATLAB script | Config | Case id | Status |
| --- | --- | --- | --- |
| `slope_stability_2D_homo_SSR.m` | [run_2D_homo_SSR_capture/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2D_homo_SSR_capture/case.toml) | `2d_homo_ssr` | runnable |
| `slope_stability_2D_homo_LL.m` | [run_2d_homo_ll/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2d_homo_ll/case.toml) | `2d_homo_ssr` + `analysis = "ll"` | runnable |
| `slope_stability_2D_Kozinec_SSR.m` | [run_2d_kozinec_ssr/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2d_kozinec_ssr/case.toml) | `2d_kozinec_ssr` | runnable path is wired; default `P4` + `DIRECT` is heavier than the short smoke suite |
| `slope_stability_2D_Kozinec_LL.m` | [run_2d_kozinec_ll/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2d_kozinec_ll/case.toml) | `2d_kozinec_ll` | runnable |
| `slope_stability_2D_Luzec_SSR.m` | [run_2d_luzec_ssr/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2d_luzec_ssr/case.toml) | `2d_luzec_ssr` | runnable |
| `slope_stability_2D_Franz_dam_SSR.m` | [run_2d_franz_dam_ssr/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2d_franz_dam_ssr/case.toml) | `2d_franz_dam_ssr` | runnable, direct SSR selectable with `continuation.method = "direct"` |
| `slope_stability_3D_homo_SSR.m` | [run_3d_homo_ssr/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3d_homo_ssr/case.toml) | `3d_homo_ssr` | runnable |
| `slope_stability_3D_homo_LL.m` | [run_3d_homo_ll/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3d_homo_ll/case.toml) | `3d_homo_ssr` + `analysis = "ll"` | runnable |
| `slope_stability_3D_hetero_SSR.m` | [run_3D_hetero_SSR_capture/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3D_hetero_SSR_capture/case.toml) | `3d_hetero_ssr` | runnable |
| `slope_stability_3D_hetero_LL.m` | [run_3d_hetero_ll/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3d_hetero_ll/case.toml) | `3d_hetero_ssr` + `analysis = "ll"` | runnable |
| `SIOPT_SSR.m` | [run_3d_siopt_ssr/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3d_siopt_ssr/case.toml) | `3d_siopt_ssr` | runnable |
| `SIOPT_LL.m` | [run_3d_siopt_ll/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3d_siopt_ll/case.toml) | `3d_siopt_ssr` + `analysis = "ll"` | runnable |
| `slope_stability_2D_Sloan2013_SSR.m` seepage subproblem | [run_2D_sloan2013_seepage_capture/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_2D_sloan2013_seepage_capture/case.toml) | `2d_sloan2013_seepage` | runnable |
| `slope_stability_3D_hetero_seepage_SSR.m` seepage subproblem | [run_3D_hetero_seepage_capture/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3D_hetero_seepage_capture/case.toml) | `3d_hetero_seepage` | runnable |
| `slope_stability_3D_hetero_seepage_SSR_comsol.m` | [run_3D_hetero_seepage_SSR_comsol_capture/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml) | `3d_hetero_seepage_ssr_comsol` | runnable |
| `slope_stability_3D_homo_seepage_SSR.m` | [run_3d_homo_seepage_ssr/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/run_3d_homo_seepage_ssr/case.toml) | `3d_homo_seepage_ssr` | runnable alias on the COMSOL path |

Every config-driven run now also writes:

- a custom debug bundle in HDF5: `exports/run_debug.h5`
- a structured continuation/debug history JSON: `exports/continuation_history.json`
- a standard VTU solution file for PyVista / meshio / ParaView: `exports/final_solution.vtu`

Element-order contract:

- `2D` configs accept `P1`, `P2`, `P4`
- `3D` configs accept `P1`, `P2`, `P4`

Current numerical status:

- `2D P1/P2/P4`: wired across the supported 2D families
- `3D P2`: production path
- `3D P1`: wired where the FE path supports it, but benchmark cases still need matching `P1` meshes
- `3D P4`: unified in config/mesh-order plumbing, but current mechanics/seepage runners still fail early with an explicit `NotImplementedError`
