# P4 L1 Continuation Transfer Follow-Up

Date: `2026-04-09`

Target case:
- `SSR indirect`
- mesh `SSR_hetero_ada_L1.msh`
- `P4`
- `omega_max_stop = 6.7e6`
- continuation Newton stop `absolute_delta_lambda < 1e-4`
- `np = 8`
- linear tolerance `1e-1`

Primary artifact bundle:
- `artifacts/comparisons/continuation_transfer_report_20260409/REPORT.md`
- `artifacts/comparisons/continuation_transfer_report_20260409/comparison_summary.json`

Plots:
- `artifacts/comparisons/continuation_transfer_report_20260409/plots/continuation_runtime.png`
- `artifacts/comparisons/continuation_transfer_report_20260409/plots/trajectory_overlay.png`
- `artifacts/comparisons/continuation_transfer_report_20260409/plots/linear_work_breakdown.png`
- `artifacts/comparisons/continuation_transfer_report_20260409/plots/step_wall_overlay.png`
- `artifacts/comparisons/continuation_transfer_report_20260409/plots/step_newton_overlay.png`
- `artifacts/comparisons/continuation_transfer_report_20260409/plots/step_linear_overlay.png`

Compared continuation runs:
- baseline current: exact `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE` + default `pmg_shell`
- exact `DFGMRES` + PMG coarse `cg`
- plain `KSPFGMRES` + tuned `pmg_shell`
- exact `DFGMRES` + PMG coarse `cg` + init-only hybrid Armijo

Result:
- no tested setup beat the current continuation baseline on the full SSR indirect solve
- baseline remained fastest at `946.96 s`
- exact `DFGMRES` + coarse `cg` finished in `980.11 s` (`+3.5%`)
- tuned `KSPFGMRES + pmg_shell` finished in `1289.09 s` (`+36.1%`)
- exact `DFGMRES` + coarse `cg` + init-only hybrid Armijo finished in `1045.64 s` (`+10.4%`)

Interpretation:
- the fixed-lambda screen did show real wins from coarse-`cg`, tuned PMG, and the corrected hybrid Armijo path
- those wins did not transfer to the full continuation trajectory
- the dominant failure mode was the hard step-8 region, where continuation Newton/globalization overwhelmed the linear gains

Recommendation:
- keep the current continuation default for `P4` on `SSR_hetero_ada_L1`
- treat continuation Newton/globalization around step 8 as the next main optimization target
