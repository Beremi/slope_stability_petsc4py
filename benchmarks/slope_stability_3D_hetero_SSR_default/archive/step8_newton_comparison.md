# Step 8 Newton Comparison

This replays the `slope_stability_3D_hetero_SSR_default` secant case only up to just beyond accepted continuation step 8 and compares the step-8 Newton solver trace for `Default` vs `Less precise x100`.

- Replay `omega_max`: `6.540e+06`
- `delta U` is plotted as the accepted free-DOF Newton correction norm `||alpha ΔU||` per Newton iteration.
- `delta U / U` is plotted as the relative accepted free-DOF Newton correction norm `||alpha ΔU|| / ||U||`, where `||U||` is the current Newton iterate free-DOF norm.

## Summary

| Case | Newton tol | Step-8 target omega | Step-8 final lambda | Step-8 final omega | Step-8 Newton iterations | Step-8 final relres |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | `1.0e-04` | 6529372.2 | 1.608521 | 6529372.2 | 11 | 4.931e-05 |
| Less precise x100 | `1.0e-02` | 6532999.5 | 1.639524 | 6532999.5 | 4 | 8.155e-03 |

## Plots

### Criterion

![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/criterion.png)

### Lambda

![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/lambda.png)

### Delta U

![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/delta_u.png)

### Delta U / U

![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/delta_u_over_u.png)

### Newton Correction Norm vs Lambda

![Newton Correction Norm vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/correction_norm_vs_lambda.png)

### Newton Correction Norm vs Criterion

![Newton Correction Norm vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/correction_norm_vs_criterion.png)

### Relative Increment vs Lambda

![Relative Increment vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/relative_increment_vs_lambda.png)

### Relative Increment vs Criterion

![Relative Increment vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/relative_increment_vs_criterion.png)

### Lambda vs Criterion

![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/report/plots/lambda_vs_criterion.png)

## Step-8 Newton Data

| Iteration | Default criterion | Less precise x100 criterion | Default lambda | Less precise x100 lambda | Default `||alpha ΔU||` | Less precise x100 `||alpha ΔU||` | Default `||alpha ΔU|| / ||U||` | Less precise x100 `||alpha ΔU|| / ||U||` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.165e+04 | 3.143e+04 | 1.505724 | 1.510811 | 3.830334 | 2.123089 | 3.303e-02 | 1.893e-02 |
| 2 | 4.607e+03 | 7.051e+03 | 1.643635 | 1.654110 | 4.733842 | 7.688424 | 4.023e-02 | 6.888e-02 |
| 3 | 2.738e+03 | 4.628e+03 | 1.641276 | 1.651538 | 17.723545 | 13.654371 | 1.465e-01 | 1.169e-01 |
| 4 | 1.505e+03 | 2.836e+03 | 1.628434 | 1.639524 | 17.935275 | n/a | 1.345e-01 | n/a |
| 5 | 1.277e+03 | n/a | 1.614807 | n/a | 1.528850 | n/a | 1.043e-02 | n/a |
| 6 | 1.144e+03 | n/a | 1.613696 | n/a | 5.307091 | n/a | 3.593e-02 | n/a |
| 7 | 3.511e+02 | n/a | 1.609828 | n/a | 1.524729 | n/a | 1.006e-02 | n/a |
| 8 | 2.179e+02 | n/a | 1.608798 | n/a | 0.377216 | n/a | 2.473e-03 | n/a |
| 9 | 7.945e+01 | n/a | 1.608586 | n/a | 0.240873 | n/a | 1.577e-03 | n/a |
| 10 | 7.933e+01 | n/a | 1.608535 | n/a | 0.084120 | n/a | 5.505e-04 | n/a |
| 11 | 1.715e+01 | n/a | 1.608521 | n/a | n/a | n/a | n/a | n/a |

## Artifacts

- Default: config `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/configs/default.toml`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/runs/default`
- Less precise x100: config `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/configs/less_precise_x100.toml`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/step8_newton_default_vs_less_precise_x100/runs/less_precise_x100`
