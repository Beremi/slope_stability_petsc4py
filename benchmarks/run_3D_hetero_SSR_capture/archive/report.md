# 3D heterogeneous SSR

## Setup

- MATLAB script: `run_3D_hetero_SSR_capture`
- PETSc config: [`case.toml`](../case.toml)
- Run command: [`run.sh`](../run.sh)
- MPI ranks: `8`

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | 264.817 | 135.706 |
| Accepted steps | 14 | 14 |
| Final lambda | 1.66609847122 | 1.66610347565 |
| Final omega | 12000000 | 12000000 |
| Final Umax | 132.800873761 | 132.655547798 |
| Relative lambda history error | 1.290e-05 | - |
| Relative omega history error | 5.787e-06 | - |
| Relative Umax history error | 3.863e-03 | - |

## Generated Comparison

![](figures/continuation_history.png)

![](figures/iterations.png)

## Accepted-Step Table

| Step | MATLAB lambda | PETSc lambda | MATLAB omega | PETSc omega | MATLAB Newton | PETSc Newton | MATLAB linear | PETSc linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 6.21494e+06 | 6.21494e+06 | 6 | 6 | 75 | 56 |
| 2 | 1.1 | 1.1 | 6.22923e+06 | 6.22923e+06 | 6 | 6 | 74 | 37 |
| 3 | 1.15986 | 1.15986 | 6.24353e+06 | 6.24353e+06 | 7 | 6 | 86 | 46 |
| 4 | 1.24496 | 1.24496 | 6.27211e+06 | 6.27211e+06 | 9 | 9 | 123 | 85 |
| 5 | 1.31141 | 1.31141 | 6.30069e+06 | 6.3007e+06 | 8 | 7 | 132 | 71 |
| 6 | 1.41874 | 1.41875 | 6.35786e+06 | 6.35787e+06 | 11 | 10 | 220 | 182 |
| 7 | 1.50571 | 1.50573 | 6.41503e+06 | 6.41503e+06 | 13 | 12 | 315 | 262 |
| 8 | 1.60856 | 1.60854 | 6.52936e+06 | 6.52937e+06 | 15 | 12 | 319 | 182 |
| 9 | 1.62554 | 1.62552 | 6.6437e+06 | 6.64371e+06 | 16 | 11 | 327 | 241 |
| 10 | 1.6386 | 1.6386 | 6.87236e+06 | 6.87239e+06 | 12 | 19 | 240 | 541 |
| 11 | 1.64886 | 1.64891 | 7.3297e+06 | 7.32974e+06 | 18 | 21 | 409 | 483 |
| 12 | 1.65684 | 1.65687 | 8.24438e+06 | 8.24445e+06 | 20 | 18 | 564 | 471 |
| 13 | 1.66304 | 1.66306 | 1.00737e+07 | 1.00739e+07 | - | - | - | - |
| 14 | 1.6661 | 1.6661 | 1.2e+07 | 1.2e+07 | - | - | - | - |

## Side-by-Side Figures

### Displacement

| MATLAB | PETSc |
| --- | --- |
| ![](../../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/matlab/matlab_plots/matlab_displacements_3D.png) | ![](../../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/petsc/plots/petsc_displacements_3D.png) |

### Strain

| MATLAB | PETSc |
| --- | --- |
| ![](../../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/matlab/matlab_plots/matlab_deviatoric_strain_3D.png) | ![](../../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/petsc/plots/petsc_deviatoric_strain_3D.png) |

### Curve

| MATLAB | PETSc |
| --- | --- |
| ![](../../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/matlab/matlab_plots/matlab_omega_lambda.png) | ![](../../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/petsc/plots/petsc_omega_lambda.png) |
## Raw Outputs

- MATLAB artifacts: `../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/matlab`
- PETSc artifacts: `../../../artifacts/benchmarks/mpi8/3d_hetero_ssr/petsc`
