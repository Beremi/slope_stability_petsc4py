# 2D homogeneous SSR

## Setup

- MATLAB script: `run_2D_homo_SSR_capture`
- PETSc config: [`case.toml`](../case.toml)
- Run command: [`run.sh`](../run.sh)
- MPI ranks: `8`

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | 9.270 | 8.286 |
| Accepted steps | 14 | 14 |
| Final lambda | 1.21132826584 | 1.21132390718 |
| Final omega | 917.51514891 | 917.464390515 |
| Final Umax | 13.2980216483 | 13.2958298409 |
| Relative lambda history error | 6.736e-06 | - |
| Relative omega history error | 3.050e-05 | - |
| Relative Umax history error | 1.561e-04 | - |

## Generated Comparison

![](figures/continuation_history.png)

![](figures/iterations.png)

## Accepted-Step Table

| Step | MATLAB lambda | PETSc lambda | MATLAB omega | PETSc omega | MATLAB Newton | PETSc Newton | MATLAB linear | PETSc linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9 | 0.9 | 425.172 | 425.172 | 6 | 8 | 0 | 71 |
| 2 | 1 | 1 | 425.412 | 425.412 | 6 | 8 | 0 | 91 |
| 3 | 1.07182 | 1.07181 | 425.652 | 425.652 | 7 | 7 | 0 | 97 |
| 4 | 1.16848 | 1.16847 | 426.132 | 426.132 | 7 | 7 | 0 | 79 |
| 5 | 1.18795 | 1.18796 | 426.613 | 426.613 | 5 | 5 | 0 | 42 |
| 6 | 1.19329 | 1.1933 | 427.573 | 427.573 | 5 | 6 | 0 | 61 |
| 7 | 1.19749 | 1.19748 | 429.495 | 429.494 | 6 | 7 | 0 | 90 |
| 8 | 1.20102 | 1.20103 | 433.337 | 433.337 | 10 | 9 | 0 | 165 |
| 9 | 1.2042 | 1.20421 | 441.023 | 441.021 | 15 | 12 | 0 | 238 |
| 10 | 1.20688 | 1.20689 | 456.394 | 456.39 | 24 | 25 | 0 | 689 |
| 11 | 1.20889 | 1.20888 | 487.135 | 487.129 | 42 | 45 | 0 | 1266 |
| 12 | 1.21023 | 1.21023 | 548.618 | 548.605 | 50 | 50 | 0 | 1351 |
| 13 | 1.21095 | 1.21094 | 671.584 | 671.558 | - | - | - | - |
| 14 | 1.21133 | 1.21132 | 917.515 | 917.464 | - | - | - | - |

## Side-by-Side Figures

### Displacement

| MATLAB | PETSc |
| --- | --- |
| ![](../../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/matlab/matlab_plots/matlab_displacements_2D.png) | ![](../../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/petsc/plots/petsc_displacements_2D.png) |

### Strain

| MATLAB | PETSc |
| --- | --- |
| ![](../../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/matlab/matlab_plots/matlab_deviatoric_strain_2D.png) | ![](../../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/petsc/plots/petsc_deviatoric_strain_2D.png) |

### Curve

| MATLAB | PETSc |
| --- | --- |
| ![](../../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/matlab/matlab_plots/matlab_omega_lambda_2D.png) | ![](../../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/petsc/plots/petsc_omega_lambda_2D.png) |
## Raw Outputs

- MATLAB artifacts: `../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/matlab`
- PETSc artifacts: `../../artifacts/benchmarks/mpi8/run_2D_homo_SSR_capture/petsc`
