# 2D Sloan2013 seepage

## Setup

- MATLAB script: `run_2D_sloan2013_seepage_capture`
- PETSc config: [`case.toml`](../case.toml)
- Run command: [`run.sh`](../run.sh)
- MPI ranks requested: `8`
- PETSc MPI mode: `root_only`

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | 0.407 | 1.213 |
| Mesh nodes | 4160 | 4160 |
| Mesh elements | 7996 | 7996 |
| Relative pore-pressure error | 3.329e-15 | - |
| Relative gradient error | 1.823e-14 | - |
| Saturation mismatch count | 0 | - |

## Side-by-Side Figures

### Pore pressure

| MATLAB | PETSc |
| --- | --- |
| ![](../../../artifacts/benchmarks/mpi8/run_2D_sloan2013_seepage_capture/matlab/matlab_pore_pressure_2D.png) | ![](../../../artifacts/benchmarks/mpi8/run_2D_sloan2013_seepage_capture/petsc/plots/petsc_pore_pressure_2D.png) |

### Saturation

| MATLAB | PETSc |
| --- | --- |
| ![](../../../artifacts/benchmarks/mpi8/run_2D_sloan2013_seepage_capture/matlab/matlab_saturation_2D.png) | ![](../../../artifacts/benchmarks/mpi8/run_2D_sloan2013_seepage_capture/petsc/plots/petsc_saturation_2D.png) |
## Raw Outputs

- MATLAB artifacts: `../../artifacts/benchmarks/mpi8/run_2D_sloan2013_seepage_capture/matlab`
- PETSc artifacts: `../../artifacts/benchmarks/mpi8/run_2D_sloan2013_seepage_capture/petsc`
