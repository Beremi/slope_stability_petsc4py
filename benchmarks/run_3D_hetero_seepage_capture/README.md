# 3D heterogeneous seepage

## Setup

- MATLAB script: `run_3D_hetero_seepage_capture`
- PETSc config: [`case.toml`](case.toml)
- Run command: [`run.sh`](run.sh)
- MPI ranks requested: `8`
- PETSc MPI mode: `root_only`

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | 15.100 | 16.017 |
| Mesh nodes | 69733 | 69733 |
| Mesh elements | 48205 | 48205 |
| Relative pore-pressure error | 1.694e-16 | - |
| Relative gradient error | 4.201e-15 | - |
| Saturation mismatch count | 0 | - |

## Side-by-Side Figures

### Pore pressure

| MATLAB | PETSc |
| --- | --- |
| ![](../../artifacts/benchmarks/mpi8/run_3D_hetero_seepage_capture/matlab/matlab_pore_pressure_3D.png) | ![](../../artifacts/benchmarks/mpi8/run_3D_hetero_seepage_capture/petsc/plots/petsc_pore_pressure_3D.png) |

### Saturation

| MATLAB | PETSc |
| --- | --- |
| ![](../../artifacts/benchmarks/mpi8/run_3D_hetero_seepage_capture/matlab/matlab_saturation_3D.png) | ![](../../artifacts/benchmarks/mpi8/run_3D_hetero_seepage_capture/petsc/plots/petsc_saturation_3D.png) |
## Raw Outputs

- MATLAB artifacts: `../../artifacts/benchmarks/mpi8/run_3D_hetero_seepage_capture/matlab`
- PETSc artifacts: `../../artifacts/benchmarks/mpi8/run_3D_hetero_seepage_capture/petsc`
