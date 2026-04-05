# 3D heterogeneous seepage SSR COMSOL

This program solves a 3D slope stability problem by the modified shear strength reduction
(SSR) method described in (Sysala et al., CAS 2025). The Mohr-Coulomb yield criterion, 3
Davis approaches (denoted by A, B, C), standard finite elements (P1 or P2) and meshes with
different densities are considered. For P2 elements, the 11-point Gauss quadrature is used.
To find the safety factor of the SSR method, two continuation techniques are available:
direct and indirect. A bechmark problem on a homogeneous slope with unconfined seepage is
considered. It is possible to change geometrical parameters and mesh density.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `run_3D_hetero_seepage_SSR_comsol_capture`
- PETSc config: [`case.toml`](case.toml)
