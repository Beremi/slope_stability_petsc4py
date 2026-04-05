# 2D homogeneous SSR

This program solves a 2D slope stability problem by the modified shear strength reduction
(SSR) method described in (Sysala et al., CAS 2025). The Mohr-Coulomb yield criterion, 3
Davis approaches (denoted by A, B, C), standard finite elements (either P1 or P2 elements)
and meshes with different densities are considered. For P2 elements, the 7-point Gauss
quadrature is used. To find the safety factor of the SSR method, two continuation techniques
are available: direct and indirect. A benchmark with a homogeneous slope is considered. It
is possible to change geometrical parameters and mesh density.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `run_2D_homo_SSR_capture`
- PETSc config: [`case.toml`](case.toml)
