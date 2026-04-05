# 3D SIOPT SSR

This program solves a 3D slope stability problem by the modified shear strength reduction
(SSR) method described in (Sysala et al., CAS 2025). The Mohr-Coulomb yield criterion, 3
Davis approaches (denoted by A, B, C), standard finite elements (either P1 or P2 elements)
and meshes with different densities are considered. For P2 elements, the 11-point Gauss
quadrature is used. To find the safety factor of the SSR method, two continuation techniques
are available: direct and indirect. The benchmark described in (Sysala et al., SIOPT 2025)
is considered.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `SIOPT_SSR.m`
- PETSc config: [`case.toml`](case.toml)
