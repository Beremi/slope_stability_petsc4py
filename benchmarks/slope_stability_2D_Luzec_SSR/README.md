# 2D Luzec SSR

This program solves a 2D slope stability problem by the modified shear strength reduction
(SSR) method described in (Sysala et al., CAS 2025). The Mohr-Coulomb yield criterion, 3
Davis approaches (denoted by A, B, C), standard finite elements (P1, P2 or P4 elements) and
meshes with different densities are considered. For P2 elements, the 7-point Gauss
quadrature is used. To find the safety factor of the SSR method, two continuation techniques
are available: direct and indirect. A case study on a heterogeneous river embankment with
unconfined seepage from the locality Luzec (Czechia) is considered, see (Sysala et al., CAS
2023).

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `slope_stability_2D_Luzec_SSR.m`
- PETSc config: [`case.toml`](case.toml)
