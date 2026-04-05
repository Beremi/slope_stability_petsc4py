# 2D homogeneous LL

This program solves a 2D slope stability problem by the limit load (LL) method described in
(Sysala et al., CAS 2025). The Mohr- Coulomb yield criterion, Davis approach, standard
finite elements (either P1 or P2 elements) and meshes with different densities are
considered. For P2 elements, the 7-point Gauss quadrature is used. To find the safety factor
of the LL method, the indirect continuation technique is used. A benchmark with a
homogeneous slope is considered. It is possible to change slope inclination and other
geometrical parameters.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `slope_stability_2D_homo_LL.m`
- PETSc config: [`case.toml`](case.toml)
