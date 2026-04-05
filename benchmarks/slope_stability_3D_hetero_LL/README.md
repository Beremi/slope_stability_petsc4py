# 3D heterogeneous LL

This program solves a 3D slope stability problem by the limit load (LL) method described in
(Sysala et al., CAS 2025). The Mohr- Coulomb yield criterion, Davis approach, standard
finite elements (either P1 or P2 elements) and meshes with different densities are
considered. For P2 elements, the 11-point Gauss quadrature is used. To find the safety
factor of the LL method, the indirect continuation technique is used. A benchmark with a
heterogeneous slope from (Sysala et al., CAS 2025) is considered.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `slope_stability_3D_hetero_LL.m`
- PETSc config: [`case.toml`](case.toml)
