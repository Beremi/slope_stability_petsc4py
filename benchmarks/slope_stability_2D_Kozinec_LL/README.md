# 2D Kozinec LL

This program solves a 2D slope stability problem by the limit load (LL) method described in
(Sysala et al., CAS 2025). The Mohr- Coulomb yield criterion, Davis approach, standard
finite elements (P1, P2 or P4 elements) and meshes with different densities are considered.
For P2 elements, the 7-point Gauss quadrature is used. To find the safety factor of the LL
method, the indirect continuation technique is used. A heterogeneous slope from the locality
Doubrava-Kozinec is considered, see (Sysala et al., NAG 2021)

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `slope_stability_2D_Kozinec_LL.m`
- PETSc config: [`case.toml`](case.toml)
