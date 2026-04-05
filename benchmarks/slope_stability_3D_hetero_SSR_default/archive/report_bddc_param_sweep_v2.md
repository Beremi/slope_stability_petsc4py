# BDDC Parameter Sweep for P4 (V2)

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Supersession note: Supersedes bddc_param_sweep_v1 because v1 benchmarked BDDC without the corrected symmetric/monolithic/coarse-redundant baseline.
- PETSc sources consulted:
  - [`PCBDDC` manual page](https://petsc.org/release/manualpages/PC/PCBDDC/): baseline BDDC requirements and documented option families.
  - [`ex56`](https://petsc.org/main/src/snes/tutorials/ex56.c.html): elasticity-oriented approximate-local BDDC and GAMG tuning.
  - [`ex71`](https://petsc.org/main/src/ksp/ksp/tutorials/ex71.c.html): deluxe scaling and adaptive-threshold elasticity examples.
  - [`ex59`](https://petsc.org/main/src/ksp/ksp/tutorials/ex59.c.html): high-order adjacency/corner-primal customization reserved as the next escalation path.
- Linear residual target: `1.0e-05`
- Linear probe: distributed `P4`, rank `2`, `A=P=K_elast`, native PETSc `CG`, `ksp_norm_type=unpreconditioned`.
- Corrected BDDC baseline: `pc_bddc_symmetric=true`, `pc_bddc_monolithic=true`, `pc_bddc_coarse_redundant_pc_type=svd`, `use_faces=true`.
- MUMPS available: `yes`

## Option Smokes

| Variant | Status | First progress [s] |
| --- | --- | ---: |
| hypre_control_v2 | completed | 115.188 |
| bddc_exact_lu_ref_v2 | runtime_failure | 115.199 |
| bddc_gamg_doc_base_v2 | completed | 115.197 |

## Phase 1: Rank-2 Linear Elastic Screen

| Variant | Status | Iter | Setup [s] | Solve [s] | Runtime [s] | Rel. residual | Peak RSS [GiB] | Faces | Coarse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hypre_control_v2 | completed | 14 | 16.218 | 25.390 | 105.432 | 7.711e-06 | 23.126 | - | - |
| bddc_gamg_doc_base_v2 | completed | 103 | 47.179 | 80.939 | 192.354 | 8.235e-06 | 25.556 | 0 | 9 |

## Phase 2: One-Sided Approximate-Local Sweep

| Variant | Status | Iter | Setup [s] | Solve [s] | Runtime [s] | Rel. residual | Peak RSS [GiB] | Faces | Coarse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bddc_gamg_doc_base_v2_dir_exact_neu_approx | completed | 122 | 43.552 | 92.634 | 200.434 | 9.483e-06 | 25.303 | 0 | 9 |
| bddc_gamg_doc_base_v2_dir_approx_neu_exact | completed | 149 | 40.160 | 103.938 | 208.591 | 9.986e-06 | 26.355 | 0 | 9 |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx | completed | 103 | 47.229 | 80.567 | 192.407 | 8.235e-06 | 26.385 | 0 | 9 |

## Phase 3: Topology and Corner-Primal Sweep

| Variant | Status | Iter | Setup [s] | Solve [s] | Runtime [s] | Rel. residual | Peak RSS [GiB] | Faces | Coarse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_none | completed | 103 | 47.244 | 79.764 | 191.245 | 8.235e-06 | 25.529 | 0 | 9 |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_csr | completed | 103 | 47.245 | 80.871 | 192.418 | 8.235e-06 | 25.271 | 0 | 9 |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology | completed | 103 | 47.483 | 81.182 | 194.625 | 8.235e-06 | 25.105 | 0 | 9 |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology_corner | completed | 181 | 60.384 | 125.722 | 251.856 | 8.338e-06 | 28.948 | 0 | 28 |

## Convergence Plots

- `bddc_gamg_doc_base_v2` vs Hypre: [bddc_gamg_doc_base_v2_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_exact_neu_approx` vs Hypre: [bddc_gamg_doc_base_v2_dir_exact_neu_approx_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_exact_neu_approx_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_approx_neu_exact` vs Hypre: [bddc_gamg_doc_base_v2_dir_approx_neu_exact_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_approx_neu_exact_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_approx_neu_approx` vs Hypre: [bddc_gamg_doc_base_v2_dir_approx_neu_approx_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_approx_neu_approx_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_none` vs Hypre: [bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_none_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_none_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_csr` vs Hypre: [bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_csr_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_csr_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology` vs Hypre: [bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology_vs_hypre.png)
- `bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology_corner` vs Hypre: [bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology_corner_vs_hypre.png](artifacts/bddc_param_sweep_v2/plots/bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_topology_corner_vs_hypre.png)
- Aggregate completed-candidate plot: [p4_linear_top_candidates.png](artifacts/bddc_param_sweep_v2/plots/p4_linear_top_candidates.png)

- Promoted candidates: `bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_none, bddc_gamg_doc_base_v2_dir_approx_neu_approx`

## Diagnostics

| Variant | Status | Runtime [s] | PETSc log |
| --- | --- | ---: | --- |
| hypre_control_v2 | runtime_failure | - | - |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx_adj_none | runtime_failure | - | - |
| bddc_gamg_doc_base_v2_dir_approx_neu_approx | runtime_failure | - | - |

## Short Nonlinear Follow-up

| Variant | Status | Accepted | Runtime [s] | Accepted states | Linear total [s] |
| --- | --- | --- | ---: | ---: | ---: |

## Conclusion

- This recovery sweep corrected the BDDC baseline before comparing variants: symmetry, monolithic correction, coarse redundant SVD, and faces were treated as mandatory, not optional.
- The next step is determined by the best completed elastic result only: continue local-GAMG tuning if iterations dropped materially, move to nonlinear mismatch only if the elastic probe is close enough to Hypre, or escalate topology classification if coarse information still looks weak.
