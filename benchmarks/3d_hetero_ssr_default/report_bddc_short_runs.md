# BDDC Short-Run Diagnostics

## Scope

- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Case: rank-2 `P2`, `step_max=10`
- Baseline control: current Hypre path
- Candidate: `pc_backend=bddc`, `preconditioner_matrix_source=elastic`, ILU local solves, LU coarse solve, no deluxe scaling

## The Runtime Bug That Was Fixed

The first short nonlinear BDDC run was correct but unusable because it rebuilt the same elastic `P` matrix over and over.

Old behavior:

- runtime: `393.264 s`
- first progress: `80.153 s`
- preconditioner setup total: `325.291 s`
- rebuild count: `75`
- reuse count: `0`
- peak RSS: `5.562 GiB`

That was not a PETSc configuration issue. It was solver lifecycle behavior:

- `preconditioner_matrix_source=elastic` was constant,
- but `current_policy` still forced rebuilds every solve,
- and `release_iteration_resources()` dropped the cached `P`.

After the fix, the same CLI candidate reuses the elastic `P`:

- runtime: `77.764 s`
- first progress: `30.062 s`
- preconditioner setup total: `8.803 s`
- rebuild count: `2`
- reuse count: `73`
- peak RSS: `5.665 GiB`

## Final P2 Short-Run Comparison

| Metric | Hypre current | BDDC elastic reuse | BDDC / Hypre |
| --- | ---: | ---: | ---: |
| Runtime [s] | 89.159 | 77.764 | `0.872x` |
| First progress [s] | 20.038 | 30.062 | `1.500x` |
| Final accepted states | 10 | 10 | match |
| Final lambda | 1.638606206 | 1.638603589 | close |
| Final omega | 6872377.551148129 | 6872365.494565467 | close |
| Final Umax | 8.437590906 | 8.448082929 | close |
| Linear total [s] | 63.505 | 50.993 | `0.803x` |
| Attempt preconditioner [s] | 37.652 | 4.464 | `0.119x` |
| Attempt solve [s] | 25.853 | 46.530 | `1.800x` |
| Preconditioner setup total [s] | 43.876 | 8.803 | `0.201x` |
| Peak RSS [GiB] | 2.481 | 5.665 | `2.283x` |
| Rebuild count | 69 | 2 | much lower |
| Reuse count | 0 | 73 | much higher |

## Gate Outcome

Against the original strict `P2` short-run gate:

- accepted-state count: pass
- final state drift: pass
- runtime `<= 2.0x` Hypre: pass
- peak RSS `<= 1.25x` Hypre: fail

So the branch is now functionally valid and runtime-competitive, but it still fails the strict memory gate on `P2`.

## Interpretation

- The important broken behavior is fixed: BDDC no longer wastes time rebuilding a static elastic `P`.
- On `P2`, that fix is enough to beat the current Hypre control on total runtime and on combined linear solve + preconditioner time.
- The remaining reason not to promote this branch is memory footprint, not correctness or setup stability.
- Since the `P4` elastic probe is still blocked upstream in SciPy CSR assembly, the BDDC track does not advance to `P4 step_max=10` yet.

## Artifacts

- Old short run:
  - `artifacts/bddc_elastic_short_runs/p2_step10_bddc_ilu_elastic`
- Reuse-fixed short run:
  - `artifacts/bddc_elastic_short_runs/p2_step10_bddc_ilu_elastic_reuse`
- Hypre control:
  - `artifacts/bddc_elastic_short_runs/p2_step10_hypre_current`
