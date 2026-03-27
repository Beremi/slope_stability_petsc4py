# Mixed `L1/L2` PMG Probe For `P2(L2)`

## Question

Can the repo construct a multilevel hierarchy even though `SSR_hetero_ada_L1.msh` and `SSR_hetero_ada_L2.msh` are not node-hierarchical, using

- fine: `P2` on `L2`
- mid: `P1` on `L2`
- coarse: `P1` on `L1`

for a real `P2(L2)` frozen tangent?

## Short Answer

Yes at rank 1.

- A geometric `P1(L1) -> P1(L2)` transfer was added using point-in-tet search plus barycentric interpolation.
- The mixed hierarchy `P1(L1) -> P1(L2) -> P2(L2)` now builds and both PMG backends can use it on a real frozen `P2(L2)` tangent.
- On rank 1, built-in PETSc `pmg` and custom `pmg_shell` both converged.

Rank-8 status is mixed.

- Built-in PETSc `pmg` fails during distributed Galerkin coarse assembly.
- `pmg_shell` sets up and enters the distributed solve, but did not finish within the time budget here.

## Mesh Non-Hierarchy Check

The `L1` and `L2` meshes are on the same domain but are not node-nested.

- `L1 P1` nodes: `3845`
- `L2 P1` nodes: `6795`
- exact shared coordinates: `26`
- `L2` nodes matching some `L1` node: about `0.38%`
- `L1` nodes matching some `L2` node: about `0.68%`

Because of that, the transfer cannot be built by node-number or node-coordinate matching. It must be geometric interpolation.

## Implementation

Changed files:

- `src/slope_stability/linear/pmg.py`
- `src/slope_stability/linear/solver.py`
- `benchmarks/3d_hetero_ssr_default/archive/probe_hypre_frozen.py`
- `tests/test_pmg_hierarchy.py`

Main additions:

- generic three-level PMG hierarchy metadata instead of a solver-side hardcoded `P1 -> P2 -> P4` assumption
- cross-mesh `P1 -> P1` prolongation via coarse-tet point location and barycentric weights
- mixed hierarchy builder for `P1(L1) -> P1(L2) -> P2(L2)`
- frozen probe support for `--pmg-coarse-mesh-path`

## Real State Used

Reference `P2(L2)` state:

- run: `artifacts/l2_p2_hypre_step1_for_mixed_pmg`
- data: `artifacts/l2_p2_hypre_step1_for_mixed_pmg/data/run_info.json`
- state: `artifacts/l2_p2_hypre_step1_for_mixed_pmg/data/petsc_run.npz`

That run used direct Hypre on `P2(L2)` with:

- accepted states: `3`
- runtime: `47.981 s`
- unknowns: `145298`
- final `lambda`: about `1.16014`
- final `omega`: about `6.243994e+06`

The frozen probes below use its final state.

## Rank-1 Frozen Results

Artifacts:

- Hypre: `artifacts/l2_p2_mixed_pmg_probe_rank1/hypre/data/run_info.json`
- PETSc PMG: `artifacts/l2_p2_mixed_pmg_probe_rank1/pmg/data/run_info.json`
- Shell PMG: `artifacts/l2_p2_mixed_pmg_probe_rank1/pmg_shell/data/run_info.json`

Results:

| Backend | Hierarchy | Iterations | Setup [s] | Solve [s] | Setup+Solve [s] | Final Rel. Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `hypre` | none | 10 | 2.780 | 2.163 | 4.944 | `7.774e-04` |
| `pmg` | `P1(L1) -> P1(L2) -> P2(L2)` | 6 | 6.411 | 0.894 | 7.305 | `7.218e-04` |
| `pmg_shell` | `P1(L1) -> P1(L2) -> P2(L2)` | 11 | 0.375 | 1.918 | 2.293 | `8.372e-04` |

Readout:

- built-in `pmg` reduced Krylov iterations most strongly
- `pmg_shell` had the cheapest setup and best total wall time on this frozen rank-1 case
- both PMG variants used the intended mixed level orders `[1, 1, 2]`

## Rank-8 Status

Built-in PETSc `pmg`:

- command ran on `8` MPI ranks against the same frozen `P2(L2)` state
- failure log: `artifacts/l2_p2_mixed_pmg_probe_rank8/pmg/stderr.log`
- failure mode: PETSc `MatGalerkin()` / `MatDiagonalSet()` inserted a new diagonal nonzero on the distributed coarse matrix and tripped preallocation

Representative error:

- `Argument out of range`
- `Inserting a new nonzero at (...) in the matrix`

So the distributed built-in Galerkin path is not currently usable for this mixed `L1/L2` hierarchy.

Custom `pmg_shell` on 8 ranks:

- setup completed and produced `artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell/data/ksp_view.txt`
- it entered the solve phase and kept all ranks busy
- it did not produce `run_info.json` within the time budget here, so convergence was not established

## Conclusion

For the specific hierarchy

- `P2(L2)` fine
- `P1(L2)` mid
- `P1(L1)` coarse

the answer is:

- yes, the repo can now construct and use that hierarchy on a real `P2(L2)` tangent
- rank-1 frozen probes work for both `pmg` and `pmg_shell`
- rank-8 built-in `pmg` is blocked by distributed Galerkin coarse-matrix preallocation
- rank-8 `pmg_shell` gets further than built-in `pmg`, but still needs a proper completion run and likely further tuning
