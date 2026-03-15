# P4 Elements and Parallel Tangent Assembly: Current Implementation Brief

This note summarizes how `P4` elements, `B` construction, tangent matrix/value assembly, index precomputation, and parallelization are implemented in the current repository. It is intended as a handoff document for an external FEM/HPC expert.

## Scope

- `P4` element definitions and mesh elevation
- Basis and quadrature for `P4`
- Construction of `B` and derivative storage
- Global tangent assembly path
- Owned-row / distributed tangent path
- Precomputed indexing and sparsity helpers
- Where parallelism is used today
- What looks most relevant for efficiency work

## Main code locations

- Element type declarations: [`src/slope_stability/core/elements.py`](../src/slope_stability/core/elements.py)
- Generic simplex Lagrange ordering / evaluation: [`src/slope_stability/core/simplex_lagrange.py`](../src/slope_stability/core/simplex_lagrange.py)
- Local basis functions: [`src/slope_stability/fem/basis.py`](../src/slope_stability/fem/basis.py)
- Quadrature rules: [`src/slope_stability/fem/quadrature.py`](../src/slope_stability/fem/quadrature.py)
- Strain operator `B` and elastic stiffness assembly: [`src/slope_stability/fem/assembly.py`](../src/slope_stability/fem/assembly.py)
- Owned-row elastic overlap assembly: [`src/slope_stability/fem/distributed_elastic.py`](../src/slope_stability/fem/distributed_elastic.py)
- Owned-row tangent precompute and assembly: [`src/slope_stability/fem/distributed_tangent.py`](../src/slope_stability/fem/distributed_tangent.py)
- Constitutive operator and `K_tangent` creation: [`src/slope_stability/constitutive/problem.py`](../src/slope_stability/constitutive/problem.py)
- 3D compiled tangent kernel bridge: [`src/slope_stability/cython/_kernels.pyx`](../src/slope_stability/cython/_kernels.pyx)
- 3D compiled tangent kernel implementation: [`src/slope_stability/cython/assemble_tangent_values_3d.c`](../src/slope_stability/cython/assemble_tangent_values_3d.c)
- 3D compiled constitutive batch kernel: [`src/slope_stability/cython/constitutive_3d_batch.c`](../src/slope_stability/cython/constitutive_3d_batch.c)
- PETSc ownership helpers: [`src/slope_stability/utils.py`](../src/slope_stability/utils.py)
- Mesh reordering / partition-aware ordering: [`src/slope_stability/mesh/reorder.py`](../src/slope_stability/mesh/reorder.py)
- Example 3D SSR runner wiring the distributed tangent path: [`src/slope_stability/cli/run_3D_hetero_SSR_capture.py`](../src/slope_stability/cli/run_3D_hetero_SSR_capture.py)

## 1. P4 element support

### 1.1 Declared element sizes

The repository treats `P4` as a supported simplex family in both 2D and 3D:

- 2D `P4` has `15` nodes per element
- 3D `P4` has `35` nodes per element
- 3D `P4` surface faces have `15` nodes

Source:

- [`src/slope_stability/core/elements.py#L8-L29`](../src/slope_stability/core/elements.py#L8-L29)

### 1.2 3D P4 node ordering

The 3D P4 ordering is generated generically from barycentric Lagrange tuples:

- `4` vertex nodes
- `6` edges, each with `3` internal edge nodes, so `18` edge nodes
- `4` faces, each with `3` internal face nodes, so `12` face-interior nodes
- `1` cell-interior node

The tuple enumeration is in:

- [`src/slope_stability/core/simplex_lagrange.py#L49-L83`](../src/slope_stability/core/simplex_lagrange.py#L49-L83)

Reference node coordinates for nodal interpolation checks are built in:

- [`src/slope_stability/core/simplex_lagrange.py#L86-L92`](../src/slope_stability/core/simplex_lagrange.py#L86-L92)

### 1.3 Mesh elevation to tet35 / tri15

For `.msh` input, the code currently expects linear tet4 source cells and elevates them to `P2` or `P4` on load.

The 3D `P4` elevation routine:

- reuses edge points through a shared `edge_map`
- reuses face-interior points through a shared `face_map`
- appends one cell centroid as the volume-interior P4 node
- builds `tet35`
- builds matching `tri15` boundary faces

Relevant code:

- [`src/slope_stability/io.py#L330-L387`](../src/slope_stability/io.py#L330-L387)
- [`src/slope_stability/io.py#L390-L449`](../src/slope_stability/io.py#L390-L449)

Practical implication:

- geometry order is not read from native high-order Gmsh tetra blocks
- instead, the code starts from tet4 geometry and deterministically inserts P4 nodes

## 2. P4 basis and quadrature

### 2.1 2D P4 basis

2D `P4` basis functions are written explicitly as closed-form formulas, including explicit derivatives. The comment states the implementation keeps the full 15-basis form to preserve MATLAB equivalence.

Source:

- [`src/slope_stability/fem/basis.py#L58-L122`](../src/slope_stability/fem/basis.py#L58-L122)

### 2.2 3D P4 basis

3D `P4` basis evaluation is generic:

- `local_basis_volume_3d("P4", xi)` calls `evaluate_tetra_lagrange_basis(4, xi)`
- the generic evaluator builds basis values and derivatives from barycentric factor products

Source:

- [`src/slope_stability/fem/basis.py#L127-L207`](../src/slope_stability/fem/basis.py#L127-L207)
- [`src/slope_stability/core/simplex_lagrange.py#L116-L161`](../src/slope_stability/core/simplex_lagrange.py#L116-L161)

### 2.3 P4 quadrature

Quadrature is upgraded for `P4`:

- 2D `P4`: 12-point triangle rule
- 3D `P4`: 24-point Keast order-7 tetrahedron rule, exact for total degree 6

Source:

- [`src/slope_stability/fem/quadrature.py#L28-L49`](../src/slope_stability/fem/quadrature.py#L28-L49)
- [`src/slope_stability/fem/quadrature.py#L92-L126`](../src/slope_stability/fem/quadrature.py#L92-L126)

Implication for cost:

- `P4` in 3D means `n_p = 35`, `n_q = 24`
- local element displacement DOFs are `3 * 35 = 105`
- local element tangent block is `105 x 105`

## 3. Construction of B

## 3.1 General role

`assemble_strain_operator` builds:

- derivative arrays `dphi*`
- quadrature weights multiplied by `|det J|`
- sparse strain-displacement matrix `B`

Source:

- [`src/slope_stability/fem/assembly.py#L18-L37`](../src/slope_stability/fem/assembly.py#L18-L37)

## 3.2 2D path

The 2D path is straightforward and more loop-based:

- loop over elements
- loop over quadrature points
- compute Jacobian from nodal coordinates and reference derivatives
- invert Jacobian
- compute physical derivatives `dphi1`, `dphi2`
- assemble COO entries for the engineering-strain `B`

Source:

- [`src/slope_stability/fem/assembly.py#L44-L130`](../src/slope_stability/fem/assembly.py#L44-L130)

Important details:

- `n_strain = 3`
- ordering is `[e11, e22, gamma12]`
- local 2D block for each basis function is `[[dN/dx, 0], [0, dN/dy], [dN/dy, dN/dx]]`

See:

- [`src/slope_stability/fem/assembly.py#L106-L110`](../src/slope_stability/fem/assembly.py#L106-L110)

## 3.3 3D path

The 3D path is more vectorized across all element-quadrature pairs:

- `coord_x`, `coord_y`, `coord_z` are gathered as `(n_p, n_elem)`
- reference derivatives are tiled across elements
- element coordinates are repeated across quadrature points
- all Jacobian entries `j11..j33` are formed in bulk
- physical derivatives `dphi1`, `dphi2`, `dphi3` are computed in bulk
- then `B` is assembled in COO form and converted to CSR

Source:

- [`src/slope_stability/fem/assembly.py#L133-L237`](../src/slope_stability/fem/assembly.py#L133-L237)

The 3D `B` layout uses:

- `n_strain = 6`
- per-node local block written implicitly into `vB`

Source:

- [`src/slope_stability/fem/assembly.py#L194-L221`](../src/slope_stability/fem/assembly.py#L194-L221)

## 3.4 3D strain ordering

This matters for any replacement kernel.

The constitutive code explicitly states the MATLAB 3D shear ordering is:

- `[11, 22, 33, 12, 23, 13]`

Source:

- [`src/slope_stability/constitutive/problem.py#L483-L486`](../src/slope_stability/constitutive/problem.py#L483-L486)

The compiled constitutive kernel also documents:

- inputs use engineering shear strains
- returned `DS` is a column-major `6x6` block matching MATLAB / Octave storage

Source:

- [`src/slope_stability/cython/constitutive_3D_kernel.h#L1-L8`](../src/slope_stability/cython/constitutive_3D_kernel.h#L1-L8)

## 4. Global K:tangent creation

## 4.1 Elastic reference path

Elastic stiffness is built as:

- per-integration-point constitutive block
- sparse block-diagonal `Dp`
- `K_elast = B^T * Dp * B`

Source:

- [`src/slope_stability/fem/assembly.py#L240-L299`](../src/slope_stability/fem/assembly.py#L240-L299)

## 4.2 Tangent matrix creation in ConstitutiveOperator

The constitutive operator precomputes sparse indexing helpers once:

- `AUX`
- `iD`
- `jD`
- `vD_pre`

These encode the block-diagonal sparse structure of `D`.

Source:

- [`src/slope_stability/constitutive/problem.py#L1126-L1131`](../src/slope_stability/constitutive/problem.py#L1126-L1131)

In the full global path, tangent assembly is:

1. compute `DS`
2. scale it by `vD_pre`
3. build sparse `D = csc_matrix((vD, (iD, jD)))`
4. compute `K_tangent = B^T * D * B`
5. symmetrize

Source:

- [`src/slope_stability/constitutive/problem.py#L1530-L1545`](../src/slope_stability/constitutive/problem.py#L1530-L1545)

There is also a reference function outside the class that does the same global `B^T D B` build:

- [`src/slope_stability/fem/distributed_tangent.py#L434-L450`](../src/slope_stability/fem/distributed_tangent.py#L434-L450)

Implication:

- the global path still explicitly builds the sparse global block-diagonal `D`
- it still multiplies by the full global `B`
- this is the simpler reference path, not the optimized distributed path

## 5. Owned-row / distributed tangent path

## 5.1 High-level idea

The optimized path is not matrix-free. It is a fixed-pattern owned-row assembly path:

- reorder mesh nodes first
- assign each MPI rank a contiguous owned node block
- build an overlap submesh with all elements touching those owned nodes
- precompute a fixed local CSR pattern for owned rows in global column numbering
- at each nonlinear step, compute only the tangent values on that fixed pattern
- wrap the local CSR rows directly as a PETSc `MPIAIJ` matrix

Main wiring example:

- [`src/slope_stability/cli/run_3D_hetero_SSR_capture.py#L356-L452`](../src/slope_stability/cli/run_3D_hetero_SSR_capture.py#L356-L452)

## 5.2 Reordering and ownership

Node reorder options are in:

- [`src/slope_stability/mesh/reorder.py#L105-L146`](../src/slope_stability/mesh/reorder.py#L105-L146)

Important current choices:

- `block_metis` uses nodal adjacency and partitions nodes by graph membership, then preserves an `xyz`-based local order inside each partition
- ownership is then assigned as contiguous node blocks

Sources:

- [`src/slope_stability/mesh/reorder.py#L85-L102`](../src/slope_stability/mesh/reorder.py#L85-L102)
- [`src/slope_stability/utils.py#L64-L71`](../src/slope_stability/utils.py#L64-L71)

This is the current MPI distribution basis.

## 5.3 Local overlap elastic assembly

Before tangent assembly, the code assembles owned elastic rows from an overlap submesh:

- find all elements that touch the owned node range
- build a local overlap mesh
- assemble its `B` and `K_overlap`
- restrict to owned rows
- map overlap-local columns back to global DOFs
- apply BC treatment by replacing constrained rows with diagonal `1`

Source:

- [`src/slope_stability/fem/distributed_elastic.py#L36-L52`](../src/slope_stability/fem/distributed_elastic.py#L36-L52)
- [`src/slope_stability/fem/distributed_elastic.py#L54-L181`](../src/slope_stability/fem/distributed_elastic.py#L54-L181)

This overlap elastic matrix is used both as:

- the local elastic operator rows
- the structural template from which tangent pattern metadata is derived

## 5.4 Tangent pattern precomputation

The central precompute is `prepare_owned_tangent_pattern`:

- [`src/slope_stability/fem/distributed_tangent.py#L176-L315`](../src/slope_stability/fem/distributed_tangent.py#L176-L315)

It stores:

- overlap nodes and overlap elements
- overlap-global DOFs
- overlap `B`
- owned overlap DOF slice
- owned free-mask and owned free-row indices
- a fixed local CSR matrix pattern
- elastic values projected onto that pattern
- quadrature weights on the overlap submesh
- contiguous `dphi1`, `dphi2`, `dphi3`
- local integration-point indices
- optional unique-element submesh data
- `scatter_map`
- constrained diagonal positions

The data structure is:

- [`src/slope_stability/fem/distributed_tangent.py#L21-L53`](../src/slope_stability/fem/distributed_tangent.py#L21-L53)

## 5.5 Structural sparsity pattern

The owned-row structural pattern is built from element connectivity and `q_mask`:

- for each overlap element, get all its global DOFs
- free owned rows connect to all free columns touched by that element
- constrained rows are collapsed to their own diagonal only

Source:

- [`src/slope_stability/fem/distributed_tangent.py#L120-L158`](../src/slope_stability/fem/distributed_tangent.py#L120-L158)

Important detail:

- this pattern is in global column numbering, not overlap-local numbering

## 5.6 Projection of elastic values onto the fixed pattern

The elastic row matrix assembled earlier may have the same logical pattern but not the same storage layout. `_project_values_onto_pattern` maps the elastic row values into the fixed CSR pattern by row-wise column lookup.

Source:

- [`src/slope_stability/fem/distributed_tangent.py#L161-L173`](../src/slope_stability/fem/distributed_tangent.py#L161-L173)

These projected values are later reused in regularized builds:

- `K_r = r * K_elast + (1-r) * K_tangent`

## 5.7 Precomputation of integration-point indices

There are two main index lists:

- `local_int_indices`: all overlap integration points in overlap-element order
- `unique_local_int_indices`: only uniquely owned integration points, used by the partitioned constitutive mode

Construction:

- [`src/slope_stability/fem/distributed_tangent.py#L236-L239`](../src/slope_stability/fem/distributed_tangent.py#L236-L239)
- [`src/slope_stability/fem/distributed_tangent.py#L273-L276`](../src/slope_stability/fem/distributed_tangent.py#L273-L276)

These are used to slice global material fields and `DS` without rebuilding the mapping each Newton step.

## 5.8 Scatter-map precomputation

This is probably the most important indexing precompute for tangent assembly.

`_build_scatter_map` builds a dense table:

- shape `(n_overlap_elements, n_local_dof * n_local_dof)`
- each entry is the position in the owned-row CSR `data` array where that local element stiffness contribution must be added
- entries are `-1` if the local entry is not active in this rank's owned rows / free-column pattern

Source:

- [`src/slope_stability/fem/distributed_tangent.py#L63-L117`](../src/slope_stability/fem/distributed_tangent.py#L63-L117)

How it is built:

- row-wise dictionaries are created from CSR columns to CSR positions
- each overlap element is expanded to global DOFs
- only owned rows are considered
- only free columns are considered
- constrained rows are skipped and later handled through `constrained_diag_positions`

For 3D P4:

- `n_p = 35`
- `dim = 3`
- `n_local_dof = 105`
- `n_local_dof^2 = 11025`

So each overlap element stores `11025` `int64` scatter slots.

That is a substantial memory commitment and is likely one of the first things an optimization expert will care about.

## 6. Constitutive side and local tangent data

## 6.1 Compiled constitutive batch path

Local constitutive evaluation for 3D can go through compiled kernels:

- Python converts `E` to contiguous `(n_int, 6)` form
- C/OpenMP processes integration points independently
- `S` and `DS` are returned as `(n_int, 6)` and `(n_int, 36)` and transposed back

Sources:

- [`src/slope_stability/constitutive/problem.py#L808-L821`](../src/slope_stability/constitutive/problem.py#L808-L821)
- [`src/slope_stability/cython/_kernels.pyx#L277-L337`](../src/slope_stability/cython/_kernels.pyx#L277-L337)
- [`src/slope_stability/cython/constitutive_3d_batch.c#L9-L50`](../src/slope_stability/cython/constitutive_3d_batch.c#L9-L50)

The compiled constitutive batch kernel is OpenMP-parallel over integration points.

## 6.2 Distributed constitutive modes

The owned tangent path supports two local constitutive modes:

- `overlap`
- `unique` / `unique_gather` / `partitioned`

Relevant code:

- [`src/slope_stability/constitutive/problem.py#L1172-L1221`](../src/slope_stability/constitutive/problem.py#L1172-L1221)
- [`src/slope_stability/constitutive/problem.py#L1223-L1300`](../src/slope_stability/constitutive/problem.py#L1223-L1300)
- [`src/slope_stability/constitutive/problem.py#L1302-L1311`](../src/slope_stability/constitutive/problem.py#L1302-L1311)

### Overlap mode

- compute `u_overlap`
- compute local strain from `overlap_B @ u_overlap`
- compute `S_local`, `DS_local` directly on all overlap integration points

Pros:

- simple
- no MPI constitutive gather

Cons:

- redundant constitutive work on overlap regions

### Unique / partitioned mode

- compute only on uniquely owned elements / integration points using `unique_B`
- MPI `allgather` local results
- reconstruct overlap-ordered local `S_local` / `DS_local`

Pros:

- removes constitutive redundancy

Cons:

- introduces explicit communication and global reconstruction of `S` / `DS`

## 7. Internal force assembly in the owned path

Internal force on owned rows is not assembled through element loops in the tangent kernel. It uses the precomputed overlap `B`:

- `load = weight * stress_local`
- `overlap_force = overlap_B.T * load`
- restrict to owned overlap DOFs
- zero constrained rows

Source:

- [`src/slope_stability/constitutive/problem.py#L878-L891`](../src/slope_stability/constitutive/problem.py#L878-L891)

Then the code either:

- returns local owned rows
- or does an MPI `allgather` to reconstruct the full vector / full free vector

Source:

- [`src/slope_stability/constitutive/problem.py#L852-L875`](../src/slope_stability/constitutive/problem.py#L852-L875)
- [`src/slope_stability/constitutive/problem.py#L1462-L1528`](../src/slope_stability/constitutive/problem.py#L1462-L1528)

So force assembly and tangent assembly are currently asymmetric:

- force uses sparse overlap `B`
- tangent uses local dense element kernels plus `scatter_map`

## 8. Tangent value assembly on the fixed pattern

## 8.1 Python fallback

The Python fallback does:

1. loop over overlap elements
2. allocate dense local `ke`
3. loop over quadrature points
4. reshape `DS_q` into dense `n_strain x n_strain`
5. rebuild dense local `B_eq`
6. accumulate `ke += B_eq.T @ D_q @ B_eq`
7. scatter `ke` entries into the precomputed output vector using `scatter_map`

Source:

- [`src/slope_stability/fem/distributed_tangent.py#L318-L364`](../src/slope_stability/fem/distributed_tangent.py#L318-L364)

Important observation:

- `B_eq` is rebuilt every quadrature point every iteration, even though geometry is fixed

## 8.2 Compiled 3D path

The fast path for tangent values is enabled only when:

- `use_compiled=True`
- compiled extension is available
- `dim == 3`
- `n_strain == 6`

Source:

- [`src/slope_stability/fem/distributed_tangent.py#L367-L398`](../src/slope_stability/fem/distributed_tangent.py#L367-L398)

Before calling the C kernel:

- `DS` is sliced to local overlap integration points if needed
- result is stored contiguously as `(n_int_local, 36)`

Then Cython forwards:

- `dphi1`, `dphi2`, `dphi3`
- `ds`
- quadrature weights
- `scatter_map`
- `nnz_out`

Source:

- [`src/slope_stability/cython/_kernels.pyx#L232-L274`](../src/slope_stability/cython/_kernels.pyx#L232-L274)

## 8.3 Current OpenMP kernel structure

The C kernel in [`src/slope_stability/cython/assemble_tangent_values_3d.c`](../src/slope_stability/cython/assemble_tangent_values_3d.c) does:

- `#pragma omp parallel`
- allocate per-thread `ke`, `beq`, `tmp`
- `#pragma omp for schedule(static)` over elements
- for each element:
  - zero `ke`
  - for each quadrature point:
    - rebuild dense `beq` from `dphi*`
    - compute `tmp = (w * D_q) * beq`
    - compute `ke += beq^T * tmp`
  - scatter-add each active `ke` entry into the output value array with `#pragma omp atomic update`

Source:

- [`src/slope_stability/cython/assemble_tangent_values_3d.c#L13-L109`](../src/slope_stability/cython/assemble_tangent_values_3d.c#L13-L109)

Important implementation facts:

- the kernel name still includes `_p2`, but it is actually generic in `n_p`
- there is no element coloring or thread-private reduction of the output value vector
- collision handling is done by atomic adds on the final output values
- geometry is fixed, but `beq` is still rebuilt inside the quadrature loop

## 9. PETSc matrix creation and reuse

Owned local CSR rows are wrapped directly into PETSc `MPIAIJ` matrices:

- first build a local CSR matrix from the fixed pattern and current values
- then create PETSc AIJ with already-owned local rows

Source:

- [`src/slope_stability/constitutive/problem.py#L1313-L1332`](../src/slope_stability/constitutive/problem.py#L1313-L1332)
- [`src/slope_stability/utils.py#L171-L195`](../src/slope_stability/utils.py#L171-L195)

For the regularized matrix path, the code reuses the matrix object:

- tangent values are rebuilt on the same pattern
- values are blended with preprojected elastic values
- `Mat.setValuesCSR` updates the existing PETSc matrix in place

Source:

- [`src/slope_stability/constitutive/problem.py#L1334-L1382`](../src/slope_stability/constitutive/problem.py#L1334-L1382)
- [`src/slope_stability/utils.py#L198-L215`](../src/slope_stability/utils.py#L198-L215)

This is one of the more efficient pieces already present.

## 10. Where parallelism exists today

### MPI-level parallelism

MPI parallelism is by node-block ownership after mesh reordering:

- contiguous owned node range per rank
- overlap subdomain assembled locally
- PETSc matrix ownership follows those rows

Relevant code:

- [`src/slope_stability/utils.py#L64-L71`](../src/slope_stability/utils.py#L64-L71)
- [`src/slope_stability/fem/distributed_elastic.py#L184-L206`](../src/slope_stability/fem/distributed_elastic.py#L184-L206)
- [`src/slope_stability/cli/run_3D_hetero_SSR_capture.py#L435-L452`](../src/slope_stability/cli/run_3D_hetero_SSR_capture.py#L435-L452)

### Shared-memory parallelism

OpenMP is used in compiled kernels:

- 3D constitutive batch kernel: parallel over integration points
- 3D tangent assembly kernel: parallel over elements

Sources:

- [`src/slope_stability/cython/constitutive_3d_batch.c#L19-L20`](../src/slope_stability/cython/constitutive_3d_batch.c#L19-L20)
- [`src/slope_stability/cython/constitutive_3d_batch.c#L41-L42`](../src/slope_stability/cython/constitutive_3d_batch.c#L41-L42)
- [`src/slope_stability/cython/assemble_tangent_values_3d.c#L33-L40`](../src/slope_stability/cython/assemble_tangent_values_3d.c#L33-L40)

Build flags:

- [`setup.py#L8-L19`](../setup.py#L8-L19)

### What is not parallelized in compiled form

- 2D tangent assembly still falls back to Python
- global `B^T D B` assembly is sparse linear algebra, not a custom parallel kernel
- some pattern construction steps are Python-heavy and one-time serial precomputes

## 11. Likely efficiency pain points in the current design

These are observations about the current implementation, not design recommendations yet.

- `P4` 3D local matrices are large: `105 x 105` per element.
- `scatter_map` is dense and large for P4, and uses `int64`.
- dense local `B_eq` is rebuilt every quadrature point even though the element geometry is fixed.
- the OpenMP tangent kernel ends with atomic updates into the shared value array, which may become a bottleneck.
- the owned path stores both sparse overlap `B` and dense `dphi*`, so geometry information is duplicated.
- the full global path explicitly constructs sparse `D` every time and multiplies by the full global `B`.
- partitioned constitutive mode removes constitutive redundancy but reintroduces `allgather` of `S` and `DS`.
- overlap mode avoids constitutive communication but does redundant overlap constitutive work.

## 12. Most useful source entry points for an external expert

If someone wants the shortest reading path, these are the most relevant files:

- [`src/slope_stability/fem/distributed_tangent.py`](../src/slope_stability/fem/distributed_tangent.py)
- [`src/slope_stability/fem/assembly.py`](../src/slope_stability/fem/assembly.py)
- [`src/slope_stability/constitutive/problem.py`](../src/slope_stability/constitutive/problem.py)
- [`src/slope_stability/cython/assemble_tangent_values_3d.c`](../src/slope_stability/cython/assemble_tangent_values_3d.c)
- [`src/slope_stability/cython/constitutive_3D_kernel.h`](../src/slope_stability/cython/constitutive_3D_kernel.h)
- [`src/slope_stability/cli/run_3D_hetero_SSR_capture.py`](../src/slope_stability/cli/run_3D_hetero_SSR_capture.py)

## 13. Minimal current pipeline summary

For a 3D `P4` nonlinear run with node-distributed tangent assembly:

1. Load tet4 mesh and elevate it to tet35 / tri15:
   - [`src/slope_stability/io.py#L330-L449`](../src/slope_stability/io.py#L330-L449)
2. Reorder nodes, usually `block_metis`:
   - [`src/slope_stability/mesh/reorder.py#L85-L146`](../src/slope_stability/mesh/reorder.py#L85-L146)
3. Define owned node range from PETSc block ownership:
   - [`src/slope_stability/utils.py#L64-L71`](../src/slope_stability/utils.py#L64-L71)
4. Build overlap elastic rows:
   - [`src/slope_stability/fem/distributed_elastic.py#L54-L181`](../src/slope_stability/fem/distributed_elastic.py#L54-L181)
5. Precompute tangent pattern, overlap `B`, `dphi*`, integration-point indices, CSR pattern, elastic values, and `scatter_map`:
   - [`src/slope_stability/fem/distributed_tangent.py#L176-L315`](../src/slope_stability/fem/distributed_tangent.py#L176-L315)
6. At each nonlinear iteration, compute local strains and local `S`, `DS`:
   - [`src/slope_stability/constitutive/problem.py#L1172-L1311`](../src/slope_stability/constitutive/problem.py#L1172-L1311)
7. Assemble local internal force via overlap `B.T`:
   - [`src/slope_stability/constitutive/problem.py#L878-L891`](../src/slope_stability/constitutive/problem.py#L878-L891)
8. Assemble tangent values on the fixed CSR pattern:
   - Python fallback: [`src/slope_stability/fem/distributed_tangent.py#L318-L364`](../src/slope_stability/fem/distributed_tangent.py#L318-L364)
   - Compiled 3D OpenMP path: [`src/slope_stability/fem/distributed_tangent.py#L367-L398`](../src/slope_stability/fem/distributed_tangent.py#L367-L398)
9. Wrap local CSR rows as PETSc `MPIAIJ`:
   - [`src/slope_stability/constitutive/problem.py#L1313-L1332`](../src/slope_stability/constitutive/problem.py#L1313-L1332)

## 14. One-line diagnosis

The current implementation is already structured around fixed-pattern owned-row assembly, but for 3D `P4` it still pays heavily for large dense local element work, a large precomputed `scatter_map`, repeated reconstruction of local `B_eq`, and atomic scatter accumulation in the OpenMP tangent kernel.
