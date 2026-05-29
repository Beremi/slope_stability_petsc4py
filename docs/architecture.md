# Architecture

The engine is organized as a PETSc-first HPC application with a small Python
control surface.

Python responsibilities:

- parse compact case TOMLs and named solver profiles;
- expose CLI commands for running, validating, and inspecting cases;
- expand suite TOMLs into resolved benchmark manifests;
- resolve suite rank sweeps to concrete resource/launcher command policy;
- build resolved run/environment manifests without importing `petsc4py`;
- expose lightweight runtime artifact readers under `petsc_ssr.runtime`;
- generate benchmark notebooks and summarize result artifacts;
- keep mesh/problem definitions scriptable through `meshes/<asset>/definition.py`.

C/PETSc responsibilities:

- own all distributed DMPlex meshes, labels, sections, matrices, and vectors;
- assemble residuals, tangents, elastic operators, loads, and lambda derivatives;
- run PMG, deflation, Krylov solves, Newton methods, continuation, and profiling;
- write continuation curves, summaries, PETSc binary vectors, and VTU outputs.

The Python-to-C boundary is a serialized PETSc option string. Python does not
copy global matrices or vectors into NumPy for the maintained solve path.
Python-side option-file parsing, artifact path insertion, and shell quoting are
centralized in `petsc_ssr.runtime.options`; runners and contexts should not
hand-roll their own resolved PETSc option streams.
Debug-only Python continuation loops reach native state through
`SsrContext.debug_engine_ops()` and the `EngineOps` compatibility wrapper;
`SsrContext` does not expose individual assembly, solve, trial, or deflation
callbacks as public API.
Run artifacts also include `data/native_problem_manifest.json`, a
coordinate-free manifest of the resolved asset, physical supports, material
assignments, Dirichlet/Neumann/head/flux rules, and boundary-geometry
declarations. `data/resolved_run_manifest.json` records the concrete case,
profile, rank-adaptive PMG choices, profile-owned PMG coarse/telescope/smoother
policy, and artifact paths for the run, while `data/environment.json` records
portable host/MPI/Git provenance. Those two
manifest shapes are owned by `petsc_ssr.config.manifest` so dry-run and suite
tooling can verify them without constructing native PETSc state.

Runs also write `data/mechanics_bc_labels.csv`, a compact DMPlex
label/tag/component table for mechanics Dirichlet rules. Assets that declare
mechanics Neumann rules also write `data/mechanics_neumann_labels.csv` with
boundary label, geometry, and value-model metadata. These files are the
migration surface for replacing remaining coordinate CSV bridges with DMPlex
labels and sections. Native mechanics startup requires the label constraint
table for manifest-declared Dirichlet rules and treats the coordinate node table
as a debug compatibility artifact. The dedicated native `assembly/neumann.c`
path validates Neumann label tables and assembles affine `constant-traction`
rules by face quadrature into the external mechanics load vector. Curved
geometry-patch Neumann rows still fail explicitly, so boundary loads are not
silently approximated or dropped. Seepage
assets write `data/seepage_boundary_labels.csv` with head/flux supports; current
coupled mechanics still consumes the pressure CSV from the hydro prepass while
the label table records and stages the future DMPlex/field source in native
`AssemblySeepageBoundaryRule` rows. That bridge is explicit:
the run passes `-seepage_pressure_source hydro_prepass_coordinate_bridge` beside
`-seepage_pressure_csv`, and native startup rejects the pressure table without
that source contract. The run passes the paths to PETSc as
`-native_problem_manifest`, `-mechanics_bc_labels_csv`,
`-mechanics_neumann_labels_csv`, and `-seepage_boundary_labels_csv`, so
`-options_left` can catch spelling or routing errors. The native startup also
validates manifest kind/schema, analysis, element order, resolved MPI size,
DMPlex label names, support counts, and declared BC rule counts, then reads the
manifest's `native_inputs` block and fills mechanics/seepage artifact paths
from it when explicit PETSc options are not supplied. If the manifest declares
Dirichlet, Neumann, or seepage boundary rules without an available label-table
bridge, startup fails before assembly. When a manifest is active, native
startup also checks the loaded label-table row counts against the manifest rule
counts so PETSc builds reject stale or hand-edited artifacts. Python preflight
does the stronger row-content parity check before writing the resolved run
bundle.

## Public Model Invariants

- Case TOMLs describe mathematical problems only.
- Mesh assets own geometry, materials, regions, labels, and BC supports.
- Continuation, Newton, and linear solver profiles own algorithm/PETSc policy.
- Suites own ranks, repeats, time limits, machines, and sweep overrides.
- Artifacts own generated output paths and resolved provenance.
- Notebook sidecars own visualization and display metadata.
- The native C/PETSc engine owns distributed mesh, vector, matrix, KSP, PC, PMG,
  deflation, Newton, and continuation state.

The legacy internal case shape is an explicit migration/debug compatibility path
behind `PETSC_SSR_ALLOW_LEGACY_CASE_SCHEMA=1`; it is not accepted as the normal
public benchmark model.

Normal runs select algorithms through resolved profiles. `--force-c-baseline`
is retained only as an explicit debug override.

## Native Layout

`src/petsc_ssr/native/` is divided by solver subsystem:

- `include/`: public C API for the Cython bridge.
- `core/`: context, option parsing, lifecycle, and CLI glue.
- `mesh/`: PETSc seepage/DMPlex mesh-facing code.
- `materials/`: Mohr-Coulomb material routines.
- `assembly/`: element basis and mechanics assembly kernels.
- `linear/`: PMG shell V-cycle, deflation, and Krylov routines.
- `nonlinear/`: fixed-load and indirect Newton methods.
- `continuation/`: indirect SSR, direct SSR, and limit-load continuation.
- `reporting/`: CSV/JSON summaries and PETSc log-facing timing helpers.
- `replay/`: debug/replay-only comparison helpers.
- `cython/`: thin C API implementation exposed to `_core.pyx`.

The implementation still builds the mechanics engine as one translation unit to
avoid numerical or performance changes from symbol visibility or call-boundary
refactors.

The public native headers `petsc_ssr_engine.h`, `petsc_ssr_problem.h`,
`petsc_ssr_profile.h`, `petsc_ssr_algorithms.h`, and `petsc_ssr_stats.h`
document the target engine, manifest, subcontext, registry, and profiling
interfaces without changing the current unity build. `petsc_ssr_problem.h` owns
the native manifest constants consumed by `io/problem_manifest.c.inc`.
`petsc_ssr_engine.h` now names the stable object-lifecycle surface
(`SsrEngineCreate`, `SsrEngineSetContinuation`, `SsrEngineSetNewton`,
`SsrEngineSetLinearSolver`, `SsrEngineRun`, and `SsrEngineDestroy`) separately
from the maintained options-string entry point. The object API is wired into the
unity build as a fail-explicit scaffold while the proven benchmark path remains
`PetscSsrEngineRunOptionsString`.
`algorithms/registry.c.inc` now provides the first concrete
continuation/Newton/linear registry surface plus typed material and Neumann
value-model registry contracts. Mechanics element assembly now evaluates
Mohr-Coulomb stress/tangent points through the material registry while reusing
the existing constitutive kernels. The affine `constant-traction` Neumann path
now parses asset-declared values into a typed value context and evaluates them
through that registry during face quadrature. Native option validation resolves
the algorithm families through the same registry. The typed runtime profile also
preserves concrete/requested PC variant policy and P1 fallback reason after
native option validation, with nested PMG and deflation views for the resolved
linear profile. The maintained CLI runner now dispatches direct versus indirect
continuation through a native continuation operation table keyed by the resolved
profile selector. Continuation algorithms in turn resolve fixed-load and
indirect SSR Newton through a native Newton operation table, while reusing the
existing Newton loop bodies unchanged. The normal Newton hot path also resolves
the concrete linear algorithm through a native linear operation table and passes
a typed `SsrLinearCtx` carrying the solver plus resolved linear profile view.
PMG setup now reads active-rank layout, telescope policy, smoother policy, and
coarse level KSP policy from the resolved `SsrPmgOptions` view instead of
reaching back into the wide app context. The shell V-cycle and legacy `pcmg`
configuration paths both consume that view, while still calling the proven
KSP/PMG/deflation solve body. The deflated Krylov method inside that body is
selected through a small native deflation operation table. Deeper deflation/KSP
state remains inside `LinearSolverCtx` until it can be lifted behind fully owned
typed subcontexts in measured numerical rewrites.

Profiling should enter native algorithms through `SsrProfiler` and
`SsrProfileTimer`, not ad hoc phase-specific timing helpers. The timer API wraps
PETSc log events and returns elapsed wall time for summary compatibility. The
current mechanics startup and Newton hot paths use this API for continuation
run wall time, Newton solve wall time, elastic assembly, affine Neumann face
quadrature, Dirichlet application, plastic residual/tangent assembly, and KSP
solves. Deflation orthogonalization, coarse initial guesses, projected PC
application, and projector updates also use shared PETSc events and the shared
elapsed-counter helper. PMG shell setup/apply, apply subphases for fine/P2
smoothing, P1 coarse solves, transfers, residual updates, and operator-update
submetrics now enter the same profiler surface while preserving the legacy
summary counters behind the stats API. The Cython compatibility
bridge also uses the same timer API for assembly and linear-solve helpers.
Hydro seepage run, assembly, and linear-solve timings now enter the same PETSc
logging surface while preserving their existing summary fields. The CLI runner
now routes top-level engine wall time through the same timer API.
Cython compatibility helpers also route engine creation, whole Newton-step wall
time, assembly, RHS build, operator build, KSP setup, KSP solve,
A-orthogonalization, and line-search timers through shared events.
Replay-debug assembly checks use the same timer API as well.
Remaining legacy timers should be migrated behind the same surface as those
algorithms are touched.

## Compatibility Boundaries

Coordinate-keyed `mechanics_bc_nodes.csv` and coupled seepage pressure CSVs are
compatibility bridges. New mesh assets should declare boundary supports through
asset definitions and physical labels so the native path can migrate to
DMPlex labels, coordinate sections, and constrained sections without expanding
case TOMLs.

The compatibility bridge is explicit in artifacts:

- `data/native_problem_manifest.json` records asset-owned labels and BC rules.
- `data/resolved_run_manifest.json` records concrete profile, PC variant, PMG
  rank and coarse/telescope/smoother policy, artifact choices, and a flat
  compatibility section for any remaining coordinate bridge used by one run.
- `data/environment.json` records portable Python/MPI/environment/Git
  provenance.
- `data/mechanics_bc_labels.csv` records coordinate-free DMPlex label/tag
  constraint rows and is preferred by the native mechanics engine.
- `data/mechanics_bc_nodes.csv` is a coordinate/component debug fallback for
  extra mechanics constraints and is emitted only when
  `--write-coordinate-bc-table` is passed. Native startup requires the matching
  `-debug_coordinate_bc_table true` guard before consuming this table; when the
  table is represented through `native_problem_manifest.json`, the same intent
  is recorded as `native_inputs.debug_coordinate_bc_table = true`.
- `data/mechanics_neumann_labels.csv` records coordinate-free mechanics
  boundary load declarations and is handled by native `assembly/neumann.c`.
  Affine `constant-traction` rules assemble through native face quadrature;
  curved geometry patches and non-constant value models fail explicitly until
  those evaluators are implemented.
- `data/seepage_boundary_labels.csv` records coordinate-free seepage head/flux
  supports and is staged by native assembly metadata; coupled mechanics still
  uses pressure CSVs as the active load bridge with
  `native_inputs.seepage_pressure_source =
  "hydro_prepass_coordinate_bridge"` recorded in manifests.
- coupled seepage pressure CSVs remain a temporary pressure-load fallback until
  native boundary/field ingestion is implemented.

## Design References

The organization follows these external patterns:

- PETSc DMPlex/DM labels for mesh topology, geometry, distribution, and boundary
  marking.
- PETSc profiling stages/events for authoritative phase timing.
- libCEED-style separation of global parallel layout, element restriction,
  basis evaluation, and pointwise physics.
- PyLith/MOOSE-style named mesh/material/boundary entities.
- ASPECT-style parameter files that specify only case-relevant deviations from
  documented defaults.
