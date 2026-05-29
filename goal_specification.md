# Architecture handout: target organization for `release/petsc-ssr-engine`

I inspected the branch through GitHub and compared the structure against current PETSc/DMPlex, PETSc profiling/options/KSP practice, and high-order finite-element library organization patterns. I did **not** build or run the branch locally, so the recommendations below are a design-spec review, not a measured performance report.

## 1. Executive assessment

The branch is already moving in the right direction: Python is intended to own case metadata, CLI, notebooks, and reporting, while C/PETSc owns distributed meshes, matrices, vectors, assembly, PMG, deflation, KSP solves, Newton, continuation, and profiling. That boundary is stated explicitly in the branch README and architecture notes.  

The best final architecture is:

**Case TOML = mathematical problem.**
**Mesh asset = geometry, materials, regions, boundary labels, boundary geometry, and physical boundary-condition declarations.**
**Solver profile = algorithm and PETSc policy.**
**Suite = benchmark sweep over cases, ranks, profiles, tolerances, refinements, and machines.**
**Native C/PETSc engine = all distributed numerical state and hot paths.**
**Python = schema validation, asset discovery, run orchestration, generated docs, and post-run summaries only.**

The main changes I recommend are not small local patches. They are a cleanup sweep around the public API boundary: remove migration-era choices from user-facing TOML, stop forcing the C baseline in the generic CLI, make benchmark suites first-class, move coordinate-matched constraints/loads into DMPlex labels/sections, expose continuation/Newton/linear solvers through small C-side operation interfaces, and replace manual timing scattered through algorithms with a centralized PETSc logging/statistics API.

The current code already contains the seeds of this direction: there are compact case TOMLs, generated benchmark READMEs, a `ProblemAssetAPI`, Gmsh physical-name loading, high-order mesh elevation, curved boundary geometry hooks, PETSc option flattening, native subsystem fragments, and PETSc log-stage registration.      

The main risk is that the current transitional architecture still lets too much “migration glue” remain part of the apparent design: very large internal config dataclasses, forced baseline flags, hard-coded PMG active ranks, Python-generated coordinate CSVs for constraints, a single huge `AppCtx`, and a mix of benchmark case metadata with notebook/display details. Those are manageable now; they will become expensive once curved high-order boundaries, Neumann conditions, more solvers, and HPC scaling studies are added.

---

## 2. What the branch already does well

The repository already distinguishes the maintained engine from old benchmark scripts. The README describes `src/petsc_ssr/native` as the maintained C/PETSc kernel area and `benchmarks/cases/<slug>` as the benchmark case surface; it also states that MPI ranks, node counts, and wall time belong to launchers or suites, not case TOMLs. 

The active CLI has the right basic commands: `run`, `case validate`, `case dry-run`, `mesh-only`, and `benchmark init`. That is a good usability foundation because it gives users a validation path before they spend time on a full parallel run. 

The benchmark case layout is simple. A case directory has `case.toml`, `README.md`, notebooks, and a tiny `run.sh` delegating to a common launcher.   

The modern case schema is already much cleaner than the internal dataclasses. It restricts active TOML files to sections like `[case]`, `[mesh]`, `[physics]`, `[continuation]`, `[newton]`, `[linear]`, `[output]`, `[notebook]`, `[seepage]`, and `[geometry]`, and rejects unknown fields. 

The mesh-asset model is the right direction. Assets expose mesh variants, mechanics, seepage, materials, boundary groups, nodesets, and boundary geometry patches through a typed API.  

The native implementation is already organized by subsystem, even though it is compiled as one translation unit for now. The single translation unit includes core context, PMG shell, deflation Krylov, reporting, Newton, continuation, Cython API, replay, and CLI runner fragments. 

The direction also matches PETSc’s intended design philosophy. PETSc DMPlex exists specifically to decouple mesh/discretization layout from solvers, treats cells/faces/edges/vertices uniformly as mesh points, and uses `PetscSection` to connect mesh entities to degrees of freedom. ([PETSc][1])

---

## 3. Main structural problems to fix

### 3.1 The public TOML is clean, but the internal config is still too wide

`LinearSolverConfig` currently carries many PMG, GAMG, Hypre, BDDC, deflation, replay, and policy fields. `ExecutionConfig` still contains migration-era choices such as `mechanics_backend="legacy_array"` and `node_ordering="block_metis"`. 

That is acceptable internally during migration, but it should not become the stable architecture. The final user-facing schema should expose only stable conceptual controls. Low-level solver details should live in solver profiles or PETSc option files, not in every case schema.

### 3.2 The CLI currently forces a baseline

The `petsc-ssr run` path appends `--force-c-baseline` before calling the maintained runner.  That is useful during transition, but it is wrong for the final architecture because it bypasses the user’s mental model: the selected `[linear].profile` should define the solver policy. The default profile can be the C baseline, but the CLI should not silently override algorithm choice.

### 3.3 Hard-coded PMG rank counts are not scalable policy

The baseline PMG-deflated profile currently sets `pmg_shell_p2_active_ranks = 64` and `pmg_shell_p1_active_ranks = 32`.  On a 32-rank local run, that is already inconsistent. On HPC, it will also be wrong for many job sizes. These should become rank-adaptive policies, for example:

```toml
[pmg.p2]
active_ranks = "min(world, 64)"

[pmg.p1]
active_ranks = "min(world, max(1, world/2))"
```

or, better, typed fields:

```toml
[pmg.p2]
active_rank_policy = "cap"
max_active_ranks = 64

[pmg.p1]
active_rank_policy = "fraction"
fraction = 0.5
max_active_ranks = 64
```

The resolved run manifest should record the concrete rank counts actually used.

### 3.4 Coordinate CSVs are a future blocker

The current runner writes `mechanics_bc_nodes.csv` from the Python-built `q_mask`, then C loads constraints by matching coordinates.   The C assembly side also contains coordinate-key lookup logic for pressure/constraint tables. 

This will become fragile for curved high-order boundaries, generated mid-edge/mid-face nodes, mesh adaptation, and large distributed runs. The final design should represent constraints, Neumann boundaries, seepage pressure supports, and curved boundary geometry through DMPlex labels, coordinate sections, and field/section data—not coordinate-matched CSV files.

### 3.5 Algorithm interfaces exist implicitly, but not yet as stable plugin contracts

The native API header already exposes fine-grained functions for creating engines, assembling, solving fixed/indirect steps, line searches, deflation snapshots, and limit-load operations.  But Python `SsrContext` still exposes many fine-grained callbacks as `NotImplementedError`. 

The final architecture should not expose dozens of ad hoc callbacks. It should expose three stable families of swappable algorithms: continuation, nonlinear/Newton, and linear/preconditioner/deflation.

---

## 4. Target repository organization

Use this as the target structure. It preserves your current direction but makes the benchmark/runtime boundary explicit.

```text
slope_stability_petsc4py/
  pyproject.toml
  README.md

  src/
    petsc_ssr/
      cli/
        main.py
        commands/
          run.py
          case.py
          mesh.py
          benchmark.py
          suite.py
          doctor.py

      config/
        schema.py              # public TOML schemas only
        resolver.py            # TOML + profile + asset -> resolved run model
        profiles.py            # solver/newton/continuation profile registry
        validators.py          # unknown fields, deprecated fields, tag policy
        manifest.py            # resolved_config + environment manifest

      assets/
        api.py                 # ProblemAssetAPI and typed specs
        gmsh.py                # Gmsh import/validation
        curved.py              # boundary geometry and high-order projection
        bcs.py                 # Dirichlet/Neumann/Head spec normalization
        registry.py            # discovery of meshes/<asset>/definition.py

      runtime/
        options.py             # typed PETSc option resolver
        runner.py              # Python orchestration only
        results.py             # summary/curve/artifact readers
        environment.py         # PETSc/MPI/git/system capture

      benchmarks/
        registry.py            # enumerate cases/suites/targets
        generate.py            # README/notebook/case skeleton generation
        suites.py              # sweep expansion
        compare.py             # target comparison, tolerances
        report.py              # local markdown/html/CSV summaries

      native/
        include/
          petsc_ssr_engine.h
          petsc_ssr_problem.h
          petsc_ssr_algorithms.h
          petsc_ssr_stats.h
          petsc_ssr_profile.h

        core/
          context.c
          options.c
          engine.c
          lifecycle.c

        mesh/
          dmplex_load.c
          labels.c
          sections.c
          distribution.c
          boundary_geometry.c

        assembly/
          basis.c
          quadrature.c
          mechanics_residual.c
          mechanics_tangent.c
          neumann.c
          constraints.c

        materials/
          mohr_coulomb.c
          davis.c
          registry.c

        algorithms/
          continuation/
            indirect.c
            direct.c
            registry.c
          nonlinear/
            newton.c
            line_search.c
            registry.c
          linear/
            ksp_driver.c
            pmg_shell.c
            deflation.c
            registry.c

        profiling/
          events.c
          stats.c

        io/
          summary_json.c
          curve_csv.c
          petsc_viewers.c
          restart.c

  meshes/
    3d_hetero_slope/
      definition.py
      README.md
      adaptive_family_a_l1.msh
      ...
    fixtures/
      ...

  benchmarks/
    cases/
      3d-heterogeneous-ssr-p4/
        case.toml
        README.md
        simulation.ipynb
        visualisation.ipynb
        run.sh

    suites/
      local-32-smoke.toml
      local-32-strong-scaling.toml
      hpc-strong-scaling.toml
      validation.toml

    targets/
      local-32/
        3d-heterogeneous-ssr-p4.json
      numerical/
        3d-heterogeneous-ssr-p4.json

    tools/
      run_standalone_case.sh

  configs/
    solver_profiles/
      pmg-deflated-baseline.toml
      gamg-p1-baseline.toml
      direct-debug.toml

    petsc/
      pmg_shell_baseline.opts
      debug_options.opts

  tests/
    unit/
    mesh/
    config/
    native_smoke/
    benchmark/
    performance/

  docs/
    architecture.md
    benchmarks.md
    case-schema.md
    solver-profiles.md
    hpc.md
    profiling.md
```

The important rule is that **`src/petsc_ssr/native` should be organized by runtime subsystem, while `benchmarks/` should be organized by user-facing case/suite/target semantics**. Avoid allowing benchmark-specific shortcuts to enter the native core.

---

## 5. Benchmark model: the most important cleanup

### 5.1 Final concepts

Use six separate concepts.

| Concept            | Owns                                                                                                                        | Must not own                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Case**           | Physical/mathematical problem: asset, element order, physics model, algorithm profile names, essential numerical tolerances | MPI ranks, wall time, machine, output folder, job scheduler, repeated sweeps |
| **Asset**          | Mesh variants, physical names, materials, region assignments, BC supports, curved boundary definitions, seepage/head data   | Solver tolerances, continuation parameters, MPI ranks                        |
| **Solver profile** | Reusable algorithm policy: KSP/PC/deflation/PMG/Newton/continuation defaults                                                | Case-specific geometry or material values                                    |
| **Suite**          | Sweep matrix: cases × profiles × ranks × refinements × tolerances × repeats                                                 | Geometry/material definitions                                                |
| **Target**         | Expected numerical/performance baselines with tolerances                                                                    | Run commands                                                                 |
| **Artifact**       | Concrete run output and provenance                                                                                          | Source-of-truth input definitions                                            |

This is already partly documented in your branch: cases should describe mathematical setup, while ranks/node counts belong to launchers or suites.  The final architecture should enforce that rule in the schema.

### 5.2 Case slug policy

Use lower-kebab slugs for directories and IDs:

```text
3d-heterogeneous-ssr-p4
2d-seepage-hydrostatic-p2
3d-curved-neumann-ssr-p2
```

But do not rely on the slug as the only metadata. The slug is for humans and paths. Queryable state should live in fields:

```toml
[case]
id = "3d-heterogeneous-ssr-p4"
title = "3D heterogeneous SSR"
tags = ["regression", "scaling"]
```

Avoid tags such as `"3d"`, `"p4"`, `"ssr"`, and `"mechanics"` if those values already exist in `[mesh]` and `[physics]`. Your current example uses tags `["3d", "p4", "mechanics", "ssr"]`; those are not terrible, but they duplicate structured state and will become annoying for filtering. 

A good rule is:

**Structured state belongs in structured fields. Tags are only for orthogonal labels such as `regression`, `scaling`, `paper-figure`, `slow`, `nightly`, `validation`, `experimental`.**

### 5.3 Clean final `case.toml`

A final case should be shorter than the current example. Do not include defaults with only one meaningful value. Do not include notebook visualization settings in the core case unless the notebook is part of the benchmark definition.

Recommended form:

```toml
[case]
id = "3d-heterogeneous-ssr-p4"
title = "3D heterogeneous SSR"
tags = ["regression", "scaling"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb"
analysis = "ssr"
davis = "B"

[continuation]
profile = "indirect-classic"
omega_max = 6.7e6

[newton]
profile = "regularized-default"
stopping_criterion = "absolute_delta_lambda"
stopping_tol = 1e-4

[linear]
profile = "pmg-deflated-baseline"

[output]
preset = "standard-continuation"
```

Fields to remove from normal cases unless they truly vary:

```toml
refine_levels = 0
partitioner = "parmetis"
tolerance = 0.1
max_iterations = 100
solution = ["vtu", "petscbin"]
history = ["summary_json", "curve_csv"]
```

These should be defaults, profile choices, or suite overrides. The current case file includes many of them. 

### 5.4 Suite TOML

Suites should be the way to create benchmarks easily.

Example:

```toml
[suite]
id = "local-32-strong-scaling"
title = "Local 32-core strong-scaling sweep"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1, 2, 4, 8, 16, 32]
repeats = 3
timeout = "00:45:00"

[overrides.continuation]
step_max = 20

[overrides.output]
preset = "performance"

[collect]
petsc_log_view = true
options_left = true
environment = true
```

This gives users a clean mental model:

```bash
petsc-ssr suite run benchmarks/suites/local-32-strong-scaling.toml
petsc-ssr suite report .local/runs/local-32-strong-scaling
```

### 5.5 Benchmark creation workflow

The final user workflow should be:

```bash
petsc-ssr asset validate meshes/3d_hetero_slope
petsc-ssr benchmark init 3d-heterogeneous-ssr-p4 --asset 3d_hetero_slope --element P4 --analysis ssr
petsc-ssr case validate benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr mesh-only benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr run benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml -n 4
petsc-ssr suite run benchmarks/suites/local-32-smoke.toml
```

Your existing `benchmark init` is a good start; it already generates README and notebooks.  It should be extended to generate a complete case skeleton from an asset and to refuse composite/duplicated tags.

---

## 6. Mesh assets, curved boundaries, and Neumann conditions

### 6.1 Keep all geometry and physical supports in assets

Your current asset model is the right place for this. `meshes/3d_hetero_slope/definition.py` already defines mesh variants, materials, region assignments, and mechanics Dirichlet rules. 

The asset should become the only place for:

```text
mesh variants
region names
boundary names
nodeset names
boundary geometry patches
material models
material-region assignment
Dirichlet support definitions
Neumann support definitions
seepage head/flux definitions
```

The case should merely select:

```toml
[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"
profile = "default"
```

### 6.2 Promote curved boundary geometry to a first-class asset feature

Your code already has the right primitive types: `BoundaryGeometrySpec`, `BoundaryGeometryPatch`, `CanonicalMesh.boundary_geometry`, and `SolverMesh.boundary_geometry`. 

The Gmsh support already detects `boundary_geom` physical names and can apply curved triangle geometry to elevated high-order surface nodes.  

Make this a documented asset declaration:

```python
ASSET = build_problem_asset_3d(
    asset_id="3d_curved_slope",
    asset_dir=ASSET_DIR,
    default_variant="base_l1.msh",
    mesh_variants={...},
    materials={...},
    region_assignment={...},
    boundary_geometry={
        "slope_curve": {
            "support_boundary": "slope_face",
            "geometry_order": 2,
        }
    },
    mechanics={
        "dirichlet": [
            {"target": "base", "components": ["y"]},
            {"target": "x_lock", "components": ["x"]},
            {"target": "z_lock", "components": ["z"]},
        ],
        "neumann": [
            {
                "target": "slope_face",
                "kind": "traction",
                "geometry": "slope_curve",
                "value_model": {"type": "constant", "value": [0.0, -10.0, 0.0]},
            }
        ],
    },
)
```

### 6.3 Replace coordinate-matched side tables with DMPlex-native labels/sections

For the final high-order/curved-boundary implementation, avoid:

```text
Python q_mask -> mechanics_bc_nodes.csv -> C coordinate lookup
Python pressure table -> CSV -> C coordinate lookup
```

Use:

```text
Gmsh physical names -> DMPlex DMLabel
BoundaryGeometryPatch -> coordinate section / coordinate DM
Dirichlet rules -> constrained IS from DMLabel + PetscSection
Neumann rules -> face quadrature over DMLabel support
Seepage pressure/head -> field vector or label-supported boundary data
```

This better matches PETSc DMPlex: DMPlex represents all mesh entities as points, and `PetscSection` maps those points to dofs. ([PETSc][1]) It also removes fragile floating-point coordinate matching in the native hot path.

### 6.4 Neumann assembly design

Add a native module:

```text
native/assembly/neumann.c
```

with responsibilities:

```c
SsrNeumannBCCreateFromAssetLabels(...)
SsrAssembleNeumannResidual(...)
SsrAssembleNeumannTangentIfNeeded(...)
```

Supported value models should be registry-based:

```text
constant_traction
normal_pressure
hydrostatic_pressure
piecewise_linear_head
table_on_boundary
function_pointer_debug
```

Each Neumann BC should know:

```text
target DMLabel name
boundary geometry patch name, optional
quadrature order
value model
coordinate frame: global, normal/tangential, hydrostatic
```

For high-order tetrahedra, surface quadrature should evaluate the geometric map on the boundary face using the same coordinate section used by the volume assembly. Do not generate midpoint loads in TOML; generate them from basis/geometry at assembly time.

---

## 7. Native engine architecture

### 7.1 Keep the one-translation-unit build only as a temporary implementation detail

The current `engine_main.c` intentionally includes ordered `.c.inc` fragments into one translation unit.  That is acceptable during migration, especially if you are protecting performance and static helper visibility.

The final source organization should still expose clean headers and module boundaries even if the build keeps a unity compilation mode:

```c
/* petsc_ssr_engine.h */
typedef struct _p_SsrEngine *SsrEngine;

PetscErrorCode SsrEngineCreate(MPI_Comm comm, const SsrProblemSpec*, const SsrRunOptions*, SsrEngine*);
PetscErrorCode SsrEngineSetContinuation(SsrEngine, const char name[]);
PetscErrorCode SsrEngineSetNewton(SsrEngine, const char name[]);
PetscErrorCode SsrEngineSetLinearSolver(SsrEngine, const char name[]);
PetscErrorCode SsrEngineRun(SsrEngine, SsrRunResult*);
PetscErrorCode SsrEngineDestroy(SsrEngine*);
```

The build can still do:

```c
#include "core/context.c"
#include "assembly/mechanics_tangent.c"
#include "algorithms/continuation/indirect.c"
```

but the architecture should be documented as if modules are independent.

### 7.2 Split `AppCtx` into typed subcontexts

The current native `AppCtx` contains mesh paths, output paths, Newton controls, continuation controls, PMG controls, BDDC controls, replay/debug options, deflation controls, and more. 

Split it conceptually:

```c
typedef struct {
  char mesh[PETSC_MAX_PATH_LEN];
  PetscInt dim;
  PetscInt element_degree;
  PetscInt refine_levels;
  char partitioner[32];
} SsrMeshOptions;

typedef struct {
  char analysis[16];
  char material_model[32];
  char davis_type[8];
  PetscBool seepage_enabled;
} SsrPhysicsOptions;

typedef struct {
  char method[32];
  char predictor[32];
  char step_controller[32];
  PetscReal omega_max;
  PetscInt step_max;
} SsrContinuationOptions;

typedef struct {
  char method[32];
  char stopping_criterion[32];
  PetscReal stopping_tol;
  PetscInt max_it;
  PetscInt damp_max;
} SsrNewtonOptions;

typedef struct {
  char profile[64];
  char ksp_type[32];
  PetscReal rtol;
  PetscInt max_it;
  PetscBool reuse_preconditioner;
  SsrPmgOptions pmg;
  SsrDeflationOptions deflation;
} SsrLinearOptions;

typedef struct {
  char output_dir[PETSC_MAX_PATH_LEN];
  char curve_csv[PETSC_MAX_PATH_LEN];
  char summary_json[PETSC_MAX_PATH_LEN];
  PetscBool write_solution;
  PetscBool write_log_view;
} SsrOutputOptions;
```

Then the engine context becomes understandable:

```c
typedef struct {
  MPI_Comm comm;
  SsrMesh mesh;
  SsrPhysics physics;
  SsrDiscretization disc;
  SsrAssembly assembly;
  SsrLinearSolver linear;
  SsrContinuation continuation;
  SsrNewton newton;
  SsrStats stats;
  SsrProfiler profiler;
  SsrOutput output;
} SsrEngine_;
```

### 7.3 Algorithm operation tables

Continuation, Newton, and linear solvers should be swappable by operation tables.

```c
typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine);
  PetscErrorCode (*run)(SsrEngine, SsrRunResult*);
  PetscErrorCode (*destroy)(SsrEngine);
} SsrContinuationOps;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine);
  PetscErrorCode (*solve)(SsrEngine, SsrNewtonInput*, SsrNewtonResult*);
  PetscErrorCode (*destroy)(SsrEngine);
} SsrNewtonOps;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine);
  PetscErrorCode (*solve)(SsrEngine, Mat A, Vec b, Vec x, SsrLinearResult*);
  PetscErrorCode (*recycle)(SsrEngine, Vec update);
  PetscErrorCode (*destroy)(SsrEngine);
} SsrLinearOps;
```

Profiles then select these names:

```toml
[continuation]
algorithm = "indirect_ssr"
predictor = "secant"
step_controller = "classic"

[newton]
algorithm = "regularized_newton"
line_search = "alg5"

[linear]
algorithm = "ksp_deflated"
pc = "pmg_shell"
```

This lets you add a new continuation or Newton method without editing the case schema.

---

## 8. PETSc solver/options policy

### 8.1 Use PETSc prefixes aggressively

PETSc’s options database supports prefixes for nested objects such as multigrid coarse solvers and application-defined solver objects. ([PETSc][2]) Use this to avoid custom ad hoc names wherever possible.

Recommended prefixes:

```text
-ssr_...                    global engine options
-cont_...                   continuation controller
-newton_...                 nonlinear solve
-ls_...                     line search
-lin_...                    main KSP
-lin_pc_...                 main PC
-pmg_fine_...               PMG fine smoother
-pmg_p2_...                 PMG P2 level
-pmg_p1_...                 PMG P1/coarse level
-defl_...                   deflation layer
-replay_...                 replay/debug tools
```

Then generated PETSc options become inspectable with:

```bash
-options_view
-options_left
```

PETSc explicitly recommends `-options_left` to catch options that were never requested, which is ideal for your “no unused options” goal. ([PETSc][2])

### 8.2 Solver profile files should be profiles, not case extensions

Current `baseline-pmg-deflated.toml` is conceptually correct: it describes a reusable solver profile.  The final profile should separate high-level policy from raw PETSc options:

```toml
description = "Rank-adaptive PMG shell with cached deflation"

[linear]
algorithm = "ksp_deflated"
ksp_type = "fgmres"
rtol = 1e-1
max_it = 200
reuse_preconditioner = true

[deflation]
enabled = true
basis_tolerance = 1e-3
max_vectors = 48
recycle = "accepted_steps"

[pc]
type = "pmg_shell"

[pmg]
rank_policy = "adaptive"

[pmg.p2]
max_active_ranks = 64
ksp_max_it = 10

[pmg.p1]
active_rank_fraction = 0.5
max_active_ranks = 64

[petsc]
options_file = "configs/petsc/pmg_shell_baseline.opts"
extra = []
```

### 8.3 Reuse preconditioners deliberately

PETSc supports preconditioner reuse for repeated solves where matrix changes are small, via `-ksp_reuse_preconditioner true` or `KSPSetReusePreconditioner()`. ([PETSc][3]) That maps directly onto continuation/Newton workloads, but it should be an explicit policy:

```toml
[linear.reuse]
preconditioner = "within_newton"
matrix_policy = "same_sparsity_reuse_pc"
rebuild_on = ["failed_solve", "accepted_step", "iteration_growth"]
iteration_growth_factor = 2.0
```

The final output summary should include:

```json
{
  "linear": {
    "pc_setups": 12,
    "pc_reuses": 87,
    "ksp_solves": 99,
    "total_iterations": 3421
  }
}
```

---

## 9. Profiling and statistics architecture

The current C code registers PETSc log stages for deflation and PMG operations.  That is good, but the final profiling API should hide timing and counters from algorithms as much as possible.

PETSc already supports user-defined events via `PetscLogEventRegister`, `PetscLogEventBegin`, and `PetscLogEventEnd`; the code between begin/end is automatically timed and logged. ([PETSc][4]) PETSc also supports stages with `PetscLogStageRegister`, `PetscLogStagePush`, and `PetscLogStagePop`, and those names appear in `-log_view` output. ([PETSc][4])

Add a centralized profiler module:

```c
typedef enum {
  SSR_EVENT_ASSEMBLE_ELASTIC,
  SSR_EVENT_ASSEMBLE_TANGENT,
  SSR_EVENT_ASSEMBLE_RESIDUAL,
  SSR_EVENT_APPLY_DIRICHLET,
  SSR_EVENT_KSP_SOLVE,
  SSR_EVENT_PMG_SETUP,
  SSR_EVENT_PMG_APPLY,
  SSR_EVENT_DEFLATION_ORTHO,
  SSR_EVENT_DEFLATION_PROJECT,
  SSR_EVENT_LINE_SEARCH,
  SSR_EVENT_OUTPUT_WRITE,
  SSR_EVENT_COUNT
} SsrEvent;

PetscErrorCode SsrProfilerRegister(SsrProfiler*);
PetscErrorCode SsrProfilerBegin(SsrProfiler*, SsrEvent, PetscObject, PetscObject);
PetscErrorCode SsrProfilerEnd(SsrProfiler*, SsrEvent, PetscObject, PetscObject);
```

Then use macros:

```c
SSR_PROFILE_BEGIN(engine, SSR_EVENT_ASSEMBLE_TANGENT, A, u);
PetscCall(SsrAssembleTangent(engine, lambda, u, A));
SSR_PROFILE_END(engine, SSR_EVENT_ASSEMBLE_TANGENT, A, u);
```

Iteration counts should not be updated manually in every algorithm. Use local API calls:

```c
SsrStatsAddNewtonIteration(&engine->stats, phase);
SsrStatsAddLinearSolve(&engine->stats, ksp_its, converged_reason);
SsrStatsAddLineSearchIteration(&engine->stats);
SsrStatsAcceptContinuationStep(&engine->stats, &step_result);
```

The algorithm sees only domain concepts. The statistics writer decides how those become CSV/JSON.

Always support a benchmark option equivalent to:

```bash
-log_view :logs/petsc_log.txt
-options_view :logs/options_view.txt
-options_left
```

PETSc’s `-log_view` is intended as the primary low-overhead performance-monitoring path for PETSc codes. ([PETSc][4])

---

## 10. High-order finite-element kernel organization

For high-order methods, copy the conceptual separation used by libCEED even if you do not depend on libCEED. libCEED decomposes finite-element operators into parallel restriction, element restriction, basis evaluation, and quadrature-point physics; that separation explicitly isolates MPI parallelism, mesh topology, finite-element basis, geometry, and pointwise physics. ([libCEED][5])

For this project, that means:

```text
DMPlex / PetscSF / global Vec
  -> local section / closure
  -> element dof extraction
  -> basis/gradient evaluation
  -> constitutive model at quadrature points
  -> element residual/tangent
  -> PETSc Mat/Vec insertion
```

Map that to modules:

```text
mesh/distribution.c          DMPlex, labels, ownership, overlap
assembly/element_closure.c   closure extraction and orientation
assembly/basis.c             P1/P2/P4 shape functions, quadrature
materials/*.c                Mohr-Coulomb, Davis reduction, seepage coupling
assembly/mechanics_*.c       residual/tangent
linear/*.c                   KSP/PC/deflation/PMG
```

This helps performance because you can optimize the kernel without changing continuation/Newton, and you can change continuation/Newton without touching quadrature or material code.

---

## 11. Packaging and dependency footprint

Your `pyproject.toml` currently lists runtime dependencies including `numpy`, `scipy`, `mpi4py`, `petsc4py`, `h5py`, `meshio`, and `matplotlib`.  For a small local implementation footprint, split these:

```toml
dependencies = [
  "mpi4py",
  "petsc4py",
  "numpy",
]

[project.optional-dependencies]
mesh = ["meshio"]
reports = ["matplotlib", "h5py"]
notebooks = ["jupyter", "ipykernel", "nbformat", "nbclient"]
dev = ["pytest", "ruff", "mypy"]
```

The production run path should need only:

```text
PETSc
petsc4py
mpi4py
native extension
numpy, only if unavoidable for Python-side metadata/reporting
```

Mesh conversion, notebook visualization, HDF5 debug bundles, and plotting should be optional extras. This matters on HPC systems where users often load PETSc/MPI modules and want minimal Python package conflicts.

---

## 12. User-facing commands in the final state

Keep your existing commands and add suite/doctor/report commands.

```bash
petsc-ssr doctor
petsc-ssr case validate benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr case explain benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr mesh-only benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr run benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml --output .local/runs/manual
petsc-ssr benchmark init 3d-curved-neumann-ssr-p2 --asset 3d_curved_slope
petsc-ssr suite run benchmarks/suites/local-32-smoke.toml
petsc-ssr suite report .local/runs/local-32-smoke
petsc-ssr targets compare .local/runs/local-32-smoke benchmarks/targets/local-32
```

`case explain` should print the resolved model:

```json
{
  "case": "3d-heterogeneous-ssr-p4",
  "asset": "3d_hetero_slope",
  "mesh_variant": "adaptive_family_a_l1",
  "element": "P4",
  "analysis": "ssr",
  "continuation_profile": "indirect-classic",
  "newton_profile": "regularized-default",
  "linear_profile": "pmg-deflated-baseline",
  "petsc_options_file": "configs/petsc/pmg_shell_baseline.opts",
  "resolved_pmg": {
    "p2_active_ranks": 32,
    "p1_active_ranks": 16
  }
}
```

Every run artifact should contain:

```text
data/resolved_config.toml
data/resolved_options.txt
data/environment.json
data/problem.json
data/summary.json
data/continuation_curve.csv
logs/petsc_log.txt
logs/options_view.txt
logs/options_left.txt
exports/final_solution.vtu
```

---

## 13. Documentation structure

Write the docs as a user workflow, not as an implementation diary.

```text
docs/
  quickstart.md
  create-a-benchmark.md
  case-schema.md
  assets.md
  curved-boundaries.md
  neumann-bcs.md
  solver-profiles.md
  suite-runs.md
  local-32-testing.md
  hpc.md
  profiling.md
  architecture.md
```

`create-a-benchmark.md` should be a five-step handout:

1. Choose or create a mesh asset.
2. Validate asset physical names and materials.
3. Create a case TOML from a template.
4. Validate/dry-run/mesh-only.
5. Add it to a suite and target file.

Large frameworks such as MOOSE and ASPECT succeed partly because users have documented input structures, examples, and cookbook/benchmark collections rather than only API code. MOOSE documents a strict block-style input syntax and override/include rules; ASPECT’s documentation separates running, parameter files, cookbooks, and benchmarks. ([mooseframework.inl.gov][6]) ([aspect-documentation.readthedocs.io][7])

---

## 14. Local 32-core testing protocol

This is the testing protocol I would use before calling the architecture stable.

### 14.1 Static/schema gate

Run on every commit:

```bash
python -m compileall src tests benchmarks
python -m pytest tests/config tests/benchmark tests/mesh -q
petsc-ssr case validate benchmarks/cases/*/case.toml
petsc-ssr benchmark init --check
```

Checks:

```text
No unknown TOML fields.
No deprecated fields in committed cases.
No duplicate state in tags.
No composite tags.
No case-level MPI ranks/node counts/wall time.
No case-level raw PETSc options except explicitly allowed escape hatch.
Every case resolves to exactly one asset.
Every asset variant exists.
Every solver profile exists.
Every output preset exists.
```

### 14.2 Mesh/asset validation

For each asset and variant:

```bash
petsc-ssr mesh-only benchmarks/cases/<case>/case.toml --output .local/mesh/<case>.json
```

Validate:

```text
All region physical names have material assignments.
All boundary physical names used by BCs exist.
All nodesets used by BCs exist.
P1/P2/P4 elevation gives expected node/cell/surface counts.
Boundary labels are stable across ranks.
Curved boundary control patches match support faces.
Generated high-order boundary nodes lie on the intended curved patch within tolerance.
No coordinate-duplicate ambiguity in canonical mesh.
```

### 14.3 Native smoke tests

Use tiny fixtures, not the expensive 3D case:

```bash
mpiexec -n 1 petsc-ssr run benchmarks/cases/tiny-3d-ssr-p1/case.toml --output .local/smoke/r1
mpiexec -n 2 petsc-ssr run benchmarks/cases/tiny-3d-ssr-p2/case.toml --output .local/smoke/r2
mpiexec -n 4 petsc-ssr run benchmarks/cases/tiny-3d-ssr-p4/case.toml --output .local/smoke/r4
```

Require:

```text
Run completes.
No PETSc unused options.
No NaN/Inf in summary.
Expected number of continuation rows.
Final omega/lambda within tolerance.
Summary JSON schema valid.
VTU/binary outputs readable.
```

### 14.4 Algorithm regression matrix

For a tiny or medium case:

```text
continuation: indirect, direct
newton stopping: relative_residual, relative_correction, absolute_delta_lambda
line search: alg5, armijo_residual
linear: pmg-deflated, pmg-no-deflation, gamg-p1, direct-debug
deflation: on, off, recycle on/off
```

Pass criteria:

```text
All supported combinations validate.
Unsupported combinations fail during validation, not during iteration 50.
All supported combinations produce a summary.
Numerical deltas remain within defined tolerances.
```

### 14.5 Local 32-core strong scaling

Use the current local runner idea, but turn it into a suite. Your branch already has a 32-rank local script. 

Recommended sweep:

```text
Ranks: 1, 2, 4, 8, 16, 32
OMP_NUM_THREADS: 1
Repeats: 3
Case: 3d-heterogeneous-ssr-p4
Profiles: pmg-deflated-baseline
Continuation cap for routine test: step_max = 10 or 20
Full validation run: step_max = 100
```

Collect:

```text
wall_time
setup_time
mesh_distribution_time
elastic_assembly_time
tangent_assembly_time
residual_assembly_time
pc_setup_time
pc_apply_time
ksp_solve_time
deflation_orthogonalization_time
deflation_projector_time
total_newton_iterations
total_linear_iterations
accepted_steps
failed_steps
global_dofs
dofs_per_rank min/avg/max
memory high-water mark if available
MPI reductions from PETSc log
```

Pass/fail logic:

```text
No rank count crashes.
No unused PETSc options.
Final omega/lambda agree across rank counts within tolerance.
Accepted step count is stable or explainably different.
Linear iterations do not grow unexpectedly with ranks.
Median wall time does not regress beyond threshold against stored target.
Performance target uses medians, not single-run wall time.
```

Recommended thresholds:

```text
Numerical:
  abs(lambda_last_delta) <= 1e-6 or case-specific
  abs(omega_last_delta) / omega_ref <= 1e-5
  final_rel <= configured stopping tolerance envelope

Iterations:
  total_newton_its <= target * 1.10 + 2
  total_linear_its <= target * 1.20 + 20

Performance:
  wall_time median <= target * 1.30 for normal CI/perf machine
  assembly_time median <= target * 1.20
  ksp_time median <= target * 1.30
```

On developer laptops, do not fail hard on absolute wall time unless the machine is pinned and documented. Fail hard on numerical changes, unused options, crashes, and large iteration-count regressions.

### 14.6 Full 32-core “release candidate” protocol

Before merging a major architecture sweep:

```bash
petsc-ssr doctor
petsc-ssr suite run benchmarks/suites/local-32-smoke.toml
petsc-ssr suite run benchmarks/suites/local-32-strong-scaling.toml
petsc-ssr suite report .local/runs/local-32-strong-scaling
petsc-ssr targets compare .local/runs/local-32-strong-scaling benchmarks/targets/local-32
```

The report should show:

```text
case/profile/ranks table
speedup table
parallel efficiency table
iteration table
PETSc log summary
top 10 time-consuming PETSc events
options_left status
numerical comparison against target
artifact paths
```

---

## 15. Migration roadmap

### Phase 0 — Freeze the intended public model

Document these invariants:

```text
Cases do not contain launcher/machine settings.
Assets own geometry/material/BC supports.
Profiles own algorithm policy.
Suites own sweeps.
Artifacts own resolved provenance.
Native engine owns distributed numerical data.
```

Update `docs/architecture.md`, `docs/benchmarks.md`, and `docs/case-schema.md` first. Your existing docs already say most of this; tighten them into enforceable rules.   

### Phase 1 — Clean case schema

Remove or deprecate public fields that are defaults or profile details:

```text
mesh.refine_levels = 0
mesh.partitioner = "parmetis"
linear.tolerance when profile already sets it
linear.max_iterations when profile already sets it
output.solution arrays when preset is enough
output.history arrays when preset is enough
notebook visualization from core case
```

Keep override support, but make it explicit:

```toml
[overrides.linear]
rtol = 1e-1
```

### Phase 2 — First-class suites

Add:

```text
benchmarks/suites/*.toml
benchmarks/targets/**/*.json
petsc-ssr suite run
petsc-ssr suite report
petsc-ssr targets compare
```

Stop adding one-off benchmark shell scripts except thin compatibility wrappers.

### Phase 3 — Solver profile resolver

Make profile resolution produce one resolved object:

```text
case.toml + solver_profile.toml + PETSc opts + CLI overrides
  -> resolved_config.toml
  -> resolved_options.txt
```

Remove `--force-c-baseline` from normal `run`. Keep a debug-only `--profile` override.

### Phase 4 — Profiling/stats centralization

Create native `profiling/events.c` and `profiling/stats.c`.

Replace manual timing fields scattered through algorithm code with:

```text
SsrProfilerBegin/End
SsrStatsAdd...
```

Keep PETSc `-log_view` as the authoritative low-level performance source.

### Phase 5 — DMPlex-native BC and curved geometry path

Replace coordinate CSV constraints with:

```text
DMLabel + PetscSection + constrained IS
```

Add native Neumann face integration over boundary labels.

Make curved boundary patches part of the DMPlex/coordinate-section contract, not a visualization/preprocessing side effect.

### Phase 6 — Algorithm operation tables

Introduce registries:

```text
continuation registry
newton registry
linear registry
material registry
neumann value-model registry
```

Move direct/indirect SSR, fixed load, PMG shell, and deflation into these interfaces.

### Phase 7 — Dependency split and installation profiles

Make production install minimal:

```bash
pip install .
pip install .[mesh]
pip install .[notebooks]
pip install .[dev]
```

Keep HPC instructions based on PETSc module + petsc4py + native extension.

---

## 16. Definition of the final “perfect state”

The project is in the desired state when all of the following are true:

```text
A new benchmark can be created by adding one asset definition and one short case TOML.

The case TOML contains no MPI ranks, machine names, wall times, generated output paths, or raw PETSc tuning unless explicitly marked as an override.

Every committed case validates without unknown, deprecated, duplicate, or single-option fields.

Every algorithm choice is made through a continuation/newton/linear profile, not by editing engine code.

Continuation, Newton, and linear solver implementations can be swapped by changing profile names.

All distributed mesh, matrix, vector, KSP, PC, PMG, and deflation state lives in C/PETSc.

Python never copies global matrices/vectors into NumPy on the maintained solve path.

Constraints, materials, curved boundaries, Neumann faces, and seepage supports are label/section driven, not coordinate-CSV driven.

PETSc -options_left is clean on every benchmark.

PETSc -log_view is collected for every performance suite.

The local 32-core suite runs ranks 1, 2, 4, 8, 16, 32 and produces a reproducible report.

The package can be installed in a minimal HPC environment without notebook/plotting dependencies.

Generated READMEs and notebooks are reproducible from case metadata.

The native source tree is browsable by subsystem even if compiled in unity mode for performance.

Performance-critical changes are judged by numerical targets, iteration-count targets, and PETSc profiling, not by anecdotal wall time.
```

The single most important architectural decision is to make **benchmark cases declarative and boring**. All complexity should move either downward into assets/profiles/native PETSc objects, or upward into suites/reports. That gives you fast local experimentation, clean HPC scaling, and a codebase where adding curved tetrahedral boundaries or a new deflated KSP strategy does not require redesigning the benchmark format.

[1]: https://petsc.org/release/manual/dmplex/ "DMPlex: Unstructured Grids — PETSc 3.25.1 documentation"
[2]: https://petsc.org/release/manual/other/ "Other PETSc Features — PETSc 3.25.1 documentation"
[3]: https://petsc.org/release/manual/ksp/ "KSP: Linear System Solvers — PETSc 3.25.1 documentation"
[4]: https://petsc.org/release/manual/profiling/ "Profiling — PETSc 3.25.1 documentation"
[5]: https://libceed.org/en/latest/libCEEDapi/ "Interface Concepts — libCEED 0.12.0
 documentation"
[6]: https://mooseframework.inl.gov/application_usage/input_syntax.html "Input File Syntax | MOOSE"
[7]: https://aspect-documentation.readthedocs.io/en/latest/user/cookbooks/index.html "Cookbooks — ASPECT 3.1.0-pre"
