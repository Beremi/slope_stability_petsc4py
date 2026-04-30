# PETSc Rewrite Presentation Speaker Notes

- Mainline pacing target: about 80-85 minutes for slides 1-57, leaving discussion time.
- Appendix pacing target: backup only.
- Repeat this sentence early and again at the speed section: architecture slides use the current default `P4` mainline; performance slides use the committed `P2` study.
- Keep the tone technical and practical: focus on what changed, what is new, how new cases are added, and where code is edited.
- If asked about seepage `P4` reruns: the concave `L2` seepage mesh is much larger than the dry `L1` meshes on this host, and the local run hit about 119 GB RSS before finishing SSR.

## Slide 01. PETSc Rewrite of Slope Stability
- Target: 1 minute.
- Key message: frame the presentation as a translation from the legacy MATLAB workflow into the current PETSc runtime.
- Point at visually: the two opening bullets only.
- Fallback detail: this is a repository-specific architecture and workflow presentation, not a generic PETSc tutorial.

## Slide 02. Agenda
- Target: 1 minute.
- Key message: the talk is organized around practical extension questions.
- Point at visually: the eight agenda items.
- Fallback detail: low-value repo-internal detail has been pushed into appendix or notes.

## Slide 03. Repository Sources Used Here
- Target: 1 minute.
- Key message: three source layers are used, and each answers a different question.
- Point at visually: legacy MATLAB entry points, PETSc runtime entry points, and local docs/studies.
- Fallback detail: the architecture anchor is the default `3d_hetero_ssr` `P4` benchmark; timing claims later come from the committed `P2` study.

## Slide 04. Section Divider: MATLAB Baseline
- Target: 20 seconds.
- Key message: rebuild the shared MATLAB mental model before explaining the rewrite.
- Point at visually: the section title only.

## Slide 05. The Legacy Execution Model Is Script-Centric
- Target: 2 minutes.
- Key message: one script typically declares the case, prepares FE inputs, launches continuation, and plots figures.
- Point at visually: the right-hand flow ending in MATLAB figures.
- Fallback detail: modular packages existed, but orchestration still lived in the case script.

## Slide 06. MATLAB Package Map
- Target: 1.5 minutes.
- Key message: the old code already had meaningful responsibility splits, just not a unified runner surface.
- Point at visually: `+ASSEMBLY`, `+CONTINUATION`, `+NEWTON`, `+CONSTITUTIVE_PROBLEM`, `+LINEAR_SOLVERS`, `+MESH`, `+VIZ`.
- Fallback detail: the rewrite largely preserves these responsibility categories and changes their packaging.

## Slide 07. MATLAB Continuation And Newton Paths
- Target: 2 minutes.
- Key message: the MATLAB repository already contains multiple nonlinear branches, not only indirect SSR.
- Point at visually: common solve flow first, then the continuation/Newton mapping table.
- Fallback detail: direct SSR enters through `SSR_direct_continuation.m` plus `NEWTON.newton`; indirect SSR and indirect LL each have their own continuation and Newton entry points.

## Slide 08. MATLAB Constitutive Object Is Already A Central Kernel
- Target: 1.5 minutes.
- Key message: the constitutive wrapper is already a central object in MATLAB, so the rewrite preserves that structural idea.
- Point at visually: the constitutive kernel bullets and the short flow on the right.
- Fallback detail: `CONSTITUTIVE.m` mediates reduction, stress, tangent, residual, and timing collection, then calls the lower-level 3D constitutive functions.

## Slide 09. Mesh, Materials, And Dirichlet Labels In MATLAB Are Closely Coupled
- Target: 2 minutes.
- Key message: MATLAB tends to bind mesh storage, FE degree, BC interpretation, and material expansion into the same loader path.
- Point at visually: the loader rows and the consequences column.
- Fallback detail: preview the PETSc contrast: canonical mesh-family storage plus solver-side degree selection.

## Slide 10. MATLAB Visualisation Still Lives Inside The Case Script
- Target: 1.5 minutes.
- Key message: MATLAB already produced useful figures, but the figure path stayed attached to live script state.
- Point at visually: left boxes first, then the short workflow sketch on the right.
- Fallback detail: the rewrite keeps comparable outputs, but relocates them behind exports and notebook helpers.

## Slide 11. Section Divider: What Changed Structurally
- Target: 20 seconds.
- Key message: this is the architectural core.
- Point at visually: the section title only.

## Slide 12. What Stays Familiar, What Changes
- Target: 1.5 minutes.
- Key message: the nonlinear mathematics is familiar; execution packaging is what changes.
- Point at visually: familiar versus changed tiles.
- Fallback detail: this is the cleanest way to avoid a false “full rewrite of the mathematics” reading.

## Slide 13. Responsibility Map: MATLAB Driver To PETSc Modules
- Target: 2 minutes.
- Key message: each familiar MATLAB responsibility still exists, but behind different public surfaces.
- Point at visually: case declaration, runner dispatch, mesh/material preprocessing, constitutive operator, continuation, solver factory.
- Fallback detail: the map is tied to concrete files, not just concepts.

## Slide 14. Architecture Anchor Benchmark
- Target: 1.5 minutes.
- Key message: the structural story is anchored on one real checked-in benchmark path.
- Point at visually: the configuration box first, then the short start-to-continuation pipeline.
- Fallback detail: this benchmark is for explaining the current architecture, not for proving timing claims.

## Slide 15. Config Load And Runner Dispatch
- Target: 1.5 minutes.
- Key message: the new public runtime surface starts at TOML load/validation and explicit case-runner dispatch.
- Point at visually: config-side bullets and the pseudocode path.
- Fallback detail: material rows can be explicit in the benchmark or inherited from mesh-family metadata.

## Slide 16. Assembly Policy And Node Ownership Are First-Class Decisions
- Target: 2 minutes.
- Key message: ownership and ordering are selected before the nonlinear loop because they determine where work lives.
- Point at visually: left policy inputs and right callout.
- Fallback detail: in MATLAB, row ownership never had to be stated. In PETSc, it is a first-order architectural decision.

## Slide 17. Owned Elastic Rows And A Fixed Tangent Pattern
- Target: 2 minutes.
- Key message: structure is prepared once; values are refreshed each Newton step.
- Point at visually: prepared-once versus refreshed-each-Newton lists.
- Fallback detail: this is one of the deepest differences from a “rebuild global sparse tangent every iteration” mindset.

## Slide 18. Overlap Constitutive Ownership And The Rows Kernel
- Target: 2 minutes.
- Key message: the default path places constitutive work on overlap data and writes tangent values directly into a fixed CSR pattern.
- Point at visually: overlap mode on the left and row-slot metadata on the right.
- Fallback detail: the bilinear form is unchanged; the dataflow is what became parallel-native.

## Slide 19. What `pmg_shell` Means On This Mainline
- Target: 1.5 minutes.
- Key message: `pmg_shell` means a shell V-cycle under the outer deflated FGMRES wrapper, operating on the reduced free-space operator.
- Point at visually: MATLAB intuition versus PETSc mainline comparison.
- Fallback detail: in this repository, “parallel PMG” refers to this shell-preconditioned reduced solve path.

## Slide 20. How The PMG Hierarchy Is Built And Rebuilt
- Target: 1.5 minutes.
- Key message: hierarchy geometry and transfers are built once from the reordered mesh family, but the shell is configured on the live matrix each Newton step.
- Point at visually: startup versus Newton-step pseudocode.
- Fallback detail: `_ensure_pmg_state()` is where level transfers become PETSc matrices.

## Slide 21. Current PMG Shell Versus MATLAB HYPRE Intuition
- Target: 2 minutes.
- Key message: MATLAB HYPRE remains the right coarse-solve intuition, but the current mainline solve is now outer FGMRES plus shell PMG plus coarse HYPRE.
- Point at visually: outer Krylov, preconditioner core, coarse solve, smoother row, rebuild timing row.
- Fallback detail: for the robust parallel shell case, the smoother switches to `chebyshev + jacobi` on MPI runs with hierarchy orders `(1,2,4)` or `(1,1,2)`; otherwise the shell keeps `richardson + sor`.

## Slide 22. Section Divider: What Is New And Reusable
- Target: 20 seconds.
- Key message: move from architecture into reusable public capability.
- Point at visually: the section title only.

## Slide 23. What Became Public And Reproducible
- Target: 1.5 minutes.
- Key message: continuation, Newton, linear solver, execution, and export policies are now public config surfaces.
- Point at visually: the three-column table.
- Fallback detail: these policies used to require script edits; now they can be compared across cases.

## Slide 24. Structured Exports And Notebook Workflow
- Target: 1.5 minutes.
- Key message: standard outputs replace workspace-local state.
- Point at visually: the export pipeline first, then the two boxes below.
- Fallback detail: these exports make postprocessing reproducible and rerun-free.

## Slide 25. P4, Quadrature, And 3D Export Are Real Runtime Paths
- Target: 1.5 minutes.
- Key message: higher order and `P4` visualisation are exercised runtime paths, not conceptual placeholders.
- Point at visually: the quadrature figure and the bullets on `P1/P2/P4`, VTK Lagrange export, and pointwise deviatoric strain.
- Fallback detail: keep the claim scoped to the exercised path rather than every possible 3D runner.

## Slide 26. Section Divider: How To Add A New Benchmark
- Target: 20 seconds.
- Key message: this is the practical extension surface.
- Point at visually: the section title only.

## Slide 27. MATLAB Script Versus PETSc Benchmark Folder
- Target: 2 minutes.
- Key message: benchmark authoring moved from editing one top-level script to declaring a benchmark folder with one public config surface.
- Point at visually: asset selection, mesh variant/profile, analysis, numerical settings, then run/inspect outputs.
- Fallback detail: the practical translation is script-local problem data moves into `meshes/<asset>/definition.py`; `case.toml` selects `problem.asset`, `problem.mesh_variant`, optional `problem.profile`, and `problem.analysis`.

## Slide 28. Benchmark Folder Contract And First Run
- Target: 1.5 minutes.
- Key message: the folder is the unit of execution, but `case.toml` is the only document the runner consumes directly.
- Point at visually: bootstrap commands first, then the folder tree, then `./run.sh` and `run_case_from_config`.
- Fallback detail: `run.sh` is a convenience wrapper. `README.md`, `simulation.ipynb`, and `visualisation.ipynb` describe or reuse the config, but they do not define solver behavior. `exports/*` is the stable postprocess surface.

## Slide 29. Choose The Runner Family First
- Target: 1.5 minutes.
- Key message: `problem.asset` plus `problem.analysis` select the generic execution route; the asset definition owns mesh loading, materials, hydraulics, and boundary handling.
- Point at visually: `problem.asset`, `problem.mesh_variant`, optional `problem.profile`, `problem.analysis`, then the split between asset data and numerical settings.
- Fallback detail: dispatch happens in `src/slope_stability/execution/asset_case/runner.py` by resolved dimension, analysis, and seepage capability. The runner receives arrays and masks resolved from the asset, not raw mesh paths or local physics.

## Slide 30. What The Current Public Surface Actually Supports
- Target: 2 minutes.
- Key message: the public config surface is intentionally thin and asset-first; all supported cases share the same selector pattern even when their solver routes differ.
- Point at visually: the mechanics, seepage-only, and seepage-coupled rows, especially which fields stay in `case.toml` and which live in asset definitions.
- Fallback detail: `problem.analysis` accepts `ssr`, `ll`, and `seepage`. Seepage-coupled 3D SSR requires a seepage-capable asset and `analysis = "ssr"`; unsupported combinations are rejected instead of falling through to hidden runner conventions.

## Slide 31. Dry Mechanical Example: 3D Heterogeneous SSR
- Target: 2 minutes.
- Key message: one real `case.toml` is enough to explain the dry 3D mainline: runner family, FE degree, mesh family, continuation method, Newton stop, and linear backend.
- Point at visually: the `[problem]` block first, then `method = "indirect"`, then the `absolute_delta_lambda` stop, then `pc_backend = "pmg_shell"`.
- Fallback detail: this is `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`. The same file also carries `[execution]` for ordering and ownership, plus `[notebook]` for display defaults rather than solver logic.

## Slide 32. Hydro Paths: Seepage Only Versus Seepage-Coupled SSR
- Target: 2 minutes.
- Key message: hydro is selected by the asset capability plus analysis route; hydraulic values and head/flux conditions live with the mesh definition.
- Point at visually: seepage-only on the left, seepage-coupled SSR on the right, then the callout about boundary handling.
- Fallback detail: `case.toml` selects `problem.asset`, `problem.mesh_variant`, optional `problem.profile`, and `problem.analysis`. The generic runner receives already-resolved seepage arrays; it does not own water levels or conductivity defaults.

## Slide 33. Boundary Labels, Materials, And 2D Text Meshes
- Target: 2 minutes.
- Key message: mechanical BC labels, materials, hydraulics, and profiles come from `meshes/<asset>/definition.py`.
- Point at visually: the asset `definition.py`, then the canonical Gmsh physical names, then the `[problem]` selector in `case.toml`.
- Fallback detail: runtime loads a registered asset, promotes the canonical linear Gmsh mesh to the requested element order, and applies declared mechanics/seepage masks from the asset. Generic mesh IO only reads geometry and labels.

## Slide 34. Benchmark Authoring Checklist
- Target: 2 minutes.
- Key message: benchmark authoring is now a short, inspectable sequence, but the order matters.
- Point at visually: the checklist from `[benchmark]` and `[problem]` through the run and export verification.
- Fallback detail: start with `problem.asset`, `problem.mesh_variant`, optional `problem.profile`, and `problem.analysis`; numerical tolerances and solver settings come after the physical problem is resolved.

## Slide 35. Section Divider: Unified Visualisation
- Target: 20 seconds.
- Key message: move from run definition to result inspection.
- Point at visually: the section title only.

## Slide 36. Unified Visualisation Pipeline
- Target: 1.5 minutes.
- Key message: one shared export and reconstruction path now feeds notebooks and viewers.
- Point at visually: the pipeline pseudocode and the three helper paths.
- Fallback detail: `rebuild_case_mesh()` and `build_field_exports()` are the key runtime-side abstractions.

## Slide 37. PETSc 3D Views: Geometry And Warped Displacement
- Target: 1 minute.
- Key message: the rewrite can reproduce the 3D geometry and warped displacement products from standard exports.
- Point at visually: left mesh-outline view, right warped displacement view.
- Fallback detail: both come from the same default indirect 3D SSR visualisation notebook path.

## Slide 38. PETSc Localisation Surface And Top-View Slices
- Target: 1.5 minutes.
- Key message: the same exported deviatoric field supports both full 3D localisation views and top-view analysis slices.
- Point at visually: left 3D deviatoric surface, right slice montage.
- Fallback detail: the top-view slices are configured from benchmark metadata, not hard-coded inside one case script.

## Slide 39. MATLAB And PETSc Reach Similar Slice Products Through Different Workflows
- Target: 1.5 minutes.
- Key message: the visual end product stays familiar, but the production path is now shared and reusable.
- Point at visually: MATLAB image versus PETSc image.
- Fallback detail: this is a workflow comparison slide, not a claim that both plotting stacks are implemented the same way.

## Slide 40. Section Divider: Unified Meshes
- Target: 20 seconds.
- Key message: move from outputs back to how the underlying 3D mesh story changed.
- Point at visually: the section title only.

## Slide 41. MATLAB Mesh Handling And Element Degree Are Entangled
- Target: 1.5 minutes.
- Key message: in MATLAB, geometry storage and FE order tend to move together.
- Point at visually: each row and its consequence.
- Fallback detail: this made higher-order growth more loader-specific.

## Slide 42. PETSc Mesh Families Carry Their Own Metadata
- Target: 1.5 minutes.
- Key message: a mesh family is now a reusable asset with canonical storage, BC labels, and default materials.
- Point at visually: the `DEFINITION` snippet.
- Fallback detail: `problem_assets.py` is the runtime bridge from file path back to family metadata.

## Slide 43. Mesh Family And `elem_type` Are Now Separate Concepts
- Target: 1.5 minutes.
- Key message: geometry/material family selection and FE degree selection are now different decisions.
- Point at visually: the concept table.
- Fallback detail: this is the cleanest contrast with the MATLAB `P2`-centric mesh path.

## Slide 44. Boundary Tags, Material Tags, And Reordering Stay Explicit
- Target: 1.5 minutes.
- Key message: reordering for ownership does not erase the physics labels.
- Point at visually: boundary labels, material ids, and node reorder tiles.
- Fallback detail: labels stay explicit first and the discrete system is reordered around them afterwards.

## Slide 45. Section Divider: Speed Comparison
- Target: 20 seconds.
- Key message: now switch from architecture to controlled performance evidence.
- Point at visually: the section title only.

## Slide 46. Locked Runtime Study Protocol
- Target: 1.5 minutes.
- Key message: all timing claims in the main speed section are from the committed `P2` study, not the `P4` architecture anchor.
- Point at visually: the five protocol bullets.
- Fallback detail: repeat the distinction out loud because otherwise architecture and performance evidence will naturally get mixed.

## Slide 47. Locked P2 Study: Headline View Across The Three Cases
- Target: 1 minute.
- Key message: PETSc is already ahead on the two main dry 3D cases, and seepage has one completed level with a documented limitation.
- Point at visually: the headline table.
- Fallback detail: do not linger here; the later plots carry the story better.

## Slide 48. Homogeneous 3D SSR: PETSc Pulls Ahead As The Mesh Grows
- Target: 1.5 minutes.
- Key message: on the homogeneous dry ladder, PETSc pulls further ahead as level grows.
- Point at visually: continuation on the left, timings on the right.
- Fallback detail: cite the ratio growth from `1.24x` to `3.47x`.

## Slide 49. Homogeneous 3D SSR: Committed P2 Ladder Versus The New P4(L1) Curve
- Target: 1.5 minutes.
- Key message: the black `P4(L1)` rerun is for continuation-shape comparison against the committed `P2` ladder, not for runtime parity claims.
- Point at visually: the black line versus the colored committed curves.
- Fallback detail: mention the fixed PMG shell and `absolute_delta_lambda` stop rule used in the rerun.

## Slide 50. Heterogeneous 3D SSR: Same Trend, Stronger Advantage On Larger Levels
- Target: 1.5 minutes.
- Key message: the same PETSc advantage appears on the heterogeneous dry ladder.
- Point at visually: continuation on the left, timings on the right.
- Fallback detail: cite the ratio growth from `1.35x` to `3.08x`.

## Slide 51. Heterogeneous 3D SSR: Where The New P4(L1) Curve Sits Relative To P2
- Target: 1.5 minutes.
- Key message: the new higher-order dry rerun sits in the same continuation-family comparison space as the committed `P2` curves.
- Point at visually: black curve against the `P2` ladders.
- Fallback detail: this is a continuation-shape comparison, not a replacement for the committed timing protocol.

## Slide 52. Seepage 3D SSR: Only One Level Completed Under The Locked Protocol
- Target: 1.5 minutes.
- Key message: seepage has one committed `P2` comparison level, and the local `P4` rerun stayed incomplete because of host memory limits.
- Point at visually: committed seepage continuation/timing plots and the note in the bullets.
- Fallback detail: if asked, cite the local peak memory issue and note that this is why the slide keeps `P2` seepage evidence only.

## Slide 53. Additional Speedup Gain: The Delta-Lambda Stop Rule
- Target: 1.5 minutes.
- Key message: changing the stop rule yields another controlled performance gain inside the same study branch.
- Point at visually: continuation and timing figures together.
- Fallback detail: this is a protocol sensitivity result, not a different architecture.

## Slide 54. Section Divider: Where To Edit The Code And Close
- Target: 20 seconds.
- Key message: finish with the practical file map that is most useful after the presentation.
- Point at visually: the section title only.

## Slide 55. Code Entry Points For Common Changes
- Target: 2 minutes.
- Key message: most extension questions map cleanly to a small set of entry files.
- Point at visually: question-to-path table.
- Fallback detail: this is the slide worth photographing for a practical starting map after the presentation.

## Slide 56. Algorithm Changes Live In A Few Concentrated Files
- Target: 2 minutes.
- Key message: the main numerical behavior is concentrated enough that adaptation does not require reading the whole repository.
- Point at visually: constitutive, Newton, continuation, linear solver/PMG, runner wiring, and postprocess rows.
- Fallback detail: the `cli/` capture runners are where benchmark-side wiring and algorithm modules meet.

## Slide 57. Final Summary
- Target: 1 minute.
- Key message: the rewrite keeps the old problem recognizable, but changes configuration, ownership-aware assembly, solver layering, and outputs into first-class interfaces.
- Point at visually: the three summary tiles.
- Fallback detail: close by saying that the shortest path for adaptation is now benchmark folder plus mesh-family metadata plus a few concentrated algorithm files.

## Slide 58. Appendix Divider
- Target: only if needed.
- Key message: everything after this is backup.

## Slide 59. Architecture Mainline Versus Performance Study Protocol
- Use only if the `P4` architecture anchor and the committed `P2` timing study still get mixed.

## Slide 60. Optional And Inactive Paths Present In The Repository
- Use only if someone asks what alternative branches still exist outside the mainline story.

## Slide 61. Global Assembly And The Legacy Tangent Kernel
- Use only if someone asks how much legacy-style assembly is still present in the repository.

## Slide 62. Alternative Linear-Solver Branches
- Use only if someone asks about `hypre`, `gamg`, `bddc`, or non-mainline solver branches.

## Slide 63. Seepage Caveat In The Committed Performance Report
- Use only if someone wants the exact wording or limitation context behind the seepage protocol.

## Slide 64. Delta-Lambda Appendix Numbers
- Use only if someone wants the exact appendix timing numbers behind the stop-rule comparison.

## Slide 65. Extra Source Map And Reading Order
- Use only if someone asks for a concrete reading order after the presentation.
