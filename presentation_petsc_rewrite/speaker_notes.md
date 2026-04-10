# PETSc Rewrite Presentation Speaker Notes

- Mainline pacing target: about 84-86 minutes for slides 1-63, leaving a few minutes for transitions and discussion.
- Appendix pacing target: backup material only for discussion, not part of the 90-minute core.
- Repeat this sentence early and again at the speed section: architecture slides use the current default `P4` mainline; performance slides use the committed `P2` study.
- Speak in first person when bridging sections: “I want to keep this distinction clear”, “I keep coming back to this benchmark”, “this is the path I mean by parallel PMG”.

## Slide 01. PETSc Rewrite of Slope Stability
- Target: 1 minute.
- Key message: frame the talk as a translation from the legacy MATLAB mental model into the current PETSc rewrite.
- Point at visually: the shorter opening bullets on translating the MATLAB mental model and on the main design changes I will focus on.
- Fallback detail: note that every claim in the deck comes from repository-local code or docs, not a fresh rerun.

## Slide 02. How I Structure This Talk
- Target: 1 minute.
- Key message: I am locking the reading method early, so the audience knows which benchmark anchor I am using and which comparison I am deliberately not mixing into it.
- Point at visually: the three bullets on MATLAB baseline, the checked-in `P4`/PMG architecture anchor, and the separation from the committed `P2` study.
- Fallback detail: say explicitly that the current architecture story follows the default `P4` benchmark, while runtime claims later follow the locked `P2` report protocol.

## Slide 03. Agenda
- Target: 1 minute.
- Key message: give the audience a clean route through the rewrite from baseline to runtime, architecture, interfaces, visualisation, meshes, and speed.
- Point at visually: the numbered agenda only.
- Fallback detail: say the appendix remains backup material, but the mainline route is linear and deliberate.

## Slide 04. What I Use As Evidence
- Target: 1 minute.
- Key message: there are three source layers and each one answers a different question.
- Point at visually: legacy MATLAB sources, PETSc runtime entrypoints, and the distillation docs/studies.
- Fallback detail: explain that the MATLAB tree gives the original mental model, the runtime code gives executable truth, and the docs give the high-yield mapping.

## Slide 05. Section Divider: MATLAB Baseline
- Target: 20 seconds.
- Key message: the next block rebuilds the old MATLAB mental model first.
- Point at visually: the centered section title and the full-width progress underline.
- Fallback detail: say this is necessary because the rewrite keeps the nonlinear problem but changes how responsibilities are packaged.

## Slide 06. The Legacy Execution Model Is Script-Centric
- Target: 2 minutes.
- Key message: in MATLAB, one script usually declares the case, assembles the operators, runs continuation, and plots results.
- Point at visually: the right-hand flow from quadrature through continuation to figures.
- Fallback detail: emphasize that modular kernels existed, but orchestration was still script-local rather than registry-driven.

## Slide 07. MATLAB Package Map
- Target: 1.5 minutes.
- Key message: MATLAB already had useful modularity, but it was organized as helper packages around the script rather than around one common runner interface.
- Point at visually: `+ASSEMBLY`, `+CONTINUATION`, `+NEWTON`, `+CONSTITUTIVE_PROBLEM`, `+LINEAR_SOLVERS`, `+MESH`, and `+VIZ`.
- Fallback detail: call out that the PETSc rewrite preserves these responsibility categories, but relocates them into runtime modules and config surfaces.

## Slide 08. MATLAB Core Solver Loop
- Target: 2 minutes.
- Key message: the nonlinear lifecycle itself is familiar: predictor, Newton solve, accept or shrink, append history.
- Point at visually: accepted-step lifecycle on the left and the main MATLAB data objects on the right.
- Fallback detail: remind the audience that the historical arrays for `lambda`, `omega`, `u_max`, and counters are the ancestors of the current JSON and HDF5 exports.

## Slide 09. The MATLAB Constitutive Object Is The Main Stateful Kernel Wrapper
- Target: 1.5 minutes.
- Key message: MATLAB centralizes constitutive state in one handle object, but feeds that object with globally assembled FE operators.
- Point at visually: the pseudocode chain from reduction to stress to tangent and residual/tangent return.
- Fallback detail: this is why the rewrite still needs a strong constitutive operator, even though the assembly strategy changes completely.

## Slide 10. Mesh Handling In MATLAB Is Closely Coupled To Element Degree
- Target: 2 minutes.
- Key message: MATLAB tends to tie mesh storage, element degree, and boundary-condition conventions together.
- Point at visually: `load_mesh_P2.m`, `load_mesh_gmsh_waterlevels.m`, and the midpoint utilities.
- Fallback detail: preview the later PETSc contrast: canonical mesh-family storage plus solver-side degree selection.

## Slide 11. MATLAB Visualisation Still Lives Inside The Case Script
- Target: 2 minutes.
- Key message: MATLAB already had useful figure products, but the production path stayed attached to benchmark-local plotting code and live workspace state.
- Point at visually: the left-side split between what MATLAB does well and what remains coupled, then the short right-side workflow sketch.
- Fallback detail: tell the audience that the rewrite keeps comparable outputs, but moves them behind shared exports and shared notebook helpers.

## Slide 12. Section Divider: Prerequisites And Getting Running
- Target: 20 seconds.
- Key message: switch from historical orientation to practical PETSc-side setup.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: say this section is for someone who wants to run one case before reading architecture internals.

## Slide 13. Environment And Bootstrap
- Target: 2 minutes.
- Key message: the repository expects a real PETSc build with HYPRE, not just a lightweight pure-Python environment.
- Point at visually: the two bootstrap commands and the list of what the heavy first run installs.
- Fallback detail: explain that wheel mode can be lighter, but the benchmark-capable path is the full bootstrap.

## Slide 14. Run One Case Or Run The Whole Benchmark Suite
- Target: 2 minutes.
- Key message: both single-case execution and suite execution now flow through the same config surface.
- Point at visually: the `run_case_from_config` invocation and the `run_benchmark_suite` command.
- Fallback detail: mention that `run.sh` inside a benchmark folder is just a convenience wrapper around the same `case.toml`.

## Slide 15. Devcontainer And Standard Outputs
- Target: 1.5 minutes.
- Key message: the rewrite standardizes both environment onboarding and run outputs.
- Point at visually: the devcontainer validation entrypoint and the four standard output files.
- Fallback detail: contrast this with MATLAB, where final arrays and figure state often remained in the live workspace.

## Slide 16. Benchmark Folders Are The User-Facing Contract
- Target: 1.25 minutes.
- Key message: the benchmark folder is now the unit of execution, documentation, notebook generation, and reproducibility.
- Point at visually: `case.toml`, `run.sh`, `README.md`, `[benchmark]`, `[notebook]`, and generated artifacts.
- Fallback detail: say that this folder contract is the rewrite’s replacement for “which top-level MATLAB script should I run?”

## Slide 17. Section Divider: Structural Redesign For Parallelism
- Target: 20 seconds.
- Key message: this is the architectural core of the talk.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: tell the audience this is where the rewrite stops being a translation and becomes a new execution architecture.

## Slide 18. Responsibility Map: MATLAB Driver To PETSc Modules
- Target: 2 minutes.
- Key message: every familiar MATLAB responsibility still exists, but it now lives behind a different public surface.
- Point at visually: the row-by-row mapping from script responsibilities to `case.toml`, runner dispatch, mesh/fem setup, constitutive operator, continuation, and solver factory.
- Fallback detail: stress that the map is not conceptual only; the locations are directly tied to the current runtime tree.

## Slide 19. This Is The Architecture Anchor I Keep Coming Back To
- Target: 1.5 minutes.
- Key message: anchor the architectural discussion on one concrete benchmark path instead of speaking in abstractions.
- Point at visually: the benchmark settings box first, then the short runner pipeline, then the callout that separates this architecture anchor from the later `P2` speed evidence.
- Fallback detail: say out loud that I am using this checked-in benchmark to explain structure, not claiming that every 3D `P4` branch is equally mature.

## Slide 20. Config Load And Runner Dispatch
- Target: 1.5 minutes.
- Key message: the runner stack starts by loading and validating TOML, then selecting a case runner explicitly.
- Point at visually: the config-side bullets and the pseudocode path from config load to exports.
- Fallback detail: mention that material rows can come from the config itself or be inherited from mesh-family metadata.

## Slide 21. Assembly Policy And Node Ownership Are First-Class Decisions
- Target: 2 minutes.
- Key message: ownership and ordering are chosen up front because they determine where work and data live.
- Point at visually: policy inputs on the left and the why-this-matters callout on the right.
- Fallback detail: emphasize that MATLAB assumed one global sparse operator in one address space, whereas PETSc has to decide row ownership early.

## Slide 22. Owned Elastic Rows And A Fixed Tangent Pattern
- Target: 2 minutes.
- Key message: the rewrite separates structure preparation from Newton-step value refresh.
- Point at visually: the “prepared once” list versus the “refreshed every Newton step” list.
- Fallback detail: this is one of the deepest changes relative to MATLAB, because the tangent is no longer a fresh global sparse rebuild every iteration.

## Slide 23. Overlap Constitutive Ownership And The Rows Kernel
- Target: 2 minutes.
- Key message: constitutive work is placed on overlap data needed by owned rows, and the rows kernel writes directly into a fixed CSR pattern.
- Point at visually: the overlap mode bullets and the row-slot metadata names.
- Fallback detail: say the bilinear form is not changing; the dataflow and ownership discipline are changing.

## Slide 24. What `pmg_shell` Means On This Mainline
- Target: 1.75 minutes.
- Key message: when I say `pmg_shell`, I mean a shell V-cycle wrapped around the current reduced free-space operator, under the MATLAB-style outer Krylov interface.
- Point at visually: the left bullets on reduced operator, outer wrapper, shell V-cycle, and shared free-space numbering, then the right comparison between the MATLAB intuition and the PETSc mainline.
- Fallback detail: remind the audience that this slide defines what “parallel PMG” means in the repository today.

## Slide 25. How The PMG Hierarchy Is Built And Rebuilt
- Target: 1.75 minutes.
- Key message: the multilevel geometry and transfers are built once from the reordered case, but the shell preconditioner is still rebuilt on the current operator each Newton step.
- Point at visually: the startup-to-Newton pseudocode on the left and the bullets on stable hierarchy metadata versus live matrix rebuild on the right.
- Fallback detail: say that `_ensure_pmg_state()` caches level transfers, while the shell configuration is refreshed on the current matrix.

## Slide 26. How The Outer Krylov Solve Sits On Top Of PMG
- Target: 1.5 minutes.
- Key message: I have to read the outer solver and the PMG backend together, because the real linear solve lives in that combination.
- Point at visually: the left solve stack from predictor to shell V-cycle, then the right table for `solver_type`, `pc_backend`, smoother choices, and the coarse Hypre solve.
- Fallback detail: this is the right place to say that `solver_type` no longer tells the whole story by itself.

## Slide 27. What Changed Structurally To Enable Parallel Capabilities
- Target: 2 minutes.
- Key message: parallelism comes from four structural extractions: case spec extraction, explicit ownership, split assembly, and solve-stack decoupling.
- Point at visually: the four metric tiles.
- Fallback detail: say this is why the rewrite is a native distributed architecture rather than a MATLAB script with MPI wrapped around it.

## Slide 28. Section Divider: Main New Functionality
- Target: 20 seconds.
- Key message: move from architectural change into new capabilities exposed by that architecture.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: note that these are additions beyond a literal MATLAB translation.

## Slide 29. Shared Registry Across Benchmark Folders
- Target: 1.25 minutes.
- Key message: once the folder contract exists, the registry becomes the cross-case surface for suites, parity reporting, notebook generation, and automation.
- Point at visually: the three tiles for parity suite, additional runnable cases, and shared metadata.
- Fallback detail: distinguish this from the earlier folder-contract slide: here the point is not “what is inside one benchmark?”, but “what can I do uniformly across many benchmarks?”

## Slide 30. Richer Continuation And Solver Controls
- Target: 1.5 minutes.
- Key message: continuation, Newton, linear-solver, and execution policies are now declarative and comparable across cases.
- Point at visually: the table rows for `[continuation]`, `[newton]`, `[linear_solver]`, and `[execution]`.
- Fallback detail: note that stop criteria, warm starts, fine-switch logic, and solver backends are no longer buried in local script edits.

## Slide 31. Structured Exports And Notebook Workflow
- Target: 1.5 minutes.
- Key message: the run now produces stable artifacts that can be consumed later without rerunning the case.
- Point at visually: the export pipeline pseudocode and the “what this improves” box.
- Fallback detail: say this is the enabling step for unified visualisation and for better reproducibility.

## Slide 32. The Rewrite Also Carries A P4 Export And 3D Display Path
- Target: 1.5 minutes.
- Key message: the benchmark I use in the architecture story does carry a real `P4` export and display path, even though I do not want to oversell it as uniform maturity across every 3D runner.
- Point at visually: the quadrature figure and the bullets on `P1`/`P2`/`P4`, VTK Lagrange export, pointwise deviatoric strain export, and the maturity caveat.
- Fallback detail: remind the audience that this talk uses a benchmark that exercises the `P4` path deliberately, while the active docs still keep broad production claims focused on `P2`.

## Slide 33. Mainline Versus Appendix Features
- Target: 1 minute.
- Key message: keep the default story narrow enough to stay readable, and push alternatives into appendix material.
- Point at visually: mainline versus appendix rows for mechanics path, preconditioning, continuation, constitutive ownership, and speed study.
- Fallback detail: say this is a presentation choice, not a claim that appendix branches are unimportant.

## Slide 34. Section Divider: Unified Runners
- Target: 20 seconds.
- Key message: shift from architecture to the user-facing execution contract.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: mention that this is the practical extension surface for new benchmarks.

## Slide 35. Benchmark Folder Contract
- Target: 1.25 minutes.
- Key message: return to the folder layout here only as the practical skeleton a contributor must create or inspect.
- Point at visually: the folder tree and the bullets about one `case.toml` powering CLI, suites, notebooks, and exports.
- Fallback detail: contrast this with slide 16: there the point was the public contract; here the point is the concrete extension skeleton.

## Slide 36. Public TOML Sections
- Target: 1.5 minutes.
- Key message: the runtime surface is intentionally explicit and grouped by responsibility.
- Point at visually: the table of `[problem]`, `[execution]`, `[continuation]`, `[newton]`, `[linear_solver]`, `[seepage]`, `[export]`, and `[[materials]]`.
- Fallback detail: this slide is the answer to “what can I configure without editing Python code?”

## Slide 37. Walkthrough Of The Architecture-Anchor `case.toml`
- Target: 1.5 minutes.
- Key message: one real config is now the shortest path to understanding a benchmark.
- Point at visually: the example `P4` problem block, execution block, and `pc_backend = "pmg_shell"`, then the right-side bullets that explain why this one file is enough to orient the benchmark.
- Fallback detail: mention that notebook metadata lives in the same TOML and therefore shares the same case identity.

## Slide 38. Dispatch Through `run_case_from_config`
- Target: 1.5 minutes.
- Key message: dispatch is explicit, readable, and centered on `problem.case`.
- Point at visually: the pseudocode from config load through case-runner mapping to exports.
- Fallback detail: say this replaces choosing between many top-level MATLAB entry scripts.

## Slide 39. How To Add A New Benchmark: Steps 1 To 3
- Target: 1.5 minutes.
- Key message: adding a case begins with folder creation, benchmark metadata, and filling the runtime blocks.
- Point at visually: the three numbered steps and the callout about inheriting material tables from mesh-family definitions.
- Fallback detail: emphasize that the benchmark contract is documentation plus execution surface, not just a config file.

## Slide 40. How To Add A New Benchmark: Steps 4 To 5
- Target: 1.25 minutes.
- Key message: the last steps are notebook metadata and one real execution that populates reusable artifacts.
- Point at visually: the two numbered steps and the final checklist.
- Fallback detail: say the important validation question is whether `run.sh`, the notebooks, and the exports all point to the same config truth.

## Slide 41. Section Divider: Unified Visualisation
- Target: 20 seconds.
- Key message: move from running the case to consuming the outputs.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: tell the audience the next slides focus on 3D because that is where the MATLAB authors will feel the biggest workflow change.

## Slide 42. Unified Visualisation Pipeline
- Target: 1.5 minutes.
- Key message: visualisation is now a shared postprocess pipeline rather than per-case plotting code.
- Point at visually: the top pipeline pseudocode and the helper modules listed below.
- Fallback detail: say this is the practical meaning of “unified visualisation”: one mesh reconstruction surface, one field naming surface, and one consumer surface.

## Slide 43. PETSc 3D Views: Geometry And Warped Displacement
- Target: 1.5 minutes.
- Key message: these are the notebook-style views of the default indirect benchmark that I actually want the audience to see, not recycled comparison crops.
- Point at visually: the mesh-outline view on the left and the warped displacement view on the right.
- Fallback detail: say that these screenshots came from the `slope_stability_3D_hetero_SSR_default` visualisation path and I used them as the visual target for the deck.

## Slide 44. PETSc Localisation Surface And Top-View Slices
- Target: 1.5 minutes.
- Key message: this is the PETSc view family I want to standardise on: one full localisation surface view and one slice view family from the same exported field.
- Point at visually: the notebook-style boundary-surface plot on the left, then the slice view on the right.
- Fallback detail: say that the slice planes still come from `[notebook]` metadata in the benchmark config, even though the screenshot itself was taken from the notebook output.

## Slide 45. MATLAB And PETSc Reach Similar Slice Products Through Different Workflows
- Target: 1.25 minutes.
- Key message: the slice product can look comparable, but the workflow beneath it is very different now.
- Point at visually: MATLAB on the left, PETSc on the right, then read the callout sentence about where the slice product comes from.
- Fallback detail: say this is one of the most important maintenance changes in the rewrite.

## Slide 46. Reuse In PyVista And ParaView
- Target: 1.5 minutes.
- Key message: standard outputs now support both in-repo notebooks and external viewers cleanly.
- Point at visually: the four output files and the note about rebuilding the case mesh with attached fields.
- Fallback detail: compare this with the old MATLAB workflow of hoping the right arrays still existed in the live workspace.

## Slide 47. Section Divider: Unified Meshes
- Target: 20 seconds.
- Key message: transition from postprocessing to geometric input design.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: tell the audience this is the second biggest conceptual shift after owned-row assembly.

## Slide 48. MATLAB Mesh Handling And Element Degree Are Entangled
- Target: 1.5 minutes.
- Key message: in MATLAB, geometry storage, FE degree, and loader assumptions are tightly coupled.
- Point at visually: the question/answer/consequence table.
- Fallback detail: mention that this is why higher-order growth in MATLAB tends to proliferate utilities and special-case loaders.

## Slide 49. PETSc Mesh Families Carry Their Own Metadata
- Target: 1.5 minutes.
- Key message: a mesh family is now a reusable asset that carries storage conventions, labels, defaults, and materials.
- Point at visually: the `DEFINITION` pseudocode and the bullets about canonical storage and inherited materials.
- Fallback detail: say this is what lets multiple benchmarks share one family without copying material tables everywhere.

## Slide 50. Canonical Gmsh Tet4 Storage With Loader-Side Elevation
- Target: 1.5 minutes.
- Key message: one canonical low-order family file can serve multiple finite-element degrees.
- Point at visually: the pipeline from Gmsh tet4 physical groups to loader-side elevation to `P1`/`P2`/`P4`.
- Fallback detail: explicitly contrast this with the MATLAB `P2` HDF5 path and degree-specific midpoint utilities.

## Slide 51. Mesh Family And `elem_type` Are Now Separate Concepts
- Target: 1.5 minutes.
- Key message: geometry/material-family selection and FE-degree selection are now distinct runtime decisions.
- Point at visually: the table rows for mesh family, `problem.mesh_path`, `problem.elem_type`, and export cell type.
- Fallback detail: say this separation is the core of the new 3D mesh design.

## Slide 52. Boundary Tags, Material Tags, And Reordering Stay Explicit
- Target: 1.5 minutes.
- Key message: the rewrite keeps physical meaning explicit even while reordering for ownership.
- Point at visually: boundary labels, material identifiers, and node reorder tiles.
- Fallback detail: say the system does not bury physics labels under generic partitioning; it reorders around explicit labels.

## Slide 53. Section Divider: Speed Comparison
- Target: 20 seconds.
- Key message: now switch from architecture to the committed evidence.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: repeat that the next section is about the controlled `P2` study, not the current `P4` default.

## Slide 54. Before I Quote Runtimes, Here Is The Locked Study Protocol
- Target: 1.5 minutes.
- Key message: define the protocol before quoting any runtime numbers.
- Point at visually: indirect SSR only, `P2` only, PETSc MPI-8, MATLAB OMP-8, and the three study cases.
- Fallback detail: use the callout sentence verbatim if the audience seems likely to conflate the study with the architecture mainline.

## Slide 55. Locked P2 Study: Headline View Across The Three Cases
- Target: 1 minute.
- Key message: before I dive into the plots, I want one study-only table that summarizes what the locked `P2` report actually established across the three cases.
- Point at visually: completed levels first, then the MATLAB-over-PETSc ratios, then the takeaway column.
- Fallback detail: use this frame to restate that the seepage row is deliberately narrow and that the next level remains part of the reported limitation.

## Slide 56. Homogeneous 3D SSR: PETSc Pulls Ahead As The Mesh Grows
- Target: 1 minute.
- Key message: on the homogeneous 3D dry case, PETSc’s advantage widens as the mesh gets larger.
- Point at visually: the continuation plot and timing plot, then the ratio bullet from `1.24x` to `3.47x`.
- Fallback detail: mention the absolute numbers only if asked; the trend is the main point.

## Slide 57. Heterogeneous 3D SSR: Same Trend, Stronger Advantage On Larger Levels
- Target: 1 minute.
- Key message: the heterogeneous dry case shows the same scaling pattern, with PETSc clearly ahead on larger levels.
- Point at visually: the timing growth and the ratio bullet from `1.35x` to `3.08x`.
- Fallback detail: say this is the closest committed study analogue to the current default heterogeneous architecture benchmark.

## Slide 58. Seepage 3D SSR: Only One Level Completed Under The Locked Protocol
- Target: 1 minute.
- Key message: the seepage story is promising at the completed level but must be presented honestly with its current limitation.
- Point at visually: the completed `concave_L2` comparison and the note that the next level failed under the locked study protocol.
- Fallback detail: stress that the report deliberately stops at completed evidence rather than smoothing over the limitation.

## Slide 59. Additional Speedup Gain: The Delta-Lambda Stop Rule
- Target: 1 minute.
- Key message: PETSc speed depends not only on backend choice but also on stopping protocol, and the delta-lambda variant is a concrete extra speedup on top of the main study branch.
- Point at visually: the continuation and timing figures first, then the bullets comparing delta-lambda against the residual baseline and against MATLAB.
- Fallback detail: say explicitly that this is a stop-policy gain, not a different solver architecture.

## Slide 60. Section Divider: Close
- Target: 20 seconds.
- Key message: move from evidence back to what the MATLAB authors should retain.
- Point at visually: the centered section title and the progress underline.
- Fallback detail: say the appendix is available for side branches and caveats after the main summary.

## Slide 61. Key Takeaways For The Original MATLAB Authors
- Target: 1 minute.
- Key message: summarize the five durable takeaways on packaging, config, owned-row assembly, standard exports, and current speed evidence.
- Point at visually: all five bullets in sequence.
- Fallback detail: if time is short, read bullets two, three, and four only; those are the biggest workflow changes.

## Slide 62. Recommended Reading Order For New Contributors
- Target: 30 seconds.
- Key message: give the audience a practical map for follow-up reading after the talk.
- Point at visually: the question-to-file table.
- Fallback detail: recommend starting with the default `P4` architecture benchmark config, then `run_case_from_config.py`, then the map document.

## Slide 63. Final Summary
- Target: 30 seconds.
- Key message: close with what stayed the same, what changed, and what should be discussed next.
- Point at visually: the three metric tiles.
- Fallback detail: use the rightmost tile to prompt discussion about production defaults versus appendix branches.

## Slide 64. Appendix Divider: Optional Paths, Caveats, And Additional Reading
- Target: 10 seconds.
- Key message: the remaining slides are backup material for questions.
- Point at visually: the centered appendix title and the full-width underline.
- Fallback detail: say you will use these only if the room wants to go deeper on side branches or report caveats.

## Slide 65. Architecture Mainline Versus Performance Study Protocol
- Target: 1 minute if used.
- Key message: restate the benchmark split one final time in tabular form.
- Point at visually: purpose, case anchor, element degree, backend focus, and what each path proves.
- Fallback detail: if someone quotes a runtime from the speed study against a `P4` architecture slide, come back to this frame.

## Slide 66. Optional And Inactive Paths Present In The Repository
- Target: 1 minute if used.
- Key message: several alternative branches remain in the repo, but they are not part of the default story.
- Point at visually: direct SSR, LL, BDDC, Hypre, GAMG, and alternative constitutive ownership modes.
- Fallback detail: mention that the map document annex preserves these paths without letting them dominate first-time orientation.

## Slide 67. Global Assembly And The Legacy Tangent Kernel
- Target: 1 minute if used.
- Key message: legacy-style global rebuild and tangent paths still exist, but they are no longer the mainline architecture.
- Point at visually: the comparison rows for global rebuild, legacy tangent kernel, and legacy scatter-style assembly.
- Fallback detail: this frame is useful if the audience asks whether the rewrite still has a more MATLAB-like fallback path.

## Slide 68. Alternative Linear-Solver Branches
- Target: 1 minute if used.
- Key message: multiple solver backends remain available for studies, but `pmg_shell` is the mainline narrative.
- Point at visually: `hypre`, `gamg`, `bddc`, and `DIRECT`.
- Fallback detail: use this frame if the audience wants to discuss where future tuning or production hardening should happen.

## Slide 69. Seepage Caveat In The Committed Performance Report
- Target: 1 minute if used.
- Key message: the seepage limitation is specific, documented, and intentionally left visible.
- Point at visually: the bullets about `concave_L2`, the next-level crash after hierarchy construction and solver setup, and the same-mesh fallback diagnostics.
- Fallback detail: say this is exactly the kind of report honesty that should guide future performance communication.

## Slide 70. Delta-Lambda Appendix Numbers
- Target: 1 minute if used.
- Key message: the delta-lambda stopping rule materially changes PETSc runtimes in the heterogeneous dry case.
- Point at visually: the side-by-side residual and delta-lambda PETSc numbers against MATLAB.
- Fallback detail: keep the interpretation narrow: it is a protocol-sensitivity note, not a replacement baseline.

## Slide 71. Extra Source Map And Reading Order
- Target: 30 seconds if used.
- Key message: end with the highest-yield files for self-study.
- Point at visually: the ordered list from the map document through runner, tangent, solver, and notebook support.
- Fallback detail: if the audience only remembers two starting points, recommend the map document and the default `P4` architecture benchmark config.
