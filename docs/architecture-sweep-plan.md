# Architecture Sweep Plan

This sweep follows `goal_specification.md` and keeps the numerical path intact.

## Implement Now

1. Freeze the public model: cases are mathematical, assets own geometry/material/BC supports, profiles own solver policy, suites own sweeps/resource launch policy, artifacts own provenance, and C/PETSc owns distributed numerical state.
2. Tighten case validation: reject launcher/machine/artifact fields, duplicated structured tags, and case-level linear solver tuning.
3. Resolve continuation/Newton/linear profiles before launch, including explicit algorithm selectors, adaptive PMG active-rank policy, concrete PMG coarse/telescope/smoother policy, requested-vs-concrete PC variant choices, and concrete choices in run artifacts.
4. Remove normal-run reliance on `--force-c-baseline`; keep it as an explicit debug override.
5. Add suite parsing, local 32-core sweep definitions, resource/launcher-aware resolved manifest and per-run command provenance generation, direct-run command provenance, report scaffolding, and target-comparison scaffolding.
6. Move notebook display metadata into per-case sidecars so source case TOMLs stay mathematical.
7. Add `case explain`, `asset validate`, and `doctor` commands for the public workflow.
8. Add native header scaffolding for typed subcontexts, continuation/Newton/linear registries, and centralized stats/profiling, including shared PETSc event and stage registration.
9. Add native-ready problem artifacts that record asset-owned labels, materials, BC rules, boundary geometry, and label/tag constraint rows without coordinate tables.
10. Split optional Python dependencies so the maintained PETSc runtime has a small HPC footprint.
11. Keep generated run options clean: project-owned options must be consumed by the native engine or explicitly classified as PETSc-owned prefixed KSP/PC/DM/logging options.

## Defer Deliberately

1. Expand native `native_problem_manifest.json` ingestion beyond the current metadata, topology-label, support-count, BC-rule-count, artifact-path consistency, label-table row-count validation, and label-table row-fingerprint parity checks; affine mechanics `constant-traction` Neumann rows now assemble through native face quadrature and evaluate through the Neumann value-model registry, while curved geometry-patch Neumann and non-constant value models remain explicit native deferrals; then retire coordinate-matched mechanics/seepage CSV compatibility bridges.
2. Continue moving existing C timing fields behind the new `SsrProfiler`/`SsrStats` API throughout all hot paths. CLI engine run wall time, continuation run wall time, Newton solve wall time, Newton assembly/solves/line-search phases, elastic setup, deflation orthogonalization/coarse/projector/PC-apply timing, PMG shell setup/apply, PMG apply subphases, PMG operator-update submetrics, hydro seepage run/assembly/linear-solve timing, replay-debug assembly checks, and Cython engine-create/whole-step/assembly/RHS/operator/KSP setup/solve/A-orthogonalization/line-search compatibility helpers now enter the shared profiler; PMG/deflation PETSc stages are now registered through that same profiling surface, and preserved PMG/deflation summary elapsed counters now accumulate through a shared stats helper.
3. Implement native curved/high-order Neumann face integration and additional Neumann value-model evaluators over asset-declared boundary supports.
4. Continue lifting existing implementations behind typed native registry operation tables. Direct versus indirect continuation dispatch, fixed-load versus indirect SSR Newton dispatch, normal Newton-to-linear solve dispatch, deflated Krylov method dispatch, and Mohr-Coulomb material point evaluation are now registry-owned. Newton now passes a typed `SsrLinearCtx` carrying the native solver plus resolved linear profile view into the linear registry, the resolved profile has nested PMG/deflation subcontext views, and PMG setup policy for shell V-cycle plus `pcmg` now reads the resolved PMG view. Python profile artifacts now expose the same PMG coarse/telescope/smoother policy that the native view consumes; the deeper unity-build PMG/KSP bodies remain unchanged.
5. Convert the unity-build C fragments into separately browsable modules only after profiling shows no regression risk.
6. Populate local 32-core numerical/performance targets from measured runs.

## Review Gates

- `python -m compileall -q src tests benchmarks/tools`
- `python -m pytest tests/config tests/benchmark tests/assets tests/native -q`
- `python -m pytest tests/config/test_native_options_contract.py -q`
- `petsc-ssr profile validate`
- `petsc-ssr benchmark init --check`
- `petsc-ssr case validate --all`
- `petsc-ssr suite validate benchmarks/suites/local-32-smoke.toml`
- `petsc-ssr suite expand benchmarks/suites/local-32-smoke.toml`
- Practical smoke runs with PETSc/MPI when the native extension is built.
