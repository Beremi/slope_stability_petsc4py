# Profiling

PETSc logging is the authoritative low-level performance source. Performance
suites should collect `-log_view` and `-options_left`, and reports should treat
timings as measured data rather than claims.

The native mechanics engine exposes a small profiling API in
`src/petsc_ssr/native/include/petsc_ssr_stats.h`:

- `SsrProfilerRegister`
- `SsrProfilerBegin` / `SsrProfilerEnd`
- `SsrProfilerStagePush` / `SsrProfilerStagePop`
- `SsrProfileTimerBegin` / `SsrProfileTimerEnd`
- `SsrStatsAccumulateElapsed`
- `SsrStatsAddNewtonIteration`
- `SsrStatsAddLinearSolve`
- `SsrStatsAcceptContinuationStep`
- `SsrStatsAddNewtonStepAssembly`
- `SsrStatsAddNewtonStepLinearSolve`
- `SsrStatsAddNewtonStepLineSearch`
- `SsrStatsAddHydroAssembly`
- `SsrStatsAddHydroLinearSolve`
- `SsrStatsAddNeumannAssembly`

Algorithm code should call this generic API for checkpoints, counters, and
timed regions. New algorithm code should not introduce phase-local `PetscTime`
bookkeeping or local `PetscLogStageRegister` state when a shared event, stage,
or counter is available.

Current shared events cover:

- elastic, affine Neumann, residual, tangent, Dirichlet, and KSP solve phases;
- engine creation/run, operator build, RHS build, and KSP setup helper phases;
- Newton line-search phases, including Cython compatibility bridge helpers;
- continuation run and Newton solve wall-time phases;
- PMG setup/apply and V-cycle subphases;
- PMG operator update, Galerkin products, redistribution, submatrix extraction,
  and concatenation;
- deflation orthogonalization, coarse initial guesses, projected PC application,
  and projector updates.
- hydro seepage run, assembly, and linear-solve phases.
- replay-debug assembly checks.

Hydro seepage summary fields such as assembly time, solve time, and total
linear iterations are accumulated through `SsrHydroStats` helpers. PMG and
deflation subphase summaries still preserve their existing output fields, but
elapsed-time additions now go through `SsrStatsAccumulateElapsed` rather than
open-coded `+=` updates inside the algorithms. The solvers may still format
those values in run summaries, but the counters themselves live behind the
shared stats API.

Affine mechanics Neumann face-quadrature counts and assembly time are
accumulated through `SsrNeumannStats` and logged as the
`SSR Assemble Neumann` PETSc event. Curved/high-order Neumann rows still fail
explicitly before assembly rather than entering this event with an approximate
load.

Shared PETSc stages currently cover deflation orthogonalization/coarse/projector
phases and PMG residual, transfer, P2, P1, and fine-smoothing phases. The stage
names are registered by the profiling module so algorithm fragments call stable
`SSR_PROFILE_STAGE_*` helpers instead of owning stage globals.

Run artifacts connect profiling to benchmark reports:

```text
logs/stdout.txt
logs/options_left.txt
logs/options_view.txt
logs/petsc_log.txt
data/summary.json
data/resolved_run_manifest.json
```

`logs/options_left.txt` is materialized by the suite runner from captured
`stdout.txt` when `options_left` collection is enabled, because PETSc's
unused-option diagnostics are part of normal finalization output in this run
mode. Reports prefer the materialized file and fall back to `stdout.txt` for
older run roots.

`suite report` reads completed `summary.json` files and writes median report
sections plus `report.csv` and `report.scaling.csv`. When `logs/petsc_log.txt`
exists, it also extracts the top event timings per run into the Markdown
`PETSc Log Events` section and `report.petsc-events.csv`; the full PETSc log
remains the authoritative source. `targets compare` uses the same median
grouping, so performance targets are not checked against a single repeat.

Native PETSc smoke and performance checks require a working `petsc4py` install
and the compiled `petsc_ssr.native._core` extension. Static tests still verify
that native algorithms use the shared profiling surface and that option tokens
have consumers.
