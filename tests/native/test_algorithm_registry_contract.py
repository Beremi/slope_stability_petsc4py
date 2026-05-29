from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src" / "petsc_ssr" / "native"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_algorithm_header_uses_typed_context_and_result_handles() -> None:
    header = _read(NATIVE / "include" / "petsc_ssr_algorithms.h")

    assert "typedef struct _p_SsrContinuationCtx *SsrContinuationCtx;" in header
    assert "typedef struct _p_SsrNewtonCtx *SsrNewtonCtx;" in header
    assert "typedef struct _p_SsrLinearCtx *SsrLinearCtx;" in header
    assert "typedef struct _p_SsrMaterialCtx *SsrMaterialCtx;" in header
    assert "typedef struct _p_SsrNeumannValueCtx *SsrNeumannValueCtx;" in header
    assert "SsrContinuationResult" in header
    assert "SsrNewtonInput" in header
    assert "SsrNewtonResult" in header
    assert "SsrLinearResult" in header
    assert "SsrMaterialPointInput" in header
    assert "SsrMaterialPointResult" in header
    assert "PetscBool        plastic;" in header
    assert "PetscReal  gamma_sat;" in header
    assert "PetscReal  gamma_unsat;" in header
    assert "SsrNeumannValueInput" in header
    assert "SsrNeumannValueResult" in header
    assert "void *result" not in header
    assert "void *input" not in header


def test_native_app_context_exposes_typed_resolved_profile_view() -> None:
    profile_header = _read(NATIVE / "include" / "petsc_ssr_profile.h")
    context = _read(NATIVE / "core" / "context.c.inc")

    assert "SsrRuntimeProfile" in profile_header
    assert "char      algorithm[32];" in profile_header
    assert "char      pc_variant[32];" in profile_header
    assert "char      requested_pc_variant[32];" in profile_header
    assert "char      pc_variant_fallback_reason[64];" in profile_header
    assert "SsrPmgOptions" in profile_header
    assert "SsrDeflationOptions" in profile_header
    assert "SsrPmgOptions       pmg;" in profile_header
    assert "SsrDeflationOptions deflation;" in profile_header
    assert "char      apply_backend[32];" in profile_header
    assert "char      coarse_ksp_type[32];" in profile_header
    assert "char      coarse_inner_pc_type[32];" in profile_header
    assert "char      coarse_telescope_subcomm_type[32];" in profile_header
    assert "char      p2_telescope_subcomm_type[32];" in profile_header
    assert "char      p2_telescope_ksp_type[32];" in profile_header
    assert "char      p2_telescope_pc_type[32];" in profile_header
    assert "PetscInt  p2_active_ranks;" in profile_header
    assert "PetscInt  p1_active_ranks;" in profile_header
    assert "PetscInt  coarse_lu_max_dofs;" in profile_header
    assert "PetscInt  coarse_telescope_active_ranks;" in profile_header
    assert "PetscReal coarse_ksp_rtol;" in profile_header
    assert "PetscInt  coarse_ksp_max_it;" in profile_header
    assert "PetscInt  p2_telescope_active_ranks;" in profile_header
    assert "PetscReal p2_telescope_ksp_rtol;" in profile_header
    assert "PetscInt  p2_telescope_ksp_max_it;" in profile_header
    assert "PetscBool coarse_gamg_aggressive_square_graph;" in profile_header
    assert "PetscBool check_coarse_transfers;" in profile_header
    assert "PetscBool enabled;" in profile_header
    assert "char      solver[32];" in profile_header
    assert "char      projector[32];" in profile_header
    assert "PetscBool krylov_persistent;" in profile_header
    for field in (
        "SsrMeshOptions         mesh;",
        "SsrPhysicsOptions      physics;",
        "SsrContinuationOptions continuation;",
        "SsrNewtonOptions       newton;",
        "SsrLinearOptions       linear;",
        "SsrOutputOptions       output;",
    ):
        assert field in profile_header

    assert '#include "petsc_ssr_profile.h"' in context
    assert "SsrRuntimeProfile resolved_profile;" in context
    assert "AppCtxSyncResolvedProfile" in context
    assert "app->resolved_profile.continuation.algorithm" in context
    assert "app->resolved_profile.continuation.omega_max = app->omega_max" in context
    assert "app->resolved_profile.newton.algorithm" in context
    assert "app->resolved_profile.newton.max_it = app->newton_max_it" in context
    assert "app->resolved_profile.linear.algorithm" in context
    assert "app->resolved_profile.linear.pc_variant" in context
    assert "app->resolved_profile.linear.requested_pc_variant" in context
    assert "app->resolved_profile.linear.pc_variant_fallback_reason" in context
    assert "requested_variant_name" in context
    assert "variant_fallback_reason" in context
    assert "app->resolved_profile.linear.pmg_p2_active_ranks = app->pmg_shell_p2_active_ranks" in context
    assert "app->resolved_profile.linear.pmg.p2_active_ranks = app->pmg_shell_p2_active_ranks" in context
    assert "app->resolved_profile.linear.pmg.p1_active_ranks = app->pmg_shell_p1_active_ranks" in context
    assert "app->resolved_profile.linear.pmg.lag_preconditioner = app->pmg_lag_preconditioner" in context
    assert "app->resolved_profile.linear.pmg.coarse_telescope_active_ranks = app->pmg_coarse_telescope_active_ranks" in context
    assert "app->resolved_profile.linear.pmg.coarse_ksp_rtol = app->pmg_coarse_telescope_ksp_rtol" in context
    assert "app->resolved_profile.linear.pmg.coarse_ksp_max_it = app->pmg_coarse_telescope_ksp_max_it" in context
    assert "app->resolved_profile.linear.pmg.p2_telescope_active_ranks = app->pmg_p2_telescope_active_ranks" in context
    assert "app->resolved_profile.linear.pmg.p2_telescope_ksp_rtol = app->pmg_p2_telescope_ksp_rtol" in context
    assert "app->resolved_profile.linear.pmg.p2_telescope_ksp_max_it = app->pmg_p2_telescope_ksp_max_it" in context
    assert "app->resolved_profile.linear.pmg.coarse_gamg_aggressive_square_graph = app->pmg_coarse_gamg_aggressive_square_graph" in context
    assert "app->resolved_profile.linear.pmg.check_coarse_transfers = app->pmg_check_coarse_transfers" in context
    assert "app->resolved_profile.linear.deflation.enabled = app->use_deflation" in context
    assert "app->resolved_profile.linear.deflation.basis_tol = app->deflation_basis_tol" in context
    assert "app->resolved_profile.linear.deflation.krylov_persistent = app->deflation_krylov_persistent" in context
    assert "PetscCall(AppCtxSyncResolvedProfile(app));" in context


def test_native_pmg_active_rank_defaults_are_mpi_adaptive() -> None:
    context = _read(NATIVE / "core" / "context.c.inc")

    assert "ResolveDefaultPmgShellActiveRanks" in context
    assert "PetscCallMPI(MPI_Comm_size(comm, &comm_size));" in context
    assert "app->pmg_shell_p2_active_ranks                = 0;" in context
    assert "app->pmg_shell_p1_active_ranks                = 0;" in context
    assert "pmg_shell_p2_active_ranks_set" in context
    assert "pmg_shell_p1_active_ranks_set" in context
    assert "world / 2" in context
    assert "app->pmg_shell_p2_active_ranks                = 64;" not in context
    assert "app->pmg_shell_p1_active_ranks                = 32;" not in context


def test_native_continuation_step_controller_uses_classic_public_name() -> None:
    context = _read(NATIVE / "core" / "context.c.inc")

    assert 'PetscStrncpy(app->omega_step_controller, "classic"' in context
    assert "Omega step controller policy, currently classic" in context
    assert '-omega_step_controller currently supports only classic' in context
    assert 'PetscStrncpy(app->omega_step_controller, "legacy"' not in context


def test_native_algorithm_registry_is_in_unity_build_and_names_default_families() -> None:
    engine = _read(NATIVE / "core" / "engine_main.c")
    registry = _read(NATIVE / "algorithms" / "registry.c.inc")
    context = _read(NATIVE / "core" / "context.c.inc")

    assert "../algorithms/registry.c.inc" in engine
    assert "SsrContinuationRegistryFind" in registry
    assert "SsrNewtonRegistryFind" in registry
    assert "SsrLinearRegistryFind" in registry
    assert "SsrMaterialRegistryFind" in registry
    assert "SsrNeumannValueRegistryFind" in registry
    assert "SsrNativeContinuationOps" in registry
    assert "SsrNativeContinuationRegistryFind" in registry
    assert "SsrNativeNewtonInput" in registry
    assert "SsrNativeNewtonOps" in registry
    assert "SsrNativeNewtonRegistryFind" in registry
    assert "SsrNativeLinearInput" in registry
    assert "SsrNativeLinearResult" in registry
    assert "SsrNativeLinearOps" in registry
    assert "SsrNativeLinearRegistryFind" in registry
    assert "struct _p_SsrLinearCtx" in registry
    assert "LinearSolverCtx        *solver;" in registry
    assert "const SsrLinearOptions *options;" in registry
    assert "struct _p_SsrLinearResult" in registry
    assert "SsrNativeLinearCtxBind" in registry
    assert "SsrLinearCtx     linear;" in registry
    assert "input->linear->solver" in registry
    assert "linear->options->algorithm" in registry
    assert "Resolved linear context algorithm %s does not match native linear operations %s" in registry
    assert "SsrNativeLinearSolve" in registry
    assert "SsrNativeDeflationInput" in registry
    assert "SsrNativeDeflationResult" in registry
    assert "SsrNativeDeflationOps" in registry
    assert "SsrNativeDeflationRegistryFind" in registry
    assert "SsrNativeDeflationSolve" in registry
    assert '{"indirect", SSRContinuationSolve}' in registry
    assert '{"direct", SSRDirectContinuationSolve}' in registry
    assert '{"fixed-load", SsrNativeFixedLoadNewtonSolve}' in registry
    assert '{"indirect-ssr", SsrNativeIndirectNewtonSolve}' in registry
    assert '{"fgmres", DEFLATION_SOLVER_FGMRES, SsrNativeDeflatedFGMRESSolve}' in registry
    assert '{"matlab_dfgmres", DEFLATION_SOLVER_MATLAB_DFGMRES, SsrNativeDeflatedMatlabDFGMRESSolve}' in registry
    assert '{"dfgmres", DEFLATION_SOLVER_MATLAB_DFGMRES, SsrNativeDeflatedMatlabDFGMRESSolve}' in registry
    assert '{"cg", DEFLATION_SOLVER_CG, SsrNativeDeflatedCGSolve}' in registry
    assert '{"pmg-deflated", SsrNativeKspLinearSolve}' in registry
    assert '{"pmg", SsrNativeKspLinearSolve}' in registry
    assert '{"gamg", SsrNativeKspLinearSolve}' in registry
    assert '{"bddc", SsrNativeKspLinearSolve}' in registry
    assert '{"fetidp", SsrNativeKspLinearSolve}' in registry
    assert '{"none", SsrNativeKspLinearSolve}' in registry
    assert '{"debug-direct", SsrNativeKspLinearSolve}' in registry
    assert "PetscStrcasecmp(name, ssr_continuation_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_native_continuation_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_native_newton_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_native_linear_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_newton_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_linear_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_material_registry[i].name" in registry
    assert "PetscStrcasecmp(name, ssr_neumann_value_registry[i].name" in registry
    assert '{"indirect", NULL, NULL, NULL}' in registry
    assert '{"direct", NULL, NULL, NULL}' in registry
    assert '{"fixed-load", NULL, NULL, NULL}' in registry
    assert '{"indirect-ssr", NULL, NULL, NULL}' in registry
    assert '{"pmg-deflated", NULL, NULL, NULL, NULL}' in registry
    assert '{"pmg", NULL, NULL, NULL, NULL}' in registry
    assert '{"gamg", NULL, NULL, NULL, NULL}' in registry
    assert '{"bddc", NULL, NULL, NULL, NULL}' in registry
    assert '{"fetidp", NULL, NULL, NULL, NULL}' in registry
    assert '{"none", NULL, NULL, NULL, NULL}' in registry
    assert "SsrMaterialMohrCoulombEvaluate" in registry
    assert '{"mohr_coulomb", NULL, SsrMaterialMohrCoulombEvaluate, NULL}' in registry
    assert '{"mohr_coulomb_ssr", NULL, SsrMaterialMohrCoulombEvaluate, NULL}' in registry
    assert '{"mohr_coulomb_limit_load", NULL, SsrMaterialMohrCoulombEvaluate, NULL}' in registry
    assert '{"constant", NULL, NULL, NULL}' in registry
    assert '{"constant-traction", NULL, SsrNeumannConstantTractionEvaluate, NULL}' in registry
    assert '{"normal-pressure", NULL, NULL, NULL}' in registry
    assert '{"hydrostatic-pressure", NULL, NULL, NULL}' in registry
    assert '{"piecewise-linear-head", NULL, NULL, NULL}' in registry
    assert '{"table-on-boundary", NULL, NULL, NULL}' in registry
    assert '{"function-pointer-debug", NULL, NULL, NULL}' in registry
    assert "PETSC_ERR_ARG_UNKNOWN_TYPE" in registry
    assert "ValidateAlgorithmRegistrySelection" in context
    assert "PetscOptionsString(\"-continuation_algorithm\"" in context
    assert "PetscOptionsString(\"-newton_algorithm\"" in context
    assert "PetscOptionsString(\"-linear_algorithm\"" in context
    assert "AppCtxNormalizeAlgorithmSelectors" in context
    assert "SsrContinuationRegistryFind(app->continuation_algorithm" in context
    assert "SsrNewtonRegistryFind(app->newton_algorithm" in context
    assert "SsrLinearRegistryFind(app->linear_algorithm" in context
    assert 'SsrNewtonRegistryFind("fixed-load"' in context
    assert 'SsrMaterialRegistryFind("mohr_coulomb"' in context
    assert 'SsrNeumannValueRegistryFind("constant"' not in context


def test_native_cli_dispatches_continuation_through_registry() -> None:
    runner = _read(NATIVE / "core" / "cli_runner.c.inc")

    assert "const SsrNativeContinuationOps *continuation_ops = NULL;" in runner
    assert "SsrNativeContinuationRegistryFind(app.continuation_algorithm, &continuation_ops)" in runner
    assert "continuation_ops->solve(" in runner
    assert "PetscStrcasecmp(app.continuation_method, \"direct\"" not in runner


def test_native_continuations_dispatch_newton_through_registry() -> None:
    indirect = _read(NATIVE / "continuation" / "continuation_indirect_ssr.c.inc")
    direct = _read(NATIVE / "continuation" / "continuation_direct_ssr.c.inc")

    assert 'SsrNativeNewtonRegistryFind("fixed-load", &fixed_newton_ops)' in indirect
    assert "SsrNativeNewtonRegistryFind(app->newton_algorithm, &continuation_newton_ops)" in indirect
    assert "fixed_newton_ops->solve(" in indirect
    assert "continuation_newton_ops->solve(" in indirect
    assert "PetscCall(FixedLambdaNewtonSolve" not in indirect
    assert "PetscCall(IndirectNewtonSolve" not in indirect

    assert "SsrNativeNewtonRegistryFind(app->newton_algorithm, &newton_ops)" in direct
    assert "newton_ops->solve(" in direct
    assert "PetscCall(FixedLambdaNewtonSolve" not in direct


def test_native_newton_dispatches_linear_through_registry() -> None:
    fixed = _read(NATIVE / "nonlinear" / "newton_fixed_load.c.inc")
    indirect = _read(NATIVE / "nonlinear" / "newton_indirect_ssr.c.inc")

    for source in (fixed, indirect):
        assert "const SsrNativeLinearOps *linear_ops = NULL;" in source
        assert "struct _p_SsrLinearCtx    linear_ctx;" in source
        assert "SsrNativeLinearRegistryFind(app->linear_algorithm, &linear_ops)" in source
        assert "SsrNativeLinearCtxBind(solver, &app->resolved_profile.linear, &linear_ctx)" in source
        assert "SsrNativeLinearSolve(linear_ops, &linear_ctx" in source
        assert "SsrNativeLinearSolve(linear_ops, solver" not in source
        assert "PetscCall(SolveLinearSystem" not in source


def test_pmg_shell_vcycle_uses_resolved_pmg_options_subcontext() -> None:
    pmg = _read(NATIVE / "linear" / "pmg_shell.c.inc")
    create = pmg.split("static PetscErrorCode PMGShellCreateHierarchy", 1)[1].split(
        "static PetscErrorCode PMGShellUpdateOperators", 1
    )[0]
    update = pmg.split("static PetscErrorCode PMGShellUpdateOperators", 1)[1].split(
        "static PetscErrorCode PMGShellResidual", 1
    )[0]

    assert "const SsrPmgOptions *pmg;" in pmg
    assert "ctx->pmg  = &app->resolved_profile.linear.pmg;" in pmg
    assert "PMGShellConfigureSmootherKSP(KSP ksp, const SsrPmgOptions *pmg" in pmg
    assert "PMGShellConfigureCoarseKSP(KSP ksp, const SsrPmgOptions *pmg" in pmg
    assert "pmg->smoother_ksp_type" in pmg
    assert "pmg->smoother_pc_type" in pmg
    assert "pmg->smoother_max_it" in pmg
    assert "pmg->coarse_ksp_type" in pmg
    assert "pmg->coarse_inner_pc_type" in pmg
    assert "pmg->coarse_ksp_rtol" in pmg
    assert "pmg->coarse_ksp_max_it" in pmg
    assert "pmg->coarse_gamg_aggressive_square_graph" in pmg

    assert "const SsrPmgOptions *pmg = ctx->pmg;" in create
    assert "PMG shell hierarchy requires resolved PMG options" in create
    assert "PMGActiveLayoutCreate(comm, p2_vec, pmg->p2_active_ranks, pmg->shell_subcomm_type" in create
    assert "PMGActiveLayoutCreate(comm, p1_vec, pmg->p1_active_ranks, pmg->shell_subcomm_type" in create
    assert "if (pmg->check_coarse_transfers)" in create
    assert "PMGShellConfigureSmootherKSP(ctx->smooth4, pmg" in create
    assert "ctx->app->pmg_" not in create

    assert "const SsrPmgOptions *pmg = ctx->pmg;" in update
    assert "PMG shell operator update requires resolved PMG options" in update
    assert "PMGShellConfigureSmootherKSP(ctx->smooth2, pmg" in update
    assert "PMGShellConfigureCoarseKSP(ctx->coarse1, pmg" in update
    assert "ctx->app->pmg_" not in update


def test_pmg_setup_policy_uses_resolved_pmg_options_subcontext() -> None:
    pmg = _read(NATIVE / "linear" / "pmg_shell.c.inc")
    configure = pmg.split("static PetscErrorCode ConfigurePMG", 1)[1]

    assert "app->pmg_" not in pmg
    assert "ChoosePMGCoarsePC(const SsrPmgOptions *pmg" in pmg
    assert "SetPMGTelescopeDefaults(const SsrPmgOptions *pmg" in pmg
    assert "SetPMGP2TelescopeDefaults(const SsrPmgOptions *pmg" in pmg
    assert "LinearSolverSetupPMGTransferChecks(LinearSolverCtx *solver, DM dm, const AppCtx *app, const SsrPmgOptions *pmg)" in pmg
    assert "const SsrPmgOptions *pmg = &app->resolved_profile.linear.pmg;" in configure
    assert "PMG setup requires resolved PMG options" in configure
    assert "SetPMGTelescopeDefaults(pmg, comm)" in configure
    assert "SetPMGP2TelescopeDefaults(pmg, comm)" in configure
    assert "ChoosePMGCoarsePC(pmg, dm_p1" in configure
    assert "if (pmg->check_coarse_transfers)" in configure
    assert "pmg->coarse_redundant_group_size" in configure
    assert "pmg->coarse_gamg_aggressive_square_graph" in configure
    assert "pmg->smoother_ksp_type" in configure
    assert "pmg->smoother_pc_type" in configure
    assert "pmg->smoother_max_it" in configure


def test_native_deflated_krylov_dispatches_through_registry() -> None:
    linear = _read(NATIVE / "linear" / "deflation_krylov.c.inc")
    prepared = linear.split(
        "static PetscErrorCode LinearSolverSolvePrepared(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char *label, PetscBool nonlinear_tangent, PetscBool basis_already_orthogonalized, PetscInt *its)\n{",
        1,
    )[1].split(
        "static PetscErrorCode LinearSolverFinishPreparedKSP", 1
    )[0]

    assert "const SsrNativeDeflationOps *deflation_ops = NULL;" in prepared
    assert "SsrNativeDeflationRegistryFind(solver->app->deflation_solver, &deflation_ops)" in prepared
    assert "SsrNativeDeflationSolve(deflation_ops, solver, ksp, rhs, x, label, its, &reported_rel)" in prepared
    assert "DeflatedFGMRESSolve(solver, ksp" not in prepared
    assert "DeflatedMatlabDFGMRESSolve(solver, ksp" not in prepared
    assert "DeflatedCGSolve(solver, ksp" not in prepared


def test_native_assembly_dispatches_material_points_through_registry() -> None:
    assembly = _read(NATIVE / "assembly" / "assembly.c")
    cells = assembly.split("static PetscErrorCode AssembleCells", 1)[1].split(
        "PetscErrorCode AssembleElasticProblem", 1
    )[0]

    assert '#include "petsc_ssr_algorithms.h"' in assembly
    assert "const SsrMaterialOps *material_ops = NULL;" in cells
    assert 'SsrMaterialRegistryFind("mohr_coulomb", &material_ops)' in cells
    assert "SsrMaterialPointInput  material_input;" in cells
    assert "SsrMaterialPointResult material_result;" in cells
    assert "material_input.plastic = plastic;" in cells
    assert "material_ops->evaluate(NULL, &material_input, &material_result)" in cells
    assert "material_result.gamma_sat" in cells
    assert "material_result.gamma_unsat" in cells
    assert "MaterialMCFromRegion" not in cells
    assert "MaterialMCPlasticStressTangent" not in cells
    assert "MaterialMCElasticStressTangent" not in cells


def test_native_summary_records_resolved_linear_and_pmg_provenance() -> None:
    reporting = _read(NATIVE / "reporting" / "reporting.c.inc")

    assert '\\"native_linear_algorithm\\": \\"%s\\"' in reporting
    assert '\\"linear_algorithm\\": \\"%s\\"' in reporting
    assert '\\"pc_variant\\": \\"%s\\"' in reporting
    assert '\\"requested_pc_variant\\": \\"%s\\"' in reporting
    assert '\\"pc_variant_fallback_reason\\": \\"%s\\"' in reporting
    assert '\\"pmg_p2_active_ranks\\": %" PetscInt_FMT' in reporting
    assert '\\"pmg_p1_active_ranks\\": %" PetscInt_FMT' in reporting
    assert '\\"reuse_preconditioner\\": %s' in reporting
    assert '\\"pmg_lag_preconditioner\\": %" PetscInt_FMT' in reporting
    assert '\\"deflation_projector\\": \\"%s\\"' in reporting
    assert '\\"deflation_max_vectors\\": %" PetscInt_FMT' in reporting
    assert "app->linear_algorithm, app->linear_algorithm" in reporting
    assert "app->variant_name, app->requested_variant_name" in reporting
    assert "app->requested_variant_name, app->variant_fallback_reason" in reporting
    assert "app->pmg_shell_p2_active_ranks" in reporting
    assert "app->pmg_shell_p1_active_ranks" in reporting
    assert 'app->reuse_linear_solver ? "true" : "false"' in reporting
    assert "app->deflation_projector_name" in reporting


def test_fine_grained_engine_api_is_marked_as_debug_compatibility() -> None:
    header = _read(NATIVE / "include" / "engine_api.h")

    assert "Cython debug-loop compatibility bridge." in header
    assert "do not treat them as the public case or solver API" in header
    assert "petsc_ssr_algorithms.h" in header


def test_python_context_exposes_debug_engine_ops_instead_of_ad_hoc_callbacks() -> None:
    context = _read(ROOT / "src" / "petsc_ssr" / "context.py")
    continuation = _read(ROOT / "src" / "petsc_ssr" / "continuation.py")
    limit_load = _read(ROOT / "src" / "petsc_ssr" / "limit_load.py")

    assert "def debug_engine_ops" in context
    assert "EngineOps(self.create_engine())" in context
    run_method = context.split("    def run(self) -> EngineRunResult:", 1)[1].split("    def run_python_loop", 1)[0]
    assert "return self.run_monolithic()" in run_method
    assert "run_python_loop" not in run_method
    assert "ctx.debug_engine_ops()" in continuation
    assert "ctx.debug_engine_ops()" in limit_load
    assert "fine-grained" not in context
    for callback_name in (
        "assemble",
        "form_regularized_operator",
        "solve_indirect_pair",
        "evaluate_trial",
        "accept_trial",
        "rescale_to_omega",
        "snapshot_deflation",
        "restore_deflation",
        "append_deflation_from_update",
    ):
        assert f"def {callback_name}" not in context
