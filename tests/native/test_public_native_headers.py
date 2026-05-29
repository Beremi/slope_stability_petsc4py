from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src" / "petsc_ssr" / "native"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_native_engine_header_names_supported_entry_surface() -> None:
    header = _read(NATIVE / "include" / "petsc_ssr_engine.h")
    context = _read(NATIVE / "core" / "context.c.inc")
    engine = _read(NATIVE / "core" / "engine_main.c")
    public_engine = _read(NATIVE / "core" / "public_engine.c.inc")

    assert "#ifndef PETSC_SSR_ENGINE_H" in header
    assert '#include "petsc_ssr_algorithms.h"' in header
    assert '#include "petsc_ssr_problem.h"' in header
    assert '#include "petsc_ssr_profile.h"' in header
    assert '#include "petsc_ssr_stats.h"' in header
    assert "Public native engine facade." in header
    assert "unity translation" in header
    assert "P4IndirectSSRRunOptionsString" in header
    assert "#define PetscSsrEngineRunOptionsString P4IndirectSSRRunOptionsString" in header
    assert "engine_api.h" in header
    assert '#include "petsc_ssr_engine.h"' in context
    assert "SsrRunResult" in header
    assert "PetscErrorCode SsrEngineCreate(MPI_Comm comm, const SsrRuntimeProfile *profile, SsrEngine *engine);" in header
    assert "PetscErrorCode SsrEngineSetContinuation(SsrEngine engine, const char name[]);" in header
    assert "PetscErrorCode SsrEngineSetNewton(SsrEngine engine, const char name[]);" in header
    assert "PetscErrorCode SsrEngineSetLinearSolver(SsrEngine engine, const char name[]);" in header
    assert "PetscErrorCode SsrEngineRun(SsrEngine engine, SsrRunResult *result);" in header
    assert "PetscErrorCode SsrEngineDestroy(SsrEngine *engine);" in header

    assert '#include "public_engine.c.inc"' in engine
    assert "struct _p_SsrEngine" in public_engine
    assert "SsrContinuationRegistryFind(name, &ops)" in public_engine
    assert "SsrNewtonRegistryFind(name, &ops)" in public_engine
    assert "SsrLinearRegistryFind(name, &ops)" in public_engine
    assert "PETSC_ERR_SUP" in public_engine
    assert "use PetscSsrEngineRunOptionsString until it is wired" in public_engine
    assert "P4SsrEngine" not in public_engine


def test_public_native_problem_header_owns_manifest_constants() -> None:
    header = _read(NATIVE / "include" / "petsc_ssr_problem.h")
    manifest = _read(NATIVE / "io" / "problem_manifest.c.inc")

    assert "#ifndef PETSC_SSR_PROBLEM_H" in header
    assert 'PETSC_SSR_NATIVE_PROBLEM_MANIFEST_KIND "petsc_ssr_native_problem_manifest"' in header
    assert "PETSC_SSR_NATIVE_PROBLEM_MANIFEST_SCHEMA_VERSION 1" in header
    assert 'PETSC_SSR_DMPLEX_REGION_LABEL "Cell Sets"' in header
    assert 'PETSC_SSR_DMPLEX_BOUNDARY_LABEL "Face Sets"' in header
    assert 'PETSC_SSR_DMPLEX_NATIVE_BOUNDARY_MARKER_LABEL "boundary_marker"' in header
    assert "SsrNativeProblemTopologyStats" in header
    assert "SsrNativeProblemRuleStats" in header

    assert '#include "petsc_ssr_problem.h"' in manifest
    assert "PETSC_SSR_NATIVE_PROBLEM_MANIFEST_KIND" in manifest
    assert "PETSC_SSR_NATIVE_PROBLEM_MANIFEST_SCHEMA_VERSION" in manifest
    assert "PETSC_SSR_DMPLEX_REGION_LABEL" in manifest
    assert "PETSC_SSR_DMPLEX_BOUNDARY_LABEL" in manifest
    assert "PETSC_SSR_DMPLEX_NATIVE_BOUNDARY_MARKER_LABEL" in manifest
    assert "SsrNativeProblemTopologyStats topology_stats" in manifest
    assert "SsrNativeProblemRuleStats     rule_stats" in manifest
    assert "NativeProblemManifestValidateLabelTableFingerprint" in manifest
    assert "NativeManifestFileFingerprint" in manifest
    assert "NATIVE_PROBLEM_MANIFEST_ROW_FINGERPRINT" in manifest
    assert "NativeProblemManifestTopologyStats" not in manifest
    assert "NativeProblemManifestRuleStats" not in manifest


def test_native_neumann_boundary_path_has_dedicated_assembly_module() -> None:
    setup_py = _read(ROOT / "setup.py")
    header = _read(NATIVE / "assembly" / "assembly.h")
    neumann = _read(NATIVE / "assembly" / "neumann.c")
    assembly = _read(NATIVE / "assembly" / "assembly.c")

    assert '_rel(NATIVE_DIR / "assembly" / "neumann.c")' in setup_py
    assert "AssemblyNeumannRule" in header
    assert "AssemblyNeumannLabelStats" in header
    assert "neumann_rule_count" in header
    assert "neumann_rules" in header
    assert "AssemblySeepageBoundaryRule" in header
    assert "seepage_boundary_rule_count" in header
    assert "seepage_boundary_rules" in header
    assert "AssemblyCtxLoadNeumannLabelsCSV" in header
    assert "AssemblyCtxAssembleNeumannResidual" in header
    assert "AssemblyCtxLoadNeumannLabelsCSV" in neumann
    assert "AssemblyCtxValidateNeumannLabelsCSV" in neumann
    assert "AssemblyCtxAssembleNeumannResidual" in neumann
    assert "AssemblyCtxAppendNeumannRule" in neumann
    assert "AssemblyCtxClearNeumannRules" in neumann
    assert "value_model_name" in header
    assert '#include "petsc_ssr_algorithms.h"' in neumann
    assert "NeumannSplitCsvFields" in neumann
    assert "PetscBool in_quotes = PETSC_FALSE" in neumann
    assert "if (in_quotes && *src == '\"')" in neumann
    assert "!in_quotes && (c == ','" in neumann
    assert "nfields == 9" in neumann
    assert "expected exactly 9" in neumann
    assert "NeumannParseGeometryOrder" in neumann
    assert "NeumannValidateNativeStatus" in neumann
    assert "NeumannValueRegistryName" in neumann
    assert "SsrNeumannValueRegistryFind(model_name" in neumann
    assert "last_geometry_order=%\" PetscInt_FMT" in neumann
    assert "native_status=%s" in neumann
    assert "staged_rules=%\" PetscInt_FMT" in neumann
    assert "constant-traction" in neumann
    assert "label_table_validated native_status=%s" in neumann
    assert "ctx->neumann_rule_count == 0" in neumann
    assert "struct _p_SsrNeumannValueCtx" in neumann
    assert "SsrNeumannConstantTractionEvaluate" in neumann
    assert "NeumannPrepareValueModel" in neumann
    assert "NeumannParseConstantTraction" in neumann
    assert "SsrNeumannValueRegistryFind(rule->value_model_name, ops)" in neumann
    assert "value_ops->evaluate((SsrNeumannValueCtx)&value_ctx" in neumann
    assert "NeumannFaceQuadrature" in neumann
    assert "NeumannBuildBasisAlphas" in neumann
    assert "DMPlexVecSetClosure(dm, lsec, rhs_loc, cell, elem_vec, ADD_VALUES)" in neumann
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_NEUMANN" in neumann
    assert "SsrStatsAddNeumannAssembly(&ctx->neumann_stats" in neumann
    assert "MECHANICS_NEUMANN_ASSEMBLY" in neumann
    assert "native_face_quadrature_affine" in neumann
    assert "pending_native_curved_face_quadrature" in neumann
    assert "native curved face quadrature is not implemented yet" in neumann
    assert "AssemblyCtxAssembleNeumannResidual(ctx, f_ext)" in assembly
    assert "AssemblyCtxAppendSeepageBoundaryRule" in assembly
    assert "AssemblyCtxClearSeepageBoundaryRules" in assembly
    assert "SplitQuotedCsvFields" in assembly
    assert "expected exactly 10" in assembly
    assert "label_ready_coordinate_pressure_bridge_active" in assembly
    assert "seepage_boundary_rule_count" in assembly
    assert "AssemblyCtxValidateNeumannLabelsCSV" not in assembly
