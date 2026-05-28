# cython: language_level=3
"""Cython bridge for the self-contained PETSc SSR engine."""

ctypedef int PetscErrorCode

cdef extern from "engine_api.h":
    ctypedef struct P4SsrEngine
    ctypedef struct P4SsrEngineInfo:
        int ranks
        int global_dofs
        int basis_cols
        double rhs_norm
        double elastic_assembly_time
        double create_time
        double deflation_orthogonalization_time
        double deflation_coarse_initial_time
        double deflation_pc_apply_time
        double deflation_projector_time
        int deflation_coarse_initial_calls
        int deflation_projected_pc_calls
    ctypedef struct P4SsrStepResult:
        int converged
        int solved
        int failed
        int stop
        int compute_diffs_out
        double rel_residual
        double rel_correction
        double alpha
        double r_out
        double lambda_out
        double delta_lambda
        double trial_rel
        double abs_delta_lambda
        double initial_decrease
        int linear_its
        int linear_its_w
        int linear_its_v
        int line_search_its
        int newton_its
        double assembly_time
        double solve_time
        double wall_time
    PetscErrorCode P4IndirectSSRRunOptionsString(const char options[])
    PetscErrorCode P4SsrEngineCreateOptionsString(const char options[], P4SsrEngine **out)
    PetscErrorCode P4SsrEngineDestroy(P4SsrEngine **ctxp)
    PetscErrorCode P4SsrEngineGetInfo(P4SsrEngine *ctx, P4SsrEngineInfo *info)
    PetscErrorCode P4SsrEngineBasisCols(P4SsrEngine *ctx, int *cols)
    PetscErrorCode P4SsrEngineTruncateBasis(P4SsrEngine *ctx, int n_keep)
    PetscErrorCode P4SsrEngineAppendBasisFromSlot(P4SsrEngine *ctx, int slot, const char label[])
    PetscErrorCode P4SsrEngineVecZero(P4SsrEngine *ctx, int slot)
    PetscErrorCode P4SsrEngineVecCopy(P4SsrEngine *ctx, int src, int dst)
    PetscErrorCode P4SsrEngineVecWAXPY(P4SsrEngine *ctx, int dst, double alpha, int x, int y)
    PetscErrorCode P4SsrEngineVecAXPY(P4SsrEngine *ctx, int y, double alpha, int x)
    PetscErrorCode P4SsrEngineDotOmega(P4SsrEngine *ctx, int slot, double *omega)
    PetscErrorCode P4SsrEngineScaleToOmega(P4SsrEngine *ctx, int slot, double omega)
    PetscErrorCode P4SsrEngineDisplacementMax(P4SsrEngine *ctx, int slot, double *u_max)
    PetscErrorCode P4SsrEngineWriteSolutionFromSlot(P4SsrEngine *ctx, int slot)
    PetscErrorCode P4SsrEngineAssembleResidualJacobian(P4SsrEngine *ctx, int slot, double lambda_, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineComputeLambdaDerivative(P4SsrEngine *ctx, int slot, double lambda_, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineBuildRegularizedOperator(P4SsrEngine *ctx, double r, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineBuildFixedCorrectionRHS(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineBuildIndirectRHS(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineKSPSetup(P4SsrEngine *ctx, int force_reuse_preconditioner, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineAOrthogonalize(P4SsrEngine *ctx, const char label[], P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineKSPSolveFixedCorrection(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineKSPSolveIndirectW(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineKSPSolveIndirectV(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineFixedLineSearch(P4SsrEngine *ctx, int slot, double lambda_, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineApplyFixedCorrection(P4SsrEngine *ctx, int slot, double alpha, double r_in, int update_basis, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineFormIndirectUpdate(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineIndirectLineSearch(P4SsrEngine *ctx, int slot, double lambda_, double omega_target, double current_rel, double d_lambda, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineAcceptIndirectUpdate(P4SsrEngine *ctx, int slot, double lambda_in, double omega_target, double alpha, double d_lambda, double r_in, int update_basis, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineResidualRel(P4SsrEngine *ctx, int slot, double lambda_, double *rel)
    PetscErrorCode P4SsrEngineSolveElasticInitialGuess(P4SsrEngine *ctx, int slot, double scale, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineAssembleLimitLoad(P4SsrEngine *ctx, int slot, double lambda_ell, double load_t, double r, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineBuildLimitLoadRHS(P4SsrEngine *ctx, double load_t, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineFormLimitLoadUpdate(P4SsrEngine *ctx, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineLimitLoadLineSearch(P4SsrEngine *ctx, int slot, double lambda_ell, double load_t, P4SsrStepResult *out)
    PetscErrorCode P4SsrEngineAcceptLimitLoadUpdate(P4SsrEngine *ctx, int slot, double load_t, double omega_target, double alpha, double d_t, double r_in, int update_basis, P4SsrStepResult *out)

cdef extern from "hydro_seepage.h":
    PetscErrorCode HydroSeepageRunOptionsString(const char options[])


def run_options(str options):
    """Run the PETSc-owned C SSR engine with a serialized PETSc option string."""
    cdef bytes encoded = options.encode("utf-8")
    cdef PetscErrorCode ierr = P4IndirectSSRRunOptionsString(encoded)
    if ierr != 0:
        raise RuntimeError(f"P4IndirectSSRRunOptionsString failed with PETSc error code {ierr}")
    return None


def run_hydro_options(str options):
    """Run the PETSc-owned COMSOL seepage solver with a serialized PETSc option string."""
    cdef bytes encoded = options.encode("utf-8")
    cdef PetscErrorCode ierr = HydroSeepageRunOptionsString(encoded)
    if ierr != 0:
        raise RuntimeError(f"HydroSeepageRunOptionsString failed with PETSc error code {ierr}")
    return None


cdef dict _info_dict(P4SsrEngineInfo info):
    return {
        "ranks": info.ranks,
        "global_dofs": info.global_dofs,
        "basis_cols": info.basis_cols,
        "rhs_norm": info.rhs_norm,
        "elastic_assembly_time": info.elastic_assembly_time,
        "create_time": info.create_time,
        "deflation_orthogonalization_time": info.deflation_orthogonalization_time,
        "deflation_coarse_initial_time": info.deflation_coarse_initial_time,
        "deflation_pc_apply_time": info.deflation_pc_apply_time,
        "deflation_projector_time": info.deflation_projector_time,
        "deflation_coarse_initial_calls": info.deflation_coarse_initial_calls,
        "deflation_projected_pc_calls": info.deflation_projected_pc_calls,
    }


cdef dict _step_dict(P4SsrStepResult out):
    return {
        "converged": bool(out.converged),
        "solved": bool(out.solved),
        "failed": bool(out.failed),
        "stop": bool(out.stop),
        "compute_diffs_out": bool(out.compute_diffs_out),
        "rel_residual": out.rel_residual,
        "rel_correction": out.rel_correction,
        "alpha": out.alpha,
        "r_out": out.r_out,
        "lambda_out": out.lambda_out,
        "delta_lambda": out.delta_lambda,
        "trial_rel": out.trial_rel,
        "abs_delta_lambda": out.abs_delta_lambda,
        "initial_decrease": out.initial_decrease,
        "linear_its": out.linear_its,
        "linear_its_w": out.linear_its_w,
        "linear_its_v": out.linear_its_v,
        "line_search_its": out.line_search_its,
        "newton_its": out.newton_its,
        "assembly_time": out.assembly_time,
        "solve_time": out.solve_time,
        "wall_time": out.wall_time,
    }


cdef void _check(PetscErrorCode ierr, str name) except *:
    if ierr != 0:
        raise RuntimeError(f"{name} failed with PETSc error code {ierr}")


cdef class Engine:
    cdef P4SsrEngine *ctx

    def __cinit__(self, str options):
        cdef bytes encoded = options.encode("utf-8")
        self.ctx = NULL
        _check(P4SsrEngineCreateOptionsString(encoded, &self.ctx), "P4SsrEngineCreateOptionsString")

    def close(self):
        if self.ctx != NULL:
            _check(P4SsrEngineDestroy(&self.ctx), "P4SsrEngineDestroy")

    def __dealloc__(self):
        if self.ctx != NULL:
            P4SsrEngineDestroy(&self.ctx)

    def info(self):
        cdef P4SsrEngineInfo info
        _check(P4SsrEngineGetInfo(self.ctx, &info), "P4SsrEngineGetInfo")
        return _info_dict(info)

    def basis_cols(self):
        cdef int cols = 0
        _check(P4SsrEngineBasisCols(self.ctx, &cols), "P4SsrEngineBasisCols")
        return cols

    def truncate_basis(self, int n_keep):
        _check(P4SsrEngineTruncateBasis(self.ctx, n_keep), "P4SsrEngineTruncateBasis")

    def append_basis_from_slot(self, int slot, str label):
        cdef bytes encoded = label.encode("utf-8")
        _check(P4SsrEngineAppendBasisFromSlot(self.ctx, slot, encoded), "P4SsrEngineAppendBasisFromSlot")

    def vec_zero(self, int slot):
        _check(P4SsrEngineVecZero(self.ctx, slot), "P4SsrEngineVecZero")

    def vec_copy(self, int src, int dst):
        _check(P4SsrEngineVecCopy(self.ctx, src, dst), "P4SsrEngineVecCopy")

    def vec_waxpy(self, int dst, double alpha, int x, int y):
        _check(P4SsrEngineVecWAXPY(self.ctx, dst, alpha, x, y), "P4SsrEngineVecWAXPY")

    def vec_axpy(self, int y, double alpha, int x):
        _check(P4SsrEngineVecAXPY(self.ctx, y, alpha, x), "P4SsrEngineVecAXPY")

    def dot_omega(self, int slot):
        cdef double omega = 0.0
        _check(P4SsrEngineDotOmega(self.ctx, slot, &omega), "P4SsrEngineDotOmega")
        return omega

    def scale_to_omega(self, int slot, double omega):
        _check(P4SsrEngineScaleToOmega(self.ctx, slot, omega), "P4SsrEngineScaleToOmega")

    def displacement_max(self, int slot):
        cdef double u_max = 0.0
        _check(P4SsrEngineDisplacementMax(self.ctx, slot, &u_max), "P4SsrEngineDisplacementMax")
        return u_max

    def write_solution_from_slot(self, int slot):
        _check(P4SsrEngineWriteSolutionFromSlot(self.ctx, slot), "P4SsrEngineWriteSolutionFromSlot")

    def assemble_residual_jacobian(self, int slot, double lambda_):
        cdef P4SsrStepResult out
        _check(P4SsrEngineAssembleResidualJacobian(self.ctx, slot, lambda_, &out), "P4SsrEngineAssembleResidualJacobian")
        return _step_dict(out)

    def compute_lambda_derivative(self, int slot, double lambda_):
        cdef P4SsrStepResult out
        _check(P4SsrEngineComputeLambdaDerivative(self.ctx, slot, lambda_, &out), "P4SsrEngineComputeLambdaDerivative")
        return _step_dict(out)

    def build_regularized_operator(self, double r):
        cdef P4SsrStepResult out
        _check(P4SsrEngineBuildRegularizedOperator(self.ctx, r, &out), "P4SsrEngineBuildRegularizedOperator")
        return _step_dict(out)

    def build_fixed_correction_rhs(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineBuildFixedCorrectionRHS(self.ctx, &out), "P4SsrEngineBuildFixedCorrectionRHS")
        return _step_dict(out)

    def build_indirect_rhs(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineBuildIndirectRHS(self.ctx, &out), "P4SsrEngineBuildIndirectRHS")
        return _step_dict(out)

    def ksp_setup(self, bint force_reuse_preconditioner):
        cdef P4SsrStepResult out
        _check(P4SsrEngineKSPSetup(self.ctx, <int>force_reuse_preconditioner, &out), "P4SsrEngineKSPSetup")
        return _step_dict(out)

    def a_orthogonalize(self, str label):
        cdef bytes encoded = label.encode("utf-8")
        cdef P4SsrStepResult out
        _check(P4SsrEngineAOrthogonalize(self.ctx, encoded, &out), "P4SsrEngineAOrthogonalize")
        return _step_dict(out)

    def ksp_solve_fixed_correction(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineKSPSolveFixedCorrection(self.ctx, &out), "P4SsrEngineKSPSolveFixedCorrection")
        return _step_dict(out)

    def ksp_solve_indirect_w(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineKSPSolveIndirectW(self.ctx, &out), "P4SsrEngineKSPSolveIndirectW")
        return _step_dict(out)

    def ksp_solve_indirect_v(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineKSPSolveIndirectV(self.ctx, &out), "P4SsrEngineKSPSolveIndirectV")
        return _step_dict(out)

    def fixed_line_search(self, int slot, double lambda_):
        cdef P4SsrStepResult out
        _check(P4SsrEngineFixedLineSearch(self.ctx, slot, lambda_, &out), "P4SsrEngineFixedLineSearch")
        return _step_dict(out)

    def apply_fixed_correction(self, int slot, double alpha, double r, bint update_basis):
        cdef P4SsrStepResult out
        _check(P4SsrEngineApplyFixedCorrection(self.ctx, slot, alpha, r, <int>update_basis, &out), "P4SsrEngineApplyFixedCorrection")
        return _step_dict(out)

    def form_indirect_update(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineFormIndirectUpdate(self.ctx, &out), "P4SsrEngineFormIndirectUpdate")
        return _step_dict(out)

    def indirect_line_search(self, int slot, double lambda_, double omega_target, double current_rel, double d_lambda):
        cdef P4SsrStepResult out
        _check(P4SsrEngineIndirectLineSearch(self.ctx, slot, lambda_, omega_target, current_rel, d_lambda, &out), "P4SsrEngineIndirectLineSearch")
        return _step_dict(out)

    def accept_indirect_update(self, int slot, double lambda_, double omega_target, double alpha, double d_lambda, double r, bint update_basis):
        cdef P4SsrStepResult out
        _check(P4SsrEngineAcceptIndirectUpdate(self.ctx, slot, lambda_, omega_target, alpha, d_lambda, r, <int>update_basis, &out), "P4SsrEngineAcceptIndirectUpdate")
        return _step_dict(out)

    def residual_rel(self, int slot, double lambda_):
        cdef double rel = 0.0
        _check(P4SsrEngineResidualRel(self.ctx, slot, lambda_, &rel), "P4SsrEngineResidualRel")
        return rel

    def solve_elastic_initial_guess(self, int slot, double scale):
        cdef P4SsrStepResult out
        _check(P4SsrEngineSolveElasticInitialGuess(self.ctx, slot, scale, &out), "P4SsrEngineSolveElasticInitialGuess")
        return _step_dict(out)

    def assemble_limit_load(self, int slot, double lambda_ell, double load_t, double r):
        cdef P4SsrStepResult out
        _check(P4SsrEngineAssembleLimitLoad(self.ctx, slot, lambda_ell, load_t, r, &out), "P4SsrEngineAssembleLimitLoad")
        return _step_dict(out)

    def build_limit_load_rhs(self, double load_t):
        cdef P4SsrStepResult out
        _check(P4SsrEngineBuildLimitLoadRHS(self.ctx, load_t, &out), "P4SsrEngineBuildLimitLoadRHS")
        return _step_dict(out)

    def form_limit_load_update(self):
        cdef P4SsrStepResult out
        _check(P4SsrEngineFormLimitLoadUpdate(self.ctx, &out), "P4SsrEngineFormLimitLoadUpdate")
        return _step_dict(out)

    def limit_load_line_search(self, int slot, double lambda_ell, double load_t):
        cdef P4SsrStepResult out
        _check(P4SsrEngineLimitLoadLineSearch(self.ctx, slot, lambda_ell, load_t, &out), "P4SsrEngineLimitLoadLineSearch")
        return _step_dict(out)

    def accept_limit_load_update(self, int slot, double load_t, double omega_target, double alpha, double d_t, double r, bint update_basis):
        cdef P4SsrStepResult out
        _check(P4SsrEngineAcceptLimitLoadUpdate(self.ctx, slot, load_t, omega_target, alpha, d_t, r, <int>update_basis, &out), "P4SsrEngineAcceptLimitLoadUpdate")
        return _step_dict(out)
