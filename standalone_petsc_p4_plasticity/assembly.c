#include "assembly.h"
#include "material_mc.h"

static void GradPhys(const P4Basis *basis, PetscInt q, PetscInt b, const PetscReal invJ[9], PetscReal grad[3])
{
  const PetscReal *dref = &basis->basis_der[(q * basis->n_basis + b) * 3];
  for (PetscInt d = 0; d < 3; ++d) grad[d] = invJ[0 * 3 + d] * dref[0] + invJ[1 * 3 + d] * dref[1] + invJ[2 * 3 + d] * dref[2];
}

static void AddBTransposeStress(PetscScalar elem[], PetscInt a, const PetscReal grad[3], const PetscReal stress[6], PetscReal weight)
{
  elem[3 * a + 0] += weight * (grad[0] * stress[0] + grad[1] * stress[3] + grad[2] * stress[5]);
  elem[3 * a + 1] += weight * (grad[1] * stress[1] + grad[0] * stress[3] + grad[2] * stress[4]);
  elem[3 * a + 2] += weight * (grad[2] * stress[2] + grad[1] * stress[4] + grad[0] * stress[5]);
}

static void BColumn(PetscInt comp, const PetscReal grad[3], PetscReal bcol[6])
{
  for (PetscInt i = 0; i < 6; ++i) bcol[i] = 0.0;
  if (comp == 0) {
    bcol[0] = grad[0];
    bcol[3] = grad[1];
    bcol[5] = grad[2];
  } else if (comp == 1) {
    bcol[3] = grad[0];
    bcol[1] = grad[1];
    bcol[4] = grad[2];
  } else {
    bcol[5] = grad[0];
    bcol[4] = grad[1];
    bcol[2] = grad[2];
  }
}

static PetscReal BDB(const PetscReal bi[6], const PetscReal D[36], const PetscReal bj[6])
{
  PetscReal v = 0.0;
  for (PetscInt r = 0; r < 6; ++r) {
    for (PetscInt s = 0; s < 6; ++s) v += bi[r] * D[r + 6 * s] * bj[s];
  }
  return v;
}

static PetscErrorCode CellRegion(AssemblyCtx *ctx, PetscInt cell, PetscInt *region)
{
  PetscFunctionBeginUser;
  *region = 1;
  if (ctx->cell_sets) {
    PetscCall(DMLabelGetValue(ctx->cell_sets, cell, region));
    if (*region < 0) *region = 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CopyConstrainedGlobalIndices(AssemblyCtx *ctx)
{
  const PetscInt *idx;
  PetscBool       debug = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(ISAllGather(ctx->constrained_is, &ctx->constrained_all_is));
  PetscCall(ISSortRemoveDups(ctx->constrained_all_is));
  PetscCall(ISGetLocalSize(ctx->constrained_all_is, &ctx->n_constrained_all));
  if (ctx->n_constrained_all > 0) {
    PetscCall(PetscMalloc1(ctx->n_constrained_all, &ctx->constrained_all));
    PetscCall(ISGetIndices(ctx->constrained_all_is, &idx));
    PetscCall(PetscArraycpy(ctx->constrained_all, idx, ctx->n_constrained_all));
    PetscCall(ISRestoreIndices(ctx->constrained_all_is, &idx));
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-debug_constraints", &debug, NULL));
  if (debug) {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ctx->dm), &rank));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ctx->dm), "[%d] global constrained dofs:", rank));
    for (PetscInt i = 0; i < ctx->n_constrained_all; ++i) PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ctx->dm), " %" PetscInt_FMT, ctx->constrained_all[i]));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ctx->dm), "\n"));
    PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject)ctx->dm), PETSC_STDOUT));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestrictConstrainedISOwned(DM dm, AssemblyCtx *ctx)
{
  Vec       v;
  PetscInt lo, hi, nowned = 0, *owned = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(VecDestroy(&v));

  if (ctx->n_constrained_all > 0) PetscCall(PetscMalloc1(ctx->n_constrained_all, &owned));
  for (PetscInt i = 0; i < ctx->n_constrained_all; ++i) {
    if (ctx->constrained_all[i] >= lo && ctx->constrained_all[i] < hi) owned[nowned++] = ctx->constrained_all[i];
  }
  PetscCall(ISDestroy(&ctx->constrained_is));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nowned, owned, PETSC_OWN_POINTER, &ctx->constrained_is));
  ctx->n_constrained_local = nowned;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool IsConstrainedGlobalDof(const AssemblyCtx *ctx, PetscInt idx)
{
  PetscInt lo = 0, hi = ctx->n_constrained_all;

  if (idx < 0 || ctx->n_constrained_all == 0) return PETSC_FALSE;
  while (lo < hi) {
    const PetscInt mid = lo + (hi - lo) / 2;
    if (ctx->constrained_all[mid] == idx) return PETSC_TRUE;
    if (ctx->constrained_all[mid] < idx) lo = mid + 1;
    else hi = mid;
  }
  return PETSC_FALSE;
}

PetscErrorCode BuildConstrainedGlobalIS(DM dm, IS *is, PetscInt *nlocal)
{
  PetscFunctionBeginUser;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), 0, NULL, PETSC_COPY_VALUES, is));
  if (nlocal) *nlocal = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxCreate(DM dm, P4Basis *basis, AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(*ctx)));
  ctx->dm        = dm;
  ctx->basis     = basis;
  ctx->cell_dofs = 3 * basis->n_basis;
  PetscCall(DMGetLabel(dm, "Cell Sets", &ctx->cell_sets));
  PetscCall(BuildConstrainedGlobalIS(dm, &ctx->constrained_is, &ctx->n_constrained_local));
  PetscCall(CopyConstrainedGlobalIndices(ctx));
  PetscCall(RestrictConstrainedISOwned(dm, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxDestroy(AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(ctx->constrained_all));
  PetscCall(ISDestroy(&ctx->constrained_all_is));
  PetscCall(ISDestroy(&ctx->constrained_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ZeroConstrainedVector(IS is, Vec v)
{
  const PetscInt *idx;
  PetscInt        n, lo, hi, nf = 0;
  PetscScalar    *zeros;
  PetscInt       *filtered;

  PetscFunctionBeginUser;
  PetscCall(ISGetLocalSize(is, &n));
  if (n > 0) {
    PetscCall(ISGetIndices(is, &idx));
    PetscCall(VecGetOwnershipRange(v, &lo, &hi));
    PetscCall(PetscMalloc1(n, &filtered));
    for (PetscInt i = 0; i < n; ++i) {
      if (idx[i] >= lo && idx[i] < hi) filtered[nf++] = idx[i];
    }
    if (nf > 0) {
      PetscCall(PetscCalloc1(nf, &zeros));
      PetscCall(VecSetValues(v, nf, filtered, zeros, INSERT_VALUES));
      PetscCall(PetscFree(zeros));
    }
    PetscCall(PetscFree(filtered));
    PetscCall(ISRestoreIndices(is, &idx));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyZeroDirichlet(IS is, Mat A, Vec rhs)
{
  PetscFunctionBeginUser;
  (void)A;
  PetscCall(ZeroConstrainedVector(is, rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetClosureFreeDofs(AssemblyCtx *ctx, PetscSection lsec, PetscSection gsec, Mat A, PetscInt cell, PetscScalar elem_mat[], PetscScalar free_mat[], PetscInt free_idx[], PetscInt free_pos[])
{
  DM           dm = ctx->dm;
  PetscInt    num_indices = 0, *global_indices = NULL, nfree = 0;
  PetscScalar *values = elem_mat, *values_orig = elem_mat;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &global_indices, NULL, &values));
  PetscCheck(num_indices == ctx->cell_dofs, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
             "Unexpected matrix closure size %" PetscInt_FMT " != %" PetscInt_FMT " on cell %" PetscInt_FMT, num_indices, ctx->cell_dofs, cell);

  for (PetscInt i = 0; i < num_indices; ++i) {
    if (global_indices[i] < 0 || IsConstrainedGlobalDof(ctx, global_indices[i])) continue;
    free_pos[nfree] = i;
    free_idx[nfree] = global_indices[i];
    ++nfree;
  }

  if (nfree > 0) {
    for (PetscInt i = 0; i < nfree; ++i) {
      const PetscInt ii = free_pos[i];
      for (PetscInt j = 0; j < nfree; ++j) {
        const PetscInt jj = free_pos[j];
        free_mat[i * nfree + j] = values[ii * num_indices + jj];
      }
    }
    PetscCall(MatSetValues(A, nfree, free_idx, nfree, free_idx, free_mat, ADD_VALUES));
  }

  PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &global_indices, NULL, &values));
  if (values != values_orig) PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetConstrainedDiagonals(AssemblyCtx *ctx, PetscBool use_local_indices, PetscSection lsec, PetscSection gsec, Mat A)
{
  PetscFunctionBeginUser;
  (void)ctx;
  (void)use_local_indices;
  (void)lsec;
  (void)gsec;
  (void)A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatEnsureDiagonalSlots(DM dm, PetscBool use_local_indices, PetscSection lsec, Mat A)
{
  PetscScalar zero = 0.0;

  PetscFunctionBeginUser;
  (void)use_local_indices;
  (void)lsec;
  {
    Vec      v = NULL;
    PetscInt lo, hi;

    PetscCall(DMCreateGlobalVector(dm, &v));
    PetscCall(VecGetOwnershipRange(v, &lo, &hi));
    PetscCall(VecDestroy(&v));
    for (PetscInt i = lo; i < hi; ++i) PetscCall(MatSetValue(A, i, i, zero, ADD_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleCells(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool plastic, PetscBool assemble_jacobian)
{
  DM              dm = ctx->dm;
  P4Basis        *basis = ctx->basis;
  PetscSection    lsec, gsec;
  Vec             u_loc = NULL, probe_loc = NULL, out_loc = NULL;
  Vec             out = residual ? residual : f_ext;
  PetscInt        cStart, cEnd;
  PetscScalar    *elem_vec = NULL, *elem_mat = NULL, *free_mat = NULL;
  PetscInt       *free_idx = NULL, *free_pos = NULL;
  const PetscInt  ndof = ctx->cell_dofs;
  PetscBool       use_local_mat_indices = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  if (u) {
    PetscCall(DMGetLocalVector(dm, &u_loc));
    PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, u_loc));
    PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, u_loc));
    PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, u_loc, 0.0, NULL, NULL, NULL));
  } else {
    PetscCall(DMGetLocalVector(dm, &probe_loc));
    PetscCall(VecZeroEntries(probe_loc));
  }
  if (A && assemble_jacobian) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &use_local_mat_indices));
    PetscCall(MatZeroEntries(A));
  }
  if (out) {
    PetscCall(VecZeroEntries(out));
    PetscCall(DMGetLocalVector(dm, &out_loc));
    PetscCall(VecZeroEntries(out_loc));
  }
  PetscCall(PetscCalloc1(ndof, &elem_vec));
  if (A && assemble_jacobian) {
    PetscCall(PetscCalloc1(ndof * ndof, &elem_mat));
    PetscCall(PetscMalloc1(ndof * ndof, &free_mat));
    PetscCall(PetscMalloc1(ndof, &free_idx));
    PetscCall(PetscMalloc1(ndof, &free_pos));
  }

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal     v0[3], J[9], invJ[9], detJ;
    PetscScalar  *u_cell = NULL;
    PetscInt      u_size = 0, region;
    MaterialMC    mat;
    PetscReal     grad[35][3];

    PetscCall(PetscArrayzero(elem_vec, ndof));
    if (elem_mat) PetscCall(PetscArrayzero(elem_mat, ndof * ndof));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    detJ = PetscAbsReal(detJ);
    PetscCall(CellRegion(ctx, cell, &region));
    PetscCall(MaterialMCFromRegion(region, &mat));
    if (u_loc || probe_loc) {
      PetscCall(DMPlexVecGetClosure(dm, lsec, u_loc ? u_loc : probe_loc, cell, &u_size, &u_cell));
      PetscCheck(u_size == ndof, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected closure size %" PetscInt_FMT " != %" PetscInt_FMT, u_size, ndof);
    }

    for (PetscInt q = 0; q < basis->n_qp; ++q) {
      PetscReal strain[6] = {0, 0, 0, 0, 0, 0};
      PetscReal stress[6], tangent[36];
      PetscReal B[105][6], DB[105][6];
      PetscReal w = basis->weights[q] * detJ;

      for (PetscInt a = 0; a < basis->n_basis; ++a) {
        GradPhys(basis, q, a, invJ, grad[a]);
        for (PetscInt ca = 0; ca < 3; ++ca) BColumn(ca, grad[a], B[3 * a + ca]);
        if (u_cell) {
          const PetscReal ux = PetscRealPart(u_cell[3 * a + 0]);
          const PetscReal uy = PetscRealPart(u_cell[3 * a + 1]);
          const PetscReal uz = PetscRealPart(u_cell[3 * a + 2]);
          strain[0] += grad[a][0] * ux;
          strain[1] += grad[a][1] * uy;
          strain[2] += grad[a][2] * uz;
          strain[3] += grad[a][1] * ux + grad[a][0] * uy;
          strain[4] += grad[a][2] * uy + grad[a][1] * uz;
          strain[5] += grad[a][2] * ux + grad[a][0] * uz;
        }
      }
      if (plastic) MaterialMCPlasticStressTangent(&mat, lambda, strain, stress, tangent);
      else MaterialMCElasticStressTangent(&mat, strain, stress, tangent);

      if (residual) {
        for (PetscInt a = 0; a < basis->n_basis; ++a) AddBTransposeStress(elem_vec, a, grad[a], stress, w);
      } else {
        for (PetscInt a = 0; a < basis->n_basis; ++a) elem_vec[3 * a + 1] += w * basis->basis[q * basis->n_basis + a] * (-mat.gamma_sat);
      }

      if (elem_mat) {
        for (PetscInt j = 0; j < ndof; ++j) {
          for (PetscInt r = 0; r < 6; ++r) {
            DB[j][r] = 0.0;
            for (PetscInt s = 0; s < 6; ++s) DB[j][r] += tangent[r + 6 * s] * B[j][s];
          }
        }
        for (PetscInt i = 0; i < ndof; ++i) {
          for (PetscInt j = 0; j < ndof; ++j) {
            PetscReal val = 0.0;
            for (PetscInt r = 0; r < 6; ++r) val += B[i][r] * DB[j][r];
            elem_mat[i * ndof + j] += w * val;
          }
        }
      }
    }
    if (elem_mat) {
      /* Keep the tangent exactly symmetric for CG/GAMG despite roundoff-order noise. */
      for (PetscInt i = 0; i < ndof; ++i) {
        for (PetscInt j = i + 1; j < ndof; ++j) {
          const PetscScalar v = 0.5 * (elem_mat[i * ndof + j] + elem_mat[j * ndof + i]);
          elem_mat[i * ndof + j] = v;
          elem_mat[j * ndof + i] = v;
        }
      }
    }
    if (u_cell) PetscCall(DMPlexVecRestoreClosure(dm, lsec, u_loc ? u_loc : probe_loc, cell, &u_size, &u_cell));
    if (residual) {
      PetscCall(DMPlexVecSetClosure(dm, lsec, out_loc, cell, elem_vec, ADD_VALUES));
    } else if (f_ext) {
      PetscCall(DMPlexVecSetClosure(dm, lsec, out_loc, cell, elem_vec, ADD_VALUES));
    }
    if (elem_mat) PetscCall(MatSetClosureFreeDofs(ctx, lsec, gsec, A, cell, elem_mat, free_mat, free_idx, free_pos));
  }

  if (out) {
    PetscCall(DMLocalToGlobalBegin(dm, out_loc, ADD_VALUES, out));
    PetscCall(DMLocalToGlobalEnd(dm, out_loc, ADD_VALUES, out));
  }
  if (A && assemble_jacobian) {
    /*
      PETSc section constraints remove essential DOFs from the global algebraic
      system. Keep owned diagonal slots explicit for preconditioners without
      adding artificial constrained unit rows.
    */
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
    PetscCall(MatEnsureDiagonalSlots(dm, use_local_mat_indices, lsec, A));
    PetscCall(MatSetConstrainedDiagonals(ctx, use_local_mat_indices, lsec, gsec, A));
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  if (residual) {
    PetscCall(VecAXPY(residual, -1.0, f_ext));
  }
  PetscCall(PetscFree(elem_vec));
  PetscCall(PetscFree(elem_mat));
  PetscCall(PetscFree(free_mat));
  PetscCall(PetscFree(free_idx));
  PetscCall(PetscFree(free_pos));
  if (u_loc) PetscCall(DMRestoreLocalVector(dm, &u_loc));
  if (probe_loc) PetscCall(DMRestoreLocalVector(dm, &probe_loc));
  if (out_loc) PetscCall(DMRestoreLocalVector(dm, &out_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleElasticProblem(AssemblyCtx *ctx, Mat A, Vec f_ext)
{
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(f_ext));
  PetscCall(AssembleCells(ctx, 1.0, NULL, f_ext, A, NULL, PETSC_FALSE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblePlasticResidualJacobian(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool assemble_jacobian)
{
  PetscFunctionBeginUser;
  PetscCall(AssembleCells(ctx, lambda, u, f_ext, A, residual, PETSC_TRUE, assemble_jacobian));
  PetscFunctionReturn(PETSC_SUCCESS);
}
