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

PetscErrorCode BuildConstrainedGlobalIS(DM dm, IS *is, PetscInt *nlocal)
{
  PetscSection gsec;
  PetscInt    *idx = NULL, nidx = 0, cap = 0;
  const PetscInt face_ids[5] = {1, 2, 3, 4, 5};
  const PetscInt comps[5]    = {0, 0, 2, 2, 1};

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalSection(dm, &gsec));
  for (PetscInt k = 0; k < 5; ++k) {
    IS              faces = NULL;
    const PetscInt *fidx;
    PetscInt        nf;

    PetscCall(DMGetStratumIS(dm, "Face Sets", face_ids[k], &faces));
    if (!faces) continue;
    PetscCall(ISGetLocalSize(faces, &nf));
    PetscCall(ISGetIndices(faces, &fidx));
    for (PetscInt f = 0; f < nf; ++f) {
      PetscInt *closure = NULL, nclosure = 0;
      PetscCall(DMPlexGetTransitiveClosure(dm, fidx[f], PETSC_TRUE, &nclosure, &closure));
      for (PetscInt c = 0; c < nclosure; ++c) {
        const PetscInt p = closure[2 * c];
        PetscInt       dof, off;
        PetscCall(PetscSectionGetDof(gsec, p, &dof));
        PetscCall(PetscSectionGetOffset(gsec, p, &off));
        if (dof <= 0 || off < 0) continue;
        PetscCheck(dof % 3 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Expected vector dofs divisible by 3");
        for (PetscInt j = 0; j < dof / 3; ++j) {
          if (nidx == cap) {
            cap = cap ? 2 * cap : 1024;
            PetscCall(PetscRealloc(cap * sizeof(PetscInt), &idx));
          }
          idx[nidx++] = off + 3 * j + comps[k];
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, fidx[f], PETSC_TRUE, &nclosure, &closure));
    }
    PetscCall(ISRestoreIndices(faces, &fidx));
    PetscCall(ISDestroy(&faces));
  }
  {
    PetscInt w = 0;
    for (PetscInt i = 0; i < nidx; ++i) {
      if (idx[i] >= 0) idx[w++] = idx[i];
    }
    nidx = w;
  }
  PetscCall(PetscSortRemoveDupsInt(&nidx, idx));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nidx, idx, PETSC_OWN_POINTER, is));
  if (nlocal) *nlocal = nidx;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxDestroy(AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
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
  if (n == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISGetIndices(is, &idx));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(PetscMalloc1(n, &filtered));
  for (PetscInt i = 0; i < n; ++i) {
    if (idx[i] >= lo && idx[i] < hi) filtered[nf++] = idx[i];
  }
  if (nf == 0) {
    PetscCall(PetscFree(filtered));
    PetscCall(ISRestoreIndices(is, &idx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscCalloc1(nf, &zeros));
  PetscCall(VecSetValues(v, nf, filtered, zeros, INSERT_VALUES));
  PetscCall(PetscFree(filtered));
  PetscCall(PetscFree(zeros));
  PetscCall(ISRestoreIndices(is, &idx));
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyZeroDirichlet(IS is, Mat A, Vec rhs)
{
  const PetscInt *idx;
  PetscInt        n, lo, hi, nf = 0, *filtered;
  IS              owned_is;

  PetscFunctionBeginUser;
  PetscCall(ZeroConstrainedVector(is, rhs));
  PetscCall(ISGetLocalSize(is, &n));
  if (n == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatGetOwnershipRange(A, &lo, &hi));
  PetscCall(ISGetIndices(is, &idx));
  PetscCall(PetscMalloc1(n, &filtered));
  for (PetscInt i = 0; i < n; ++i) {
    if (idx[i] >= lo && idx[i] < hi) filtered[nf++] = idx[i];
  }
  PetscCall(ISRestoreIndices(is, &idx));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), nf, filtered, PETSC_OWN_POINTER, &owned_is));
  PetscCall(MatZeroRowsColumnsIS(A, owned_is, 1.0, NULL, rhs));
  PetscCall(ISDestroy(&owned_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleCells(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool plastic, PetscBool assemble_jacobian)
{
  DM              dm = ctx->dm;
  P4Basis        *basis = ctx->basis;
  PetscSection    lsec, gsec;
  Vec             u_loc = NULL, probe_loc = NULL;
  PetscInt        cStart, cEnd;
  PetscScalar    *elem_vec = NULL, *elem_mat = NULL;
  const PetscInt  ndof = ctx->cell_dofs;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  if (u) {
    PetscCall(DMGetLocalVector(dm, &u_loc));
    PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, u_loc));
    PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, u_loc));
  } else {
    PetscCall(DMGetLocalVector(dm, &probe_loc));
    PetscCall(VecZeroEntries(probe_loc));
  }
  if (A && assemble_jacobian) PetscCall(MatZeroEntries(A));
  if (residual) PetscCall(VecZeroEntries(residual));
  PetscCall(PetscCalloc1(ndof, &elem_vec));
  if (A && assemble_jacobian) PetscCall(PetscCalloc1(ndof * ndof, &elem_mat));

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
    if (u_cell) PetscCall(DMPlexVecRestoreClosure(dm, lsec, u_loc ? u_loc : probe_loc, cell, &u_size, &u_cell));
    if (residual) {
      PetscCall(DMPlexVecSetClosure(dm, NULL, residual, cell, elem_vec, ADD_VALUES));
    } else if (f_ext) {
      PetscCall(DMPlexVecSetClosure(dm, NULL, f_ext, cell, elem_vec, ADD_VALUES));
    }
    if (elem_mat) PetscCall(DMPlexMatSetClosure(dm, NULL, NULL, A, cell, elem_mat, ADD_VALUES));
  }

  if (A && assemble_jacobian) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  if (residual) {
    PetscCall(VecAssemblyBegin(residual));
    PetscCall(VecAssemblyEnd(residual));
    PetscCall(VecAXPY(residual, -1.0, f_ext));
  } else if (f_ext) {
    PetscCall(VecAssemblyBegin(f_ext));
    PetscCall(VecAssemblyEnd(f_ext));
  }
  PetscCall(PetscFree(elem_vec));
  PetscCall(PetscFree(elem_mat));
  if (u_loc) PetscCall(DMRestoreLocalVector(dm, &u_loc));
  if (probe_loc) PetscCall(DMRestoreLocalVector(dm, &probe_loc));
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
