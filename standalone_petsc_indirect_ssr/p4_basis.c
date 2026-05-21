#include "p4_basis.h"

#include <petscdualspace.h>
#include <petscspace.h>

static PetscErrorCode CreateTetraQuadratureFromRule(MPI_Comm comm, PetscInt npoints, PetscInt order, const PetscReal xi[][3], const PetscReal wf[], PetscQuadrature *quad)
{
  PetscReal *points, *weights;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(npoints * 3, &points));
  PetscCall(PetscMalloc1(npoints, &weights));
  /* PETSc tabulates simplex FE spaces on the biunit reference tetrahedron. */
  for (PetscInt q = 0; q < npoints; ++q) {
    for (PetscInt d = 0; d < 3; ++d) points[3 * q + d] = 2.0 * xi[q][d] - 1.0;
    weights[q] = 8.0 * wf[q];
  }
  PetscCall(PetscQuadratureCreate(comm, quad));
  PetscCall(PetscQuadratureSetCellType(*quad, DM_POLYTOPE_TETRAHEDRON));
  PetscCall(PetscQuadratureSetOrder(*quad, order));
  PetscCall(PetscQuadratureSetData(*quad, 3, 1, npoints, points, weights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTetra1Quadrature(MPI_Comm comm, PetscQuadrature *quad)
{
  static const PetscReal xi[1][3] = {
    {0.25, 0.25, 0.25},
  };
  static const PetscReal wf[1] = {
    1.0 / 6.0,
  };

  PetscFunctionBeginUser;
  PetscCall(CreateTetraQuadratureFromRule(comm, 1, 1, xi, wf, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTetra11Quadrature(MPI_Comm comm, PetscQuadrature *quad)
{
  static const PetscReal xi[11][3] = {
    {0.25, 0.25, 0.25},
    {0.0714285714285714, 0.0714285714285714, 0.0714285714285714},
    {0.7857142857142860, 0.0714285714285714, 0.0714285714285714},
    {0.0714285714285714, 0.7857142857142860, 0.0714285714285714},
    {0.0714285714285714, 0.0714285714285714, 0.7857142857142860},
    {0.3994035761667990, 0.1005964238332010, 0.1005964238332010},
    {0.1005964238332010, 0.3994035761667990, 0.1005964238332010},
    {0.1005964238332010, 0.1005964238332010, 0.3994035761667990},
    {0.3994035761667990, 0.3994035761667990, 0.1005964238332010},
    {0.3994035761667990, 0.1005964238332010, 0.3994035761667990},
    {0.1005964238332010, 0.3994035761667990, 0.3994035761667990},
  };
  static const PetscReal wf[11] = {
    -0.013155555555555,
    0.007622222222222,
    0.007622222222222,
    0.007622222222222,
    0.007622222222222,
    0.024888888888888,
    0.024888888888888,
    0.024888888888888,
    0.024888888888888,
    0.024888888888888,
    0.024888888888888,
  };

  PetscFunctionBeginUser;
  PetscCall(CreateTetraQuadratureFromRule(comm, 11, 4, xi, wf, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTetra24Quadrature(MPI_Comm comm, PetscQuadrature *quad)
{
  static const PetscReal xi[24][3] = {
    {0.3561913862225449, 0.2146028712591517, 0.2146028712591517},
    {0.2146028712591517, 0.2146028712591517, 0.2146028712591517},
    {0.2146028712591517, 0.2146028712591517, 0.3561913862225449},
    {0.2146028712591517, 0.3561913862225449, 0.2146028712591517},
    {0.8779781243961660, 0.0406739585346113, 0.0406739585346113},
    {0.0406739585346113, 0.0406739585346113, 0.0406739585346113},
    {0.0406739585346113, 0.0406739585346113, 0.8779781243961660},
    {0.0406739585346113, 0.8779781243961660, 0.0406739585346113},
    {0.0329863295731731, 0.3223378901422757, 0.3223378901422757},
    {0.3223378901422757, 0.3223378901422757, 0.3223378901422757},
    {0.3223378901422757, 0.3223378901422757, 0.0329863295731731},
    {0.3223378901422757, 0.0329863295731731, 0.3223378901422757},
    {0.2696723314583159, 0.0636610018750175, 0.0636610018750175},
    {0.0636610018750175, 0.2696723314583159, 0.0636610018750175},
    {0.0636610018750175, 0.0636610018750175, 0.2696723314583159},
    {0.6030056647916491, 0.0636610018750175, 0.0636610018750175},
    {0.0636610018750175, 0.6030056647916491, 0.0636610018750175},
    {0.0636610018750175, 0.0636610018750175, 0.6030056647916491},
    {0.0636610018750175, 0.2696723314583159, 0.6030056647916491},
    {0.2696723314583159, 0.6030056647916491, 0.0636610018750175},
    {0.6030056647916491, 0.0636610018750175, 0.2696723314583159},
    {0.0636610018750175, 0.6030056647916491, 0.2696723314583159},
    {0.2696723314583159, 0.0636610018750175, 0.6030056647916491},
    {0.6030056647916491, 0.2696723314583159, 0.0636610018750175},
  };
  static const PetscReal wf[24] = {
    0.0399227502581679 / 6.0, 0.0399227502581679 / 6.0, 0.0399227502581679 / 6.0, 0.0399227502581679 / 6.0,
    0.0100772110553207 / 6.0, 0.0100772110553207 / 6.0, 0.0100772110553207 / 6.0, 0.0100772110553207 / 6.0,
    0.0553571815436544 / 6.0, 0.0553571815436544 / 6.0, 0.0553571815436544 / 6.0, 0.0553571815436544 / 6.0,
    0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0,
    0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0,
    0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0, 0.0482142857142857 / 6.0,
  };
  PetscFunctionBeginUser;
  PetscCall(CreateTetraQuadratureFromRule(comm, 24, 6, xi, wf, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateEquispacedScalarLagrangeFE(MPI_Comm comm, PetscInt degree, PetscQuadrature quad, PetscFE *fe)
{
  PetscSpace     P = NULL;
  PetscDualSpace Q = NULL;
  DM             K = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscSpaceCreate(comm, &P));
  PetscCall(PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscSpaceSetNumVariables(P, 3));
  PetscCall(PetscSpaceSetNumComponents(P, 1));
  PetscCall(PetscSpacePolynomialSetTensor(P, PETSC_FALSE));
  PetscCall(PetscSpaceSetDegree(P, degree, PETSC_DETERMINE));
  PetscCall(PetscSpaceSetUp(P));

  PetscCall(PetscDualSpaceCreate(comm, &Q));
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DM_POLYTOPE_TETRAHEDRON, &K));
  PetscCall(PetscDualSpaceSetDM(Q, K));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDualSpaceSetNumComponents(Q, 1));
  PetscCall(PetscDualSpaceSetOrder(Q, degree));
  PetscCall(PetscDualSpaceLagrangeSetTensor(Q, PETSC_FALSE));
  PetscCall(PetscDualSpaceLagrangeSetNodeType(Q, PETSCDTNODES_EQUISPACED, PETSC_TRUE, 0.0));
  PetscCall(PetscDualSpaceSetUp(Q));

  PetscCall(PetscObjectReference((PetscObject)quad));
  PetscCall(PetscFECreateFromSpaces(P, Q, quad, NULL, fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4BasisCreateDegree(MPI_Comm comm, PetscInt degree, P4Basis *basis)
{
  PetscInt Nc, dim, npoints;

  PetscFunctionBeginUser;
  PetscCheck(degree == 1 || degree == 2 || degree == 4, comm, PETSC_ERR_ARG_OUTOFRANGE, "Only P1/P2/P4 tetrahedral bases are supported");
  PetscCall(PetscMemzero(basis, sizeof(*basis)));
  basis->dim        = 3;
  basis->components = 3;
  basis->degree     = degree;

  if (degree == 1) {
    PetscCall(CreateTetra1Quadrature(comm, &basis->quadrature));
  } else if (degree == 2) {
    PetscCall(CreateTetra11Quadrature(comm, &basis->quadrature));
  } else {
    PetscCall(CreateTetra24Quadrature(comm, &basis->quadrature));
  }
  PetscCall(CreateEquispacedScalarLagrangeFE(comm, degree, basis->quadrature, &basis->fe_scalar));
  PetscCall(PetscFESetQuadrature(basis->fe_scalar, basis->quadrature));
  PetscCall(PetscFECreateVector(basis->fe_scalar, 3, PETSC_TRUE, PETSC_FALSE, &basis->fe_vector));
  PetscCall(PetscFESetQuadrature(basis->fe_vector, basis->quadrature));
  PetscCall(PetscQuadratureGetData(basis->quadrature, &dim, &Nc, &npoints, &basis->points, &basis->weights));
  PetscCheck(dim == 3 && Nc == 1, comm, PETSC_ERR_PLIB, "Unexpected quadrature shape");
  basis->n_qp = npoints;
  PetscCall(PetscFEGetDimension(basis->fe_scalar, &basis->n_basis));
  PetscCall(PetscFECreateTabulation(basis->fe_scalar, 1, basis->n_qp, basis->points, 1, &basis->tabulation));
  basis->basis     = basis->tabulation->T[0];
  basis->basis_der = basis->tabulation->T[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4BasisCreate(MPI_Comm comm, P4Basis *basis)
{
  PetscFunctionBeginUser;
  PetscCall(P4BasisCreateDegree(comm, 4, basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4BasisDestroy(P4Basis *basis)
{
  PetscFunctionBeginUser;
  PetscCall(PetscTabulationDestroy(&basis->tabulation));
  PetscCall(PetscFEDestroy(&basis->fe_vector));
  PetscCall(PetscFEDestroy(&basis->fe_scalar));
  PetscCall(PetscQuadratureDestroy(&basis->quadrature));
  PetscFunctionReturn(PETSC_SUCCESS);
}
