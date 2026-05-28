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

static PetscErrorCode CreateTriangleQuadratureFromRule(MPI_Comm comm, PetscInt npoints, PetscInt order, const PetscReal xi[][2], const PetscReal wf[], PetscQuadrature *quad)
{
  PetscReal *points, *weights;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(npoints * 2, &points));
  PetscCall(PetscMalloc1(npoints, &weights));
  /* PETSc tabulates simplex FE spaces on the biunit reference triangle. */
  for (PetscInt q = 0; q < npoints; ++q) {
    for (PetscInt d = 0; d < 2; ++d) points[2 * q + d] = 2.0 * xi[q][d] - 1.0;
    weights[q] = 4.0 * wf[q];
  }
  PetscCall(PetscQuadratureCreate(comm, quad));
  PetscCall(PetscQuadratureSetCellType(*quad, DM_POLYTOPE_TRIANGLE));
  PetscCall(PetscQuadratureSetOrder(*quad, order));
  PetscCall(PetscQuadratureSetData(*quad, 2, 1, npoints, points, weights));
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

static PetscErrorCode CreateTriangle1Quadrature(MPI_Comm comm, PetscQuadrature *quad)
{
  static const PetscReal xi[1][2] = {
    {1.0 / 3.0, 1.0 / 3.0},
  };
  static const PetscReal wf[1] = {
    0.5,
  };

  PetscFunctionBeginUser;
  PetscCall(CreateTriangleQuadratureFromRule(comm, 1, 1, xi, wf, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTriangle7Quadrature(MPI_Comm comm, PetscQuadrature *quad)
{
  static const PetscReal xi[7][2] = {
    {0.1012865073235, 0.1012865073235},
    {0.7974269853531, 0.1012865073235},
    {0.1012865073235, 0.7974269853531},
    {0.4701420641051, 0.0597158717898},
    {0.4701420641051, 0.4701420641051},
    {0.0597158717898, 0.4701420641051},
    {1.0 / 3.0, 1.0 / 3.0},
  };
  static const PetscReal wf[7] = {
    0.1259391805448 / 2.0,
    0.1259391805448 / 2.0,
    0.1259391805448 / 2.0,
    0.1323941527885 / 2.0,
    0.1323941527885 / 2.0,
    0.1323941527885 / 2.0,
    0.2250000000000 / 2.0,
  };

  PetscFunctionBeginUser;
  PetscCall(CreateTriangleQuadratureFromRule(comm, 7, 5, xi, wf, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTriangle12Quadrature(MPI_Comm comm, PetscQuadrature *quad)
{
  static const PetscReal xi[12][2] = {
    {0.063089014491502, 0.063089014491502},
    {0.063089014491502, 0.873821971016996},
    {0.873821971016996, 0.063089014491502},
    {0.249286745170910, 0.249286745170910},
    {0.249286745170910, 0.501426509658179},
    {0.501426509658179, 0.249286745170910},
    {0.310352451033785, 0.053145049844816},
    {0.310352451033785, 0.636502499121399},
    {0.053145049844816, 0.310352451033785},
    {0.053145049844816, 0.636502499121399},
    {0.636502499121399, 0.310352451033785},
    {0.636502499121399, 0.053145049844816},
  };
  static const PetscReal wf[12] = {
    0.050844906370207 / 2.0,
    0.050844906370207 / 2.0,
    0.050844906370207 / 2.0,
    0.116786275726379 / 2.0,
    0.116786275726379 / 2.0,
    0.116786275726379 / 2.0,
    0.082851075618374 / 2.0,
    0.082851075618374 / 2.0,
    0.082851075618374 / 2.0,
    0.082851075618374 / 2.0,
    0.082851075618374 / 2.0,
    0.082851075618374 / 2.0,
  };

  PetscFunctionBeginUser;
  PetscCall(CreateTriangleQuadratureFromRule(comm, 12, 6, xi, wf, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateEquispacedScalarLagrangeFE(MPI_Comm comm, PetscInt degree, PetscInt dim, PetscQuadrature quad, PetscFE *fe)
{
  PetscSpace     P = NULL;
  PetscDualSpace Q = NULL;
  DM             K = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscSpaceCreate(comm, &P));
  PetscCall(PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscSpaceSetNumVariables(P, dim));
  PetscCall(PetscSpaceSetNumComponents(P, 1));
  PetscCall(PetscSpacePolynomialSetTensor(P, PETSC_FALSE));
  PetscCall(PetscSpaceSetDegree(P, degree, PETSC_DETERMINE));
  PetscCall(PetscSpaceSetUp(P));

  PetscCall(PetscDualSpaceCreate(comm, &Q));
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, dim == 2 ? DM_POLYTOPE_TRIANGLE : DM_POLYTOPE_TETRAHEDRON, &K));
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

PetscErrorCode P4BasisCreateDegreeDim(MPI_Comm comm, PetscInt degree, PetscInt dim, PetscInt components, P4Basis *basis)
{
  PetscInt Nc, qdim, npoints;

  PetscFunctionBeginUser;
  PetscCheck(degree == 1 || degree == 2 || degree == 4, comm, PETSC_ERR_ARG_OUTOFRANGE, "Only P1/P2/P4 simplex bases are supported");
  PetscCheck(dim == 2 || dim == 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "Only 2D triangles and 3D tetrahedra are supported");
  PetscCheck(components >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "Vector component count must be positive");
  PetscCall(PetscMemzero(basis, sizeof(*basis)));
  basis->dim        = dim;
  basis->components = components;
  basis->degree     = degree;

  if (dim == 2) {
    if (degree == 1) {
      PetscCall(CreateTriangle1Quadrature(comm, &basis->quadrature));
    } else if (degree == 2) {
      PetscCall(CreateTriangle7Quadrature(comm, &basis->quadrature));
    } else {
      PetscCall(CreateTriangle12Quadrature(comm, &basis->quadrature));
    }
  } else {
    if (degree == 1) {
      PetscCall(CreateTetra1Quadrature(comm, &basis->quadrature));
    } else if (degree == 2) {
      PetscCall(CreateTetra11Quadrature(comm, &basis->quadrature));
    } else {
      PetscCall(CreateTetra24Quadrature(comm, &basis->quadrature));
    }
  }
  PetscCall(CreateEquispacedScalarLagrangeFE(comm, degree, dim, basis->quadrature, &basis->fe_scalar));
  PetscCall(PetscFESetQuadrature(basis->fe_scalar, basis->quadrature));
  PetscCall(PetscFECreateVector(basis->fe_scalar, components, PETSC_TRUE, PETSC_FALSE, &basis->fe_vector));
  PetscCall(PetscFESetQuadrature(basis->fe_vector, basis->quadrature));
  PetscCall(PetscQuadratureGetData(basis->quadrature, &qdim, &Nc, &npoints, &basis->points, &basis->weights));
  PetscCheck(qdim == basis->dim && Nc == 1, comm, PETSC_ERR_PLIB, "Unexpected quadrature shape");
  basis->n_qp = npoints;
  PetscCall(PetscFEGetDimension(basis->fe_scalar, &basis->n_basis));
  PetscCall(PetscFECreateTabulation(basis->fe_scalar, 1, basis->n_qp, basis->points, 1, &basis->tabulation));
  basis->basis     = basis->tabulation->T[0];
  basis->basis_der = basis->tabulation->T[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4BasisCreateDegree(MPI_Comm comm, PetscInt degree, P4Basis *basis)
{
  PetscFunctionBeginUser;
  PetscCall(P4BasisCreateDegreeDim(comm, degree, 3, 3, basis));
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
