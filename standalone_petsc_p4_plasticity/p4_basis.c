#include "p4_basis.h"

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
  PetscReal *points, *weights;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(24 * 3, &points));
  PetscCall(PetscMalloc1(24, &weights));
  for (PetscInt q = 0; q < 24; ++q) {
    points[3 * q + 0] = xi[q][0];
    points[3 * q + 1] = xi[q][1];
    points[3 * q + 2] = xi[q][2];
    weights[q]        = wf[q];
  }
  PetscCall(PetscQuadratureCreate(comm, quad));
  PetscCall(PetscQuadratureSetData(*quad, 3, 1, 24, points, weights));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4BasisCreate(MPI_Comm comm, P4Basis *basis)
{
  PetscInt Nc, dim, npoints;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(basis, sizeof(*basis)));
  basis->dim        = 3;
  basis->components = 3;
  basis->degree     = 4;

  PetscCall(CreateTetra24Quadrature(comm, &basis->quadrature));
  PetscCall(PetscFECreateLagrange(comm, 3, 3, PETSC_TRUE, 4, PETSC_DETERMINE, &basis->fe_vector));
  PetscCall(PetscFESetQuadrature(basis->fe_vector, basis->quadrature));
  PetscCall(PetscFECreateLagrange(comm, 3, 1, PETSC_TRUE, 4, PETSC_DETERMINE, &basis->fe_scalar));
  PetscCall(PetscFESetQuadrature(basis->fe_scalar, basis->quadrature));
  PetscCall(PetscQuadratureGetData(basis->quadrature, &dim, &Nc, &npoints, &basis->points, &basis->weights));
  PetscCheck(dim == 3 && Nc == 1 && npoints == 24, comm, PETSC_ERR_PLIB, "Unexpected quadrature shape");
  basis->n_qp = npoints;
  PetscCall(PetscFEGetDimension(basis->fe_scalar, &basis->n_basis));
  PetscCall(PetscFECreateTabulation(basis->fe_scalar, 1, basis->n_qp, basis->points, 1, &basis->tabulation));
  basis->basis     = basis->tabulation->T[0];
  basis->basis_der = basis->tabulation->T[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4BasisDestroy(P4Basis *basis)
{
  PetscFunctionBeginUser;
  PetscCall(PetscTabulationDestroy(&basis->tabulation));
  PetscCall(PetscFEDestroy(&basis->fe_scalar));
  PetscCall(PetscFEDestroy(&basis->fe_vector));
  PetscCall(PetscQuadratureDestroy(&basis->quadrature));
  PetscFunctionReturn(PETSC_SUCCESS);
}
