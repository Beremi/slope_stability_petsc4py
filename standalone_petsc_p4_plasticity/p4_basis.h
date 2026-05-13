#pragma once

#include <petscdmplex.h>
#include <petscfe.h>

typedef struct {
  PetscFE         fe_vector;
  PetscFE         fe_scalar;
  PetscQuadrature quadrature;
  PetscTabulation tabulation;
  PetscInt        dim;
  PetscInt        components;
  PetscInt        degree;
  PetscInt        n_basis;
  PetscInt        n_qp;
  const PetscReal *points;
  const PetscReal *weights;
  const PetscReal *basis;
  const PetscReal *basis_der;
} P4Basis;

PetscErrorCode P4BasisCreate(MPI_Comm comm, P4Basis *basis);
PetscErrorCode P4BasisDestroy(P4Basis *basis);
