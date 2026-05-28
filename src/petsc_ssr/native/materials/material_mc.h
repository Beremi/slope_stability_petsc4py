#pragma once

#include <petscsys.h>

typedef struct {
  PetscReal c0;
  PetscReal phi_deg;
  PetscReal psi_deg;
  PetscReal young;
  PetscReal poisson;
  PetscReal gamma_sat;
  PetscReal gamma_unsat;
  PetscReal shear;
  PetscReal bulk;
  PetscReal lame;
} MaterialMC;

PetscErrorCode MaterialMCFromRegion(PetscInt region, MaterialMC *mat);
PetscErrorCode MaterialMCConfigureDefaults(void);
PetscErrorCode MaterialMCConfigureFromOptions(MPI_Comm comm);
PetscErrorCode MaterialMCReduceDavisB(const MaterialMC *mat, PetscReal lambda, PetscReal *c_bar, PetscReal *sin_phi);
void           MaterialMCElasticStressTangent(const MaterialMC *mat, const PetscReal strain[6], PetscReal stress[6], PetscReal tangent[36]);
void           MaterialMCPlasticStressTangent(const MaterialMC *mat, PetscReal lambda, const PetscReal strain[6], PetscReal stress[6], PetscReal tangent[36]);
void           MaterialMCElasticStressTangent2D(const MaterialMC *mat, const PetscReal strain[3], PetscReal stress[3], PetscReal tangent[9]);
void           MaterialMCPlasticStressTangent2D(const MaterialMC *mat, PetscReal lambda, const PetscReal strain[3], PetscReal stress[3], PetscReal tangent[9]);
