#pragma once

#include <petscsys.h>

typedef struct {
  PetscReal c0;
  PetscReal phi_deg;
  PetscReal psi_deg;
  PetscReal young;
  PetscReal poisson;
  PetscReal gamma_sat;
  PetscReal shear;
  PetscReal bulk;
  PetscReal lame;
} MaterialMC;

PetscErrorCode MaterialMCFromRegion(PetscInt region, MaterialMC *mat);
PetscErrorCode MaterialMCReduceDavisB(const MaterialMC *mat, PetscReal lambda, PetscReal *c_bar, PetscReal *sin_phi);
void           MaterialMCElasticStressTangent(const MaterialMC *mat, const PetscReal strain[6], PetscReal stress[6], PetscReal tangent[36]);
void           MaterialMCPlasticStressTangent(const MaterialMC *mat, PetscReal lambda, const PetscReal strain[6], PetscReal stress[6], PetscReal tangent[36]);
