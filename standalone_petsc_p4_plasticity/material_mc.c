#include "material_mc.h"
#include "material_mc_kernel.h"

static void MaterialMCFinish(MaterialMC *mat)
{
  mat->shear = mat->young / (2.0 * (1.0 + mat->poisson));
  mat->bulk  = mat->young / (3.0 * (1.0 - 2.0 * mat->poisson));
  mat->lame  = mat->bulk - (2.0 / 3.0) * mat->shear;
}

PetscErrorCode MaterialMCFromRegion(PetscInt region, MaterialMC *mat)
{
  PetscFunctionBeginUser;
  PetscCheck(mat, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Material pointer is NULL");
  switch (region) {
  case 1: /* region:general_foundation */
    *mat = (MaterialMC){15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 0.0, 0.0, 0.0};
    break;
  case 2: /* region:weak_foundation */
    *mat = (MaterialMC){10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 0.0, 0.0, 0.0};
    break;
  case 3: /* region:slope_mass */
    *mat = (MaterialMC){18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 0.0, 0.0, 0.0};
    break;
  case 4: /* region:cover_layer */
    *mat = (MaterialMC){15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 0.0, 0.0, 0.0};
    break;
  default:
    *mat = (MaterialMC){15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 0.0, 0.0, 0.0};
    break;
  }
  MaterialMCFinish(mat);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MaterialMCReduceDavisB(const MaterialMC *mat, PetscReal lambda, PetscReal *c_bar, PetscReal *sin_phi)
{
  const PetscReal deg = PETSC_PI / 180.0;
  PetscReal       phi, psi, c01, phi1, psi1, beta, c0_lambda, phi_lambda;

  PetscFunctionBeginUser;
  PetscCheck(mat, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Material pointer is NULL");
  PetscCheck(lambda > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "lambda must be positive");
  phi        = mat->phi_deg * deg;
  psi        = mat->psi_deg * deg;
  c01        = mat->c0 / lambda;
  phi1       = PetscAtanReal(PetscTanReal(phi) / lambda);
  psi1       = PetscAtanReal(PetscTanReal(psi) / lambda);
  beta       = PetscCosReal(phi1) * PetscCosReal(psi1) / (1.0 - PetscSinReal(phi1) * PetscSinReal(psi1));
  c0_lambda  = beta * c01;
  phi_lambda = PetscAtanReal(beta * PetscTanReal(phi1));
  *c_bar     = 2.0 * c0_lambda * PetscCosReal(phi_lambda);
  *sin_phi   = PetscSinReal(phi_lambda);
  PetscFunctionReturn(PETSC_SUCCESS);
}

void MaterialMCElasticStressTangent(const MaterialMC *mat, const PetscReal strain[6], PetscReal stress[6], PetscReal tangent[36])
{
  const PetscReal s2  = 2.0 * mat->shear;
  const PetscReal tr  = strain[0] + strain[1] + strain[2];
  const PetscReal ltr = mat->lame * tr;

  stress[0] = ltr + s2 * strain[0];
  stress[1] = ltr + s2 * strain[1];
  stress[2] = ltr + s2 * strain[2];
  stress[3] = mat->shear * strain[3];
  stress[4] = mat->shear * strain[4];
  stress[5] = mat->shear * strain[5];

  if (tangent) {
    PetscMemzero(tangent, 36 * sizeof(PetscReal));
    tangent[0]  = s2 + mat->lame;
    tangent[7]  = s2 + mat->lame;
    tangent[14] = s2 + mat->lame;
    tangent[1] = tangent[6] = mat->lame;
    tangent[2] = tangent[12] = mat->lame;
    tangent[8] = tangent[13] = mat->lame;
    tangent[21] = mat->shear;
    tangent[28] = mat->shear;
    tangent[35] = mat->shear;
  }
}

void MaterialMCPlasticStressTangent(const MaterialMC *mat, PetscReal lambda, const PetscReal strain[6], PetscReal stress[6], PetscReal tangent[36])
{
  PetscReal c_bar, sin_phi;

  (void)MaterialMCReduceDavisB(mat, lambda, &c_bar, &sin_phi);
  constitutive_3D_point(strain, c_bar, sin_phi, mat->shear, mat->bulk, mat->lame, stress, tangent);
}
