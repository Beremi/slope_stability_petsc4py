#include "material_mc.h"
#include "material_mc_kernel.h"

static void MaterialMCFinish(MaterialMC *mat)
{
  mat->shear = mat->young / (2.0 * (1.0 + mat->poisson));
  mat->bulk  = mat->young / (3.0 * (1.0 - 2.0 * mat->poisson));
  mat->lame  = mat->bulk - (2.0 / 3.0) * mat->shear;
}

static MaterialMC material_table[17];
static PetscBool  material_table_ready = PETSC_FALSE;

static PetscErrorCode MaterialMCDefaultForRegion(PetscInt region, MaterialMC *mat)
{
  PetscFunctionBeginUser;
  PetscCheck(mat, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Material pointer is NULL");
  switch (region) {
  case 1: /* region:general_foundation */
    *mat = (MaterialMC){15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0, 0.0, 0.0, 0.0};
    break;
  case 2: /* region:weak_foundation */
    *mat = (MaterialMC){10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0, 0.0, 0.0, 0.0};
    break;
  case 3: /* region:slope_mass */
    *mat = (MaterialMC){18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0, 0.0, 0.0, 0.0};
    break;
  case 4: /* region:cover_layer */
    *mat = (MaterialMC){15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0, 0.0, 0.0, 0.0};
    break;
  default:
    *mat = (MaterialMC){15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0, 0.0, 0.0, 0.0};
    break;
  }
  MaterialMCFinish(mat);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MaterialMCConfigureDefaults(void)
{
  PetscFunctionBeginUser;
  for (PetscInt region = 0; region < 17; ++region) PetscCall(MaterialMCDefaultForRegion(region, &material_table[region]));
  material_table_ready = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MaterialMCConfigureFromOptions(MPI_Comm comm)
{
  PetscFunctionBeginUser;
  PetscCall(MaterialMCConfigureDefaults());
  for (PetscInt region = 1; region < 17; ++region) {
    char      key[64];
    PetscReal values[7];
    PetscInt  n = 7;
    PetscBool found = PETSC_FALSE;

    PetscCall(PetscSNPrintf(key, sizeof(key), "-material_region_%" PetscInt_FMT, region));
    PetscCall(PetscOptionsGetRealArray(NULL, NULL, key, values, &n, &found));
    if (!found) continue;
    PetscCheck(n == 6 || n == 7, comm, PETSC_ERR_ARG_SIZ,
               "%s expects six or seven values: c0 phi_deg psi_deg young poisson gamma_sat [gamma_unsat]", key);
    if (n == 6) values[6] = values[5];
    material_table[region] = (MaterialMC){values[0], values[1], values[2], values[3], values[4], values[5], values[6], 0.0, 0.0, 0.0};
    MaterialMCFinish(&material_table[region]);
    PetscCall(PetscPrintf(comm,
                          "MATERIAL_CONFIG region=%" PetscInt_FMT " c0=%.12g phi_deg=%.12g psi_deg=%.12g young=%.12g poisson=%.12g gamma_sat=%.12g gamma_unsat=%.12g\n",
                          region, (double)material_table[region].c0, (double)material_table[region].phi_deg,
                          (double)material_table[region].psi_deg, (double)material_table[region].young,
                          (double)material_table[region].poisson, (double)material_table[region].gamma_sat,
                          (double)material_table[region].gamma_unsat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MaterialMCFromRegion(PetscInt region, MaterialMC *mat)
{
  PetscFunctionBeginUser;
  PetscCheck(mat, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Material pointer is NULL");
  if (!material_table_ready) PetscCall(MaterialMCConfigureDefaults());
  if (region < 1 || region >= 17) region = 1;
  *mat = material_table[region];
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

static void MaterialMCExtractPlaneStrain2D(const PetscReal stress6[6], const PetscReal tangent6[36], PetscReal stress2[3], PetscReal tangent2[9])
{
  const PetscInt map[3] = {0, 1, 3};

  stress2[0] = stress6[0];
  stress2[1] = stress6[1];
  stress2[2] = stress6[3];
  if (tangent2) {
    for (PetscInt j = 0; j < 3; ++j) {
      for (PetscInt i = 0; i < 3; ++i) tangent2[i + 3 * j] = tangent6[map[i] + 6 * map[j]];
    }
  }
}

void MaterialMCElasticStressTangent2D(const MaterialMC *mat, const PetscReal strain[3], PetscReal stress[3], PetscReal tangent[9])
{
  PetscReal strain6[6] = {strain[0], strain[1], 0.0, strain[2], 0.0, 0.0};
  PetscReal stress6[6], tangent6[36];

  MaterialMCElasticStressTangent(mat, strain6, stress6, tangent ? tangent6 : NULL);
  if (tangent) MaterialMCExtractPlaneStrain2D(stress6, tangent6, stress, tangent);
  else {
    stress[0] = stress6[0];
    stress[1] = stress6[1];
    stress[2] = stress6[3];
  }
}

void MaterialMCPlasticStressTangent2D(const MaterialMC *mat, PetscReal lambda, const PetscReal strain[3], PetscReal stress[3], PetscReal tangent[9])
{
  PetscReal strain6[6] = {strain[0], strain[1], 0.0, strain[2], 0.0, 0.0};
  PetscReal stress6[6], tangent6[36];

  MaterialMCPlasticStressTangent(mat, lambda, strain6, stress6, tangent ? tangent6 : NULL);
  if (tangent) MaterialMCExtractPlaneStrain2D(stress6, tangent6, stress, tangent);
  else {
    stress[0] = stress6[0];
    stress[1] = stress6[1];
    stress[2] = stress6[3];
  }
}
