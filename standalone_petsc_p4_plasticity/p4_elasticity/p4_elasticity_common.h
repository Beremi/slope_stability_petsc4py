#pragma once

#include <petscsys.h>

typedef enum {
  P4_ELASTICITY_CUBE,
  P4_ELASTICITY_L1_MESH
} P4ElasticityCaseKind;

typedef struct {
  P4ElasticityCaseKind kind;
  const char          *name;
  const char          *default_mesh;
  const char          *default_bc_mode;
  PetscReal            default_pressure;
  PetscReal            default_gravity;
} P4ElasticityCase;

PetscErrorCode P4ElasticityRun(int argc, char **argv, const P4ElasticityCase *spec, const char help[]);
