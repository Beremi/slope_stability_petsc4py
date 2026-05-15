#include "p4_elasticity_common.h"

static const char help[] =
  "Pure PETSc P4 tetrahedral elasticity generated-cube case.\n"
  "The common solver generates a structured tetra cube, clamps the lower z face,\n"
  "and applies a downward traction on the upper z face.\n\n";

int main(int argc, char **argv)
{
  const P4ElasticityCase spec = {
    P4_ELASTICITY_CUBE,
    "cube",
    NULL,
    "cube_clamped_bottom",
    1.0,
    0.0
  };

  return P4ElasticityRun(argc, argv, &spec, help);
}
