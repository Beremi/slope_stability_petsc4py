#include "p4_elasticity_common.h"

static const char help[] =
  "Pure PETSc P4 tetrahedral elasticity L1 Gmsh-mesh case.\n"
  "The common solver reads the copied L1 mesh, glues the bottom, applies side\n"
  "rollers by default, and uses downward gravity as the load.\n\n";

int main(int argc, char **argv)
{
  const P4ElasticityCase spec = {
    P4_ELASTICITY_L1_MESH,
    "l1",
    "../data/adaptive_family_a_l1.msh",
    "rollers",
    0.0,
    1.0
  };

  return P4ElasticityRun(argc, argv, &spec, help);
}
