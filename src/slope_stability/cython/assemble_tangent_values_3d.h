#ifndef SLOPE_STABILITY_PETSC_ASSEMBLE_TANGENT_VALUES_3D_H
#define SLOPE_STABILITY_PETSC_ASSEMBLE_TANGENT_VALUES_3D_H

#include <stdint.h>

void assemble_tangent_values_3d_p2_c(
    const double *dphi1,
    const double *dphi2,
    const double *dphi3,
    const double *ds,
    const double *weight,
    const int64_t *scatter_map,
    int n_int,
    int n_elem,
    int n_p,
    int n_q,
    int nnz_out,
    double *out_values
);

#endif
