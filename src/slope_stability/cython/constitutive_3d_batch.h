#ifndef SLOPE_STABILITY_PETSC_CONSTITUTIVE_3D_BATCH_H
#define SLOPE_STABILITY_PETSC_CONSTITUTIVE_3D_BATCH_H

void constitutive_problem_3d_s_batch_c(
    const double *E,
    const double *c_bar,
    const double *sin_phi,
    const double *shear,
    const double *bulk,
    const double *lame,
    int n_int,
    double *S_out
);

void constitutive_problem_3d_sds_batch_c(
    const double *E,
    const double *c_bar,
    const double *sin_phi,
    const double *shear,
    const double *bulk,
    const double *lame,
    int n_int,
    double *S_out,
    double *DS_out
);

#endif
