#include "constitutive_3d_batch.h"

#include "constitutive_3D_kernel.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void constitutive_problem_3d_s_batch_c(
    const double *E,
    const double *c_bar,
    const double *sin_phi,
    const double *shear,
    const double *bulk,
    const double *lame,
    int n_int,
    double *S_out
) {
#pragma omp parallel for schedule(static)
    for (int p = 0; p < n_int; ++p) {
        constitutive_3D_point(
            E + (size_t)6 * (size_t)p,
            c_bar[p], sin_phi[p], shear[p], bulk[p], lame[p],
            S_out + (size_t)6 * (size_t)p,
            NULL
        );
    }
}

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
) {
#pragma omp parallel for schedule(static)
    for (int p = 0; p < n_int; ++p) {
        constitutive_3D_point(
            E + (size_t)6 * (size_t)p,
            c_bar[p], sin_phi[p], shear[p], bulk[p], lame[p],
            S_out + (size_t)6 * (size_t)p,
            DS_out + (size_t)36 * (size_t)p
        );
    }
}
