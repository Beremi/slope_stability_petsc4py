#include "assemble_tangent_values_3d.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NS 6
#define DIM 3

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
) {
    (void)n_int;
    (void)nnz_out;

    const int n_local_dof = DIM * n_p;
    const int n_ld2 = n_local_dof * n_local_dof;

#pragma omp parallel
    {
        double *ke = (double *)malloc((size_t)n_ld2 * sizeof(double));
        double *beq = (double *)malloc((size_t)(NS * n_local_dof) * sizeof(double));
        double *tmp = (double *)malloc((size_t)(NS * n_local_dof) * sizeof(double));
        const int alloc_ok = (ke != NULL && beq != NULL && tmp != NULL);

#pragma omp for schedule(static)
        for (int e = 0; e < n_elem; ++e) {
            if (!alloc_ok) {
                continue;
            }
            memset(ke, 0, (size_t)n_ld2 * sizeof(double));
            const int g_base = e * n_q;
            const int64_t *smap = scatter_map + (size_t)e * (size_t)n_ld2;

            for (int q = 0; q < n_q; ++q) {
                const int g = g_base + q;
                const double *dp1 = dphi1 + (size_t)g * (size_t)n_p;
                const double *dp2 = dphi2 + (size_t)g * (size_t)n_p;
                const double *dp3 = dphi3 + (size_t)g * (size_t)n_p;
                const double *dsg = ds + (size_t)g * (size_t)(NS * NS);
                const double w = weight[g];

                memset(beq, 0, (size_t)(NS * n_local_dof) * sizeof(double));
                for (int i = 0; i < n_p; ++i) {
                    const double dN1 = dp1[i];
                    const double dN2 = dp2[i];
                    const double dN3 = dp3[i];
                    const int c = DIM * i;

                    beq[0 * n_local_dof + (c + 0)] = dN1;
                    beq[1 * n_local_dof + (c + 1)] = dN2;
                    beq[2 * n_local_dof + (c + 2)] = dN3;
                    beq[3 * n_local_dof + (c + 0)] = dN2;
                    beq[3 * n_local_dof + (c + 1)] = dN1;
                    beq[4 * n_local_dof + (c + 1)] = dN3;
                    beq[4 * n_local_dof + (c + 2)] = dN2;
                    beq[5 * n_local_dof + (c + 0)] = dN3;
                    beq[5 * n_local_dof + (c + 2)] = dN1;
                }

                for (int ii = 0; ii < NS; ++ii) {
                    for (int j = 0; j < n_local_dof; ++j) {
                        double acc = 0.0;
                        for (int kk = 0; kk < NS; ++kk) {
                            acc += (w * dsg[ii + kk * NS]) * beq[kk * n_local_dof + j];
                        }
                        tmp[ii * n_local_dof + j] = acc;
                    }
                }

                for (int ii = 0; ii < n_local_dof; ++ii) {
                    for (int j = 0; j < n_local_dof; ++j) {
                        double acc = 0.0;
                        for (int kk = 0; kk < NS; ++kk) {
                            acc += beq[kk * n_local_dof + ii] * tmp[kk * n_local_dof + j];
                        }
                        ke[ii * n_local_dof + j] += acc;
                    }
                }
            }

            for (int k = 0; k < n_ld2; ++k) {
                const int64_t idx = smap[k];
                if (idx >= 0) {
#pragma omp atomic update
                    out_values[idx] += ke[k];
                }
            }
        }

        free(ke);
        free(beq);
        free(tmp);
    }
}
