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
    const int32_t *scatter_map,
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
            const int32_t *smap = scatter_map + (size_t)e * (size_t)n_ld2;

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
                const int32_t idx = smap[k];
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

void assemble_tangent_values_3d_rows_c(
    const double *dphi1,
    const double *dphi2,
    const double *dphi3,
    const double *ds,
    const double *weight,
    const int32_t *row_slot_ptr,
    const int32_t *slot_elem,
    const uint8_t *slot_lrow,
    const int32_t *slot_pos,
    int n_int,
    int n_rows,
    int n_slots,
    int n_p,
    int n_q,
    int nnz_out,
    double *out_values
) {
    (void)n_int;
    (void)n_slots;
    (void)nnz_out;

    const int n_local_dof = DIM * n_p;

#pragma omp parallel for schedule(static)
    for (int row = 0; row < n_rows; ++row) {
        const int slot_start = row_slot_ptr[row];
        const int slot_end = row_slot_ptr[row + 1];
        if (slot_start == slot_end) {
            continue;
        }

        for (int slot = slot_start; slot < slot_end; ++slot) {
            const int elem_id = slot_elem[slot];
            const int alpha = (int)slot_lrow[slot];
            const int node_i = alpha / DIM;
            const int comp = alpha % DIM;
            const int32_t *pos = slot_pos + (size_t)slot * (size_t)n_local_dof;
            const int g_base = elem_id * n_q;

            for (int q = 0; q < n_q; ++q) {
                const int g = g_base + q;
                const double *dp1 = dphi1 + (size_t)g * (size_t)n_p;
                const double *dp2 = dphi2 + (size_t)g * (size_t)n_p;
                const double *dp3 = dphi3 + (size_t)g * (size_t)n_p;
                const double *dsg = ds + (size_t)g * (size_t)(NS * NS);
                const double w = weight[g];
                const double dxi = dp1[node_i];
                const double dyi = dp2[node_i];
                const double dzi = dp3[node_i];

                double t0;
                double t1;
                double t2;
                double t3;
                double t4;
                double t5;

                if (comp == 0) {
                    t0 = w * (dxi * dsg[0] + dyi * dsg[3] + dzi * dsg[5]);
                    t1 = w * (dxi * dsg[6] + dyi * dsg[9] + dzi * dsg[11]);
                    t2 = w * (dxi * dsg[12] + dyi * dsg[15] + dzi * dsg[17]);
                    t3 = w * (dxi * dsg[18] + dyi * dsg[21] + dzi * dsg[23]);
                    t4 = w * (dxi * dsg[24] + dyi * dsg[27] + dzi * dsg[29]);
                    t5 = w * (dxi * dsg[30] + dyi * dsg[33] + dzi * dsg[35]);
                } else if (comp == 1) {
                    t0 = w * (dyi * dsg[1] + dxi * dsg[3] + dzi * dsg[4]);
                    t1 = w * (dyi * dsg[7] + dxi * dsg[9] + dzi * dsg[10]);
                    t2 = w * (dyi * dsg[13] + dxi * dsg[15] + dzi * dsg[16]);
                    t3 = w * (dyi * dsg[19] + dxi * dsg[21] + dzi * dsg[22]);
                    t4 = w * (dyi * dsg[25] + dxi * dsg[27] + dzi * dsg[28]);
                    t5 = w * (dyi * dsg[31] + dxi * dsg[33] + dzi * dsg[34]);
                } else {
                    t0 = w * (dzi * dsg[2] + dyi * dsg[4] + dxi * dsg[5]);
                    t1 = w * (dzi * dsg[8] + dyi * dsg[10] + dxi * dsg[11]);
                    t2 = w * (dzi * dsg[14] + dyi * dsg[16] + dxi * dsg[17]);
                    t3 = w * (dzi * dsg[20] + dyi * dsg[22] + dxi * dsg[23]);
                    t4 = w * (dzi * dsg[26] + dyi * dsg[28] + dxi * dsg[29]);
                    t5 = w * (dzi * dsg[32] + dyi * dsg[34] + dxi * dsg[35]);
                }

                for (int j = 0; j < n_p; ++j) {
                    const double dxj = dp1[j];
                    const double dyj = dp2[j];
                    const double dzj = dp3[j];
                    const int base = DIM * j;
                    const int32_t pos_x = pos[base + 0];
                    const int32_t pos_y = pos[base + 1];
                    const int32_t pos_z = pos[base + 2];

                    if (pos_x >= 0) {
                        out_values[pos_x] += t0 * dxj + t3 * dyj + t5 * dzj;
                    }
                    if (pos_y >= 0) {
                        out_values[pos_y] += t3 * dxj + t1 * dyj + t4 * dzj;
                    }
                    if (pos_z >= 0) {
                        out_values[pos_z] += t5 * dxj + t4 * dyj + t2 * dzj;
                    }
                }
            }
        }
    }
}

void assemble_overlap_strain_3d_c(
    const double *dphi1,
    const double *dphi2,
    const double *dphi3,
    const double *u_overlap,
    const int32_t *elem_dof_lids,
    int n_int,
    int n_elem,
    int n_p,
    int n_q,
    double *out_values
) {
    (void)n_int;

    const int n_local_dof = DIM * n_p;

#pragma omp parallel for schedule(static)
    for (int e = 0; e < n_elem; ++e) {
        const int32_t *dofs = elem_dof_lids + (size_t)e * (size_t)n_local_dof;
        const int g_base = e * n_q;
        for (int q = 0; q < n_q; ++q) {
            const int g = g_base + q;
            const double *dp1 = dphi1 + (size_t)g * (size_t)n_p;
            const double *dp2 = dphi2 + (size_t)g * (size_t)n_p;
            const double *dp3 = dphi3 + (size_t)g * (size_t)n_p;
            double e11 = 0.0;
            double e22 = 0.0;
            double e33 = 0.0;
            double e12 = 0.0;
            double e23 = 0.0;
            double e13 = 0.0;

            for (int j = 0; j < n_p; ++j) {
                const int base = DIM * j;
                const double ux = u_overlap[dofs[base + 0]];
                const double uy = u_overlap[dofs[base + 1]];
                const double uz = u_overlap[dofs[base + 2]];
                const double dx = dp1[j];
                const double dy = dp2[j];
                const double dz = dp3[j];

                e11 += dx * ux;
                e22 += dy * uy;
                e33 += dz * uz;
                e12 += dy * ux + dx * uy;
                e23 += dz * uy + dy * uz;
                e13 += dz * ux + dx * uz;
            }

            out_values[(size_t)g * (size_t)NS + 0] = e11;
            out_values[(size_t)g * (size_t)NS + 1] = e22;
            out_values[(size_t)g * (size_t)NS + 2] = e33;
            out_values[(size_t)g * (size_t)NS + 3] = e12;
            out_values[(size_t)g * (size_t)NS + 4] = e23;
            out_values[(size_t)g * (size_t)NS + 5] = e13;
        }
    }
}

void assemble_force_3d_rows_c(
    const double *dphi1,
    const double *dphi2,
    const double *dphi3,
    const double *stress,
    const double *weight,
    const int32_t *row_slot_ptr,
    const int32_t *slot_elem,
    const uint8_t *slot_lrow,
    int n_int,
    int n_rows,
    int n_slots,
    int n_p,
    int n_q,
    double *out_values
) {
    (void)n_int;
    (void)n_slots;

#pragma omp parallel for schedule(static)
    for (int row = 0; row < n_rows; ++row) {
        const int slot_start = row_slot_ptr[row];
        const int slot_end = row_slot_ptr[row + 1];
        double acc = 0.0;

        for (int slot = slot_start; slot < slot_end; ++slot) {
            const int elem_id = slot_elem[slot];
            const int alpha = (int)slot_lrow[slot];
            const int node_i = alpha / DIM;
            const int comp = alpha % DIM;
            const int g_base = elem_id * n_q;

            for (int q = 0; q < n_q; ++q) {
                const int g = g_base + q;
                const double *sig = stress + (size_t)g * (size_t)NS;
                const double w = weight[g];
                const double dxi = dphi1[(size_t)g * (size_t)n_p + (size_t)node_i];
                const double dyi = dphi2[(size_t)g * (size_t)n_p + (size_t)node_i];
                const double dzi = dphi3[(size_t)g * (size_t)n_p + (size_t)node_i];

                if (comp == 0) {
                    acc += w * (dxi * sig[0] + dyi * sig[3] + dzi * sig[5]);
                } else if (comp == 1) {
                    acc += w * (dxi * sig[3] + dyi * sig[1] + dzi * sig[4]);
                } else {
                    acc += w * (dxi * sig[5] + dyi * sig[4] + dzi * sig[2]);
                }
            }
        }

        out_values[row] = acc;
    }
}
