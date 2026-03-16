#ifndef SLOPE_STABILITY_PETSC_ASSEMBLE_TANGENT_VALUES_3D_H
#define SLOPE_STABILITY_PETSC_ASSEMBLE_TANGENT_VALUES_3D_H

#include <stdint.h>

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
);

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
);

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
);

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
);

#endif
