#ifndef PETSC_SSR_PROBLEM_H
#define PETSC_SSR_PROBLEM_H

#include <petscsys.h>

#define PETSC_SSR_NATIVE_PROBLEM_MANIFEST_KIND "petsc_ssr_native_problem_manifest"
#define PETSC_SSR_NATIVE_PROBLEM_MANIFEST_SCHEMA_VERSION 1
#define PETSC_SSR_DMPLEX_REGION_LABEL "Cell Sets"
#define PETSC_SSR_DMPLEX_BOUNDARY_LABEL "Face Sets"
#define PETSC_SSR_DMPLEX_NATIVE_BOUNDARY_MARKER_LABEL "boundary_marker"

typedef struct {
  char     region_label[64];
  char     boundary_label[64];
  char     native_boundary_marker_label[64];
  PetscInt regions;
  PetscInt boundaries;
  PetscInt nodesets;
  PetscInt boundary_geometry;
} SsrNativeProblemTopologyStats;

typedef struct {
  PetscInt mechanics_dirichlet;
  PetscInt mechanics_neumann;
  PetscInt seepage_head;
  PetscInt seepage_flux;
} SsrNativeProblemRuleStats;

#endif
