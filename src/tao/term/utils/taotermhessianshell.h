#pragma once

#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/

typedef struct _n_TaoTermHessianShell TaoTermHessianShell;

struct _n_TaoTermHessianShell {
  TaoTerm          term;
  TaoTermMapping  *mt; /* copy of the mapping; when set, MatMult() applies map^T (grad^2 f)(map x) map matrix-free */
  Vec              x;
  Vec              params;
  Vec              Ax; /* work vector for map * x, used only when mt is set */
  PetscObjectState x_state;
  PetscObjectState params_state;
  MatState         map_state;
  PetscBool        x_state_change_warning;
  PetscBool        params_state_change_warning;
  PetscBool        map_state_change_warning;
};

PETSC_INTERN PetscErrorCode TaoTermHessianShellCheck(TaoTermHessianShell *, PetscBool, PetscBool);
