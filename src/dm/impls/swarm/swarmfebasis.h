#pragma once

#include <petscdm.h>

/* Largest number of basis functions evaluated on the device; a thread keeps the prime basis at its point in local memory */
#define DMSWARM_FEBASIS_MAX_PDIM 64
/* Largest number of prime functions of one factor of a tensor product basis */
#define DMSWARM_FEBASIS_MAX_FACTOR_PDIM 8
/* Rows and columns of a tile of the per-cell blocks of the particle mass matrix square computed by one device thread block */
#define DMSWARM_MASS_SQUARE_TILE 32

typedef enum {
  DMSWARM_FEBASIS_CONSTANT, /* The constant 1, a PetscFV cell value */
  DMSWARM_FEBASIS_LINEAR,   /* 1, x_0, ..., x_{dim-1}: the degree 1 prime basis of PETSCSPACEPOLYNOMIAL */
  DMSWARM_FEBASIS_PKD       /* The PKD basis of PetscDTPKDEvalJet(), the prime basis of PETSCSPACEPOLYNOMIAL for other degrees */
} DMSwarmFEBasisPrimeType;

/*
  A scalar finite element (or finite volume) basis flattened for evaluation on the device, reproducing
  PetscFECreateTabulation() (B only) bitwise: the prime basis, as a single space on simplices or as a product of
  one-variable factors on tensor cells, followed by the contraction with the prime-to-nodal matrix invV
*/
typedef struct _n_DMSwarmFEBasis *DMSwarmFEBasis;
struct _n_DMSwarmFEBasis {
  PetscBool               supported; /* The discretization can be evaluated on the device; nothing below is set otherwise */
  PetscInt                dim;       /* Reference dimension */
  PetscInt                pdim;      /* Number of nodal basis functions */
  PetscBool               tensor;    /* The prime basis is the product of dim one-variable factor bases */
  DMSwarmFEBasisPrimeType ptype;     /* Prime basis of the factor space */
  PetscInt                fdim;      /* Variables of the factor space: dim, or 1 if tensor */
  PetscInt                fpdim;     /* Prime functions of the factor space */
  PetscInt               *pkd;       /* [3 * fpdim] PKD recurrence variable, and the m1 and m2 indices (-1 if none) */
  PetscReal              *coef;      /* [4 * fpdim] PKD cnm1, cnm1x, cnm2, and scale */
  PetscReal              *invV;      /* [pdim * pdim] prime-to-nodal matrix, or NULL for the constant basis */
  void                   *pkd_d, *coef_d, *invV_d;
  PetscErrorCode (*freedevice)(DMSwarmFEBasis);
};

/*
  The affine cell geometry the particle mass matrix assembly uses, DMPlexComputeCellGeometryFEM() at no quadrature point
  for every cell, tabulated for the device. It is cached on the DMPLEX and rebuilt when the coordinates change.
*/
typedef struct _n_DMSwarmCellGeometry *DMSwarmCellGeometry;
struct _n_DMSwarmCellGeometry {
  PetscInt         cStart, cEnd, dim;
  PetscObjectId    coordId[2]; /* Local coordinate and cellwise coordinate Vecs the table was built from */
  PetscObjectState coordState[2];
  PetscReal       *v0;   /* [(cEnd - cStart) * dim] */
  PetscReal       *invJ; /* [(cEnd - cStart) * dim * dim] */
  PetscReal       *detJ; /* [cEnd - cStart] */
  void            *v0_d, *invJ_d, *detJ_d;
  PetscErrorCode (*freedevice)(DMSwarmCellGeometry);
};

PETSC_INTERN PetscErrorCode DMSwarmFEBasisGet_Internal(PetscObject, DMSwarmFEBasis *);
PETSC_INTERN PetscErrorCode DMSwarmFEBasisGetDevice_Internal(PetscObject, DMSwarmFEBasis *);
PETSC_INTERN PetscErrorCode DMSwarmCellGeometryGetDevice_Internal(DM, DMSwarmCellGeometry *);
#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode DMSwarmFEBasisUpload_HIP(DMSwarmFEBasis);
PETSC_INTERN PetscErrorCode DMSwarmFEBasisTabulate_Internal_HIP(DMSwarmFEBasis, PetscInt, const PetscReal[], PetscReal[]);
PETSC_INTERN PetscErrorCode DMSwarmCellGeometryUpload_HIP(DMSwarmCellGeometry);
PETSC_INTERN PetscErrorCode DMSwarmMassMatrixSetValuesCOO_Internal_HIP(Mat, DMSwarmFEBasis, DMSwarmCellGeometry, PetscInt, const PetscInt[], const PetscInt[], const PetscReal[], PetscInt, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMassMatrixSquareSetValuesCOO_Internal_HIP(Mat, DMSwarmFEBasis, DMSwarmCellGeometry, PetscInt, const PetscInt[], const PetscInt[], const PetscReal[], PetscInt, PetscBool, PetscInt, const PetscInt[], const PetscCount[], const PetscCount[]);
#endif
