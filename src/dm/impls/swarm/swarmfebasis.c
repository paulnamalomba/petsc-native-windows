#include <petsc/private/petscfeimpl.h>
#include <petsc/private/dtimpl.h>
#include <petsc/private/dmimpl.h>
#include <petscdmplex.h>
#include "swarmfebasis.h"

static PetscErrorCode DMSwarmFEBasisDestroy_Private(PetscCtxRt ctx)
{
  DMSwarmFEBasis basis = *(DMSwarmFEBasis *)ctx;

  PetscFunctionBegin;
  if (basis->freedevice) PetscCall((*basis->freedevice)(basis));
  PetscCall(PetscFree2(basis->pkd, basis->coef));
  PetscCall(PetscFree(basis->invV));
  PetscCall(PetscFree(basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set the factor prime basis of a scalar PETSCSPACEPOLYNOMIAL, mirroring PetscSpaceEvaluate_Polynomial() */
static PetscErrorCode DMSwarmFEBasisSetFactor_Private(DMSwarmFEBasis basis, PetscSpace sp, PetscBool *supported)
{
  PetscBool isPoly, tensor;
  PetscInt  Nv, Nc, degree;

  PetscFunctionBegin;
  *supported = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)sp, PETSCSPACEPOLYNOMIAL, &isPoly));
  if (!isPoly) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSpacePolynomialGetTensor(sp, &tensor));
  PetscCall(PetscSpaceGetNumVariables(sp, &Nv));
  PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));
  if ((tensor && Nv > 1) || Nc != 1 || Nv != basis->fdim) PetscFunctionReturn(PETSC_SUCCESS);
  basis->ptype = degree == 1 ? DMSWARM_FEBASIS_LINEAR : DMSWARM_FEBASIS_PKD;
  if (degree == 1) basis->fpdim = Nv + 1;
  else PetscCall(PetscDTBinomialInt(Nv + degree, Nv, &basis->fpdim));
  if (basis->fpdim > (basis->tensor ? DMSWARM_FEBASIS_MAX_FACTOR_PDIM : DMSWARM_FEBASIS_MAX_PDIM)) PetscFunctionReturn(PETSC_SUCCESS);
  if (basis->ptype == DMSWARM_FEBASIS_PKD) {
    PetscInt  *dvar, *m1, *m2;
    PetscReal *cnm1, *cnm1x, *cnm2, *scales;

    PetscCall(PetscMalloc2(3 * basis->fpdim, &basis->pkd, 4 * basis->fpdim, &basis->coef));
    PetscCall(PetscMalloc7(basis->fpdim, &dvar, basis->fpdim, &m1, basis->fpdim, &m2, basis->fpdim, &cnm1, basis->fpdim, &cnm1x, basis->fpdim, &cnm2, basis->fpdim, &scales));
    PetscCall(PetscDTPKDRecurrence_Internal(Nv, degree, dvar, m1, m2, cnm1, cnm1x, cnm2, scales));
    for (PetscInt i = 0; i < basis->fpdim; ++i) {
      basis->pkd[3 * i + 0]  = dvar[i];
      basis->pkd[3 * i + 1]  = m1[i];
      basis->pkd[3 * i + 2]  = m2[i];
      basis->coef[4 * i + 0] = cnm1[i];
      basis->coef[4 * i + 1] = cnm1x[i];
      basis->coef[4 * i + 2] = cnm2[i];
      basis->coef[4 * i + 3] = scales[i];
    }
    PetscCall(PetscFree7(dvar, m1, m2, cnm1, cnm1x, cnm2, scales));
  }
  *supported = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmFEBasisCreate_Private(PetscObject disc, DMSwarmFEBasis basis)
{
  PetscClassId id;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetClassId(disc, &id));
  if (id == PETSCFV_CLASSID) {
    PetscInt Nc;

    PetscCall(PetscFVGetNumComponents((PetscFV)disc, &Nc));
    if (Nc != 1) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscFVGetSpatialDimension((PetscFV)disc, &basis->dim));
    basis->ptype     = DMSWARM_FEBASIS_CONSTANT;
    basis->pdim      = 1;
    basis->fpdim     = 1;
    basis->supported = PETSC_TRUE;
  } else if (id == PETSCFE_CLASSID) {
    PetscFE    fe = (PetscFE)disc;
    PetscSpace sp;
    PetscBool  isBasic, isTensor, supported = PETSC_TRUE;
    PetscInt   Nc;

    PetscCall(PetscObjectTypeCompare(disc, PETSCFEBASIC, &isBasic));
    PetscCall(PetscFEGetNumComponents(fe, &Nc));
    if (!isBasic || Nc != 1) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscFESetUp(fe));
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetNumVariables(sp, &basis->dim));
    PetscCall(PetscFEGetDimension(fe, &basis->pdim));
    if (basis->pdim > DMSWARM_FEBASIS_MAX_PDIM) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscObjectTypeCompare((PetscObject)sp, PETSCSPACETENSOR, &isTensor));
    if (isTensor) {
      PetscInt Ns, fdeg = -1;

      /* PetscSpaceEvaluate_Tensor() of dim identical one-variable polynomial factors */
      basis->tensor = PETSC_TRUE;
      basis->fdim   = 1;
      PetscCall(PetscSpaceTensorGetNumSubspaces(sp, &Ns));
      if (Ns != basis->dim) PetscFunctionReturn(PETSC_SUCCESS);
      for (PetscInt s = 0; s < Ns && supported; ++s) {
        PetscSpace subsp;
        PetscInt   deg;

        PetscCall(PetscSpaceTensorGetSubspace(sp, s, &subsp));
        PetscCall(PetscSpaceGetDegree(subsp, &deg, NULL));
        if (s == 0) {
          fdeg = deg;
          PetscCall(DMSwarmFEBasisSetFactor_Private(basis, subsp, &supported));
        } else {
          PetscBool isPoly;
          PetscInt  Nv, sNc;

          PetscCall(PetscObjectTypeCompare((PetscObject)subsp, PETSCSPACEPOLYNOMIAL, &isPoly));
          PetscCall(PetscSpaceGetNumVariables(subsp, &Nv));
          PetscCall(PetscSpaceGetNumComponents(subsp, &sNc));
          supported = isPoly && Nv == 1 && sNc == 1 && deg == fdeg ? PETSC_TRUE : PETSC_FALSE;
        }
      }
      if (supported && PetscPowInt(basis->fpdim, basis->dim) != basis->pdim) supported = PETSC_FALSE;
    } else {
      basis->fdim = basis->dim;
      PetscCall(DMSwarmFEBasisSetFactor_Private(basis, sp, &supported));
      if (supported && basis->fpdim != basis->pdim) supported = PETSC_FALSE;
    }
    if (!supported) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscMalloc1(basis->pdim * basis->pdim, &basis->invV));
    PetscCall(PetscArraycpy(basis->invV, fe->invV, basis->pdim * basis->pdim));
    basis->supported = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMSwarmFEBasisGet_Internal - Get the device evaluation description of a discretization, cached on it

  Input Parameter:
. disc - A PetscFE or PetscFV

  Output Parameter:
. basis - The description; basis->supported is PETSC_FALSE if the discretization cannot be evaluated on the device

  Note:
  Supported are scalar PETSCFEBASIC elements whose basis space is a PETSCSPACEPOLYNOMIAL (simplices) or a PETSCSPACETENSOR
  of identical one-variable PETSCSPACEPOLYNOMIAL factors (tensor cells), with at most DMSWARM_FEBASIS_MAX_PDIM basis
  functions, and scalar PetscFV.
*/
PetscErrorCode DMSwarmFEBasisGet_Internal(PetscObject disc, DMSwarmFEBasis *basis)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectContainerQuery(disc, "_DMSwarmFEBasis", basis));
  if (!*basis) {
    PetscCall(PetscNew(basis));
    PetscCall(DMSwarmFEBasisCreate_Private(disc, *basis));
    PetscCall(PetscObjectContainerCompose(disc, "_DMSwarmFEBasis", *basis, DMSwarmFEBasisDestroy_Private));
    PetscCall(PetscInfo(disc, "Device basis evaluation %s\n", (*basis)->supported ? "supported" : "not supported"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* As DMSwarmFEBasisGet_Internal(), with the device mirrors uploaded if the basis is supported */
PetscErrorCode DMSwarmFEBasisGetDevice_Internal(PetscObject disc, DMSwarmFEBasis *basis)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmFEBasisGet_Internal(disc, basis));
#if PetscDefined(HAVE_HIP)
  if ((*basis)->supported && !(*basis)->freedevice) PetscCall(DMSwarmFEBasisUpload_HIP(*basis));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmCellGeometryDestroy_Private(PetscCtxRt ctx)
{
  DMSwarmCellGeometry geom = *(DMSwarmCellGeometry *)ctx;

  PetscFunctionBegin;
  if (geom->freedevice) PetscCall((*geom->freedevice)(geom));
  PetscCall(PetscFree3(geom->v0, geom->invJ, geom->detJ));
  PetscCall(PetscFree(geom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMSwarmCellGeometryGetDevice_Internal - Get the cell geometry table of a DMPLEX with its device mirrors, cached on the DM

  Input Parameter:
. dm - The DMPLEX, whose coordinate dimension must be its dimension

  Output Parameter:
. geom - The table: v0, invJ, and detJ of DMPlexComputeCellGeometryFEM(dm, c, NULL, ...) for every cell c

  Note:
  The table is rebuilt when the (id, state) of the local or cellwise coordinate Vec changed since it was built, the same
  test as DMPlexInvalidateStaleLocationCache_Internal(), so that it reproduces the per-cell host computation bitwise.
*/
PetscErrorCode DMSwarmCellGeometryGetDevice_Internal(DM dm, DMSwarmCellGeometry *geom)
{
  DMSwarmCellGeometry g;
  PetscObjectId       id[2]    = {0, 0};
  PetscObjectState    state[2] = {0, 0};
  PetscInt            cStart, cEnd, dim;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMGetCellCoordinatesLocalSetUp(dm));
  for (PetscInt i = 0; i < 2; ++i) {
    Vec xl = dm->coordinates[i].xl;

    if (xl) {
      PetscCall(PetscObjectGetId((PetscObject)xl, &id[i]));
      PetscCall(PetscObjectStateGet((PetscObject)xl, &state[i]));
    }
  }
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(PetscObjectContainerQuery((PetscObject)dm, "_DMSwarmCellGeometry", &g));
  if (!g || g->cStart != cStart || g->cEnd != cEnd || g->dim != dim || g->coordId[0] != id[0] || g->coordState[0] != state[0] || g->coordId[1] != id[1] || g->coordState[1] != state[1]) {
    PetscReal J[9];

    PetscCall(PetscNew(&g));
    g->cStart = cStart;
    g->cEnd   = cEnd;
    g->dim    = dim;
    for (PetscInt i = 0; i < 2; ++i) {
      g->coordId[i]    = id[i];
      g->coordState[i] = state[i];
    }
    PetscCall(PetscMalloc3((cEnd - cStart) * dim, &g->v0, (cEnd - cStart) * dim * dim, &g->invJ, cEnd - cStart, &g->detJ));
    for (PetscInt c = cStart; c < cEnd; ++c) PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, &g->v0[(c - cStart) * dim], J, &g->invJ[(c - cStart) * dim * dim], &g->detJ[c - cStart]));
    /* Replacing the container destroys any stale table */
    PetscCall(PetscObjectContainerCompose((PetscObject)dm, "_DMSwarmCellGeometry", g, DMSwarmCellGeometryDestroy_Private));
    PetscCall(PetscInfo(dm, "Built the particle mass matrix cell geometry for %" PetscInt_FMT " cells\n", cEnd - cStart));
  }
#if PetscDefined(HAVE_HIP)
  if (!g->freedevice) PetscCall(DMSwarmCellGeometryUpload_HIP(g));
#endif
  *geom = g;
  PetscFunctionReturn(PETSC_SUCCESS);
}
