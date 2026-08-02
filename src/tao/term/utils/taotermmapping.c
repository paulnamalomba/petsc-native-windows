#include <petsc/private/taoimpl.h>
#include <petsc/private/matimpl.h>

PETSC_INTERN PetscErrorCode TaoTermMappingSetData(TaoTermMapping *mt, const char *prefix, PetscReal scale, TaoTerm term, Mat map)
{
  PetscBool same_name;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(prefix, mt->prefix, &same_name));
  if (!same_name) {
    PetscCall(PetscFree(mt->prefix));
    PetscCall(PetscStrallocpy(prefix, &mt->prefix));
  }
  if (term != mt->term) {
    PetscCall(VecDestroy(&mt->_unmapped_vec_work));
    PetscCall(MatDestroy(&mt->_unmapped_H));
    PetscCall(MatDestroy(&mt->_unmapped_Hpre));
    PetscCall(MatDestroy(&mt->_mapped_H));
    PetscCall(MatDestroy(&mt->_mapped_Hpre));
  }
  PetscCall(PetscObjectReference((PetscObject)term));
  PetscCall(TaoTermDestroy(&mt->term));
  mt->term  = term;
  mt->scale = scale;
  if (map != mt->map) PetscCall(VecDestroy(&mt->_map_output));
  PetscCall(PetscObjectReference((PetscObject)map));
  PetscCall(MatDestroy(&mt->map));
  mt->map = map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingReset(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermMappingSetData(mt, NULL, 0.0, NULL, NULL));
  PetscCall(VecDestroy(&mt->_mapped_vec_work));
  PetscCall(MatDestroy(&mt->_unmapped_H));
  PetscCall(MatDestroy(&mt->_unmapped_Hpre));
  PetscCall(MatDestroy(&mt->_mapped_H));
  PetscCall(MatDestroy(&mt->_mapped_Hpre));
  mt->mask = TAOTERM_MASK_NONE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingGetData(TaoTermMapping *mt, const char **prefix, PetscReal *scale, TaoTerm *term, Mat *map)
{
  PetscFunctionBegin;
  if (prefix) *prefix = mt->prefix;
  if (term) *term = mt->term;
  if (scale) *scale = mt->scale;
  if (map) *map = mt->map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TaoTermMappingCheckInsertMode(mt, mode) \
  do { \
    PetscCheck((mt)->term, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "TaoTermMapping has no TaoTerm set"); \
    PetscCheck((mode) == INSERT_VALUES || (mode) == ADD_VALUES, PetscObjectComm((PetscObject)(mt)->term), PETSC_ERR_ARG_OUTOFRANGE, "insert mode must be INSERT_VALUES or ADD_VALUES"); \
  } while (0)

static PetscErrorCode TaoTermMappingMap(TaoTermMapping *mt, Vec x, Vec *Ax)
{
  PetscFunctionBegin;
  *Ax = x;
  if (mt->map) {
    if (!mt->_map_output) PetscCall(MatCreateVecs(mt->map, NULL, &mt->_map_output));
    PetscCall(MatMult(mt->map, x, mt->_map_output));
    *Ax = mt->_map_output;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjective(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, PetscReal *value)
{
  Vec       Ax;
  PetscReal v;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjective(mt->term, Ax, params, &v));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingGetWorkVec - Get the column-space work vector for a `TaoTerm`

  Collective

  Input Parameters:
+ mt   - the `TaoTermMapping`
. mode - `INSERT_VALUES` or `ADD_VALUES`
- g    - the destination vector, in the row space of `mt->map`

  Output Parameter:
. unmapped_g - column-space buffer the `TaoTerm` writes its raw result into

  Level: developer

  Notes:
  For `INSERT_VALUES`, `unmapped_g` is `g` itself.
  For `ADD_VALUES`, `unmapped_g` is a separate internal vector.

  If `mt->map` is set, `unmapped_g` is instead a separate vector in the column space of `mt->map`.

  The internal vectors are allocated on first use.  Pair every call with
  `TaoTermMappingAccumulateWorkVec()`.

.seealso: `TaoTermMapping`, `TaoTermMappingAccumulateWorkVec()`
*/
static PetscErrorCode TaoTermMappingGetWorkVec(TaoTermMapping *mt, InsertMode mode, Vec g, Vec *unmapped_g)
{
  PetscFunctionBegin;
  *unmapped_g = g;
  if (mode == ADD_VALUES) {
    if (!mt->_mapped_vec_work) PetscCall(VecDuplicate(g, &mt->_mapped_vec_work));
    *unmapped_g = mt->_mapped_vec_work;
  }
  if (mt->map) {
    if (!mt->_unmapped_vec_work) PetscCall(TaoTermCreateSolutionVec(mt->term, &mt->_unmapped_vec_work));
    *unmapped_g = mt->_unmapped_vec_work;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingAccumulateWorkVec - Combine the staged work vector into `g`,
  applying `mt->map`, `mt->scale`, and the requested `InsertMode`

  Collective

  Input Parameters:
+ mt         - the `TaoTermMapping`
. mode       - `INSERT_VALUES` or `ADD_VALUES`
- unmapped_g - column-space work vector from `TaoTermMappingGetWorkVec()`

  Output Parameter:
. g - the destination vector, in the row space of `mt->map`

  Level: developer

  Notes:
  When `mt->map` is set, `mapped_g <- map^H * unmapped_g`; otherwise the two are the same vector.

  For `INSERT_VALUES`, `mapped_g` is `g` itself, and is scaled in place by `mt->scale`.
  For `ADD_VALUES`, `g <- g + mt->scale * mapped_g`.

  This is the counterpart to `TaoTermMappingGetWorkVec()` and must be called with the vector
  it returned.

.seealso: `TaoTermMapping`, `TaoTermMappingGetWorkVec()`
*/
static PetscErrorCode TaoTermMappingAccumulateWorkVec(TaoTermMapping *mt, InsertMode mode, Vec unmapped_g, Vec g)
{
  Vec mapped_g = unmapped_g;

  PetscFunctionBegin;
  if (mt->map) {
    mapped_g = mode == ADD_VALUES ? mt->_mapped_vec_work : g;
    PetscCall(MatMultHermitianTranspose(mt->map, unmapped_g, mapped_g));
  }
  if (mode == ADD_VALUES) PetscCall(VecAXPY(g, mt->scale, mapped_g));
  else if (mt->scale != 1.0) PetscCall(VecScale(g, mt->scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeGradient(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Vec g)
{
  Vec Ax, unmapped_g = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetWorkVec(mt, mode, g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeGradient(mt->term, Ax, params, unmapped_g));
  PetscCall(TaoTermMappingAccumulateWorkVec(mt, mode, unmapped_g, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjectiveAndGradient(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, PetscReal *value, Vec g)
{
  Vec       Ax, unmapped_g = NULL;
  PetscReal v;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask) && TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      *value = 0.0;
      PetscCall(VecZeroEntries(g));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscCall(TaoTermMappingComputeGradient(mt, x, params, mode, g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscCall(TaoTermMappingComputeObjective(mt, x, params, mode, value));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetWorkVec(mt, mode, g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjectiveAndGradient(mt->term, Ax, params, &v, unmapped_g));
  PetscCall(TaoTermMappingAccumulateWorkVec(mt, mode, unmapped_g, g));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingMatPtAP(Mat unmapped_H, Mat map, Mat mapped_H)
{
  PetscFunctionBegin;
  PetscCall(MatPtAP(unmapped_H, map, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped_H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingGetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat *mapped_H, Mat *mapped_Hpre, Mat *unmapped_H, Mat *unmapped_Hpre)
{
  PetscFunctionBegin;
  *mapped_H    = H;
  *mapped_Hpre = Hpre;
  if (mode == ADD_VALUES || mt->map) {
    // we will need _unmapped_H / _unmapped_Hpre
    if (!mt->_unmapped_H) {
      PetscBool is_defined = PETSC_FALSE;

      PetscCall(TaoTermIsCreateHessianMatricesDefined(mt->term, &is_defined));
      if (is_defined) {
        PetscCall(MatDestroy(&mt->_unmapped_Hpre));
        PetscCall(TaoTermCreateHessianMatrices(mt->term, &mt->_unmapped_H, &mt->_unmapped_Hpre));
      }
      if (!mt->map) {
        PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
        PetscCall(MatDestroy(&mt->_mapped_H));
        mt->_mapped_H = mt->_unmapped_H;

        PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
        PetscCall(MatDestroy(&mt->_mapped_Hpre));
        mt->_mapped_Hpre = mt->_unmapped_Hpre;
      }
    }
  }
  if (mode == ADD_VALUES) {
    if (H) {
      if (!mt->_mapped_H) PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &mt->_mapped_H));
      *mapped_H = mt->_mapped_H;
    }
    if (Hpre) {
      if (!mt->_mapped_Hpre) PetscCall(MatDuplicate(Hpre, MAT_DO_NOT_COPY_VALUES, &mt->_mapped_Hpre));
      *mapped_Hpre = mt->_mapped_Hpre;
    }
  }
  *unmapped_H    = *mapped_H;
  *unmapped_Hpre = *mapped_Hpre;
  if (mt->map) {
    if (H) *unmapped_H = mt->_unmapped_H;
    if (Hpre) *unmapped_Hpre = mt->_unmapped_Hpre;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// if (map) mapped_H \gets map^T @ unmapped_H @ map
// else (assumes that unmapped == mapped.
//
// if INSERT
//   H \gets mapped_H
// else if ADD
//   H \gets H + scale * mapped_H
static PetscErrorCode TaoTermMappingSetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat mapped_H, Mat mapped_Hpre, Mat unmapped_H, Mat unmapped_Hpre)
{
  PetscFunctionBegin;
  if (mt->map) {
    // currently only implements Gauss-Newton Hessian approximation
    if (mapped_H) PetscCall(TaoTermMappingMatPtAP(unmapped_H, mt->map, mapped_H));
    if (mapped_Hpre && (mapped_Hpre != mapped_H)) PetscCall(TaoTermMappingMatPtAP(unmapped_Hpre, mt->map, mapped_Hpre));
  }
  if (mode == ADD_VALUES) {
    if (H) PetscCall(MatAXPY(H, mt->scale, mapped_H, UNKNOWN_NONZERO_PATTERN));
    if (Hpre) PetscCall(MatAXPY(Hpre, mt->scale, mapped_Hpre, UNKNOWN_NONZERO_PATTERN));
  } else {
    if (H) PetscCall(MatCopy(mapped_H, H, DIFFERENT_NONZERO_PATTERN));
    if (Hpre && (H != Hpre)) PetscCall(MatCopy(mapped_Hpre, Hpre, DIFFERENT_NONZERO_PATTERN));
    if (mt->scale != 1.0) {
      if (H) PetscCall(MatScale(H, mt->scale));
      if (Hpre && Hpre != H) PetscCall(MatScale(Hpre, mt->scale));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Either called by TaoComputeHessian (one term in Tao), or by TAOTERMSUM
//
// First case: (one term in Tao)
// TaoComputeHessian
//   -> TaoTermMappingComputeHessian
//      (unmapped_H == mapped_H)
//
// Second case: TAOTERMSUM, (more than one term in Tao)
// TaoComputeHessian
//   -> TaoTermMappingComputeHessian
//     -> (mt->_unmapped_H == mt->_mapped_H == tao->hessian) (SUM does not take mapping)
//     -> TaoTermComputeHessian
//       -> TaoTermComputeHessian_Sum
//         -> for(i:n_terms)
//         -> TaoTermMappingComputeHessian
//           -> (unmapped_H may not == mapped_H)
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessian(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Mat H, Mat Hpre)
{
  Vec Ax;
  Mat mapped_H, mapped_Hpre, unmapped_H = NULL, unmapped_Hpre = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      if (H) {
        PetscCall(MatZeroEntries(H));
        PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
      }
      if (Hpre && Hpre != H) {
        PetscCall(MatZeroEntries(Hpre));
        PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermMappingGetHessians(mt, mode, H, Hpre, &mapped_H, &mapped_Hpre, &unmapped_H, &unmapped_Hpre));
  PetscCall(TaoTermComputeHessian(mt->term, Ax, params, unmapped_H, unmapped_Hpre));
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingSetUp(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermSetUp(mt->term));
  if (mt->map) PetscCall(MatSetUp(mt->map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingCreateSolutionVec(TaoTermMapping *mt, Vec *solution)
{
  PetscFunctionBegin;
  if (mt->map) PetscCall(MatCreateVecs(mt->map, solution, NULL));
  else PetscCall(TaoTermCreateSolutionVec(mt->term, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingCreateParametersVec(TaoTermMapping *mt, Vec *params)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreateParametersVec(mt->term, params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This function takes in unmapped_H, map, and returns matrix for mapped_H, which is PtAP
static PetscErrorCode TaoTermMappingCreatePtAP(Mat unmapped_H, Mat map, Mat *H)
{
  PetscFunctionBegin;
  PetscCall(MatPtAP(unmapped_H, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, H));
  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Internal function to create Hessian matrices for TaoTermMapping
 *
 * map: m x n
 *
 * This function will internally create unmapped, and mapped  H and Hpre,
 * and return H \gets mt->_mapped_H, and Hpre \gets mt->_mapped_Hpre.
 *
 * if (mt->map)
 *   mapped:   n x n
 *   unmapped: m x m
 *
 * else
 *   mapped:   n x n
 *   unmapped: n x n
 *
 */
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianMatrices(TaoTermMapping *mt, Mat *H, Mat *Hpre)
{
  Mat       uH, uHpre, mH, mHpre;
  PetscBool is_sum;

  PetscFunctionBegin;
  uH    = mt->_unmapped_H;
  uHpre = mt->_unmapped_Hpre;
  mH    = mt->_mapped_H;
  mHpre = mt->_mapped_Hpre;
  PetscCall(PetscObjectTypeCompare((PetscObject)mt->term, TAOTERMSUM, &is_sum));
  if (is_sum && mt->map) PetscCall(PetscInfo(mt->term, "%s: TaoTermType is TAOTERMSUM, but Map is given. Ignoring it.\n", ((PetscObject)mt->term)->prefix));
  PetscCheck(H, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input for H");
  PetscCheck(Hpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input Hpre");
  if (!mt->map) {
    // mt->_unmapped_{H,Hpre} == mt->_unmapped_{H,Hpre}
    if (uH && mH) PetscCheck(uH == mH, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian and unmapped Hessian must be same");
    if (uHpre && mHpre) PetscCheck(uHpre == mHpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian preconditioner and unmapped Hessian preconditioner needs to be same");

    // If mapped matrices are present, it should be set to unmapped matrices
    if (mt->_mapped_H && !mt->_unmapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
      mt->_unmapped_H = mt->_mapped_H;
    }
    if (mt->_mapped_Hpre && !mt->_unmapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      mt->_unmapped_Hpre = mt->_mapped_Hpre;
    }
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    // If mapped matrices are NULL, it should be set to mapped matrices
    if (mt->_unmapped_H && !mt->_mapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
      mt->_mapped_H = mt->_unmapped_H;
    }
    if (mt->_unmapped_Hpre && !mt->_mapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
      mt->_mapped_Hpre = mt->_unmapped_Hpre;
    }

    // always returns Hpre, even if same as H
    if (*H != mt->_unmapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
    if (*Hpre != mt->_unmapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
    *H    = mt->_unmapped_H;
    *Hpre = mt->_unmapped_Hpre;
  } else {
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    // Hack to support  AIJ.... TODO
    PetscCall(MatAssemblyBegin(mt->_unmapped_H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mt->_unmapped_H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(mt->_unmapped_H, 1.));
    // Create PtAP only if mt->_mapped_H is empty
    if (mt->_unmapped_H && !mt->_mapped_H) PetscCall(TaoTermMappingCreatePtAP(mt->_unmapped_H, mt->map, &mt->_mapped_H));
    if (*H != mt->_mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
    *H = mt->_mapped_H;
    if (mt->_unmapped_Hpre == mt->_unmapped_H) {
      // Hpre_is_H true, so mapped_H = mapped_Hpre
      if (!mt->_mapped_Hpre) {
        PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
        mt->_mapped_Hpre = mt->_mapped_H;
      }
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)*H));
      *Hpre = *H;
    } else {
      PetscCall(MatAssemblyBegin(mt->_unmapped_Hpre, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(mt->_unmapped_Hpre, MAT_FINAL_ASSEMBLY));
      PetscCall(MatShift(mt->_unmapped_Hpre, 1.));
      if (!mt->_mapped_Hpre) PetscCall(TaoTermMappingCreatePtAP(mt->_unmapped_Hpre, mt->map, &mt->_mapped_Hpre));
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      *Hpre = mt->_mapped_Hpre;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
