#include <petsc/private/taoimpl.h>
#include <petsc/private/matimpl.h>

/*
  Notation: F(x) = alpha f(Ax), with f = mt->term, alpha = mt->scale, A = mt->map (identity if NULL).

  unmapped_*: in the term solution space (range of A), e.g. grad f, Hess f
  mapped_*:   in the outer solution space (domain of A), e.g. A^T grad f, A^T Hess f A
*/

static PetscErrorCode TaoTermMappedHessianStateGet(Mat unmapped, Mat map, Mat mapped, TaoTermMappedHessianState *state)
{
  PetscFunctionBegin;
  PetscCall(MatGetState(unmapped, &state->unmapped));
  PetscCall(MatGetState(map, &state->map));
  PetscCall(MatGetState(mapped, &state->mapped));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappedHessianStateInvalidate(TaoTermMappedHessianState *state)
{
  PetscFunctionBegin;
  PetscCall(MatStateInvalidate(state->unmapped));
  PetscCall(MatStateInvalidate(state->map));
  PetscCall(MatStateInvalidate(state->mapped));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingCreateMappedHessianPlaceholder(TaoTermMapping *mt, Mat *mapped)
{
  PetscLayout layout;
  VecType     vec_type;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)mt->map), mapped));
  PetscCall(MatGetLayouts(mt->map, NULL, &layout));
  PetscCall(MatSetLayouts(*mapped, layout, layout));
  PetscCall(MatGetVecType(mt->map, &vec_type));
  PetscCall(MatSetType(*mapped, MATAIJ));
  PetscCall(MatSetVecType(*mapped, vec_type));
  PetscCall(MatSetUp(*mapped));
  PetscCall(MatAssemblyBegin(*mapped, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mapped, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingUpdateMappedHessian(Mat unmapped, Mat map, Mat mapped, TaoTermMappedHessianState *cached, PetscBool *refreshed)
{
  TaoTermMappedHessianState current;
  Mat                       fresh = NULL;
  PetscBool                 unmapped_same, map_same, mapped_same, structure_changed, values_changed;

  PetscFunctionBegin;
  PetscCall(TaoTermMappedHessianStateGet(unmapped, map, mapped, &current));
  PetscCall(MatStateCompare(current.unmapped, cached->unmapped, &unmapped_same));
  PetscCall(MatStateCompare(current.map, cached->map, &map_same));
  PetscCall(MatStateCompare(current.mapped, cached->mapped, &mapped_same));
  structure_changed = PETSC_FALSE;
  if (cached->unmapped.id != current.unmapped.id || cached->unmapped.nonzerostate != current.unmapped.nonzerostate) structure_changed = PETSC_TRUE;
  if (cached->map.id != current.map.id || cached->map.nonzerostate != current.map.nonzerostate) structure_changed = PETSC_TRUE;
  if (cached->mapped.id != current.mapped.id || cached->mapped.nonzerostate != current.mapped.nonzerostate) structure_changed = PETSC_TRUE;
  values_changed = PetscNot(unmapped_same && map_same && mapped_same);

  *refreshed = PETSC_FALSE;

  if (structure_changed) {
    PetscCall(MatPtAP(unmapped, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &fresh));
    /* Tao and KSP may hold references to `mapped`, so do not replace the Mat object.
       Instead, move the newly computed PtAP matrix into the existing object. */
    PetscCall(MatHeaderReplace(mapped, &fresh));
    *refreshed = PETSC_TRUE;
  } else if (values_changed) {
    PetscCall(MatPtAP(unmapped, map, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped));
    *refreshed = PETSC_TRUE;
  }
  /* The caller records the final cache state after any term scaling has modified `mapped`. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
    PetscCall(VecDestroy(&mt->unmapped_vec_work));
    PetscCall(MatDestroy(&mt->unmapped_H));
    PetscCall(MatDestroy(&mt->unmapped_Hpre));
    PetscCall(MatDestroy(&mt->mapped_H));
    PetscCall(MatDestroy(&mt->mapped_Hpre));
  }
  if (term != mt->term || map != mt->map) {
    PetscCall(TaoTermMappedHessianStateInvalidate(&mt->mapped_H_state));
    PetscCall(TaoTermMappedHessianStateInvalidate(&mt->mapped_Hpre_state));
  }
  PetscCall(PetscObjectReference((PetscObject)term));
  PetscCall(TaoTermDestroy(&mt->term));
  mt->term  = term;
  mt->scale = scale;
  if (map != mt->map) PetscCall(VecDestroy(&mt->map_output));
  PetscCall(PetscObjectReference((PetscObject)map));
  PetscCall(MatDestroy(&mt->map));
  mt->map = map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingReset(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermMappingSetData(mt, NULL, 0.0, NULL, NULL));
  PetscCall(VecDestroy(&mt->mapped_vec_work));
  PetscCall(MatDestroy(&mt->unmapped_H));
  PetscCall(MatDestroy(&mt->unmapped_Hpre));
  PetscCall(MatDestroy(&mt->mapped_H));
  PetscCall(MatDestroy(&mt->mapped_Hpre));
  PetscCall(TaoTermMappedHessianStateInvalidate(&mt->mapped_H_state));
  PetscCall(TaoTermMappedHessianStateInvalidate(&mt->mapped_Hpre_state));
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
    if (!mt->map_output) PetscCall(MatCreateVecs(mt->map, NULL, &mt->map_output));
    PetscCall(MatMult(mt->map, x, mt->map_output));
    *Ax = mt->map_output;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingComputeObjective - Compute the objective value of a `TaoTermMapping`

  Collective

  Input Parameters:
+ mt     - the `TaoTermMapping`
. x      - the point in the outer solution space
. params - (optional) the parameters of the term
- mode   - `INSERT_VALUES` or `ADD_VALUES`

  Output Parameter:
. value - the objective value

  Level: developer

  Notes:
  For `INSERT_VALUES`, this routine computes $value = \alpha f(Ax)$,
  where $\alpha$ is the scale, $A$ is the mapping matrix, and $f$ is the term of `mt`.
  For `ADD_VALUES`, this routine computes $value \leftarrow value + \alpha f(Ax)$.
  If the objective is masked with `TAOTERM_MASK_OBJECTIVE`, the objective value is taken to be zero.

.seealso: `TaoTermMapping`, `TaoTermComputeObjective()`, `TaoTermMappingComputeObjectiveAndGradient()`
*/
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
  Get where to store y (unmapped_g, term solution space),
  where y = grad f(Ax) or y = Hess f(Ax) A v. Usage:

    TaoTermMappingGetWorkVec(mt, mode, g, &unmapped_g);
    TaoTermComputeGradient(mt->term, Ax, params, unmapped_g);
    TaoTermMappingAccumulateWorkVec(mt, mode, unmapped_g, g); // g = alpha A^T y, or g += alpha A^T y

  The caller writes unmapped_g the same way for both modes. It may alias g:

  A         Mode            unmapped_g
  -------   -------------   ---------------------
  none      INSERT_VALUES   g
  none      ADD_VALUES      mt->mapped_vec_work
  present   INSERT_VALUES   mt->unmapped_vec_work
  present   ADD_VALUES      mt->unmapped_vec_work
*/
static PetscErrorCode TaoTermMappingGetWorkVec(TaoTermMapping *mt, InsertMode mode, Vec g, Vec *unmapped_g)
{
  PetscFunctionBegin;
  *unmapped_g = g;
  if (mode == ADD_VALUES) {
    if (!mt->mapped_vec_work) PetscCall(VecDuplicate(g, &mt->mapped_vec_work));
    *unmapped_g = mt->mapped_vec_work;
  }
  if (mt->map) {
    if (!mt->unmapped_vec_work) PetscCall(TaoTermCreateSolutionVec(mt->term, &mt->unmapped_vec_work));
    *unmapped_g = mt->unmapped_vec_work;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Complete the operation started by TaoTermMappingGetWorkVec():

  1. If a mapping matrix A is present, compute

    mapped_g = A^T unmapped_g.

  2. Combine the scaled result with g according to mode:

    INSERT_VALUES: g =  alpha mapped_g
    ADD_VALUES:    g += alpha mapped_g
*/
static PetscErrorCode TaoTermMappingAccumulateWorkVec(TaoTermMapping *mt, InsertMode mode, Vec unmapped_g, Vec g)
{
  Vec mapped_g = unmapped_g;

  PetscFunctionBegin;
  if (mt->map) {
    mapped_g = mode == ADD_VALUES ? mt->mapped_vec_work : g;
    PetscCall(MatMultTranspose(mt->map, unmapped_g, mapped_g));
  }
  if (mode == ADD_VALUES) PetscCall(VecAXPY(g, mt->scale, mapped_g));
  else if (mt->scale != 1.0) PetscCall(VecScale(g, mt->scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingComputeGradient - Compute the gradient of a `TaoTermMapping`

  Collective

  Input Parameters:
+ mt     - the `TaoTermMapping`
. x      - the point in the outer solution space
. params - (optional) the parameters of the term
- mode   - `INSERT_VALUES` or `ADD_VALUES`

  Output Parameter:
. g - the gradient in the outer solution space

  Level: developer

  Notes:
  For `INSERT_VALUES`, this routine computes $g = \alpha A^T \nabla f(Ax)$,
  where $\alpha$ is the scale, $A$ is the mapping matrix, and $f$ is the term of `mt`.
  For `ADD_VALUES`, this routine computes $g \leftarrow g + \alpha A^T \nabla f(Ax)$.
  If the gradient is masked with `TAOTERM_MASK_GRADIENT`, the gradient is taken to be zero.

.seealso: `TaoTermMapping`, `TaoTermComputeGradient()`, `TaoTermMappingComputeObjectiveAndGradient()`
*/
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

/*
  TaoTermMappingComputeObjectiveAndGradient - Compute the objective value and gradient of a `TaoTermMapping`

  Collective

  Input Parameters:
+ mt     - the `TaoTermMapping`
. x      - the point in the outer solution space
. params - (optional) the parameters of the term
- mode   - `INSERT_VALUES` or `ADD_VALUES`

  Output Parameters:
+ value - the objective value
- g     - the gradient in the outer solution space

  Level: developer

  Notes:
  For `INSERT_VALUES`, this routine computes $value = \alpha f(Ax)$ and $g = \alpha A^T \nabla f(Ax)$,
  where $\alpha$ is the scale, $A$ is the mapping matrix, and $f$ is the term of `mt`.
  For `ADD_VALUES`, this routine computes $value \leftarrow value + \alpha f(Ax)$ and
  $g \leftarrow g + \alpha A^T \nabla f(Ax)$.
  If the objective is masked with `TAOTERM_MASK_OBJECTIVE`, the objective value is taken to be zero.
  If the gradient is masked with `TAOTERM_MASK_GRADIENT`, the gradient is taken to be zero.

.seealso: `TaoTermMapping`, `TaoTermComputeObjectiveAndGradient()`, `TaoTermMappingComputeObjective()`, `TaoTermMappingComputeGradient()`
*/
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

/*
  Create the unmapped Hessian and preconditioning matrices that do not exist yet, if the term can create
  them. Nothing is computed. When A is the identity, the unmapped and mapped matrices are the same Mat.
*/
static PetscErrorCode TaoTermMappingEnsureUnmappedHessians(TaoTermMapping *mt)
{
  Mat       existing_H, existing_Hpre, H = NULL, Hpre = NULL;
  PetscBool is_defined = PETSC_FALSE, Hpre_is_H, create_H, create_Hpre;

  PetscFunctionBegin;
  existing_H    = mt->unmapped_H ? mt->unmapped_H : (!mt->map ? mt->mapped_H : NULL);
  existing_Hpre = mt->unmapped_Hpre ? mt->unmapped_Hpre : (!mt->map ? mt->mapped_Hpre : NULL);
  PetscCall(TaoTermGetCreateHessianMode(mt->term, &Hpre_is_H, NULL, NULL));
  create_H    = PetscNot(existing_H);
  create_Hpre = PetscNot(existing_Hpre || (!create_H && Hpre_is_H));
  if (!create_H && !create_Hpre) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoTermIsCreateHessianMatricesDefined(mt->term, &is_defined));
  if (!is_defined) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoTermCreateHessianMatrices(mt->term, create_H ? &H : NULL, create_Hpre ? &Hpre : NULL));
  if (H) mt->unmapped_H = H;
  if (Hpre) mt->unmapped_Hpre = Hpre;
  if (!mt->map) {
    if (H && !mt->mapped_H) {
      PetscCall(PetscObjectReference((PetscObject)H));
      mt->mapped_H = H;
    }
    if (Hpre && !mt->mapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)Hpre));
      mt->mapped_Hpre = Hpre;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Get the unmapped and mapped Hessian work matrices. The caller computes the
  unmapped Hessian and preconditioner into unmapped_H and unmapped_Hpre, then
  passes all returned matrices to TaoTermMappingSetHessians(), which forms
  A^T H A and updates the destination matrices.

  A         Mode            Term writes into       Mapped result stored in
  -------   -------------   ---------------------  -------------------------
  none      INSERT_VALUES   destination matrices   same matrices
  none      ADD_VALUES      work matrices          same work matrices
  present   INSERT_VALUES   unmapped matrices      destination or cached matrices
  present   ADD_VALUES      unmapped matrices      mapped work matrices

  The cached mapped matrices are kept so that `MatPtAP()` can reuse them.
  `H` and `Hpre` may be the same matrix.
*/
static PetscErrorCode TaoTermMappingGetHessians(TaoTermMapping *mt, InsertMode mode, Mat Hdest, Mat Hpredest, Mat *mapped_H, Mat *mapped_Hpre, Mat *unmapped_H, Mat *unmapped_Hpre)
{
  PetscFunctionBegin;
  *mapped_H    = Hdest;
  *mapped_Hpre = Hpredest;
  /* For INSERT_VALUES, use the cached mapped matrices when they exist, so that MatPtAP() can reuse them */
  if (mode == INSERT_VALUES) {
    if (Hdest && mt->mapped_H) *mapped_H = mt->mapped_H;
    if (Hpredest && mt->mapped_Hpre) *mapped_Hpre = mt->mapped_Hpre;
  }
  // ADD_VALUES accumulation and mapped terms both need internal unmapped storage
  if (mode == ADD_VALUES || mt->map) PetscCall(TaoTermMappingEnsureUnmappedHessians(mt));
  if (mode == ADD_VALUES) {
    if (Hdest) {
      if (!mt->mapped_H) PetscCall(MatDuplicate(Hdest, MAT_DO_NOT_COPY_VALUES, &mt->mapped_H));
      *mapped_H = mt->mapped_H;
    }
    if (Hpredest) {
      if (!mt->mapped_Hpre) PetscCall(MatDuplicate(Hpredest, MAT_DO_NOT_COPY_VALUES, &mt->mapped_Hpre));
      *mapped_Hpre = mt->mapped_Hpre;
    }
  }
  /* A TaoHessianFn expects a valid H even when only Hpre is requested, so pass mt->mapped_H */
  if (!*mapped_H && Hpredest && !mt->map && mt->mapped_H) *mapped_H = mt->mapped_H;
  *unmapped_H    = *mapped_H;
  *unmapped_Hpre = *mapped_Hpre;
  if (mt->map) {
    if (Hdest) *unmapped_H = mt->unmapped_H;
    if (Hpredest) *unmapped_Hpre = mt->unmapped_Hpre;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Complete the operation started by TaoTermMappingGetHessians():

  1. If a mapping matrix A is present, compute

    mapped_H = A^T unmapped_H A.

  2. Combine the scaled result with Hdest according to mode:

    INSERT_VALUES: Hdest =  alpha mapped_H
    ADD_VALUES:    Hdest += alpha mapped_H

  Apply the same operation to a distinct preconditioning matrix.
*/
static PetscErrorCode TaoTermMappingSetHessians(TaoTermMapping *mt, InsertMode mode, Mat Hdest, Mat Hpredest, Mat mapped_H, Mat mapped_Hpre, Mat unmapped_H, Mat unmapped_Hpre)
{
  PetscBool H_refreshed = PETSC_FALSE, Hpre_refreshed = PETSC_FALSE;

  PetscFunctionBegin;
  if (mt->map) {
    if (mapped_H) PetscCall(TaoTermMappingUpdateMappedHessian(unmapped_H, mt->map, mapped_H, &mt->mapped_H_state, &H_refreshed));
    if (mapped_Hpre && mapped_Hpre != mapped_H) PetscCall(TaoTermMappingUpdateMappedHessian(unmapped_Hpre, mt->map, mapped_Hpre, &mt->mapped_Hpre_state, &Hpre_refreshed));
  }
  if (mode == ADD_VALUES) {
    if (Hdest) PetscCall(MatAXPY(Hdest, mt->scale, mapped_H, UNKNOWN_NONZERO_PATTERN));
    if (Hpredest) PetscCall(MatAXPY(Hpredest, mt->scale, mapped_Hpre, UNKNOWN_NONZERO_PATTERN));
  } else {
    if (Hdest && (mapped_H != Hdest)) PetscCall(MatCopy(mapped_H, Hdest, DIFFERENT_NONZERO_PATTERN));
    if (Hpredest && (Hdest != Hpredest) && (mapped_Hpre != Hpredest)) PetscCall(MatCopy(mapped_Hpre, Hpredest, DIFFERENT_NONZERO_PATTERN));
    if (mt->scale != 1.0) {
      if (Hdest && (!mt->map || mapped_H != Hdest || H_refreshed)) PetscCall(MatScale(Hdest, mt->scale));
      if (Hpredest && Hpredest != Hdest && (!mt->map || mapped_Hpre != Hpredest || Hpre_refreshed)) PetscCall(MatScale(Hpredest, mt->scale));
    }
  }
  if (mt->map) {
    if (mapped_H) PetscCall(TaoTermMappedHessianStateGet(unmapped_H, mt->map, mapped_H, &mt->mapped_H_state));
    if (mapped_Hpre && mapped_Hpre != mapped_H) PetscCall(TaoTermMappedHessianStateGet(unmapped_Hpre, mt->map, mapped_Hpre, &mt->mapped_Hpre_state));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingGetUnmappedHessians - Get the unmapped Hessian matrices of a `TaoTermMapping`

  Collective

  Input Parameter:
. mt - the `TaoTermMapping`

  Output Parameters:
+ unmapped_H    - the unmapped, unscaled Hessian matrix, or `NULL` if none can be created
- unmapped_Hpre - the unmapped, unscaled preconditioning matrix, or `NULL` if `unmapped_H` is `NULL`

  Level: developer

  Notes:
  The matrices are in the term solution space and are owned by `mt`, so do not destroy them.
  They are created on first use if the term supports `TaoTermCreateHessianMatrices()`. If the term
  has no separate preconditioning matrix, `unmapped_Hpre` is `unmapped_H`.

.seealso: `TaoTermMapping`, `TaoTermCreateHessianMatrices()`, `TaoTermMappingApplyHessian()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingGetUnmappedHessians(TaoTermMapping *mt, Mat *unmapped_H, Mat *unmapped_Hpre)
{
  PetscFunctionBegin;
  PetscCall(TaoTermMappingEnsureUnmappedHessians(mt));
  if (mt->map) {
    *unmapped_H    = mt->unmapped_H;
    *unmapped_Hpre = mt->unmapped_Hpre;
  } else {
    *unmapped_H    = mt->unmapped_H ? mt->unmapped_H : mt->mapped_H;
    *unmapped_Hpre = mt->unmapped_Hpre ? mt->unmapped_Hpre : mt->mapped_Hpre;
  }
  if (!*unmapped_Hpre) *unmapped_Hpre = *unmapped_H;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingApplyHessian - Map, scale, and accumulate unmapped Hessian matrices into destination matrices

  Collective

  Input Parameters:
+ mt            - the `TaoTermMapping`
. mode          - `INSERT_VALUES` or `ADD_VALUES`
. unmapped_H    - (optional) the unmapped, unscaled Hessian in the term solution space
- unmapped_Hpre - (optional) the unmapped, unscaled preconditioning matrix in the term solution space

  Output Parameters:
+ H    - (optional) the destination Hessian in the outer solution space
- Hpre - (optional) the destination preconditioning matrix in the outer solution space

  Level: developer

  Notes:
  For `INSERT_VALUES`, this routine computes $H = \alpha A^T H_u A$,
  where $H_u$ is `unmapped_H`, $\alpha$ is the scale, and $A$ is the mapping matrix of `mt`.
  For `ADD_VALUES`, this routine computes $H \leftarrow H + \alpha A^T H_u A$.
  `Hpre` is computed the same way from `unmapped_Hpre`.

.seealso: `TaoTermMapping`, `TaoTermMappingGetUnmappedHessians()`, `TaoTermMappingComputeHessian()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingApplyHessian(TaoTermMapping *mt, InsertMode mode, Mat unmapped_H, Mat unmapped_Hpre, Mat H, Mat Hpre)
{
  Mat mapped_H = unmapped_H, mapped_Hpre = unmapped_Hpre;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (mt->map) {
    if (H && unmapped_H) {
      if (!mt->mapped_H) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->mapped_H));
      mapped_H = mt->mapped_H;
    } else mapped_H = NULL;
    if (Hpre && unmapped_Hpre) {
      if (unmapped_Hpre == unmapped_H) mapped_Hpre = mapped_H;
      else {
        if (!mt->mapped_Hpre) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->mapped_Hpre));
        mapped_Hpre = mt->mapped_Hpre;
      }
    } else mapped_Hpre = NULL;
  }
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Call paths.

  Tao has a single term.
    TaoComputeHessian()
      -> TaoTermMappingComputeHessian()
         (without a map, unmapped_H and mapped_H are the same matrix)

  Tao has several terms. They are held in a TAOTERMSUM.
    TaoComputeHessian()
      -> TaoTermMappingComputeHessian()
        -> TaoTermComputeHessian()
          -> TaoTermComputeHessian_Sum(), which for each summand calls
            -> TaoTermSumHessCacheGetHessian()
            -> TaoTermMappingApplyHessian()
             or, if the summand cannot create unmapped Hessian matrices,
            -> TaoTermMappingComputeHessian()
*/

/*
  TaoTermMappingComputeHessian - Compute the Hessian of a `TaoTermMapping`

  Collective

  Input Parameters:
+ mt     - the `TaoTermMapping`
. x      - the point in the outer solution space
. params - (optional) the parameters of the term
- mode   - `INSERT_VALUES` or `ADD_VALUES`

  Output Parameters:
+ H    - (optional) the Hessian in the outer solution space
- Hpre - (optional) the preconditioning matrix in the outer solution space, may be `H`

  Level: developer

  Notes:
  For `INSERT_VALUES`, this routine computes $H = \alpha A^T \nabla^2 f(Ax) A$,
  where $\alpha$ is the scale, $A$ is the mapping matrix, and $f$ is the term of `mt`.
  For `ADD_VALUES`, this routine computes $H \leftarrow H + \alpha A^T \nabla^2 f(Ax) A$.
  `Hpre` is computed the same way from the preconditioning matrix of the term.
  If the Hessian is masked with `TAOTERM_MASK_HESSIAN`, the Hessian is taken to be zero.
  A matrix-free `H` or `Hpre` is not assembled, only its evaluation point is updated.

.seealso: `TaoTermMapping`, `TaoTermComputeHessian()`, `TaoTermMappingApplyHessian()`, `TaoTermMappingCreateHessianMatrices()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessian(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Mat H, Mat Hpre)
{
  Vec Ax;
  Mat mapped_H, mapped_Hpre, unmapped_H = NULL, unmapped_Hpre = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (mt->map) {
    /* A matrix-free (shell) outer Hessian applies map^T (grad^2 f)(map x) map in MatMult(). Update its
       (x, params) here and skip assembly, as TaoTermComputeHessian() does. */
    PetscCall(TaoTermMappingPreprocessHessianShells(mt, x, params, &H, &Hpre));
    if (!H && !Hpre) PetscFunctionReturn(PETSC_SUCCESS);
  }
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

/*
  TaoTermMappingComputeHessianMult - Compute a Hessian-vector product of a `TaoTermMapping`

  Collective

  Input Parameters:
+ mt         - the `TaoTermMapping`
. Ax         - the point in the term solution space, which is `x` if there is no mapping matrix
. params     - (optional) the parameters of the term
. unmapped_H - (optional) the unmapped, unscaled Hessian at `Ax`, used instead of `TaoTermComputeHessianMult()`
. v          - the vector to multiply, in the outer solution space
- mode       - `INSERT_VALUES` or `ADD_VALUES`

  Output Parameter:
. Hv - the Hessian-vector product in the outer solution space

  Level: developer

  Notes:
  For `INSERT_VALUES`, this routine computes $Hv = \alpha A^T \nabla^2 f(Ax) A v$,
  where $\alpha$ is the scale, $A$ is the mapping matrix, and $f$ is the term of `mt`.
  For `ADD_VALUES`, this routine computes $Hv \leftarrow Hv + \alpha A^T \nabla^2 f(Ax) A v$.
  If the Hessian is masked with `TAOTERM_MASK_HESSIAN`, the Hessian is taken to be zero.

.seealso: `TaoTermMapping`, `TaoTermComputeHessianMult()`, `TaoTermMappingComputeHessian()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessianMult(TaoTermMapping *mt, Vec Ax, Vec params, Mat unmapped_H, Vec v, InsertMode mode, Vec Hv)
{
  Vec unmapped_Hv = NULL, Av;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(Hv));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, v, &Av));
  PetscCall(TaoTermMappingGetWorkVec(mt, mode, Hv, &unmapped_Hv));
  if (unmapped_H) PetscCall(MatMult(unmapped_H, Av, unmapped_Hv));
  else PetscCall(TaoTermComputeHessianMult(mt->term, Ax, params, Av, unmapped_Hv));
  PetscCall(TaoTermMappingAccumulateWorkVec(mt, mode, unmapped_Hv, Hv));
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

/*
  TaoTermMappingCreateHessianMatrices - Create the unmapped and mapped Hessian matrices of a `TaoTermMapping`

  Collective

  Input Parameter:
. mt - the `TaoTermMapping`

  Output Parameters:
+ H    - (optional) the mapped Hessian matrix, pass `NULL` if only `Hpre` is needed
- Hpre - the mapped preconditioning matrix, which is `H` if the term has no separate preconditioning matrix

  Level: developer

  Notes:
  With a mapping matrix $A \in \mathbb{R}^{m \times n}$, the unmapped matrices are $m \times m$ and
  the mapped matrices are $n \times n$. Without one, they are the same `Mat`.

  `H` and `Hpre` return the mapped matrices. Calling this routine again reuses the existing matrices.

.seealso: `TaoTermMapping`, `TaoTermCreateHessianMatrices()`, `TaoTermMappingComputeHessian()`, `TaoTermMappingGetUnmappedHessians()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianMatrices(TaoTermMapping *mt, Mat *H, Mat *Hpre)
{
  Mat       uH, uHpre, mH, mHpre;
  PetscBool is_sum;

  PetscFunctionBegin;
  uH    = mt->unmapped_H;
  uHpre = mt->unmapped_Hpre;
  mH    = mt->mapped_H;
  mHpre = mt->mapped_Hpre;
  PetscCall(PetscObjectTypeCompare((PetscObject)mt->term, TAOTERMSUM, &is_sum));
  if (is_sum && mt->map) PetscCall(PetscInfo(mt->term, "%s: TaoTermType is TAOTERMSUM, but Map is given. Ignoring it.\n", ((PetscObject)mt->term)->prefix));
  /* H may be NULL to request only the Hpre (preconditioner) matrices -- used when the outer
     Hessian is matrix-free (a shell) and the mapped Hessian matrix itself is not needed. */
  PetscCheck(Hpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices() does not take NULL input Hpre");
  if (!mt->map) {
    PetscCheck(!uH || !mH || uH == mH, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian and unmapped Hessian must be the same");
    PetscCheck(!uHpre || !mHpre || uHpre == mHpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian preconditioner and unmapped Hessian preconditioner must be the same");

    // without a map, use existing mapped matrices as the unmapped ones
    if (mt->mapped_H && !mt->unmapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->mapped_H));
      mt->unmapped_H = mt->mapped_H;
    }
    if (mt->mapped_Hpre && !mt->unmapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->mapped_Hpre));
      mt->unmapped_Hpre = mt->mapped_Hpre;
    }
    // create unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->unmapped_H) ? NULL : &mt->unmapped_H, (mt->unmapped_Hpre) ? NULL : &mt->unmapped_Hpre));
    // If mapped matrices are NULL, they should be set to unmapped matrices
    if (mt->unmapped_H && !mt->mapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->unmapped_H));
      mt->mapped_H = mt->unmapped_H;
    }
    if (mt->unmapped_Hpre && !mt->mapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->unmapped_Hpre));
      mt->mapped_Hpre = mt->unmapped_Hpre;
    }

    // always returns Hpre, even if same as H
    if (H && *H != mt->unmapped_H) PetscCall(PetscObjectReference((PetscObject)mt->unmapped_H));
    if (*Hpre != mt->unmapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->unmapped_Hpre));
    if (H) *H = mt->unmapped_H;
    *Hpre = mt->unmapped_Hpre;
  } else {
    PetscBool is_outer_shell = PETSC_FALSE;

    /* A matrix-free (shell) outer Hessian: build a shell that applies map^T (grad^2 f)(map x) map
       through the term's Hessian-vector product, instead of assembling map^T H map (PtAP). */
    if (mt->term->H_mattype) PetscCall(PetscStrcmp(mt->term->H_mattype, MATSHELL, &is_outer_shell));
    if (is_outer_shell) {
      PetscCheck(mt->term->Hpre_is_H, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "A separate preconditioner matrix with a matrix-free (shell) Hessian is not supported for a mapped term; use -tao_term_hessian_pre_is_hessian true");
      if (!mt->mapped_H) PetscCall(TaoTermMappingCreateHessianShell(mt, &mt->mapped_H));
      if (!mt->mapped_Hpre) {
        PetscCall(PetscObjectReference((PetscObject)mt->mapped_H));
        mt->mapped_Hpre = mt->mapped_H;
      }
      if (H) {
        if (*H != mt->mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->mapped_H));
        *H = mt->mapped_H;
      }
      if (*Hpre != mt->mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->mapped_Hpre));
      *Hpre = mt->mapped_Hpre;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    // create unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->unmapped_H) ? NULL : &mt->unmapped_H, (mt->unmapped_Hpre) ? NULL : &mt->unmapped_Hpre));
    /* The mapped matrices must exist now because Tao and KSP keep references to them, but A^T H A cannot be
       formed until H has values. Create empty matrices that the first evaluation fills with MatHeaderReplace(). */
    if (mt->unmapped_H && !mt->mapped_H) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->mapped_H));
    if (H) {
      if (*H != mt->mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->mapped_H));
      *H = mt->mapped_H;
    }
    if (mt->unmapped_Hpre == mt->unmapped_H) {
      // Hpre_is_H true, so mapped_Hpre = mapped_H
      if (!mt->mapped_Hpre) {
        PetscCall(PetscObjectReference((PetscObject)mt->mapped_H));
        mt->mapped_Hpre = mt->mapped_H;
      }
      if (*Hpre != mt->mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->mapped_Hpre));
      *Hpre = mt->mapped_Hpre;
    } else {
      if (!mt->mapped_Hpre) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->mapped_Hpre));
      if (*Hpre != mt->mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->mapped_Hpre));
      *Hpre = mt->mapped_Hpre;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
