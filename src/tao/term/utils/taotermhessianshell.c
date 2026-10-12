#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/
#include "taotermhessianshell.h"

static PetscErrorCode TaoTermHessianShellDestroy(PetscCtxRt ctx)
{
  TaoTermHessianShell *hess = *(TaoTermHessianShell **)ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&hess->x));
  PetscCall(VecDestroy(&hess->params));
  PetscCall(VecDestroy(&hess->Ax));
  if (hess->mt) {
    PetscCall(TaoTermMappingReset(hess->mt));
    PetscCall(PetscFree(hess->mt));
  }
  PetscCall(TaoTermDestroy(&hess->term));
  PetscCall(PetscFree(hess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermHessianShellCheck - Log with `PetscInfo()` if the vectors or mapping matrix
  of a Hessian shell changed since it was last updated

  Not Collective

  Input Parameters:
+ hess             - the Hessian shell context
. check_solution   - check the solution vector
- check_parameters - check the parameters vector

  Level: developer

.seealso: `TaoTermUpdateHessianShell()`, `TaoTermCreateHessianShell()`
*/
PETSC_INTERN PetscErrorCode TaoTermHessianShellCheck(TaoTermHessianShell *hess, PetscBool check_solution, PetscBool check_parameters)
{
  PetscFunctionBegin;
  if (check_solution) {
    PetscObjectState x_state;

    PetscCall(PetscObjectStateGet((PetscObject)hess->x, &x_state));
    if (!hess->x_state_change_warning && x_state != hess->x_state) {
      hess->x_state_change_warning = PETSC_TRUE;
      PetscCall(PetscInfo(hess->term, "x vector may have changed since TaoTermUpdateHessianShell() was called\n"));
    }
  }
  if (check_parameters) {
    PetscObjectState params_state = 0;

    if (hess->params) PetscCall(PetscObjectStateGet((PetscObject)hess->params, &params_state));
    if (!hess->params_state_change_warning && params_state != hess->params_state) {
      hess->params_state_change_warning = PETSC_TRUE;
      PetscCall(PetscInfo(hess->term, "parameter vector may have changed since TaoTermUpdateHessianShell() was called\n"));
    }
  }
  if (hess->mt && hess->mt->map) {
    MatState  map_state;
    PetscBool map_same;

    PetscCall(MatGetState(hess->mt->map, &map_state));
    PetscCall(MatStateCompare(map_state, hess->map_state, &map_same));
    if (!hess->map_state_change_warning && !map_same) {
      hess->map_state_change_warning = PETSC_TRUE;
      PetscCall(PetscInfo(hess->term, "mapping matrix has changed since the Hessian shell was refreshed\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_TaoTermHessianShell(Mat shell, Vec v, Vec y)
{
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &hess));
  PetscCheck(hess->x, PetscObjectComm((PetscObject)shell), PETSC_ERR_ARG_WRONGSTATE, "TaoTermUpdateHessianShell() must be called before applying this Hessian shell");
  PetscCall(TaoTermHessianShellCheck(hess, PETSC_TRUE, PETSC_TRUE));
  if (hess->mt) {
    Vec Ax = hess->x;

    /* mapped term: y = alpha map^T (grad^2 f)(map x) map v, using the term's Hessian-vector product */
    if (hess->mt->map) {
      if (!hess->Ax) PetscCall(MatCreateVecs(hess->mt->map, NULL, &hess->Ax));
      PetscCall(MatMult(hess->mt->map, hess->x, hess->Ax));
      Ax = hess->Ax;
    }
    PetscCall(TaoTermMappingComputeHessianMult(hess->mt, Ax, hess->params, NULL, v, INSERT_VALUES, y));
  } else PetscCall(TaoTermComputeHessianMult(hess->term, hess->x, hess->params, v, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianShell - Create a `MATSHELL` for `TaoTermComputeHessianMult()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. shell - a `Mat` of type `MATSHELL`

  Level: developer

  Note:
  `MatMult()` on `shell` evaluates the Hessian at the solution and parameters vectors last passed to
  `TaoTermUpdateHessianShell()`, which `TaoTermComputeHessian()` calls when `shell` is its Hessian matrix.
  `shell` keeps references to these vectors, not copies, so modifying them in place changes the product.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianMult()`, `TaoTermUpdateHessianShell()`
@*/
PetscErrorCode TaoTermCreateHessianShell(TaoTerm term, Mat *shell)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(shell, 2);
  PetscCall(MatCreate(PetscObjectComm((PetscObject)term), shell));
  PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
  PetscCall(MatSetLayouts(*shell, sol_layout, sol_layout));
  PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
  PetscCall(MatSetVecType(*shell, sol_vec_type));
  PetscCall(MatSetType(*shell, MATSHELL));
  PetscCall(PetscNew(&hess));
  PetscCall(PetscObjectReference((PetscObject)term));
  hess->term = term;
  PetscCall(MatShellSetContext(*shell, hess));
  PetscCall(MatShellSetOperation(*shell, MATOP_MULT, (void (*)(void))MatMult_TaoTermHessianShell));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &container));
  PetscCall(PetscContainerSetPointer(container, hess));
  PetscCall(PetscContainerSetCtxDestroy(container, TaoTermHessianShellDestroy));
  PetscCall(PetscObjectCompose((PetscObject)*shell, "__TaoTermHessianShell", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingCreateHessianShell - Create a `MATSHELL` for `TaoTermMappingComputeHessianMult()`

  Collective

  Input Parameter:
. mt - a `TaoTermMapping` with a mapping matrix

  Output Parameter:
. shell - a `Mat` of type `MATSHELL` in the outer solution space

  Level: developer

  Notes:
  `MatMult()` on `shell` computes $\alpha A^T \nabla^2 f(Ax) A v$ without assembling the matrix,
  where $\alpha$ is the scale, $A$ is the mapping matrix, and $f$ is the term of `mt`.
  The shell keeps a copy of the scale, mapping matrix, and mask of `mt`, which
  `TaoTermMappingPreprocessHessianShells()` updates.

.seealso: [](sec_tao_term), `TaoTermMapping`, `TaoTermCreateHessianShell()`, `TaoTermMappingComputeHessianMult()`, `TaoTermMappingPreprocessHessianShells()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianShell(TaoTermMapping *mt, Mat *shell)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscAssertPointer(shell, 2);
  PetscCheck(mt->map, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_ARG_WRONGSTATE, "TaoTermMappingCreateHessianShell() requires a mapping matrix");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)mt->term), shell));
  PetscCall(MatGetLayouts(mt->map, NULL, &sol_layout)); /* solution (outer) space is the domain of map */
  PetscCall(MatSetLayouts(*shell, sol_layout, sol_layout));
  PetscCall(MatGetVecType(mt->map, &sol_vec_type));
  PetscCall(MatSetVecType(*shell, sol_vec_type));
  PetscCall(MatSetType(*shell, MATSHELL));
  PetscCall(PetscNew(&hess));
  PetscCall(PetscObjectReference((PetscObject)mt->term));
  hess->term = mt->term;
  PetscCall(PetscNew(&hess->mt));
  PetscCall(TaoTermMappingSetData(hess->mt, NULL, mt->scale, mt->term, mt->map));
  hess->mt->mask = mt->mask;
  PetscCall(MatShellSetContext(*shell, hess));
  PetscCall(MatShellSetOperation(*shell, MATOP_MULT, (void (*)(void))MatMult_TaoTermHessianShell));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)mt->term), &container));
  PetscCall(PetscContainerSetPointer(container, hess));
  PetscCall(PetscContainerSetCtxDestroy(container, TaoTermHessianShellDestroy));
  PetscCall(PetscObjectCompose((PetscObject)*shell, "__TaoTermHessianShell", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermUpdateHessianShell - Update the solution and parameter vectors for a shell matrix constructed with `TaoTermCreateHessianShell()`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. shell  - the matrix created with `TaoTermCreateHessianShell()`
. x      - a solution vector
- params - a parameters vector (may be `NULL`)

  Level: developer

  Notes:
  After this is called, `MatMult()` on `shell` performs `TaoTermComputeHessianMult()` with `x` and `params`.
  `shell` keeps references to `x` and `params`, not copies.
  Calling this discards any modification previously applied to `shell`, such as `MatShift()` or `MatScale()`.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianMult()`, `TaoTermCreateHessianShell()`
@*/
PetscErrorCode TaoTermUpdateHessianShell(TaoTerm term, Mat shell, Vec x, Vec params)
{
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(shell, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscCall(MatShellGetContext(shell, &hess));
  PetscCheck(hess->term == term, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Hessian shell matrix does not come from this TaoTerm");
  PetscCall(PetscObjectReference((PetscObject)x));
  PetscCall(VecDestroy(&hess->x));
  hess->x = x;
  PetscCall(PetscObjectStateGet((PetscObject)x, &hess->x_state));
  PetscCall(PetscObjectReference((PetscObject)params));
  PetscCall(VecDestroy(&hess->params));
  hess->params = params;
  if (params) PetscCall(PetscObjectStateGet((PetscObject)params, &hess->params_state));
  else hess->params_state = 0;
  hess->x_state_change_warning      = PETSC_FALSE;
  hess->params_state_change_warning = PETSC_FALSE;
  /* MATSHELL only clears an accumulated MatShift() or MatScale() on assembly. Assemble so that a shift the
   solver applied at the previous point (e.g. TAONLS) is dropped, as TaoTermComputeHessianMFFD() does. */
  PetscCall(MatAssemblyBegin(shell, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(shell, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* If *M is a Hessian shell, update it with x, params, and mt (if given), then set *M to NULL. */
static PetscErrorCode MaybeUpdateOneShell(TaoTerm term, TaoTermMapping *mt, Vec x, Vec params, Mat *M)
{
  PetscContainer       marker = NULL;
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  if (*M) PetscCall(PetscObjectQuery((PetscObject)*M, "__TaoTermHessianShell", (PetscObject *)&marker));
  if (marker) {
    PetscCall(MatShellGetContext(*M, &hess));
    if (mt) {
      PetscCheck(hess->mt, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Hessian shell matrix does not come from a TaoTermMapping");
      if (hess->mt->map != mt->map) PetscCall(VecDestroy(&hess->Ax));
      PetscCall(TaoTermMappingSetData(hess->mt, NULL, mt->scale, mt->term, mt->map));
      hess->mt->mask = mt->mask;
      PetscCall(MatGetState(hess->mt->map, &hess->map_state));
      hess->map_state_change_warning = PETSC_FALSE;
    }
    PetscCall(TaoTermUpdateHessianShell(term, *M, x, params));
    *M = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermPreprocessHessianShells - Update the Hessian shells among `H` and `Hpre` and set them to `NULL`

  Collective

  Input Parameters:
+ term   - the `TaoTerm`
. x      - the solution vector
- params - (optional) the parameters vector

  Input/Output Parameters:
+ H    - the Hessian matrix, set to `NULL` if it is a Hessian shell
- Hpre - the preconditioning matrix, set to `NULL` if it is a Hessian shell

  Level: developer

  Notes:
  `TaoTermComputeHessian()` calls this first. A Hessian shell, created with
  `TaoTermCreateHessianShell()`, has no entries to compute, so this routine only passes `x` and
  `params` to it with `TaoTermUpdateHessianShell()`. It then sets that argument to `NULL`, which
  the Hessian routine of the `TaoTermType` treats as not requested. An argument that is not a
  Hessian shell is left unchanged.

.seealso: `TaoTermComputeHessian()`, `TaoTermCreateHessianShell()`, `TaoTermUpdateHessianShell()`
*/
PETSC_INTERN PetscErrorCode TaoTermPreprocessHessianShells(TaoTerm term, Vec x, Vec params, Mat *H, Mat *Hpre)
{
  PetscBool Hpre_is_H = (*Hpre == *H) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MaybeUpdateOneShell(term, NULL, x, params, H));
  if (Hpre_is_H && !*H) *Hpre = NULL;
  PetscCall(MaybeUpdateOneShell(term, NULL, x, params, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingPreprocessHessianShells - Update the Hessian shells among `H` and `Hpre` and set them to `NULL`

  Collective

  Input Parameters:
+ mt     - the `TaoTermMapping`
. x      - the solution vector in the outer solution space
- params - (optional) the parameters vector

  Input/Output Parameters:
+ H    - the Hessian matrix, set to `NULL` if it is a Hessian shell
- Hpre - the preconditioning matrix, set to `NULL` if it is a Hessian shell

  Level: developer

  Notes:
  `TaoTermMappingComputeHessian()` calls this first when `mt` has a mapping matrix. It does what
  `TaoTermPreprocessHessianShells()` does, for Hessian shells created with
  `TaoTermMappingCreateHessianShell()`, and also updates each shell's copy of the scale, mapping
  matrix, and mask from `mt`.

.seealso: `TaoTermPreprocessHessianShells()`, `TaoTermMappingCreateHessianShell()`, `TaoTermMappingComputeHessian()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingPreprocessHessianShells(TaoTermMapping *mt, Vec x, Vec params, Mat *H, Mat *Hpre)
{
  PetscBool Hpre_is_H = (*Hpre == *H) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MaybeUpdateOneShell(mt->term, mt, x, params, H));
  if (Hpre_is_H && !*H) *Hpre = NULL;
  PetscCall(MaybeUpdateOneShell(mt->term, mt, x, params, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}
