const char help[] = "Solve a least-squares problem with regularizers added through TaoAddTerm().\n";

#include <petsctao.h>
#include "taotermtestcommon.h"

typedef struct {
  MPI_Comm    comm;
  PetscInt    n;
  PetscInt    callback_hessian_evals;
  PetscInt    callback_hessianmult_evals;
  PetscInt    term1_hessian_evals;
  PetscInt    term1_hessianmult_evals;
  Vec         target;
  PetscBool   use_term1;
  PetscBool   term1_shell;
  ExampleTerm term1;
  PetscBool   separate_hpre;
  PetscBool   callback_hessian_mult;
  PetscBool   repeat_setfromoptions;
} TestCtx;

typedef struct {
  TestCtx          ctx;
  Tao              tao;
  Vec              x;
  Mat              H;
  Mat              Hpre;
  ExampleCtx       reference_ctx;
  ExampleReference reference;
} TestState;

/* Forward declarations */
static PetscErrorCode TestCtxProcessOptions(MPI_Comm, TestCtx *);
static PetscErrorCode TestCreateData(TestState *);
static PetscErrorCode TestStateDestroy(TestState *);
static PetscErrorCode CheckRepeatedSetFromOptions(TestState *);
static PetscErrorCode CreateTaoTermWithOptions(TestCtx *, ExampleTerm *, const char *, const char *);
static PetscErrorCode FormFunctionGradient_Callback(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_Callback(Tao, Vec, Mat, Mat, void *);
static PetscErrorCode FormHessianMult_Callback(Tao, Vec, Vec, Vec, void *);
static PetscErrorCode Hessian_Term1(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode HessianMult_Term1(TaoTerm, Vec, Vec, Vec, Vec);
static PetscErrorCode CheckReferenceHessianOperators(TestState *);
static PetscErrorCode CheckMaskedSubtermsUntouched(TestState *);
static PetscErrorCode CheckReferenceTermType(TaoTerm);

int main(int argc, char **argv)
{
  TestState state;
  TaoTerm   objective;
  MPI_Comm  comm;
  TestCtx  *ctx = &state.ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscMemzero(&state, sizeof(state)));
  comm = PETSC_COMM_WORLD;

  PetscCall(TestCtxProcessOptions(comm, ctx));
  PetscCall(TaoCreate(comm, &state.tao));
  PetscCall(TaoSetOptionsPrefix(state.tao, "t_"));
  PetscCall(TaoSetType(state.tao, TAOLMVM));

  /* Create a least-squares objective using the traditional TaoSet interface */
  PetscCall(TestCreateData(&state));
  PetscCall(TaoSetSolution(state.tao, state.x));
  PetscCall(TaoSetObjectiveAndGradient(state.tao, NULL, FormFunctionGradient_Callback, ctx));
  PetscCall(TaoSetHessian(state.tao, state.H, state.Hpre, FormHessian_Callback, ctx));
  if (ctx->callback_hessian_mult) PetscCall(TaoSetHessianMult(state.tao, FormHessianMult_Callback, ctx));

  /* Add term 1 if requested */
  if (ctx->use_term1) {
    PetscCall(CreateTaoTermWithOptions(ctx, &ctx->term1, "reg1_", "A1_"));
    PetscCall(TaoAddTerm(state.tao, "reg1_", ctx->term1.scale, ctx->term1.term, ctx->term1.parameters, ctx->term1.map));
  }
  PetscCall(TaoSetFromOptions(state.tao));

  state.reference_ctx.nsubterms              = ctx->use_term1 ? 2 : 1;
  state.reference_ctx.subterms[0].parameters = ctx->target;
  state.reference_ctx.subterms[0].scale      = 1.0;

  if (ctx->use_term1) {
    PetscCall(CheckReferenceTermType(ctx->term1.term));
    PetscCall(ExampleTermSetSubterm(&ctx->term1, &state.reference_ctx.subterms[1]));
  }
  if (state.reference_ctx.nsubterms > 1) {
    PetscCall(TaoGetTerm(state.tao, NULL, &objective, NULL, NULL));
    for (PetscInt i = 0; i < state.reference_ctx.nsubterms; i++) PetscCall(TaoTermSumGetTermMask(objective, i, &state.reference_ctx.subterms[i].mask));
  }
  PetscCall(ExampleReferenceCreate(comm, state.tao, state.x, &state.reference_ctx, &state.reference));
  PetscCall(ExampleReferenceSolveAndCompare(state.tao, state.x, &state.reference));
  PetscCall(CheckReferenceHessianOperators(&state));

  if (ctx->repeat_setfromoptions) PetscCall(CheckRepeatedSetFromOptions(&state));
  PetscCall(CheckMaskedSubtermsUntouched(&state));
  PetscCall(TestStateDestroy(&state));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode TestCtxProcessOptions(MPI_Comm comm, TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(TestCtx)));

  ctx->comm                       = comm;
  ctx->n                          = 10;
  ctx->use_term1                  = PETSC_FALSE;
  ctx->term1_shell                = PETSC_FALSE;
  ctx->term1.size                 = ctx->n - 1;
  ctx->term1.scale                = 0.1;
  ctx->term1.provide_hessian_mult = PETSC_TRUE;
  ctx->separate_hpre              = PETSC_FALSE;
  ctx->callback_hessian_mult      = PETSC_FALSE;
  ctx->repeat_setfromoptions      = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "TaoTerm Coverage Test Options", "TAO");
  PetscCall(PetscOptionsBool("-use_term1", "Use first additional term", "", ctx->use_term1, &ctx->use_term1, NULL));
  PetscCall(PetscOptionsBool("-term1_has_A", "Term 1 has a map matrix A", "", ctx->term1.use_map, &ctx->term1.use_map, NULL));
  PetscCall(PetscOptionsBool("-term1_has_params", "Term 1 has parameters", "", ctx->term1.supply_parameters, &ctx->term1.supply_parameters, NULL));
  PetscCall(PetscOptionsBool("-term1_shell", "Create term 1 as a least-squares shell term", "", ctx->term1_shell, &ctx->term1_shell, NULL));
  PetscCall(PetscOptionsBool("-term1_hessian_mult", "Register HessianMult on shell term 1", "", ctx->term1.provide_hessian_mult, &ctx->term1.provide_hessian_mult, NULL));
  PetscCall(PetscOptionsReal("-term1_scale", "Scaling for term 1", "", ctx->term1.scale, &ctx->term1.scale, NULL));
  PetscCall(PetscOptionsInt("-map_row_size", "Row size of mapping matrix", "", ctx->term1.size, &ctx->term1.size, NULL));
  PetscCall(PetscOptionsBool("-separate_hpre", "Use a separate preconditioning matrix for the legacy callback term", "", ctx->separate_hpre, &ctx->separate_hpre, NULL));
  PetscCall(PetscOptionsBool("-callback_hessian_mult", "Register HessianMult on the legacy callback term", "", ctx->callback_hessian_mult, &ctx->callback_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-repeat_setfromoptions", "Call TaoSetFromOptions() again after TaoSolve()", "", ctx->repeat_setfromoptions, &ctx->repeat_setfromoptions, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCreateData(TestState *state)
{
  TestCtx *ctx = &state->ctx;
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(ctx->comm, PETSC_DECIDE, PETSC_DECIDE, ctx->n, ctx->n, 1, NULL, 0, NULL, &state->H));
  if (ctx->separate_hpre) {
    PetscCall(MatAssemblyBegin(state->H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(state->H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatDuplicate(state->H, MAT_COPY_VALUES, &state->Hpre));
  } else state->Hpre = state->H;
  PetscCall(MatCreateVecs(state->H, &state->x, NULL));
  PetscCall(VecCreateMPI(ctx->comm, PETSC_DECIDE, ctx->n, &ctx->target));
  PetscCall(VecGetOwnershipRange(ctx->target, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(VecSetValue(ctx->target, i, 1.0 + 0.05 * (i + 1), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(ctx->target));
  PetscCall(VecAssemblyEnd(ctx->target));
  PetscCall(VecZeroEntries(state->x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestStateDestroy(TestState *state)
{
  PetscFunctionBeginUser;
  PetscCall(ExampleReferenceDestroy(&state->reference));
  PetscCall(ExampleTermDestroy(&state->ctx.term1));
  PetscCall(TaoDestroy(&state->tao));
  PetscCall(VecDestroy(&state->x));
  if (state->Hpre != state->H) PetscCall(MatDestroy(&state->Hpre));
  PetscCall(MatDestroy(&state->H));
  PetscCall(VecDestroy(&state->ctx.target));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckRepeatedSetFromOptions(TestState *state)
{
  TaoTerm     objective;
  TaoTermMask mask_before = TAOTERM_MASK_NONE, mask_after;
  PetscInt    max_it;

  PetscFunctionBeginUser;
  if (state->ctx.use_term1) {
    PetscCall(TaoGetTerm(state->tao, NULL, &objective, NULL, NULL));
    PetscCall(TaoTermSumGetTermMask(objective, 1, &mask_before));
    PetscCall(PetscOptionsSetValue(NULL, "-t_tao_term_sum_reg1_mask", "none"));
  }
  PetscCall(PetscOptionsSetValue(NULL, "-t_tao_max_it", "7"));
  PetscCall(TaoSetFromOptions(state->tao));
  PetscCall(TaoGetMaximumIterations(state->tao, &max_it));
  PetscCheck(max_it == 7, state->ctx.comm, PETSC_ERR_PLIB, "Repeated TaoSetFromOptions() did not update -t_tao_max_it");
  if (state->ctx.use_term1) {
    PetscCall(TaoTermSumGetTermMask(objective, 1, &mask_after));
    PetscCheck(mask_after == mask_before, state->ctx.comm, PETSC_ERR_PLIB, "Repeated TaoSetFromOptions() changed structural TaoTerm options after setup");
  }
  PetscCall(PetscPrintf(state->ctx.comm, "Repeated TaoSetFromOptions() check passed\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTaoTermWithOptions(TestCtx *ctx, ExampleTerm *term, const char *term_prefix, const char *A_prefix)
{
  MPI_Comm    comm = ctx->comm;
  PetscInt    rstart, rend;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Create parameters if requested */
  if (term->supply_parameters) {
    PetscCall(VecCreate(comm, &term->parameters));
    PetscCall(VecSetSizes(term->parameters, PETSC_DECIDE, term->use_map ? term->size : ctx->n));
    PetscCall(VecSetFromOptions(term->parameters));
    PetscCall(VecGetOwnershipRange(term->parameters, &rstart, &rend));
    for (PetscInt i = rstart; i < rend; i++) PetscCall(VecSetValue(term->parameters, i, 0.5 + 0.1 * (i + 1), INSERT_VALUES));
    PetscCall(VecAssemblyBegin(term->parameters));
    PetscCall(VecAssemblyEnd(term->parameters));
  }

  /* Create map matrix A if requested */
  if (term->use_map) {
    PetscCall(MatCreate(comm, &term->map));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term->map, A_prefix));
    PetscCall(MatSetSizes(term->map, PETSC_DECIDE, PETSC_DECIDE, term->size, ctx->n));
    PetscCall(MatSetType(term->map, MATAIJ)); /* Set default type before SetFromOptions */
    PetscCall(MatSetFromOptions(term->map));
    /* Check matrix type and set up accordingly */
    if (size == 1) PetscCall(MatSeqAIJSetPreallocation(term->map, PETSC_DEFAULT, NULL));
    else PetscCall(MatMPIAIJSetPreallocation(term->map, 5, NULL, 5, NULL));
    PetscCall(MatSetUp(term->map));
    PetscCall(MatGetOwnershipRange(term->map, &rstart, &rend));
    for (PetscInt i = rstart; i < rend; i++) {
      PetscInt          cols[4];
      const PetscScalar vals[] = {2.0, -0.3, 0.2, -0.1};

      for (PetscInt k = 0; k < 4; k++) cols[k] = (i + k) % ctx->n;
      PetscCall(MatSetValues(term->map, 1, &i, 4, cols, vals, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(term->map, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(term->map, MAT_FINAL_ASSEMBLY));
  }
  /* Create TaoTerm, set prefix, and configure from options */
  PetscCall(TaoTermCreate(comm, &term->term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term->term, term_prefix));
  PetscCall(TaoTermSetSolutionSizes(term->term, PETSC_DECIDE, term->use_map ? term->size : ctx->n, 1));
  if (ctx->term1_shell) {
    PetscCall(TaoTermSetType(term->term, TAOTERMSHELL));
    PetscCall(TaoTermShellSetContext(term->term, ctx));
    PetscCall(TaoTermSetParametersSizes(term->term, PETSC_DECIDE, term->use_map ? term->size : ctx->n, 1));
    PetscCall(TaoTermShellSetObjectiveAndGradient(term->term, ExampleIdentityLeastSquaresObjectiveGradient));
    PetscCall(TaoTermShellSetCreateHessianMatrices(term->term, TaoTermCreateHessianMatricesDefault));
    PetscCall(TaoTermSetCreateHessianMode(term->term, PETSC_TRUE, MATAIJ, NULL));
    PetscCall(TaoTermShellSetHessian(term->term, Hessian_Term1));
    if (term->provide_hessian_mult) PetscCall(TaoTermShellSetHessianMult(term->term, HessianMult_Term1));
  }
  PetscCall(TaoTermSetFromOptions(term->term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunctionGradient_Callback(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx    *ctx = (TestCtx *)ptr;
  Vec         diff;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(X, &diff));
  PetscCall(VecWAXPY(diff, -1.0, ctx->target, X));
  PetscCall(VecDot(diff, diff, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscCall(VecCopy(diff, G));
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian_Callback(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  ctx->callback_hessian_evals++;
  if (ctx->separate_hpre) PetscCheck(!Hpre || Hpre != H, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Distinct callback Hessian storage was aliased during evaluation");
  PetscCall(MatZeroEntries(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, 1.0));
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMult_Callback(Tao tao, Vec X, Vec V, Vec HV, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  ctx->callback_hessianmult_evals++;
  PetscCall(VecCopy(V, HV));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Counting wrappers used to assert that a Hessian-masked shell term does not invoke
   its Hessian callbacks. */
static PetscErrorCode Hessian_Term1(TaoTerm term, Vec x, Vec parameters, Mat H, Mat Hpre)
{
  TestCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->term1_hessian_evals++;
  PetscCall(ExampleIdentityLeastSquaresHessian(term, x, parameters, H, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HessianMult_Term1(TaoTerm term, Vec x, Vec parameters, Vec v, Vec Hv)
{
  TestCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->term1_hessianmult_evals++;
  PetscCall(ExampleIdentityLeastSquaresHessianMult(term, x, parameters, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckReferenceHessianOperators(TestState *state)
{
  TestCtx  *ctx = &state->ctx;
  Mat       cH, cHpre, tH, tHpre;
  PetscBool equal, c_is_nls, t_is_nls;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)state->tao, TAONLS, &t_is_nls));
  PetscCall(PetscObjectTypeCompare((PetscObject)state->reference.tao, TAONLS, &c_is_nls));
  if (!t_is_nls || !c_is_nls) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoGetHessianMatrices(state->reference.tao, &cH, &cHpre));
  PetscCall(TaoGetHessianMatrices(state->tao, &tH, &tHpre));
  PetscCall(TaoComputeHessian(state->reference.tao, state->x, cH, cHpre));
  PetscCall(TaoComputeHessian(state->tao, state->x, tH, tHpre));
  PetscCall(MatMultEqual(cH, tH, 5, &equal));
  PetscCheck(equal, ctx->comm, PETSC_ERR_PLIB, "Reference and TaoTerm Hessians differ");
  PetscCall(MatMultEqual(cHpre, tHpre, 5, &equal));
  PetscCheck(equal, ctx->comm, PETSC_ERR_PLIB, "Reference and TaoTerm Hessian preconditioners differ");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMaskedSubtermsUntouched(TestState *state)
{
  TestCtx *ctx = &state->ctx;

  PetscFunctionBeginUser;
  if (!ctx->use_term1) PetscFunctionReturn(PETSC_SUCCESS);
  if (state->reference_ctx.subterms[0].mask & TAOTERM_MASK_HESSIAN)
    PetscCheck(ctx->callback_hessian_evals == 0 && ctx->callback_hessianmult_evals == 0, ctx->comm, PETSC_ERR_PLIB, "Hessian-masked callback term was evaluated (%" PetscInt_FMT " Hessian, %" PetscInt_FMT " HessianMult calls)", ctx->callback_hessian_evals,
               ctx->callback_hessianmult_evals);
  if (ctx->term1_shell && (state->reference_ctx.subterms[1].mask & TAOTERM_MASK_HESSIAN))
    PetscCheck(ctx->term1_hessian_evals == 0 && ctx->term1_hessianmult_evals == 0, ctx->comm, PETSC_ERR_PLIB, "Hessian-masked term 1 was evaluated (%" PetscInt_FMT " Hessian, %" PetscInt_FMT " HessianMult calls)", ctx->term1_hessian_evals, ctx->term1_hessianmult_evals);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckReferenceTermType(TaoTerm term)
{
  TaoTermType type;
  PetscBool   supported;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)term, &supported, TAOTERMHALFL2SQUARED, TAOTERMSHELL, ""));
  PetscCall(TaoTermGetType(term, &type));
  PetscCheck(supported, PetscObjectComm((PetscObject)term), PETSC_ERR_SUP, "Reference supports only half-L2 and shell least-squares added terms, not %s", type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: shell
    args: -t_tao_type nls -use_term1 -term1_shell -t_tao_term_hessian_mat_type shell

  test:
    suffix: shell_mapped
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -t_tao_term_hessian_mat_type shell

  test:
    suffix: shell_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_hessian_mat_type shell -t_tao_term_hessian_pre_is_hessian false
    args: -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: shell_mapped_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A
    args: -t_tao_term_hessian_mat_type shell -t_tao_term_hessian_pre_is_hessian false
    args: -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: masked_hessian
    args: -t_tao_type nls -use_term1 -reg1_tao_term_type halfl2squared -t_tao_term_sum_reg1_mask hessian

  test:
    suffix: promoted_separate_hpre
    args: -t_tao_type nls -separate_hpre -use_term1 -reg1_tao_term_type halfl2squared
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: promoted_separate_hpre_shell
    args: -t_tao_type nls -separate_hpre -use_term1 -reg1_tao_term_type halfl2squared -t_tao_term_hessian_mat_type shell

  test:
    suffix: halfl2_parameters
    args: -t_tao_type nls -use_term1 -term1_has_params -reg1_tao_term_type halfl2squared

  test:
    suffix: halfl2_mapped_parameters
    args: -t_tao_type nls -use_term1 -term1_has_params -term1_has_A -reg1_tao_term_type halfl2squared

  test:
    suffix: repeat_setfromoptions
    args: -repeat_setfromoptions -t_tao_fd_gradient

  test:
    suffix: repeat_setfromoptions_sum
    args: -t_tao_type nls -use_term1 -repeat_setfromoptions
    args: -reg1_tao_term_type halfl2squared -t_tao_term_sum_reg1_mask hessian

  test:
    suffix: promoted_shell_fd_direct
    args: -t_tao_type nls -use_term1 -term1_shell -t_callbacks_tao_term_hessian_use_fd
    args: -term1_hessian_mult -t_tao_term_hessian_mat_type shell

  test:
    suffix: promoted_shell_all_direct
    args: -t_tao_type nls -use_term1 -term1_shell -callback_hessian_mult
    args: -term1_hessian_mult -t_tao_term_hessian_mat_type shell

  test:
    suffix: promoted_shell_mapped_fallback
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A
    args: -term1_hessian_mult false -t_tao_term_sum_t_callbacks_mask objective,gradient,hessian
    args: -t_tao_term_hessian_mat_type shell

  test:
    suffix: callback_fd
    args: -t_tao_type nls -t_tao_fd_hessian

  test:
    suffix: callback_fd_separate_hpre
    args: -t_tao_type nls -t_tao_fd_hessian -separate_hpre

  test:
    suffix: r021_added_analytic
    args: -t_tao_type nls -use_term1 -term1_shell -t_tao_term_sum_t_callbacks_mask hessian

  test:
    suffix: r022_added_fd
    args: -t_tao_type nls -use_term1 -term1_shell -reg1_tao_term_hessian_use_fd true -t_tao_term_sum_t_callbacks_mask hessian

  test:
    suffix: r025_callback_analytic
    args: -t_tao_type nls -use_term1 -term1_shell -t_tao_term_sum_reg1_mask hessian

  test:
    suffix: r026_callback_fd
    args: -t_tao_type nls -use_term1 -term1_shell -t_callbacks_tao_term_hessian_use_fd true -t_tao_term_sum_reg1_mask hessian

  test:
    suffix: r028_analytic_fd
    args: -t_tao_type nls -use_term1 -term1_shell -reg1_tao_term_hessian_use_fd true

  test:
    suffix: r029_fd_fd
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true

  test:
    suffix: r033_added_analytic_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -t_tao_term_sum_t_callbacks_mask hessian
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r034_added_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_sum_t_callbacks_mask hessian -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r037_callback_analytic_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre -t_tao_term_sum_reg1_mask hessian
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r038_callback_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre
    args: -t_tao_term_sum_reg1_mask hessian -t_callbacks_tao_term_hessian_use_fd true
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r040_analytic_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r041_fd_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r023_mapped_added_analytic
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -t_tao_term_sum_t_callbacks_mask hessian

  test:
    suffix: r024_mapped_added_fd
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 11 -term1_scale 100
    args: -reg1_tao_term_hessian_use_fd true -t_tao_term_sum_t_callbacks_mask hessian

  test:
    suffix: r031_analytic_mapped_fd
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -reg1_tao_term_hessian_use_fd true

  test:
    suffix: r032_fd_mapped_fd
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true

  test:
    suffix: r035_mapped_added_analytic_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_tao_term_sum_t_callbacks_mask hessian
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r036_mapped_added_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_tao_term_sum_t_callbacks_mask hessian -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r042_mapped_analytic_analytic_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -separate_hpre
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r043_analytic_mapped_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -separate_hpre
    args: -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  test:
    suffix: r044_fd_mapped_fd_separate_hpre
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -separate_hpre
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

  testset:
    args: -t_tao_type nls -use_term1 -term1_shell -t_tao_term_hessian_mat_type shell

    test:
      suffix: r045_added_analytic
      args: -term1_hessian_mult false -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r046_added_fd
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r047_added_hessianmult
      args: -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r049_mapped_added_fd
      args: -term1_has_A -map_row_size 11 -term1_scale 100 -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r050_mapped_added_hessianmult
      args: -term1_has_A -map_row_size 11 -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r051_callback_analytic
      args: -t_tao_term_sum_reg1_mask hessian

    test:
      suffix: r052_callback_fd
      args: -t_callbacks_tao_term_hessian_use_fd true -t_tao_term_sum_reg1_mask hessian

    test:
      suffix: r053_callback_hessianmult
      args: -callback_hessian_mult -t_tao_term_sum_reg1_mask hessian

    test:
      suffix: r054_analytic_analytic
      args: -term1_hessian_mult false

    test:
      suffix: r055_analytic_fd
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r057_fd_fd
      args: -t_callbacks_tao_term_hessian_use_fd true -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r060_mapped_analytic_analytic
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false

    test:
      suffix: r061_mapped_analytic_fd
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r063_mapped_fd_fd
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r064_mapped_fd_hessianmult
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true

    test:
      suffix: r065_mapped_hessianmult_hessianmult
      args: -term1_has_A -map_row_size 11 -callback_hessian_mult

    test:
      suffix: r066_added_analytic_separate_hpre
      args: -term1_hessian_mult false -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r067_added_fd_separate_hpre
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r068_added_hessianmult_separate_hpre
      args: -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r069_mapped_added_analytic_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r070_mapped_added_fd_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false -term1_scale 100
      args: -reg1_tao_term_hessian_use_fd true -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r071_mapped_added_hessianmult_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r072_callback_analytic_separate_hpre
      args: -t_tao_term_sum_reg1_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r073_callback_fd_separate_hpre
      args: -t_callbacks_tao_term_hessian_use_fd true -t_tao_term_sum_reg1_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r074_callback_hessianmult_separate_hpre
      args: -callback_hessian_mult -t_tao_term_sum_reg1_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r075_analytic_analytic_separate_hpre
      args: -term1_hessian_mult false
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r076_analytic_fd_separate_hpre
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r078_fd_fd_separate_hpre
      args: -t_callbacks_tao_term_hessian_use_fd true -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r079_fd_hessianmult_separate_hpre
      args: -t_callbacks_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r080_hessianmult_hessianmult_separate_hpre
      args: -callback_hessian_mult
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r081_mapped_analytic_analytic_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r082_mapped_analytic_fd_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r084_mapped_fd_fd_separate_hpre
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r085_mapped_fd_hessianmult_separate_hpre
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r086_mapped_hessianmult_hessianmult_separate_hpre
      args: -term1_has_A -map_row_size 11 -callback_hessian_mult
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

TEST*/
