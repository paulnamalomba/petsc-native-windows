#include <petsctao.h>
#include "taotermtestcommon.h"

static char help[] = "Solve one or two linear least-squares data terms through TaoAddTerm().\n";

typedef struct {
  PetscBool separate_callbacks;
  PetscBool split_hpre;
  PetscInt *hessian_evals;
  PetscInt *hessianmult_evals;
} TermCtx;

typedef struct {
  ExampleTerm           terms[2];
  PetscInt              nterms;
  PetscBool             separate_callbacks;
  PetscBool             split_hpre;
  PetscBool             check_hessian_mult;
  PetscBool             check_hessian_cache;
  TaoTermParametersMode parameters_mode;
} TestOptions;

static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode FormHessianMult(TaoTerm, Vec, Vec, Vec, Vec);
static PetscErrorCode TestOptionsSetFromOptions(MPI_Comm, TestOptions *);
static PetscErrorCode TestCreateData(MPI_Comm, PetscInt, const TestOptions *, TermCtx[], PetscInt[], PetscInt[], Vec *);
static PetscErrorCode TestAddTerms(MPI_Comm, PetscInt, TestOptions *, TermCtx[], Tao);
static PetscErrorCode TestSetReferenceContext(Tao, TestOptions *, ExampleCtx *);
static PetscErrorCode TestDestroy(TestOptions *, ExampleReference *, Tao *, Vec *);
static PetscErrorCode CreateMap(MPI_Comm, PetscInt, PetscInt, PetscReal, Mat *);
static PetscErrorCode CreateDataTerm(MPI_Comm, const char[], TermCtx *, ExampleTerm *);
static PetscErrorCode AddScaledATAx(Vec, Vec, Mat, PetscReal, Vec);
static PetscErrorCode CheckHessianAction(Tao, Vec, Mat, Mat, Vec, Vec, Vec, PetscReal);
static PetscErrorCode CheckHessianMult(Tao, Vec, Vec, Vec, PetscReal);
static PetscErrorCode CheckHessianCache(Tao, Vec, Mat, Mat, Vec, PetscInt, PetscInt, const PetscInt *);
static PetscErrorCode CheckOperator(Tao, Vec, ExampleTerm *, PetscInt, PetscBool, PetscBool, PetscBool, const PetscInt *);

int main(int argc, char **argv)
{
  const PetscInt   n = 10;
  Tao              tao;
  TermCtx          ctx[2];
  ExampleCtx       reference_ctx;
  ExampleReference reference;
  TestOptions      options;
  Vec              x;
  PetscInt         hessian_evals[2] = {0, 0}, hessianmult_evals[2] = {0, 0};
  MPI_Comm         comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscMemzero(&reference_ctx, sizeof(reference_ctx)));
  comm = PETSC_COMM_WORLD;

  PetscCall(TestOptionsSetFromOptions(comm, &options));
  PetscCall(TestCreateData(comm, n, &options, ctx, hessian_evals, hessianmult_evals, &x));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao, "shell_"));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TestAddTerms(comm, n, &options, ctx, tao));
  PetscCall(TaoSetFromOptions(tao));

  PetscCall(TestSetReferenceContext(tao, &options, &reference_ctx));
  PetscCall(ExampleReferenceCreate(comm, tao, x, &reference_ctx, &reference));
  PetscCall(ExampleReferenceSolveAndCompare(tao, x, &reference));
  PetscCall(TaoTermComputeHessianGetUseFD(options.terms[0].term, &options.terms[0].use_fd));
  if (options.nterms > 1) PetscCall(TaoTermComputeHessianGetUseFD(options.terms[1].term, &options.terms[1].use_fd));
  PetscCall(CheckOperator(tao, x, options.terms, options.nterms, options.split_hpre, options.check_hessian_mult, options.check_hessian_cache, hessian_evals));
  /* A Hessian-masked summand must not invoke its Hessian or HessianMult callback, even
     if an accidental evaluation does not affect the result. */
  for (PetscInt i = 0; i < options.nterms; i++)
    if (options.terms[i].mask & TAOTERM_MASK_HESSIAN)
      PetscCheck(hessian_evals[i] == 0 && hessianmult_evals[i] == 0, comm, PETSC_ERR_PLIB, "Hessian-masked term %" PetscInt_FMT " was evaluated (%" PetscInt_FMT " Hessian, %" PetscInt_FMT " HessianMult calls)", i, hessian_evals[i], hessianmult_evals[i]);
  PetscCall(PetscPrintf(comm, "Least-squares TaoTerm operator check passed\n"));

  PetscCall(TestDestroy(&options, &reference, &tao, &x));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode TestCreateData(MPI_Comm comm, PetscInt n, const TestOptions *options, TermCtx ctx[], PetscInt hessian_evals[], PetscInt hessianmult_evals[], Vec *x)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < options->nterms; i++) {
    ctx[i].separate_callbacks = options->separate_callbacks;
    ctx[i].split_hpre         = options->split_hpre;
    ctx[i].hessian_evals      = &hessian_evals[i];
    ctx[i].hessianmult_evals  = &hessianmult_evals[i];
  }
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, n, x));
  PetscCall(VecSet(*x, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestAddTerms(MPI_Comm comm, PetscInt n, TestOptions *options, TermCtx ctx[], Tao tao)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < options->nterms; i++) {
    ExampleTerm *term = &options->terms[i];
    char         prefix[16];

    PetscCall(PetscStrncpy(prefix, i ? "extra_" : "data_", sizeof(prefix)));
    if (term->use_map) PetscCall(CreateMap(comm, term->size, n, i ? 0.75 : 1.0, &term->map));
    PetscCall(CreateDataTerm(comm, prefix, &ctx[i], term));
    PetscCall(TaoTermSetParametersMode(term->term, options->parameters_mode));
    if (term->supply_parameters) {
      PetscCall(VecCreateMPI(comm, PETSC_DECIDE, term->size, &term->parameters));
      PetscCall(VecSet(term->parameters, 1.0 + i));
    }
    PetscCall(TaoAddTerm(tao, prefix, term->scale, term->term, term->parameters, term->map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSetReferenceContext(Tao tao, TestOptions *options, ExampleCtx *reference_ctx)
{
  TaoTerm objective;

  PetscFunctionBeginUser;
  reference_ctx->nsubterms = options->nterms;
  if (options->nterms > 1) PetscCall(TaoGetTerm(tao, NULL, &objective, NULL, NULL));
  for (PetscInt i = 0; i < options->nterms; i++) {
    if (options->nterms > 1) PetscCall(TaoTermSumGetTermMask(objective, i, &options->terms[i].mask));
    PetscCall(ExampleTermSetSubterm(&options->terms[i], &reference_ctx->subterms[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDestroy(TestOptions *options, ExampleReference *reference, Tao *tao, Vec *x)
{
  PetscFunctionBeginUser;
  PetscCall(ExampleReferenceDestroy(reference));
  for (PetscInt i = 0; i < options->nterms; i++) PetscCall(ExampleTermDestroy(&options->terms[i]));
  PetscCall(TaoDestroy(tao));
  PetscCall(VecDestroy(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  (*ctx->hessian_evals)++;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(H, 1.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, ctx->split_hpre ? 2.0 : 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  (*ctx->hessianmult_evals)++;
  PetscCall(ExampleIdentityLeastSquaresHessianMult(term, x, params, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestOptionsSetFromOptions(MPI_Comm comm, TestOptions *options)
{
  ExampleTerm *data, *extra;
  PetscBool    second_term = PETSC_FALSE, no_map = PETSC_FALSE, provide_hessian_mult = PETSC_TRUE;
  PetscInt     m = 10;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(options, sizeof(*options)));
  data                        = &options->terms[0];
  extra                       = &options->terms[1];
  data->use_map               = PETSC_TRUE;
  extra->use_map              = PETSC_TRUE;
  data->provide_hessian_mult  = PETSC_TRUE;
  extra->provide_hessian_mult = PETSC_TRUE;
  data->supply_parameters     = PETSC_TRUE;
  extra->supply_parameters    = PETSC_TRUE;
  data->scale                 = 1.0;
  extra->scale                = 0.25;
  data->size                  = 10;
  extra->size                 = 12;
  options->parameters_mode    = TAOTERM_PARAMETERS_OPTIONAL;
  PetscOptionsBegin(comm, "", help, "Tao");
  PetscCall(PetscOptionsBool("-second_term", "Add a second least-squares data term", NULL, second_term, &second_term, NULL));
  PetscCall(PetscOptionsBool("-no_map", "Add terms directly in the Tao solution space", NULL, no_map, &no_map, NULL));
  data->use_map  = !no_map;
  extra->use_map = !no_map;
  PetscCall(PetscOptionsBool("-data_use_map", "Map the first data term", NULL, data->use_map, &data->use_map, NULL));
  PetscCall(PetscOptionsBool("-extra_use_map", "Map the second data term", NULL, extra->use_map, &extra->use_map, NULL));
  PetscCall(PetscOptionsBool("-separate_callbacks", "Register separate objective and gradient callbacks", NULL, options->separate_callbacks, &options->separate_callbacks, NULL));
  PetscCall(PetscOptionsBool("-provide_hessian_mult", "Register the direct Hessian-vector callback on both terms by default", NULL, provide_hessian_mult, &provide_hessian_mult, NULL));
  data->provide_hessian_mult  = provide_hessian_mult;
  extra->provide_hessian_mult = provide_hessian_mult;
  PetscCall(PetscOptionsBool("-data_provide_hessian_mult", "Register HessianMult on the first data term", NULL, data->provide_hessian_mult, &data->provide_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-extra_provide_hessian_mult", "Register HessianMult on the second data term", NULL, extra->provide_hessian_mult, &extra->provide_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-split_hpre", "Use a distinct raw preconditioning matrix equal to two times the Hessian", NULL, options->split_hpre, &options->split_hpre, NULL));
  PetscCall(PetscOptionsBool("-check_hessian_mult", "Compare TaoComputeHessianMult() with the exact least-squares action", NULL, options->check_hessian_mult, &options->check_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-check_hessian_cache", "Check that assembled Hessian and HessianMult evaluations share cached raw Hessians", NULL, options->check_hessian_cache, &options->check_hessian_cache, NULL));
  PetscCall(PetscOptionsEnum("-parameters", "Parameter contract for all terms", NULL, TaoTermParametersModes, (PetscEnum)options->parameters_mode, (PetscEnum *)&options->parameters_mode, NULL));
  PetscCall(PetscOptionsInt("-m", "Number of observations in the first data set", NULL, m, &m, NULL));
  PetscOptionsEnd();

  options->nterms          = second_term ? 2 : 1;
  data->size               = data->use_map ? m : 10;
  extra->size              = extra->use_map ? 12 : 10;
  data->supply_parameters  = options->parameters_mode != TAOTERM_PARAMETERS_NONE;
  extra->supply_parameters = data->supply_parameters;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMap(MPI_Comm comm, PetscInt m, PetscInt n, PetscReal shift, Mat *A)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, 2, NULL, 2, NULL, A));
  PetscCall(MatGetOwnershipRange(*A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    const PetscInt j = i % n;

    PetscCall(MatSetValue(*A, i, j, shift + 0.05 * (i + 1), INSERT_VALUES));
    if (n > 1) PetscCall(MatSetValue(*A, i, (j + 1) % n, 0.1, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDataTerm(MPI_Comm comm, const char prefix[], TermCtx *ctx, ExampleTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, ctx, NULL, &term->term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term->term, prefix));
  PetscCall(TaoTermSetSolutionSizes(term->term, PETSC_DECIDE, term->size, 1));
  PetscCall(TaoTermSetParametersSizes(term->term, PETSC_DECIDE, term->size, 1));
  if (ctx->separate_callbacks) {
    PetscCall(TaoTermShellSetObjective(term->term, ExampleIdentityLeastSquaresObjective));
    PetscCall(TaoTermShellSetGradient(term->term, ExampleIdentityLeastSquaresGradient));
  } else PetscCall(TaoTermShellSetObjectiveAndGradient(term->term, ExampleIdentityLeastSquaresObjectiveGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term->term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term->term, ctx->split_hpre ? PETSC_FALSE : PETSC_TRUE, MATAIJ, ctx->split_hpre ? MATAIJ : NULL));
  PetscCall(TaoTermShellSetHessian(term->term, FormHessian));
  if (term->provide_hessian_mult) PetscCall(TaoTermShellSetHessianMult(term->term, FormHessianMult));
  PetscCall(TaoTermSetFromOptions(term->term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add scale * A^T A * source to target, treating a NULL map as the identity. */
static PetscErrorCode AddScaledATAx(Vec target, Vec source, Mat A, PetscReal scale, Vec work)
{
  PetscFunctionBeginUser;
  if (A) {
    Vec Asource;

    PetscCall(MatCreateVecs(A, NULL, &Asource));
    PetscCall(MatMult(A, source, Asource));
    PetscCall(MatMultTranspose(A, Asource, work));
    PetscCall(VecDestroy(&Asource));
  } else PetscCall(VecCopy(source, work));
  PetscCall(VecAXPY(target, scale, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check the assembled Hessian and preconditioning matrix actions. */
static PetscErrorCode CheckHessianAction(Tao tao, Vec x, Mat H, Mat Hpre, Vec v, Vec expected, Vec expected_pre, PetscReal tolerance)
{
  Vec       actual;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &actual));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(MatMult(H, v, actual));
  PetscCall(VecAXPY(actual, -1.0, expected));
  PetscCall(VecNorm(actual, NORM_2, &error));
  PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Tao Hessian action differs from the exact least-squares action by %g", (double)error);

  if (Hpre != H) {
    PetscCall(MatMult(Hpre, v, actual));
    PetscCall(VecAXPY(actual, -1.0, expected_pre));
    PetscCall(VecNorm(actual, NORM_2, &error));
    PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Tao preconditioning action differs from its exact mapped action by %g", (double)error);
  }
  PetscCall(VecDestroy(&actual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check the direct Hessian-vector product against the exact action. */
static PetscErrorCode CheckHessianMult(Tao tao, Vec x, Vec v, Vec expected, PetscReal tolerance)
{
  Vec       actual;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &actual));
  PetscCall(TaoComputeHessianMult(tao, x, v, actual));
  PetscCall(VecAXPY(actual, -1.0, expected));
  PetscCall(VecNorm(actual, NORM_2, &error));
  PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "TaoComputeHessianMult() differs from the exact sum action by %g", (double)error);
  PetscCall(VecDestroy(&actual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check that assembled Hessians and Hessian-vector products share cached subterm Hessians. */
static PetscErrorCode CheckHessianCache(Tao tao, Vec x, Mat H, Mat Hpre, Vec v, PetscInt nterms, PetscInt active_hessian_terms, const PetscInt *hessian_evals)
{
  Mat       H2;
  Vec       actual;
  PetscInt  evals_before = 0, evals_after = 0;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &actual));
  PetscCall(VecShift(x, 0.125));
  for (PetscInt i = 0; i < nterms; i++) evals_before += hessian_evals[i];
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  for (PetscInt i = 0; i < nterms; i++) evals_after += hessian_evals[i];
  PetscCheck(evals_after == evals_before + active_hessian_terms, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "First Hessian evaluation at a new point called %" PetscInt_FMT " callbacks, expected %" PetscInt_FMT, evals_after - evals_before, active_hessian_terms);
  PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &H2));
  PetscCall(MatAssemblyBegin(H2, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H2, MAT_FINAL_ASSEMBLY));
  evals_before = evals_after;
  PetscCall(TaoComputeHessian(tao, x, H2, H2));
  PetscCall(TaoComputeHessianMult(tao, x, v, actual));
  evals_after = 0;
  for (PetscInt i = 0; i < nterms; i++) evals_after += hessian_evals[i];
  PetscCheck(evals_after == evals_before, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Repeated Hessian and HessianMult evaluations called %" PetscInt_FMT " redundant Hessian callbacks", evals_after - evals_before);
  PetscCall(MatMultEqual(H, H2, 5, &equal));
  PetscCheck(equal, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Hessian assembled into a second destination differs from the original operator");
  PetscCall(VecShift(x, -0.125));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Shared Hessian cache check passed\n"));
  PetscCall(MatDestroy(&H2));
  PetscCall(VecDestroy(&actual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckOperator(Tao tao, Vec x, ExampleTerm terms[], PetscInt nterms, PetscBool split_hpre, PetscBool check_hessian_mult, PetscBool check_hessian_cache, const PetscInt *hessian_evals)
{
  const PetscReal tolerance = 2.e-5;
  Mat             H, Hpre;
  Vec             v, expected, expected_pre, work;
  PetscInt        active_hessian_terms = 0;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &expected));
  PetscCall(VecDuplicate(x, &expected_pre));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(v, 1.0));
  PetscCall(VecZeroEntries(expected));
  PetscCall(VecZeroEntries(expected_pre));
  for (PetscInt i = 0; i < nterms; i++) {
    if (terms[i].mask & TAOTERM_MASK_HESSIAN) continue;
    PetscCall(AddScaledATAx(expected, v, terms[i].map, terms[i].scale, work));
    PetscCall(AddScaledATAx(expected_pre, v, terms[i].map, split_hpre && !terms[i].use_fd ? 2.0 * terms[i].scale : terms[i].scale, work));
    active_hessian_terms++;
  }
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(CheckHessianAction(tao, x, H, Hpre, v, expected, expected_pre, tolerance));
  if (check_hessian_mult) PetscCall(CheckHessianMult(tao, x, v, expected, tolerance));
  if (check_hessian_cache) PetscCall(CheckHessianCache(tao, x, H, Hpre, v, nterms, active_hessian_terms, hessian_evals));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&expected_pre));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  testset:

    test:
      suffix: assembled
      args: -shell_tao_type nls

    test:
      suffix: rectangular_map
      args: -m 15 -shell_tao_type nls

    test:
      suffix: separate_callbacks
      args: -separate_callbacks -shell_tao_type nls

    test:
      suffix: mapped_separate_hpre
      args: -split_hpre -shell_tao_type nls

    test:
      suffix: mapped_fd
      args: -data_tao_term_hessian_use_fd -shell_tao_type nls

    test:
      suffix: mapped_fd_separate_hpre
      args: -split_hpre -data_tao_term_hessian_use_fd -shell_tao_type nls

    test:
      suffix: two_mapped_assembled
      args: -second_term -shell_tao_type nls

    test:
      suffix: fallback_unmapped
      args: -second_term -no_map -provide_hessian_mult false -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell

    test:
      suffix: fallback_mapped
      args: -second_term -provide_hessian_mult false -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell

    test:
      suffix: fallback_separate_hpre
      args: -second_term -no_map -provide_hessian_mult false -split_hpre -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_term_hessian_pre_is_hessian false
      args: -shell_tao_term_hessian_pre_mat_type aij

    test:
      suffix: fallback_mapped_separate_hpre
      args: -second_term -provide_hessian_mult false -split_hpre -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_term_hessian_pre_is_hessian false
      args: -shell_tao_term_hessian_pre_mat_type aij

    test:
      suffix: sum_hessian_mult
      args: -second_term -check_hessian_mult -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell

    test:
      suffix: shared_hessian_cache
      args: -second_term -provide_hessian_mult false -check_hessian_cache -shell_tao_type nls

    test:
      suffix: parameters_none
      args: -second_term -parameters none -shell_tao_type nls

    test:
      suffix: parameters_required
      args: -second_term -parameters required -shell_tao_type nls

    test:
      suffix: hessian_only_model
      args: -second_term -shell_tao_term_sum_extra_mask objective,gradient
      args: -shell_tao_type nls

    test:
      suffix: shell_masked_hessian
      args: -second_term -provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian -shell_tao_term_hessian_mat_type shell
      args: -shell_tao_type nls

  test:
    suffix: asymmetric_map_assembled
    args: -second_term -data_use_map true -extra_use_map false -shell_tao_type nls

  test:
    suffix: mixed_direct_fallback
    args: -second_term -data_use_map false -extra_use_map true
    args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
    args: -shell_tao_type nls -shell_tao_term_hessian_mat_type shell

  test:
    suffix: sum_fd_mapped
    args: -second_term -shell_tao_type nls -shell_tao_fd_hessian

  test:
    suffix: sum_fd_mapped_separate_callbacks
    args: -second_term -shell_tao_type nls -shell_tao_fd_hessian
    args: -separate_callbacks

  testset:
    args: -second_term -shell_tao_type nls

    test:
      suffix: r092_second_analytic
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r093_second_fd
      args: -no_map -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r094_mapped_second_analytic
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r095_mapped_second_fd
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r096_first_analytic
      args: -no_map -shell_tao_term_sum_extra_mask hessian
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r097_first_fd
      args: -no_map -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r098_mapped_first_analytic
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r099_mapped_first_fd
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r100_analytic_analytic
      args: -no_map

    test:
      suffix: r101_analytic_fd
      args: -no_map -extra_tao_term_hessian_use_fd

    test:
      suffix: r102_fd_fd
      args: -no_map -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r104_mapped_analytic_fd
      args: -data_use_map true -extra_use_map false -extra_tao_term_hessian_use_fd

    test:
      suffix: r105_mapped_fd_fd
      args: -data_use_map true -extra_use_map false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r107_both_mapped_analytic_fd
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r108_both_mapped_fd_fd
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

  testset:
    args: -second_term -split_hpre -shell_tao_type nls
    args: -shell_tao_term_hessian_pre_is_hessian false -shell_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r109_second_analytic_separate_hpre
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r110_second_fd_separate_hpre
      args: -no_map -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r111_mapped_second_analytic_separate_hpre
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r112_mapped_second_fd_separate_hpre
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r113_first_analytic_separate_hpre
      args: -no_map -shell_tao_term_sum_extra_mask hessian
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r114_first_fd_separate_hpre
      args: -no_map -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r115_mapped_first_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r116_mapped_first_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r117_analytic_analytic_separate_hpre
      args: -no_map

    test:
      suffix: r118_analytic_fd_separate_hpre
      args: -no_map -extra_tao_term_hessian_use_fd

    test:
      suffix: r119_fd_fd_separate_hpre
      args: -no_map -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r120_mapped_analytic_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false

    test:
      suffix: r121_mapped_analytic_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -extra_tao_term_hessian_use_fd

    test:
      suffix: r122_mapped_fd_fd_separate_hpre
      args: -data_use_map true -extra_use_map false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r123_both_mapped_analytic_analytic_separate_hpre

    test:
      suffix: r124_both_mapped_analytic_fd_separate_hpre
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r125_both_mapped_fd_fd_separate_hpre
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

  testset:
    args: -second_term -shell_tao_type nls -shell_tao_term_hessian_mat_type shell

    test:
      suffix: r126_second_analytic
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r127_second_fd
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r128_second_hessianmult
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r129_mapped_second_analytic
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r130_mapped_second_fd
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r131_mapped_second_hessianmult
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r132_first_analytic
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r133_first_fd
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r134_first_hessianmult
      args: -no_map -shell_tao_term_sum_extra_mask hessian
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r136_mapped_first_fd
      args: -data_use_map true -extra_use_map false -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r137_mapped_first_hessianmult
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r139_analytic_fd
      args: -no_map -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r140_analytic_hessianmult
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r141_fd_fd
      args: -no_map -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r142_fd_hessianmult
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r143_hessianmult_hessianmult
      args: -no_map

    test:
      suffix: r144_mapped_analytic_analytic
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false

    test:
      suffix: r145_mapped_analytic_fd
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r147_mapped_fd_fd
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r151_both_mapped_analytic_fd
      args: -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r152_both_mapped_analytic_hessianmult
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r153_both_mapped_fd_fd
      args: -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r154_both_mapped_fd_hessianmult
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

  testset:
    args: -second_term -split_hpre -shell_tao_type nls
    args: -shell_tao_term_hessian_mat_type shell -shell_tao_term_hessian_pre_is_hessian false
    args: -shell_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r156_second_analytic_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r157_second_fd_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r158_second_hessianmult_separate_hpre
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r159_mapped_second_analytic_separate_hpre
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r160_mapped_second_fd_separate_hpre
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r161_mapped_second_hessianmult_separate_hpre
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r162_first_analytic_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r163_first_fd_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r164_first_hessianmult_separate_hpre
      args: -no_map -shell_tao_term_sum_extra_mask hessian
      args: -shell_tao_nls_init_type constant -c_tao_nls_init_type constant

    test:
      suffix: r165_mapped_first_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r166_mapped_first_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r167_mapped_first_hessianmult_separate_hpre
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r169_analytic_fd_separate_hpre
      args: -no_map -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r170_analytic_hessianmult_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r171_fd_fd_separate_hpre
      args: -no_map -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r172_fd_hessianmult_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r173_hessianmult_hessianmult_separate_hpre
      args: -no_map

    test:
      suffix: r174_mapped_analytic_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false

    test:
      suffix: r175_mapped_analytic_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r177_mapped_fd_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r179_mapped_hessianmult_hessianmult_separate_hpre
      args: -data_use_map true -extra_use_map false

    test:
      suffix: r181_both_mapped_analytic_fd_separate_hpre
      args: -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r182_both_mapped_analytic_hessianmult_separate_hpre
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r183_both_mapped_fd_fd_separate_hpre
      args: -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r184_both_mapped_fd_hessianmult_separate_hpre
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r185_both_mapped_hessianmult_hessianmult_separate_hpre

  test:
    suffix: r018_hessianmult_separate_hpre
    args: -no_map -split_hpre -shell_tao_type nls
    args: -data_tao_term_hessian_mat_type shell -data_tao_term_hessian_pre_is_hessian false
    args: -data_tao_term_hessian_pre_mat_type aij

TEST*/
