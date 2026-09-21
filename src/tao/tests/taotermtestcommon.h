#pragma once

#include <petsctao.h>

typedef struct {
  Mat         map;
  Vec         parameters;
  PetscReal   scale;
  TaoTermMask mask;
} ExampleSubterm;

typedef struct {
  TaoTerm     term;
  Vec         parameters;
  Mat         map;
  PetscInt    size;
  PetscReal   scale;
  PetscBool   use_map;
  PetscBool   supply_parameters;
  PetscBool   provide_hessian_mult;
  PetscBool   use_fd;
  TaoTermMask mask;
} ExampleTerm;

typedef struct {
  PetscInt       nsubterms;
  ExampleSubterm subterms[3];
} ExampleCtx;

typedef struct {
  Tao         tao;
  Mat         H;
  Mat         Hpre;
  Vec         x;
  ExampleCtx *ctx;
} ExampleReference;

PETSC_UNUSED static PetscErrorCode ExampleTermDestroy(ExampleTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermDestroy(&term->term));
  PetscCall(VecDestroy(&term->parameters));
  PetscCall(MatDestroy(&term->map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode ExampleTermSetSubterm(ExampleTerm *term, ExampleSubterm *subterm)
{
  PetscFunctionBeginUser;
  subterm->map        = term->map;
  subterm->parameters = term->parameters;
  subterm->scale      = term->scale;
  subterm->mask       = term->mask;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* These callbacks operate in the term space; TaoTermMapping applies any map and its transpose. */
PETSC_UNUSED static PetscErrorCode ExampleIdentityLeastSquaresObjective(TaoTerm term, Vec x, Vec parameters, PetscReal *f)
{
  Vec         work;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &work));
  if (parameters) PetscCall(VecWAXPY(work, -1.0, parameters, x));
  else PetscCall(VecCopy(x, work));
  PetscCall(VecDot(work, work, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleIdentityLeastSquaresGradient(TaoTerm term, Vec x, Vec parameters, Vec g)
{
  PetscFunctionBeginUser;
  if (parameters) PetscCall(VecWAXPY(g, -1.0, parameters, x));
  else PetscCall(VecCopy(x, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleIdentityLeastSquaresObjectiveGradient(TaoTerm term, Vec x, Vec parameters, PetscReal *f, Vec g)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(ExampleIdentityLeastSquaresGradient(term, x, parameters, g));
  PetscCall(VecDot(g, g, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode ExampleIdentityLeastSquaresHessian(TaoTerm term, Vec x, Vec parameters, Mat H, Mat Hpre)
{
  PetscFunctionBeginUser;
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
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode ExampleIdentityLeastSquaresHessianMult(TaoTerm term, Vec x, Vec parameters, Vec v, Vec Hv)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleMapSolution(ExampleSubterm *subterm, Vec x, Vec *mapped_x, PetscBool *destroy)
{
  PetscFunctionBeginUser;
  if (subterm->map) {
    PetscCall(MatCreateVecs(subterm->map, NULL, mapped_x));
    PetscCall(MatMult(subterm->map, x, *mapped_x));
    *destroy = PETSC_TRUE;
  } else {
    *mapped_x = x;
    *destroy  = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleComputeSubterm(ExampleSubterm *subterm, Vec x, PetscBool need_f, PetscBool need_g, PetscBool need_h, PetscReal *f, Vec gx, Mat Hx)
{
  Mat         contribution = NULL;
  Vec         mapped_x, diff, mapped_g = NULL, diag = NULL;
  PetscScalar dot;
  PetscBool   destroy_mapped_x = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(ExampleMapSolution(subterm, x, &mapped_x, &destroy_mapped_x));
  PetscCall(VecDuplicate(mapped_x, &diff));
  if (subterm->parameters) PetscCall(VecWAXPY(diff, -1.0, subterm->parameters, mapped_x));
  else PetscCall(VecCopy(mapped_x, diff));
  if (need_f) {
    PetscCall(VecDot(diff, diff, &dot));
    *f = 0.5 * PetscRealPart(dot);
    *f *= subterm->scale;
  }
  if (need_g) {
    if (subterm->map) {
      PetscCall(VecDuplicate(diff, &mapped_g));
      PetscCall(VecCopy(diff, mapped_g));
      PetscCall(MatMultTranspose(subterm->map, mapped_g, gx));
    } else PetscCall(VecCopy(diff, gx));
    PetscCall(VecScale(gx, subterm->scale));
    PetscCall(VecDestroy(&mapped_g));
  }
  if (need_h) {
    if (subterm->map) PetscCall(MatTransposeMatMult(subterm->map, subterm->map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &contribution));
    else {
      PetscCall(VecDuplicate(diff, &diag));
      PetscCall(VecSet(diag, 1.0));
      PetscCall(MatCreateDiagonal(diag, &contribution));
    }
    PetscCall(MatAXPY(Hx, subterm->scale, contribution, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&contribution));
    PetscCall(VecDestroy(&diag));
  }
  PetscCall(VecDestroy(&diff));
  if (destroy_mapped_x) PetscCall(VecDestroy(&mapped_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleFormObjectiveGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *vctx)
{
  ExampleCtx *ctx = (ExampleCtx *)vctx;
  Vec         work;
  PetscReal   subterm_f;

  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscCall(VecDuplicate(g, &work));
  for (PetscInt i = 0; i < ctx->nsubterms; i++) {
    ExampleSubterm *subterm = &ctx->subterms[i];
    PetscBool       need_f  = !(subterm->mask & TAOTERM_MASK_OBJECTIVE);
    PetscBool       need_g  = !(subterm->mask & TAOTERM_MASK_GRADIENT);

    subterm_f = 0.0;
    PetscCall(ExampleComputeSubterm(subterm, x, need_f, need_g, PETSC_FALSE, &subterm_f, work, NULL));
    if (need_f) *f += subterm_f;
    if (need_g) PetscCall(VecAXPY(g, 1.0, work));
  }
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleAssembleHessian(ExampleCtx *ctx, Vec x, Mat H)
{
  Vec       work;
  PetscReal unused;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatCreateVecs(H, &work, NULL));
  for (PetscInt i = 0; i < ctx->nsubterms; i++) {
    ExampleSubterm *subterm = &ctx->subterms[i];

    if (!(subterm->mask & TAOTERM_MASK_HESSIAN)) PetscCall(ExampleComputeSubterm(subterm, x, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE, &unused, work, H));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleApplyHessian(ExampleCtx *ctx, Vec v, Vec Hv)
{
  Vec mapped_v = NULL, contribution;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(Hv));
  PetscCall(VecDuplicate(Hv, &contribution));
  for (PetscInt i = 0; i < ctx->nsubterms; i++) {
    ExampleSubterm *subterm = &ctx->subterms[i];

    if (subterm->mask & TAOTERM_MASK_HESSIAN) continue;
    if (subterm->map) {
      PetscCall(MatCreateVecs(subterm->map, NULL, &mapped_v));
      PetscCall(MatMult(subterm->map, v, mapped_v));
      PetscCall(MatMultTranspose(subterm->map, mapped_v, contribution));
    } else PetscCall(VecCopy(v, contribution));
    PetscCall(VecAXPY(Hv, subterm->scale, contribution));
    PetscCall(VecDestroy(&mapped_v));
  }
  PetscCall(VecDestroy(&contribution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleMatMult(Mat H, Vec v, Vec Hv)
{
  ExampleCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(H, &ctx));
  PetscCall(ExampleApplyHessian(ctx, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleFormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *vctx)
{
  ExampleCtx *ctx = (ExampleCtx *)vctx;
  PetscBool   is_shell;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)H, MATSHELL, &is_shell));
  if (is_shell) {
    /* Clear the shifts and scales left by the previous TAONLS iteration before computing the Hessian at x */
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    if (Hpre && Hpre != H) PetscCall(ExampleAssembleHessian(ctx, x, Hpre));
  } else {
    PetscCall(ExampleAssembleHessian(ctx, x, H));
    if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleCreateTao(MPI_Comm comm, Tao source, Vec x, ExampleCtx *ctx, Tao *tao, Mat *H, Mat *Hpre)
{
  TaoType   type;
  Mat       source_H, source_Hpre;
  PetscInt  n, nlocal;
  PetscInt  max_it, max_funcs;
  PetscReal gatol, grtol, gttol;
  PetscBool is_nls, is_mffd, is_shell;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  PetscCall(VecGetLocalSize(x, &nlocal));
  PetscCall(TaoSetUp(source));
  PetscCall(TaoGetHessianMatrices(source, &source_H, &source_Hpre));
  PetscCall(PetscObjectTypeCompare((PetscObject)source_H, MATSHELL, &is_shell));
  PetscCall(PetscObjectTypeCompare((PetscObject)source_H, MATMFFD, &is_mffd));
  if (is_mffd) {
    PetscCall(MatCreateMFFD(comm, nlocal, nlocal, n, n, H));
    PetscCall(MatSetOption(*H, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  } else if (is_shell) {
    PetscCall(MatCreateShell(comm, nlocal, nlocal, n, n, ctx, H));
    PetscCall(MatShellSetOperation(*H, MATOP_MULT, (PetscErrorCodeFn *)ExampleMatMult));
    PetscCall(MatSetOption(*H, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  } else {
    PetscCall(MatCreateAIJ(comm, nlocal, nlocal, n, n, n, NULL, n, NULL, H));
    PetscCall(MatSetOption(*H, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  }
  if (source_Hpre != source_H) {
    PetscCall(MatCreateAIJ(comm, nlocal, nlocal, n, n, n, NULL, n, NULL, Hpre));
    PetscCall(MatSetOption(*Hpre, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatAssemblyBegin(*Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*Hpre, MAT_FINAL_ASSEMBLY));
  } else *Hpre = *H;
  PetscCall(TaoCreate(comm, tao));
  PetscCall(TaoSetOptionsPrefix(*tao, "c_"));
  PetscCall(TaoGetType(source, &type));
  PetscCall(TaoSetType(*tao, type));
  PetscCall(TaoGetTolerances(source, &gatol, &grtol, &gttol));
  PetscCall(TaoSetTolerances(*tao, gatol, grtol, gttol));
  PetscCall(TaoGetMaximumIterations(source, &max_it));
  PetscCall(TaoSetMaximumIterations(*tao, max_it));
  PetscCall(TaoGetMaximumFunctionEvaluations(source, &max_funcs));
  PetscCall(TaoSetMaximumFunctionEvaluations(*tao, max_funcs));
  PetscCall(TaoSetSolution(*tao, x));
  PetscCall(TaoSetObjectiveAndGradient(*tao, NULL, ExampleFormObjectiveGradient, ctx));
  PetscCall(TaoSetFromOptions(*tao));
  PetscCall(PetscObjectTypeCompare((PetscObject)*tao, TAONLS, &is_nls));
  if (is_nls) PetscCall(TaoSetHessian(*tao, *H, *Hpre, is_mffd ? TaoDefaultComputeHessianMFFD : ExampleFormHessian, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleCompareResults(Tao ttao, Vec tx, Tao ctao, Vec cx)
{
  const PetscReal solution_rtol = 1.e-3, objective_rtol = 1.e-4, gradient_rtol = 1.e-3;
  Vec             diff;
  PetscReal       tnorm, cnorm, error, scale, tf, cf, tg, cg;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(tx, &diff));
  PetscCall(VecWAXPY(diff, -1.0, cx, tx));
  PetscCall(VecNorm(diff, NORM_2, &error));
  PetscCall(VecNorm(tx, NORM_2, &tnorm));
  PetscCall(VecNorm(cx, NORM_2, &cnorm));
  scale = 1.0 + PetscMax(tnorm, cnorm);
  PetscCheck(error <= solution_rtol * scale, PetscObjectComm((PetscObject)ttao), PETSC_ERR_PLIB, "TaoTerm and reference solutions differ by %g relative to scale %g", (double)error, (double)scale);
  PetscCall(TaoGetSolutionStatus(ttao, NULL, &tf, &tg, NULL, NULL, NULL));
  PetscCall(TaoGetSolutionStatus(ctao, NULL, &cf, &cg, NULL, NULL, NULL));
  scale = 1.0 + PetscMax(PetscAbsReal(tf), PetscAbsReal(cf));
  PetscCheck(PetscAbsReal(tf - cf) <= objective_rtol * scale, PetscObjectComm((PetscObject)ttao), PETSC_ERR_PLIB, "TaoTerm and reference final objectives differ by %g relative to scale %g", (double)PetscAbsReal(tf - cf), (double)scale);
  scale = 1.0 + PetscMax(PetscAbsReal(tg), PetscAbsReal(cg));
  PetscCheck(PetscAbsReal(tg - cg) <= gradient_rtol * scale, PetscObjectComm((PetscObject)ttao), PETSC_ERR_PLIB, "TaoTerm and reference final gradient norms differ by %g relative to scale %g", (double)PetscAbsReal(tg - cg), (double)scale);
  PetscCall(VecDestroy(&diff));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ttao), "Reference callback comparison passed\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleReferenceCreate(MPI_Comm comm, Tao source, Vec source_x, ExampleCtx *ctx, ExampleReference *reference)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(reference, sizeof(*reference)));
  reference->ctx = ctx;
  PetscCall(VecDuplicate(source_x, &reference->x));
  PetscCall(VecCopy(source_x, reference->x));
  PetscCall(ExampleCreateTao(comm, source, reference->x, ctx, &reference->tao, &reference->H, &reference->Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleReferenceSolveAndCompare(Tao source, Vec source_x, ExampleReference *reference)
{
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(TaoSolve(source));
  PetscCall(TaoSolve(reference->tao));
  PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)source), &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(TaoView(source, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(ExampleCompareResults(source, source_x, reference->tao, reference->x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleReferenceDestroy(ExampleReference *reference)
{
  PetscFunctionBeginUser;
  PetscCall(TaoDestroy(&reference->tao));
  if (reference->Hpre != reference->H) PetscCall(MatDestroy(&reference->Hpre));
  PetscCall(MatDestroy(&reference->H));
  PetscCall(VecDestroy(&reference->x));
  reference->ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
