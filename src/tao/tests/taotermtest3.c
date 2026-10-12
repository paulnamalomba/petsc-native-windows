#include <petsctao.h>
#include "taotermtestcommon.h"

static char help[] = "Resolve a regularized least-squares problem after updating its observation operator.\n";

static PetscErrorCode SetMap(Mat, PetscBool, PetscBool);
static PetscErrorCode SetTarget(Vec, PetscReal);
static PetscErrorCode CheckHessianAction(Tao, Mat, PetscReal, PetscReal, Vec);
static PetscErrorCode CheckSolution(Mat, Vec, PetscReal, Vec);

int main(int argc, char **argv)
{
  const PetscInt   n      = 8;
  const PetscReal  lambda = 0.2;
  Tao              tao;
  TaoTerm          data, regularizer;
  ExampleCtx       reference_ctx;
  ExampleReference reference;
  Mat              A;
  Vec              b, x;
  PetscReal        hessian_tolerance = 1.e-9;
  PetscBool        change_structure  = PETSC_FALSE;
  PetscBool        use_fd            = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscMemzero(&reference_ctx, sizeof(reference_ctx)));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-change_structure", &change_structure, NULL));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 2, NULL, 2, NULL, &A));
  PetscCall(SetMap(A, PETSC_FALSE, PETSC_FALSE));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(SetTarget(b, 1.0));
  PetscCall(VecZeroEntries(x));

  PetscCall(TaoTermCreateShell(PETSC_COMM_WORLD, NULL, NULL, &data));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data, "data_"));
  PetscCall(TaoTermSetSolutionSizes(data, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetParametersSizes(data, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(data, ExampleIdentityLeastSquaresObjectiveGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(data, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(data, PETSC_TRUE, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(data, ExampleIdentityLeastSquaresHessian));
  PetscCall(TaoTermSetFromOptions(data));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &regularizer));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)regularizer, "reg_"));
  PetscCall(TaoTermSetSolutionSizes(regularizer, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetType(regularizer, TAOTERMHALFL2SQUARED));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoAddTerm(tao, "data_", 1.0, data, b, A));
  PetscCall(TaoAddTerm(tao, "reg_", lambda, regularizer, NULL, NULL));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoTermComputeHessianGetUseFD(data, &use_fd));
  if (use_fd) hessian_tolerance = 2.e-5;

  reference_ctx.nsubterms              = 2;
  reference_ctx.subterms[0].map        = A;
  reference_ctx.subterms[0].parameters = b;
  reference_ctx.subterms[0].scale      = 1.0;
  reference_ctx.subterms[1].scale      = lambda;
  PetscCall(ExampleReferenceCreate(PETSC_COMM_WORLD, tao, x, &reference_ctx, &reference));

  PetscCall(ExampleReferenceSolveAndCompare(tao, x, &reference));
  PetscCall(CheckSolution(A, b, lambda, x));
  PetscCall(CheckHessianAction(tao, A, lambda, hessian_tolerance, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial least-squares solution check passed\n"));

  PetscCall(SetMap(A, PETSC_TRUE, change_structure));
  PetscCall(SetTarget(b, 1.5));
  PetscCall(CheckHessianAction(tao, A, lambda, hessian_tolerance, x));
  PetscCall(VecZeroEntries(x));
  PetscCall(VecZeroEntries(reference.x));
  PetscCall(ExampleReferenceSolveAndCompare(tao, x, &reference));
  PetscCall(CheckSolution(A, b, lambda, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Updated least-squares solution check passed\n"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(ExampleReferenceDestroy(&reference));
  PetscCall(TaoTermDestroy(&data));
  PetscCall(TaoTermDestroy(&regularizer));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode SetMap(Mat A, PetscBool update, PetscBool change_structure)
{
  PetscInt rstart, rend, n;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetSize(A, &n, NULL));
  if (change_structure) PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatZeroEntries(A));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscCall(MatSetValue(A, i, i, (update ? 1.5 : 1.2) + 0.1 * (i + 1), INSERT_VALUES));
    if (change_structure) PetscCall(MatSetValue(A, i, (i + 1) % n, 0.15, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetTarget(Vec b, PetscReal scale)
{
  PetscInt     rstart, rend;
  PetscScalar *a;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(b, &rstart, &rend));
  PetscCall(VecGetArrayWrite(b, &a));
  for (PetscInt i = rstart; i < rend; i++) a[i - rstart] = scale * (1.0 + 0.05 * (i + 1));
  PetscCall(VecRestoreArrayWrite(b, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckHessianAction(Tao tao, Mat A, PetscReal lambda, PetscReal tolerance, Vec x)
{
  Mat       H, Hpre;
  Vec       v, Av, actual, expected;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, &v, &Av));
  PetscCall(VecDuplicate(v, &actual));
  PetscCall(VecDuplicate(v, &expected));
  PetscCall(VecSet(v, 1.0));
  PetscCall(MatMult(A, v, Av));
  PetscCall(MatMultTranspose(A, Av, expected));
  PetscCall(VecAXPY(expected, lambda, v));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(MatMult(H, v, actual));
  PetscCall(VecAXPY(actual, -1.0, expected));
  PetscCall(VecNorm(actual, NORM_2, &error));
  PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Hessian action after updating the map differs from the current mapped operator by %g", (double)error);
  if (Hpre != H) {
    PetscCall(MatMult(Hpre, v, actual));
    PetscCall(VecAXPY(actual, -1.0, expected));
    PetscCall(VecNorm(actual, NORM_2, &error));
    PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Hessian preconditioning action after updating the map differs from the current mapped operator by %g", (double)error);
  }
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&actual));
  PetscCall(VecDestroy(&Av));
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolution(Mat A, Vec b, PetscReal lambda, Vec x)
{
  KSP       ksp;
  PC        pc;
  Mat       normal;
  Vec       rhs, expected;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &normal));
  PetscCall(MatShift(normal, lambda));
  PetscCall(MatCreateVecs(A, &rhs, NULL));
  PetscCall(VecDuplicate(rhs, &expected));
  PetscCall(MatMultTranspose(A, b, rhs));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)A), &ksp));
  PetscCall(KSPSetOperators(ksp, normal, normal));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, expected));
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_2, &error));
  PetscCheck(error <= 1.e-7, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Reused Tao solution differs from freshly assembled normal equations by %g", (double)error);
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&normal));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: map_values
    args: -tao_type nls

  test:
    suffix: map_structure
    args: -change_structure -tao_type nls

  test:
    suffix: map_values_shell
    args: -tao_type nls -tao_term_hessian_mat_type shell

  test:
    suffix: r148_map_values_shell_fd
    args: -tao_type nls -tao_term_hessian_mat_type shell -data_tao_term_hessian_use_fd

  test:
    suffix: r176_map_values_shell_separate_hpre
    args: -tao_type nls -tao_term_hessian_mat_type shell
    args: -tao_term_hessian_pre_is_hessian false -tao_term_hessian_pre_mat_type aij

  test:
    suffix: r178_map_values_shell_fd_separate_hpre
    args: -tao_type nls -tao_term_hessian_mat_type shell -data_tao_term_hessian_use_fd
    args: -tao_term_hessian_pre_is_hessian false -tao_term_hessian_pre_mat_type aij

TEST*/
