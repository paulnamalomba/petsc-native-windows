static char help[] = "Tests the BNTR trust-region update after the trial step is projected onto the bounds.\n";

/*
  Minimizes f(x) = 1/2 (x - 2)^2 subject to 0 <= x <= 0.5, starting from x = 0 with a unit trust
  region. Projection shortens the trust-region step s = 1 to s = 0.5. The quadratic model is exact,
  so the predicted reduction -g^T s - 1/2 s^T H s = 0.875 equals the actual reduction, and the radius
  grows by the largest factor, alpha5 = 4.
*/

#include <petsctao.h>

static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, PetscCtx ctx)
{
  PetscScalar gg;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(X, G));
  PetscCall(VecShift(G, -2.0));
  PetscCall(VecDot(G, G, &gg));
  *f = 0.5 * PetscRealPart(gg);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetValue(H, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao       tao;
  Mat       H;
  Vec       x, xl, xu;
  PetscReal radius;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &x));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecSet(x, 0.0));
  PetscCall(VecSet(xl, 0.0));
  PetscCall(VecSet(xu, 0.5));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &H));

  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOBNTR));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, NULL));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, NULL));
  PetscCall(TaoSetInitialTrustRegionRadius(tao, 1.0));
  PetscCall(TaoSetMaximumIterations(tao, 1));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetCurrentTrustRegionRadius(tao, &radius));
  PetscCheck(PetscIsCloseAtTol(radius, 4.0, PETSC_SMALL, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Trust radius %g after the projected step, expected 4", (double)radius);

  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  test:
    args: -tao_bnk_init_type constant -tao_bnk_as_type none -tao_bnk_pc_type none
    output_file: output/empty.out

TEST*/
