static char help[] = "Test TAOTERMSUM Hessian storage and assembled-output fallbacks.\n\n";

#include <petsctao.h>

static PetscErrorCode Hess_T0(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  if (H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(H, i, i, 2.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  if (Hpre && Hpre != H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(Hpre, i, i, 2.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Hess_Tridiagonal(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  for (PetscInt k = 0; k < 2; k++) {
    Mat M = k ? Hpre : H;

    if (!M || (k && M == H)) continue;
    for (PetscInt i = 0; i < n; i++) {
      PetscCall(MatSetValue(M, i, i, 2.0, INSERT_VALUES));
      if (i) PetscCall(MatSetValue(M, i, i - 1, -1.0, INSERT_VALUES));
      if (i + 1 < n) PetscCall(MatSetValue(M, i, i + 1, -1.0, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestHpreFallback(MPI_Comm comm)
{
  TaoTerm   sum, t0;
  Mat       H0, Hout, Hpre_out;
  Vec       x;
  PetscReal norm;
  PetscInt  n = 4;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &H0));
  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &Hout));
  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &Hpre_out));
  /* The outer matrices are MatCopy()/MatAXPY() destinations for the summand contributions
     and must be assembled, matching what TaoTermCreateHessianMatrices_Sum() produces. */
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Hpre_out, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hpre_out, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumSetNumberTerms(sum, 1));
  PetscCall(TaoTermSumSetTerm(sum, 0, "t0_", 1.0, t0, NULL));
  /* H-only designated storage: the summand has no Hessian preconditioning matrix anywhere */
  PetscCall(TaoTermSumSetTermHessianMatrices(sum, 0, H0, NULL, H0, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(VecCreateSeq(comm, n, &x));
  PetscCall(VecSet(x, 1.0));

  /* Assembled outer Hessian with a distinct preconditioning matrix: the summand's H-only
     storage must serve both, with the Hessian standing in for the missing Hpre */
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hpre_out));
  PetscCall(MatMultEqual(Hpre_out, Hout, 5, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "outer Hpre differs from outer Hessian");
  PetscCall(MatNorm(Hout, NORM_FROBENIUS, &norm));
  PetscCheck(PetscAbsReal(norm - 2.0 * PetscSqrtReal((PetscReal)n)) <= 1.e-9, comm, PETSC_ERR_PLIB, "outer Hessian norm %g does not match the summand Hessian", (double)norm);
  PetscCall(PetscPrintf(comm, "H-only summand storage fills a distinct outer Hpre\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hpre_out));
  PetscCall(MatDestroy(&Hout));
  PetscCall(MatDestroy(&H0));
  PetscCall(TaoTermDestroy(&t0));
  PetscCall(TaoTermDestroy(&sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMappedOnlyStorage(MPI_Comm comm)
{
  TaoTerm   sum, t0;
  Mat       map, mapped_H, Hout;
  Vec       x;
  PetscReal norm;
  PetscInt  index;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, 2, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));
  PetscCall(TaoTermShellSetCreateHessianMatrices(t0, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(t0, PETSC_TRUE, MATAIJ, NULL));

  PetscCall(MatCreateSeqAIJ(comm, 2, 3, 1, NULL, &map));
  PetscCall(MatSetValue(map, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(map, 1, 1, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, "t0_", 1.0, t0, map, &index));
  PetscCall(MatCreateSeqAIJ(comm, 3, 3, 1, NULL, &mapped_H));
  PetscCall(MatAssemblyBegin(mapped_H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mapped_H, MAT_FINAL_ASSEMBLY));
  PetscCall(TaoTermSumSetTermHessianMatrices(sum, index, NULL, NULL, mapped_H, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, 3, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(MatDuplicate(mapped_H, MAT_DO_NOT_COPY_VALUES, &Hout));
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(comm, 3, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hout));
  PetscCall(MatNorm(Hout, NORM_FROBENIUS, &norm));
  PetscCheck(PetscAbsReal(norm - 2.0 * PetscSqrtReal(2.0)) <= 1.e-9, comm, PETSC_ERR_PLIB, "mapped Hessian norm %g is incorrect", (double)norm);
  PetscCall(PetscPrintf(comm, "Mapped-only storage uses separately created raw storage\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hout));
  PetscCall(MatDestroy(&mapped_H));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(MatDestroy(&map));
  PetscCall(TaoTermDestroy(&t0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestNoStorageAssembledOutput(MPI_Comm comm)
{
  TaoTerm   sum, t0;
  Mat       Hout;
  Vec       x;
  PetscReal norm;
  PetscInt  n = 4;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, "t0_", 1.0, t0, NULL, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(MatCreateSeqAIJ(comm, n, n, 1, NULL, &Hout));
  for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(Hout, i, i, 0.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(comm, n, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hout));
  PetscCall(MatNorm(Hout, NORM_FROBENIUS, &norm));
  PetscCheck(PetscAbsReal(norm - 2.0 * PetscSqrtReal((PetscReal)n)) <= 1.e-9, comm, PETSC_ERR_PLIB, "direct-output Hessian norm %g is incorrect", (double)norm);
  PetscCall(PetscPrintf(comm, "No-storage summand evaluates into assembled caller output\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hout));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&t0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCreatedDistinctHpre(MPI_Comm comm)
{
  TaoTerm   sum, t0;
  Mat       map, supplied_H, stored_H, stored_Hpre, Hout, Hpre_out;
  Vec       x;
  PetscInt  index;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, 2, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));
  PetscCall(TaoTermShellSetCreateHessianMatrices(t0, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(t0, PETSC_FALSE, MATAIJ, MATAIJ));

  PetscCall(MatCreateSeqAIJ(comm, 2, 3, 1, NULL, &map));
  PetscCall(MatSetValue(map, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(map, 1, 1, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateSeqAIJ(comm, 2, 2, 1, NULL, &supplied_H));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, "t0_", 1.0, t0, map, &index));
  PetscCall(TaoTermSumSetTermHessianMatrices(sum, index, supplied_H, NULL, NULL, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, 3, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(MatCreateSeqAIJ(comm, 3, 3, 1, NULL, &Hout));
  PetscCall(MatCreateSeqAIJ(comm, 3, 3, 1, NULL, &Hpre_out));
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Hpre_out, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hpre_out, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(comm, 3, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hpre_out));
  PetscCall(TaoTermSumGetTermHessianMatrices(sum, index, &stored_H, &stored_Hpre, NULL, NULL));
  PetscCheck(stored_H == supplied_H, comm, PETSC_ERR_PLIB, "Supplied unmapped Hessian was replaced");
  PetscCheck(stored_Hpre && stored_Hpre != stored_H, comm, PETSC_ERR_PLIB, "A distinct unmapped preconditioning matrix was not created");
  PetscCall(MatMultEqual(Hpre_out, Hout, 5, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "Created preconditioning matrix differs from the Hessian");
  PetscCall(PetscPrintf(comm, "H-only storage with a creator preserves H and creates a distinct Hpre\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hpre_out));
  PetscCall(MatDestroy(&Hout));
  PetscCall(MatDestroy(&supplied_H));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(MatDestroy(&map));
  PetscCall(TaoTermDestroy(&t0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSumIdempotence(MPI_Comm comm)
{
  TaoTerm   sum, l1, tridiagonal;
  Mat       H, Hcopy;
  Vec       x;
  PetscInt  index, n = 4;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreate(comm, &l1));
  PetscCall(TaoTermSetType(l1, TAOTERML1));
  PetscCall(TaoTermL1SetEpsilon(l1, 0.5));
  PetscCall(TaoTermSetSolutionSizes(l1, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &tridiagonal));
  PetscCall(TaoTermSetParametersMode(tridiagonal, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(tridiagonal, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetHessian(tridiagonal, Hess_Tridiagonal));
  PetscCall(TaoTermShellSetCreateHessianMatrices(tridiagonal, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(tridiagonal, PETSC_TRUE, MATAIJ, NULL));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, "l1_", 1.0, l1, NULL, &index));
  PetscCall(TaoTermSumAddTerm(sum, "tridiagonal_", 1.0, tridiagonal, NULL, &index));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &H));
  PetscCall(MatSetOption(H, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(comm, n, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, H, NULL));
  PetscCall(MatDuplicate(H, MAT_COPY_VALUES, &Hcopy));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, H, NULL));
  PetscCall(MatMultEqual(H, Hcopy, 5, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "Repeated sum Hessian evaluation changed the matrix");
  PetscCall(PetscPrintf(comm, "Repeated partial-overwrite sum Hessian evaluation is idempotent\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hcopy));
  PetscCall(MatDestroy(&H));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&tridiagonal));
  PetscCall(TaoTermDestroy(&l1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");
  PetscCall(TestHpreFallback(comm));
  PetscCall(TestMappedOnlyStorage(comm));
  PetscCall(TestNoStorageAssembledOutput(comm));
  PetscCall(TestCreatedDistinctHpre(comm));
  PetscCall(TestSumIdempotence(comm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: hpre_fallback

TEST*/
