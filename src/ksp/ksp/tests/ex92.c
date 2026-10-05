static char help[] = "Tests CUDA GMRES in KSPHPDDM with vector and multiple right-hand sides.\n\n";

#include <petscksp.h>

static PetscErrorCode CheckSolution(KSP ksp, Mat A, Vec b, Vec x, Vec exact, Vec work, PetscBool transpose)
{
  PetscReal          norm, bnorm, xnorm;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "KSP did not converge: %s", KSPConvergedReasons[reason]);
  if (transpose) PetscCall(MatMultTranspose(A, x, work));
  else PetscCall(MatMult(A, x, work));
  PetscCall(VecAXPY(work, -1.0, b));
  PetscCall(VecNorm(work, NORM_2, &norm));
  PetscCall(VecNorm(b, NORM_2, &bnorm));
  PetscCheck(norm < 5e-7 * (1.0 + bnorm), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "True residual norm %g exceeds tolerance", (double)norm);
  PetscCall(VecWAXPY(work, -1.0, exact, x));
  PetscCall(VecNorm(work, NORM_2, &norm));
  PetscCall(VecNorm(exact, NORM_2, &xnorm));
  PetscCheck(norm < 5e-7 * (1.0 + xnorm), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution error norm %g exceeds tolerance", (double)norm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         A, B, X, E;
  Vec         b, x, exact, work;
  KSP         ksp;
  PC          pc;
  PetscInt    n = PETSC_DECIDE, N = 63, nrhs = 3, start, end;
  PetscMPIInt rank, size;
  PetscBool   initial = PETSC_FALSE, zero = PETSC_FALSE, zero_column = PETSC_FALSE, transpose = PETSC_FALSE, empty = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-initial_guess", &initial, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_rhs", &zero, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_column", &zero_column, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose", &transpose, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_last_rank", &empty, NULL));
  if (empty && size > 1) n = rank == size - 1 ? 0 : N / (size - 1) + (rank < N % (size - 1));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, n, n, N, N));
  PetscCall(MatSetType(A, MATAIJCUSPARSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt i = start; i < end; ++i) {
    PetscCall(MatSetValue(A, i, i, 4.0 + 0.05 * (i % 7), INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(A, i, i - 1, -1.1, INSERT_VALUES));
    if (i + 1 < N) PetscCall(MatSetValue(A, i, i + 1, -0.6, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A, &work, NULL));
  PetscCall(MatGetLocalSize(A, &n, NULL));
  PetscCall(MatCreateDenseCUDA(PETSC_COMM_WORLD, n, PETSC_DECIDE, N, nrhs, NULL, &B));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &E));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPHPDDM));
  PetscCall(KSPHPDDMSetType(ksp, KSP_HPDDM_TYPE_GMRES));
  PetscCall(KSPSetTolerances(ksp, 1e-9, 1e-12, PETSC_CURRENT, 200));
  PetscCall(KSPSetInitialGuessNonzero(ksp, initial));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetFromOptions(ksp));
  for (PetscInt j = 0; j < nrhs; ++j) {
    PetscCall(MatDenseGetColumnVecWrite(E, j, &exact));
    if (zero || (zero_column && !j)) PetscCall(VecSet(exact, 0.0));
    else PetscCall(VecSetRandom(exact, NULL));
    PetscCall(MatDenseGetColumnVecWrite(B, j, &b));
    if (transpose) PetscCall(MatMultTranspose(A, exact, b));
    else PetscCall(MatMult(A, exact, b));
    PetscCall(MatDenseGetColumnVecWrite(X, j, &x));
    PetscCall(VecSet(x, initial ? 0.25 : 0.0));
    if (transpose) PetscCall(KSPSolveTranspose(ksp, b, x));
    else PetscCall(KSPSolve(ksp, b, x));
    PetscCall(CheckSolution(ksp, A, b, x, exact, work, transpose));
    if (initial) {
      PetscInt its;

      PetscCall(VecCopy(exact, x));
      if (transpose) PetscCall(KSPSolveTranspose(ksp, b, x));
      else PetscCall(KSPSolve(ksp, b, x));
      PetscCall(KSPGetIterationNumber(ksp, &its));
      PetscCheck(!its, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "An exact initial guess required %" PetscInt_FMT " iterations", its);
      PetscCall(CheckSolution(ksp, A, b, x, exact, work, transpose));
    }
    PetscCall(VecSet(x, initial ? 0.25 : 0.0));
    PetscCall(MatDenseRestoreColumnVecWrite(X, j, &x));
    PetscCall(MatDenseRestoreColumnVecWrite(B, j, &b));
    PetscCall(MatDenseRestoreColumnVecWrite(E, j, &exact));
  }
  if (transpose) PetscCall(KSPMatSolveTranspose(ksp, B, X));
  else PetscCall(KSPMatSolve(ksp, B, X));
  for (PetscInt j = 0; j < nrhs; ++j) {
    PetscCall(MatDenseGetColumnVecRead(B, j, &b));
    PetscCall(MatDenseGetColumnVecRead(X, j, &x));
    PetscCall(MatDenseGetColumnVecRead(E, j, &exact));
    PetscCall(CheckSolution(ksp, A, b, x, exact, work, transpose));
    PetscCall(MatDenseRestoreColumnVecRead(E, j, &exact));
    PetscCall(MatDenseRestoreColumnVecRead(X, j, &x));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &b));
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&work));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hpddm cuda

  testset:
    requires: double
    output_file: output/empty.out
    args: -ksp_error_if_not_converged

    test:
      suffix: variants
      nsize: 2
      args: -ksp_hpddm_variant {{left right flexible}shared output} -ksp_hpddm_orthogonalization {{cgs mgs}shared output} -ksp_gmres_restart {{1 4}shared output}

    test:
      suffix: initial
      args: -initial_guess -ksp_hpddm_variant {{left right flexible}shared output} -ksp_gmres_restart 4

    test:
      suffix: zero
      args: -zero_rhs -ksp_hpddm_variant {{left right flexible}shared output}

    test:
      suffix: zero_column
      args: -zero_column -ksp_hpddm_variant {{left right flexible}shared output} -ksp_gmres_restart 4

    test:
      suffix: transpose
      args: -transpose -initial_guess -ksp_hpddm_variant {{left right flexible}shared output} -ksp_gmres_restart 4

    test:
      suffix: empty
      nsize: 2
      args: -empty_last_rank -ksp_hpddm_variant {{left right flexible}shared output} -ksp_gmres_restart 4

    test:
      suffix: host_mpi
      nsize: 2
      args: -use_gpu_aware_mpi 0 -empty_last_rank -zero_column -initial_guess -ksp_hpddm_variant flexible -ksp_hpddm_orthogonalization {{cgs mgs}shared output} -ksp_gmres_restart 4

    test:
      suffix: mpi
      nsize: 3
      args: -zero_column -initial_guess -ksp_hpddm_variant flexible -ksp_hpddm_orthogonalization {{cgs mgs}shared output} -ksp_gmres_restart 4

TEST*/
