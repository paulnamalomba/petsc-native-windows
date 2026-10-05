static const char help[] = "Tests Jacobi PCMatApply() and PCMatApplyTranspose() with padded dense right-hand sides.\n\n";

#include <petscksp.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif

static PetscScalar RHSValue(PetscInt row, PetscInt column)
{
  PetscScalar value = (row + 1) * (column + 2);

#if PetscDefined(USE_COMPLEX)
  value += 0.25 * (row - column) * PETSC_i;
#endif
  return value;
}

static PetscErrorCode FillDense(Mat B, PetscBool rhs, PetscBool cuda)
{
  PetscScalar *b;
  PetscInt     m, n, start, lda;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(B, &m, NULL));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(MatGetOwnershipRange(B, &start, NULL));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCall(MatDenseGetArrayWrite(B, &b));
  for (PetscInt j = 0; j < n; ++j) {
    for (PetscInt i = 0; i < m; ++i) b[i + j * lda] = rhs ? RHSValue(start + i, j) : -7.0;
    for (PetscInt i = m; i < lda; ++i) b[i + j * lda] = 0.0;
  }
  PetscCall(MatDenseRestoreArrayWrite(B, &b));
#if PetscDefined(HAVE_CUDA)
  if (cuda) {
    PetscCall(MatDenseCUDAGetArray(B, &b));
    PetscCallCUDA(cudaDeviceSynchronize());
    /* Dense host/device transfers skip the padding, so initialize it on the device too. */
    PetscCallCUDA(cudaMemset2D(b + m, lda * sizeof(*b), 0, (lda - m) * sizeof(*b), n));
    PetscCall(MatDenseCUDARestoreArray(B, &b));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDense(Mat B, PetscBool rhs, PetscBool cuda)
{
  const PetscScalar *b;
  PetscInt           m, n, start, lda;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(B, &m, NULL));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(MatGetOwnershipRange(B, &start, NULL));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCall(MatDenseGetArrayRead(B, &b));
  for (PetscInt j = 0; j < n; ++j) {
    if (rhs)
      for (PetscInt i = 0; i < m; ++i) PetscCheck(b[i + j * lda] == RHSValue(start + i, j), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input entry (%" PetscInt_FMT ",%" PetscInt_FMT ") changed", start + i, j);
    for (PetscInt i = m; i < lda; ++i) PetscCheck(b[i + j * lda] == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host padding entry (%" PetscInt_FMT ",%" PetscInt_FMT ") changed", i, j);
  }
  PetscCall(MatDenseRestoreArrayRead(B, &b));
#if PetscDefined(HAVE_CUDA)
  if (cuda) {
    PetscScalar *padding;
    PetscInt     width = lda - m;

    PetscCall(PetscMalloc1(width * n, &padding));
    PetscCall(MatDenseCUDAGetArrayRead(B, &b));
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCallCUDA(cudaMemcpy2D(padding, width * sizeof(*padding), b + m, lda * sizeof(*b), width * sizeof(*padding), n, cudaMemcpyDeviceToHost));
    PetscCall(MatDenseCUDARestoreArrayRead(B, &b));
    for (PetscInt i = 0; i < width * n; ++i) PetscCheck(padding[i] == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Device padding entry %" PetscInt_FMT " changed", i);
    PetscCall(PetscFree(padding));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckApply(PC pc, Mat B, Mat X, Vec reference, PetscBool transpose, PetscBool cuda)
{
  Vec       b, x;
  PetscInt  n;
  PetscReal error, norm;

  PetscFunctionBeginUser;
  PetscCall(FillDense(X, PETSC_FALSE, cuda));
  if (transpose) PetscCall(PCMatApplyTranspose(pc, B, X));
  else PetscCall(PCMatApply(pc, B, X));
  PetscCall(MatGetSize(B, NULL, &n));
  for (PetscInt j = 0; j < n; ++j) {
    PetscCall(MatDenseGetColumnVecRead(B, j, &b));
    PetscCall(MatDenseGetColumnVecRead(X, j, &x));
    if (transpose) PetscCall(PCApplyTranspose(pc, b, reference));
    else PetscCall(PCApply(pc, b, reference));
    PetscCall(VecNorm(reference, NORM_INFINITY, &norm));
    PetscCall(VecAXPY(reference, -1.0, x));
    PetscCall(VecNorm(reference, NORM_INFINITY, &error));
    PetscCheck(error <= 100.0 * PETSC_MACHINE_EPSILON * PetscMax(norm, 1.0), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " differs from vector application by %g", j, (double)error);
    PetscCall(MatDenseRestoreColumnVecRead(X, j, &x));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &b));
  }
  PetscCall(CheckDense(B, PETSC_TRUE, cuda));
  PetscCall(CheckDense(X, PETSC_FALSE, cuda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat       A, B, X;
  PC        pc;
  Vec       reference;
  PetscInt  start, end, m, n = 7, nrhs = 4;
  PetscBool cuda = PETSC_FALSE, bind_cpu = PETSC_FALSE, device;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cuda", &cuda, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bind_cpu", &bind_cpu, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCheck(n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The global row count must be positive");
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, cuda ? MATAIJCUSPARSE : MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt i = start; i < end; ++i) {
    PetscScalar diagonal = i == 0 || i == 2 ? 0.0 : i + 2.0;

#if PetscDefined(USE_COMPLEX)
    if (diagonal != 0.0) diagonal += (0.25 + 0.125 * i) * PETSC_i;
#endif
    if (i == 3) diagonal = -diagonal;
    PetscCall(MatSetValue(A, i, i, diagonal, INSERT_VALUES));
    if (i) PetscCall(MatSetValue(A, i, i - 1, 0.5, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A, &reference, NULL));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, m, PETSC_DECIDE, n, nrhs));
  PetscCall(MatSetType(B, cuda ? MATDENSECUDA : MATDENSE));
  PetscCall(MatDenseSetLDA(B, m + 2));
  PetscCall(MatSetUp(B));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, m, PETSC_DECIDE, n, nrhs));
  PetscCall(MatSetType(X, cuda ? MATDENSECUDA : MATDENSE));
  PetscCall(MatDenseSetLDA(X, m + 4));
  PetscCall(MatSetUp(X));
  PetscCall(MatBindToCPU(A, bind_cpu));
  PetscCall(MatBindToCPU(B, bind_cpu));
  PetscCall(MatBindToCPU(X, bind_cpu));
  PetscCall(VecBindToCPU(reference, bind_cpu));
  device = (PetscBool)(cuda && !bind_cpu);
  PetscCall(FillDense(B, PETSC_TRUE, device));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(PCJacobiSetFixDiagonal(pc, PETSC_TRUE));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  for (PetscInt k = 0; k < 2; ++k) {
    PetscCall(CheckApply(pc, B, X, reference, PETSC_FALSE, device));
    PetscCall(CheckApply(pc, B, X, reference, PETSC_TRUE, device));
    if (!k) {
      PetscCall(MatScale(A, 2.0));
      PetscCall(PCSetOperators(pc, A, A));
    }
  }
  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&reference));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}

    test:
      suffix: cpu
      args: -pc_jacobi_type {{diagonal rowl1 rowmax rowsum}shared output}

    test:
      suffix: cuda
      requires: cuda
      args: -cuda -pc_jacobi_type {{diagonal rowl1 rowmax rowsum}shared output}

    test:
      suffix: abs
      args: -pc_jacobi_abs

    test:
      suffix: abs_cuda
      requires: cuda
      args: -cuda -pc_jacobi_abs

    test:
      suffix: bound_cpu
      requires: cuda
      args: -cuda -bind_cpu -pc_jacobi_type rowsum

    test:
      suffix: empty
      nsize: 2
      args: -n 1

    test:
      suffix: empty_cuda
      nsize: 2
      requires: cuda
      args: -n 1 -cuda

TEST*/
