static const char help[] = "Tests cuSPARSE matrix factor solves with padded dense right-hand sides and numeric reuse.\n\n";

#include <petscmat.h>
#include <petscdevice_cuda.h>

static PetscScalar RHSValue(PetscInt row, PetscInt column)
{
  PetscScalar value = (row + 1.0) / (column + 2.0) + 0.2 * column;

#if PetscDefined(USE_COMPLEX)
  value += 0.125 * (row - column) * PETSC_i;
#endif
  return value;
}

static PetscErrorCode FillDense(Mat B, PetscBool rhs, PetscBool cuda)
{
  PetscScalar *array;
  PetscInt     m, n, lda;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(B, &m, &n));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCall(MatDenseGetArrayWrite(B, &array));
  for (PetscInt j = 0; j < n; ++j) {
    for (PetscInt i = 0; i < m; ++i) array[i + j * lda] = rhs ? RHSValue(i, j) : -7.0;
    for (PetscInt i = m; i < lda; ++i) array[i + j * lda] = 0.0;
  }
  PetscCall(MatDenseRestoreArrayWrite(B, &array));
  if (cuda) {
    PetscCall(MatDenseCUDAGetArray(B, &array));
    PetscCallCUDA(cudaDeviceSynchronize());
    // Dense host/device transfers skip padding, so initialize it on the device too.
    PetscCallCUDA(cudaMemset2D(array + m, lda * sizeof(*array), 0, (lda - m) * sizeof(*array), n));
    PetscCall(MatDenseCUDARestoreArray(B, &array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDense(Mat B, PetscBool rhs, PetscBool cuda)
{
  const PetscScalar *array;
  PetscInt           m, n, lda;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(B, &m, &n));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCall(MatDenseGetArrayRead(B, &array));
  for (PetscInt j = 0; j < n; ++j) {
    if (rhs)
      for (PetscInt i = 0; i < m; ++i) PetscCheck(array[i + j * lda] == RHSValue(i, j), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input entry (%" PetscInt_FMT ",%" PetscInt_FMT ") changed", i, j);
    for (PetscInt i = m; i < lda; ++i) PetscCheck(array[i + j * lda] == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host padding entry (%" PetscInt_FMT ",%" PetscInt_FMT ") changed", i, j);
  }
  PetscCall(MatDenseRestoreArrayRead(B, &array));
  if (cuda) {
    PetscScalar *padding;
    PetscInt     width = lda - m;

    PetscCall(PetscMalloc1(width * n, &padding));
    PetscCall(MatDenseCUDAGetArrayRead(B, &array));
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCallCUDA(cudaMemcpy2D(padding, width * sizeof(*padding), array + m, lda * sizeof(*array), width * sizeof(*padding), n, cudaMemcpyDeviceToHost));
    PetscCall(MatDenseCUDARestoreArrayRead(B, &array));
    for (PetscInt i = 0; i < width * n; ++i) PetscCheck(padding[i] == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Device padding entry %" PetscInt_FMT " changed", i);
    PetscCall(PetscFree(padding));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolve(Mat A, Mat F, Mat reference_factor, Mat B, Mat X, Vec b, Vec x, Vec reference, PetscInt numeric, PetscBool transpose, PetscBool exact, PetscBool cuda, PetscBool inplace)
{
  Vec       column;
  PetscInt  n;
  PetscReal error, norm, tol = 1000.0 * PETSC_MACHINE_EPSILON;

  PetscFunctionBeginUser;
  PetscCall(FillDense(X, inplace, cuda));
  if (transpose) PetscCall(MatMatSolveTranspose(F, inplace ? X : B, X));
  else PetscCall(MatMatSolve(F, inplace ? X : B, X));
  PetscCall(MatGetSize(B, NULL, &n));
  for (PetscInt j = 0; j < n; ++j) {
    PetscCall(MatDenseGetColumnVecRead(B, j, &column));
    PetscCall(VecCopy(column, b));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &column));
    PetscCall(MatDenseGetColumnVecRead(X, j, &column));
    PetscCall(VecCopy(column, x));
    PetscCall(MatDenseRestoreColumnVecRead(X, j, &column));
    if (transpose) PetscCall(MatSolveTranspose(reference_factor, b, reference));
    else PetscCall(MatSolve(reference_factor, b, reference));
    PetscCall(VecNorm(reference, NORM_INFINITY, &norm));
    PetscCall(VecAXPY(reference, -1.0, x));
    PetscCall(VecNorm(reference, NORM_INFINITY, &error));
    PetscCheck(error <= tol * PetscMax(norm, 1.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s column %" PetscInt_FMT " differs from CPU scalar solve by %g after numeric factorization %" PetscInt_FMT " with %" PetscInt_FMT " RHS", transpose ? "Transpose" : "Normal", j, (double)error, numeric, n);
    if (exact) {
      if (transpose) PetscCall(MatMultTranspose(A, x, reference));
      else PetscCall(MatMult(A, x, reference));
      PetscCall(VecAXPY(reference, -1.0, b));
      PetscCall(VecNorm(reference, NORM_INFINITY, &error));
      PetscCall(VecNorm(b, NORM_INFINITY, &norm));
      PetscCheck(error <= tol * PetscMax(norm, 1.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " has residual %g", j, (double)error);
    }
  }
  PetscCall(CheckDense(B, PETSC_TRUE, cuda));
  PetscCall(CheckDense(X, PETSC_FALSE, cuda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat           A, F, B, X, Acpu, Fcpu;
  Vec           b, x, reference;
  IS            row, col;
  MatFactorInfo info;
  MatFactorType factor = MAT_FACTOR_LU;
  PetscInt      n = 9, rows[9], cols[9], widths[] = {1, 16, 1, 16}, nwidths = PETSC_STATIC_ARRAY_LENGTH(widths);
  PetscMPIInt   size;
  PetscBool     permute = PETSC_FALSE, dense_cpu = PETSC_FALSE, bind_cpu = PETSC_FALSE, cuda, symmetric, exact;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a sequential test");
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-factor_type", MatFactorTypes, (PetscEnum *)&factor, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-permute", &permute, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dense_cpu", &dense_cpu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bind_cpu", &bind_cpu, NULL));
  cuda      = (PetscBool)(!dense_cpu && !bind_cpu);
  symmetric = (PetscBool)(factor == MAT_FACTOR_CHOLESKY || factor == MAT_FACTOR_ICC);
  exact     = (PetscBool)(factor == MAT_FACTOR_LU || factor == MAT_FACTOR_CHOLESKY);
  for (PetscInt i = 0; i < n; ++i) {
    rows[i] = permute ? (2 * i + 1) % n : i;
    cols[i] = permute && !symmetric ? (4 * i + 2) % n : rows[i];
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, rows, PETSC_COPY_VALUES, &row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, cols, PETSC_COPY_VALUES, &col));
  PetscCall(MatCreateSeqAIJCUSPARSE(PETSC_COMM_SELF, n, n, n, NULL, &A));
  for (PetscInt i = 0; i < n; ++i) PetscCall(MatSetValue(A, i, i, 0.0, INSERT_VALUES));
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      PetscScalar value;

      if (i != j && PetscAbsInt(i - j) != 1 && PetscAbsInt(i - j) != 3) continue;
      value = i == j ? 6.0 + 0.25 * i : symmetric ? -0.2 : i > j ? -0.08 * (j + 1) : 0.12 * (i + 1);
#if PetscDefined(USE_COMPLEX)
      if (!symmetric && i != j) value += 0.05 * (i - j) * PETSC_i;
#endif
      PetscCall(MatSetValue(A, rows[i], cols[j], value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, symmetric));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &reference));
  PetscCall(MatFactorInfoInitialize(&info));
  info.fill = 5.0;
  PetscCall(MatGetFactor(A, MATSOLVERCUSPARSE, factor, &F));
  PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &Acpu));
  PetscCall(MatGetFactor(Acpu, MATSOLVERPETSC, factor, &Fcpu));
  switch (factor) {
  case MAT_FACTOR_LU:
    PetscCall(MatLUFactorSymbolic(F, A, row, col, &info));
    PetscCall(MatLUFactorSymbolic(Fcpu, Acpu, row, col, &info));
    break;
  case MAT_FACTOR_CHOLESKY:
    PetscCall(MatCholeskyFactorSymbolic(F, A, row, &info));
    PetscCall(MatCholeskyFactorSymbolic(Fcpu, Acpu, row, &info));
    break;
  case MAT_FACTOR_ILU:
    PetscCall(MatILUFactorSymbolic(F, A, row, col, &info));
    PetscCall(MatILUFactorSymbolic(Fcpu, Acpu, row, col, &info));
    break;
  case MAT_FACTOR_ICC:
    PetscCall(MatICCFactorSymbolic(F, A, row, &info));
    PetscCall(MatICCFactorSymbolic(Fcpu, Acpu, row, &info));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported factor type %s", MatFactorTypes[factor]);
  }
  for (PetscInt numeric = 0; numeric < 2; ++numeric) {
    if (numeric) {
      PetscCall(MatScale(A, 1.25));
      PetscCall(MatShift(A, 0.75));
    }
    PetscCall(MatCopy(A, Acpu, SAME_NONZERO_PATTERN));
    if (symmetric) {
      PetscCall(MatCholeskyFactorNumeric(F, A, &info));
      PetscCall(MatCholeskyFactorNumeric(Fcpu, Acpu, &info));
    } else {
      PetscCall(MatLUFactorNumeric(F, A, &info));
      PetscCall(MatLUFactorNumeric(Fcpu, Acpu, &info));
    }
    for (PetscInt k = 0; k < nwidths; ++k) {
      PetscInt nrhs = widths[numeric ? nwidths - 1 - k : k];

      PetscCall(MatCreate(PETSC_COMM_SELF, &B));
      PetscCall(MatSetSizes(B, n, nrhs, n, nrhs));
      PetscCall(MatSetType(B, dense_cpu ? MATSEQDENSE : MATSEQDENSECUDA));
      PetscCall(MatDenseSetLDA(B, n + 2));
      PetscCall(MatSetUp(B));
      PetscCall(MatBindToCPU(B, bind_cpu));
      PetscCall(MatCreate(PETSC_COMM_SELF, &X));
      PetscCall(MatSetSizes(X, n, nrhs, n, nrhs));
      PetscCall(MatSetType(X, dense_cpu ? MATSEQDENSE : MATSEQDENSECUDA));
      PetscCall(MatDenseSetLDA(X, n + 5));
      PetscCall(MatSetUp(X));
      PetscCall(MatBindToCPU(X, bind_cpu));
      PetscCall(FillDense(B, PETSC_TRUE, cuda));
      for (PetscInt repeat = 0; repeat < 2; ++repeat) {
        for (PetscInt inplace = 0; inplace < 2; ++inplace) {
          PetscCall(CheckSolve(A, F, Fcpu, B, X, b, x, reference, numeric, PETSC_FALSE, exact, cuda, (PetscBool)inplace));
          if (!inplace) PetscCall(CheckSolve(A, F, Fcpu, B, X, b, x, reference, numeric, PETSC_TRUE, exact, cuda, PETSC_FALSE));
        }
      }
      PetscCall(MatDestroy(&X));
      PetscCall(MatDestroy(&B));
    }
  }
  PetscCall(VecDestroy(&reference));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&Fcpu));
  PetscCall(MatDestroy(&Acpu));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&A));
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cuda

  test:
    suffix: cuda
    args: -factor_type {{lu cholesky ilu icc}} -permute {{0 1}}
    output_file: output/empty.out

  test:
    suffix: host_dense
    args: -factor_type {{lu cholesky ilu icc}} -permute {{0 1}} -dense_cpu
    output_file: output/empty.out

  test:
    suffix: bound_cpu
    args: -factor_type {{lu cholesky ilu icc}} -permute 1 -bind_cpu
    output_file: output/empty.out

TEST*/
