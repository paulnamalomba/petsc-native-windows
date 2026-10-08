static const char help[] = "Tests repeated sparse-dense products with shared sparse operands and replaced dense operands.\n\n";

#include <petscmat.h>

static PetscScalar DenseValue(PetscInt row, PetscInt column, PetscInt shift)
{
  PetscScalar value = (row + 1.0) / (column + 2.0) + 0.125 * (column + shift);

#if PetscDefined(USE_COMPLEX)
  value += 0.0625 * (row - column + shift) * PETSC_i;
#endif
  return value;
}

static PetscErrorCode CreateDense(PetscInt m, PetscInt n, PetscInt lda, PetscBool cuda, Mat *B)
{
  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, B));
  PetscCall(MatSetSizes(*B, m, PETSC_DECIDE, PETSC_DETERMINE, n));
  PetscCall(MatSetType(*B, cuda ? MATDENSECUDA : MATDENSE));
  PetscCall(MatDenseSetLDA(*B, lda));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillDense(Mat B, PetscInt shift)
{
  PetscScalar *array;
  PetscInt     m, n, lda, start;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(B, &m, NULL));
  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(MatGetOwnershipRange(B, &start, NULL));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCall(MatDenseGetArrayWrite(B, &array));
  for (PetscInt j = 0; j < n; ++j) {
    for (PetscInt i = 0; i < m; ++i) array[i + j * lda] = DenseValue(start + i, j, shift);
    for (PetscInt i = m; i < lda; ++i) array[i + j * lda] = 0.0;
  }
  PetscCall(MatDenseRestoreArrayWrite(B, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateProduct(Mat A, Mat B, Mat C)
{
  PetscFunctionBeginUser;
  PetscCall(MatProductCreateWithMat(A, B, NULL, C));
  PetscCall(MatProductSetType(C, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckProduct(Mat A, Mat B, Mat C, PetscInt shift)
{
  Mat                Acpu;
  Vec                x, y, column;
  const PetscScalar *array;
  PetscInt           m, n, lda, start;
  PetscReal          norm, error;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Acpu));
  PetscCall(MatCreateVecs(Acpu, &x, &y));
  PetscCall(MatGetSize(B, NULL, &n));
  for (PetscInt j = 0; j < n; ++j) {
    PetscCall(MatDenseGetColumnVecRead(B, j, &column));
    PetscCall(VecCopy(column, x));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &column));
    PetscCall(MatMult(Acpu, x, y));
    PetscCall(VecNorm(y, NORM_INFINITY, &norm));
    PetscCall(MatDenseGetColumnVecRead(C, j, &column));
    PetscCall(VecAXPY(y, -1.0, column));
    PetscCall(MatDenseRestoreColumnVecRead(C, j, &column));
    PetscCall(VecNorm(y, NORM_INFINITY, &error));
    PetscCheck(error <= 1000.0 * PETSC_MACHINE_EPSILON * PetscMax(norm, 1.0), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " differs from scalar product by %g", j, (double)error);
  }
  PetscCall(MatGetLocalSize(B, &m, NULL));
  PetscCall(MatGetOwnershipRange(B, &start, NULL));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCall(MatDenseGetArrayRead(B, &array));
  for (PetscInt j = 0; j < n; ++j)
    for (PetscInt i = 0; i < m; ++i) PetscCheck(array[i + j * lda] == DenseValue(start + i, j, shift), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input entry (%" PetscInt_FMT ",%" PetscInt_FMT ") changed", i, j);
  PetscCall(MatDenseRestoreArrayRead(B, &array));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Acpu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         A, Anew, B[2][3], C[2], current[2];
  PetscInt    m = 6, M, start, end, widths[] = {16, 7};
  PetscMPIInt rank, size;
  PetscBool   cuda = PETSC_FALSE, padded = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cuda", &cuda, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-padded", &padded, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, m, m, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, cuda ? MATAIJCUSPARSE : MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 4, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 4, NULL, 4, NULL));
  PetscCall(MatGetSize(A, &M, NULL));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt i = start; i < end; ++i) {
    PetscCall(MatSetValue(A, i, i, 4.0 + 0.125 * i, INSERT_VALUES));
    if (i) PetscCall(MatSetValue(A, i, i - 1, -0.5, INSERT_VALUES));
    if (i + 1 < M) PetscCall(MatSetValue(A, i, i + 1, -0.75, INSERT_VALUES));
    PetscCall(MatSetValue(A, i, (i + 3) % M, 0.25, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Anew));
  PetscCall(MatScale(Anew, 0.875));
  PetscCall(MatShift(Anew, 0.25));
  for (PetscInt k = 0; k < 2; ++k) {
    for (PetscInt j = 0; j < 3; ++j) {
      PetscInt lda = m + (padded ? 2 + k + (!rank && j == 2 ? 3 : 0) : 0);

      PetscCall(CreateDense(m, widths[k], lda, cuda, &B[k][j]));
      PetscCall(FillDense(B[k][j], j));
    }
    PetscCall(CreateDense(m, widths[k], m + (padded ? 5 - k : 0), cuda, &C[k]));
    PetscCall(CreateProduct(A, B[k][0], C[k]));
    current[k] = A;
  }
  for (PetscInt stage = 0; stage < 7; ++stage) {
    PetscInt which = PetscMin(stage, 2);

    if (stage == 3) {
      PetscCall(MatScale(A, 1.25));
      PetscCall(MatShift(A, 0.5));
    } else if (stage == 4) current[0] = Anew;
    else if (stage == 5) current[1] = Anew;
    else if (stage == 6) {
      PetscCall(MatProductClear(C[0]));
      PetscCall(CreateProduct(Anew, B[0][which], C[0]));
    }
    for (PetscInt k = 0; k < 2; ++k) {
      // The MPI product scatter is set up for a fixed input leading dimension.
      if (stage == 2 && padded && size > 1) {
        PetscCall(MatProductClear(C[k]));
        PetscCall(CreateProduct(current[k], B[k][which], C[k]));
      } else PetscCall(MatProductReplaceMats(current[k], B[k][which], NULL, C[k]));
    }
    // Interleave products sharing A so that each retains its own sparse-dense workspace.
    for (PetscInt repeat = 0; repeat < 3; ++repeat)
      for (PetscInt k = 0; k < 2; ++k) {
        PetscCall(MatProductNumeric(C[k]));
        PetscCall(CheckProduct(current[k], B[k][which], C[k], which));
      }
  }
  for (PetscInt k = 0; k < 2; ++k) {
    PetscCall(MatDestroy(&C[k]));
    for (PetscInt j = 0; j < 3; ++j) PetscCall(MatDestroy(&B[k][j]));
  }
  PetscCall(MatDestroy(&Anew));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cpu
    nsize: {{1 2 3}}
    args: -padded {{0 1}} -matproduct_batch_size {{0 5}}
    output_file: output/empty.out

  test:
    suffix: cuda
    requires: cuda
    nsize: {{1 2 3}}
    args: -cuda -padded {{0 1}} -matproduct_batch_size {{0 5}}
    output_file: output/empty.out

  test:
    suffix: cuda_csr_alg1
    requires: cuda
    nsize: {{1 2}}
    args: -cuda -padded -matproduct_batch_size {{0 5}} -mat_cusparse_spmm_alg csr_alg1
    output_file: output/empty.out

TEST*/
