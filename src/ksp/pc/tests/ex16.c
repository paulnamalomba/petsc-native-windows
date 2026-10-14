static const char help[] = "Tests cached ASM block applications with padded dense matrices and changing layouts.\n\n";

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
    /* Dense host/device transfers skip the padding. */
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

static PetscErrorCode CheckApply(PC pc, Mat B, Mat X, Vec reference, PetscBool transpose, PetscBool cuda, PetscBool hostoutput)
{
  Vec       b, x;
  PetscInt  n;
  PetscReal error, norm;

  PetscFunctionBeginUser;
  PetscCall(FillDense(X, PETSC_FALSE, cuda && !hostoutput));
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
    PetscCheck(error <= 500.0 * PETSC_MACHINE_EPSILON * PetscMax(norm, 1.0), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " differs from vector application by %g", j, (double)error);
    PetscCall(MatDenseRestoreColumnVecRead(X, j, &x));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &b));
  }
  PetscCall(CheckDense(B, PETSC_TRUE, cuda));
  PetscCall(CheckDense(X, PETSC_FALSE, cuda && !hostoutput));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckWidths(PC pc, Mat A, Vec reference, PetscBool cuda, PetscBool hostoutput, PetscBool boundoutput)
{
  Mat         B[2], X[2];
  PetscInt    m, n, widths[] = {1, 16, 16, 1, 3, 16}, nwidths = PETSC_STATIC_ARRAY_LENGTH(widths);
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  for (PetscInt pass = 0; pass < nwidths; ++pass) {
    for (PetscInt pair = 0; pair < 2; ++pair) {
      PetscCall(MatCreate(PETSC_COMM_WORLD, &B[pair]));
      PetscCall(MatSetSizes(B[pair], m, PETSC_DECIDE, n, widths[pass]));
      PetscCall(MatSetType(B[pair], cuda ? MATDENSECUDA : MATDENSE));
      PetscCall(MatDenseSetLDA(B[pair], m + 1 + 2 * pair * (rank == 0) + 2 * (pass % 2)));
      PetscCall(MatSetUp(B[pair]));
      PetscCall(MatCreate(PETSC_COMM_WORLD, &X[pair]));
      PetscCall(MatSetSizes(X[pair], m, PETSC_DECIDE, n, widths[pass]));
      PetscCall(MatSetType(X[pair], cuda && !hostoutput ? MATDENSECUDA : MATDENSE));
      PetscCall(MatDenseSetLDA(X[pair], m + 3 + 2 * pair * (rank == 0) + 2 * (pass % 2)));
      PetscCall(MatSetUp(X[pair]));
      PetscCall(MatBindToCPU(X[pair], boundoutput));
      PetscCall(FillDense(B[pair], PETSC_TRUE, cuda));
    }
    /* Keep both pairs alive so changing their pointers cannot reuse an allocation. */
    for (PetscInt repeat = 0; repeat < 2; ++repeat) {
      for (PetscInt pair = 0; pair < 2; ++pair) {
        PetscCall(CheckApply(pc, B[pair], X[pair], reference, PETSC_FALSE, cuda, hostoutput || boundoutput));
        PetscCall(CheckApply(pc, B[pair], X[pair], reference, PETSC_TRUE, cuda, hostoutput || boundoutput));
      }
    }
    for (PetscInt pair = 0; pair < 2; ++pair) {
      PetscCall(MatDestroy(&X[pair]));
      PetscCall(MatDestroy(&B[pair]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat       A;
  PC        pc;
  Vec       reference;
  PetscInt  start, end, n = 24, bs = 1, custom = 0;
  PetscBool cuda = PETSC_FALSE, hostoutput = PETSC_FALSE, boundoutput = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cuda", &cuda, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-custom_subdomains", &custom, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-host_output", &hostoutput, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bound_output", &boundoutput, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSetType(A, cuda ? MATAIJCUSPARSE : MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt i = start; i < end; ++i) {
    PetscCall(MatSetValue(A, i, i, 4.0 + i / 32.0, INSERT_VALUES));
    if (i) PetscCall(MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(A, i, i + 1, -0.5, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A, &reference, NULL));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCASM));
  PetscCall(PCASMSetOverlap(pc, 0));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  for (PetscInt setup = 0; setup < 3; ++setup) {
    if (setup == 1) {
      PetscCall(MatScale(A, 1.5));
      PetscCall(PCSetOperators(pc, A, A));
    } else if (setup == 2) {
      PetscCall(PCReset(pc));
      PetscCall(PCASMSetOverlap(pc, 2));
      PetscCall(PCSetOperators(pc, A, A));
    }
    if (custom && setup != 1) {
      IS          outer, inner;
      PetscInt   *indices, count = 0, outercount = 0;
      PetscMPIInt rank, size;

      PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
      PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
      PetscCall(PetscMalloc1(n, &indices));
      if (custom != 2) {
        for (PetscInt i = 0; i < n; ++i) indices[outercount++] = n - 1 - i;
      } else {
        for (PetscInt i = end - 1; i >= start; i -= 2) indices[outercount++] = i;
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, outercount, indices, PETSC_COPY_VALUES, &outer));
      for (PetscInt i = 0; i < outercount; ++i)
        if ((custom != 2 && indices[i] % size == rank) || (custom == 2 && i % 2 == 0)) indices[count++] = indices[i];
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, count, indices, PETSC_OWN_POINTER, &inner));
      PetscCall(PCASMSetOverlap(pc, 0));
      PetscCall(PCASMSetSortIndices(pc, PETSC_FALSE));
      PetscCall(PCASMSetLocalSubdomains(pc, 1, &outer, custom == 3 && rank % 2 ? NULL : &inner));
      PetscCall(ISDestroy(&outer));
      PetscCall(ISDestroy(&inner));
    }
    PetscCall(PCSetUp(pc));
    PetscCall(CheckWidths(pc, A, reference, cuda, hostoutput, boundoutput));
  }
  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&reference));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2 3}}
    args: -pc_asm_type {{basic restrict interpolate none}shared output} -sub_pc_type jacobi -bs {{1 2}shared output}

    test:
      suffix: cpu

    test:
      suffix: cuda
      requires: cuda
      args: -cuda

    test:
      suffix: cpu_custom
      args: -custom_subdomains {{1 2 3}shared output} -sub_ksp_matsolve_batch_size 4

    test:
      suffix: cuda_custom
      requires: cuda
      args: -cuda -custom_subdomains {{1 2 3}shared output} -sub_ksp_matsolve_batch_size 4

    test:
      suffix: cuda_hostoutput
      requires: cuda
      args: -cuda -host_output

    test:
      suffix: cuda_boundoutput
      requires: cuda
      args: -cuda -bound_output

TEST*/
