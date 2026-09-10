static const char help[] = "Tests custom PCHPDDM deflation and ASM scaling against the explicit two-level operator.\n\n";

#include <petscksp.h>

static PetscScalar TestScalar(PetscReal real, PetscReal imag)
{
#if PetscDefined(USE_COMPLEX)
  return PetscCMPLX(real, imag);
#else
  (void)imag;
  return real;
#endif
}

static PetscScalar Weight(PetscInt rank, PetscInt row)
{
  const PetscReal first[] = {1.0, 1.0, 1.0, 0.65, 0.35, 0.0, 0.0, 0.0};

  return rank ? 1.0 - first[row] : first[row];
}

static PetscScalar Mode(PetscInt rank, PetscInt row)
{
  return TestScalar(1.0 + 0.15 * row, 0.3 * (rank + 1) - 0.05 * row);
}

static PetscErrorCode CreateOperator(Mat *A)
{
  PetscInt start, end;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 4, 8, 8, 3, NULL, 2, NULL, A));
  PetscCall(MatGetOwnershipRange(*A, &start, &end));
  for (PetscInt row = start; row < end; ++row) {
    PetscCall(MatSetValue(*A, row, row, TestScalar(3.0 + 0.1 * row, 0.25), INSERT_VALUES));
    if (row) PetscCall(MatSetValue(*A, row, row - 1, TestScalar(-0.9, 0.07), INSERT_VALUES));
    if (row < 7) PetscCall(MatSetValue(*A, row, row + 1, TestScalar(-0.3, -0.12), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ModifySubmatrices(PC pc, PetscInt n, const IS rows[], const IS cols[], Mat sub[], void *ctx)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < n; ++i) PetscCall(MatShift(sub[i], TestScalar(0.35, 0.2)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDeflation(PetscBool empty_rank0, IS *is, Mat *U, Vec *D)
{
  PetscMPIInt  rank;
  PetscInt     idx[6];
  PetscScalar *u, *d;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  /* Deliberately use the reverse of PCASM's final sorted ordering. */
  for (PetscInt i = 0; i < 6; ++i) idx[i] = 2 * rank + 5 - i;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 6, idx, PETSC_COPY_VALUES, is));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 6, empty_rank0 && !rank ? 0 : 1, NULL, U));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 6, D));
  PetscCall(MatDenseGetArray(*U, &u));
  PetscCall(VecGetArray(*D, &d));
  for (PetscInt i = 0; i < 6; ++i) {
    d[i] = Weight(rank, idx[i]);
    if (!empty_rank0 || rank) u[i] = Mode(rank, idx[i]);
  }
  PetscCall(VecRestoreArray(*D, &d));
  PetscCall(MatDenseRestoreArray(*U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateReference(Mat A, PetscBool empty_rank0, PetscBool reversed, Mat *T)
{
  Mat                B, Z, R, AZ, E, Q, AB, QE, *sub;
  KSP                ksp;
  PC                 subpc;
  Vec                x, y;
  IS                 is;
  PetscMPIInt        rank;
  PetscInt           idx[6], start, end;
  const PetscScalar *values;
  PetscScalar        column[6];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 4, 8, 8, 4, NULL, 4, NULL, &B));
  for (PetscInt i = 0; i < 6; ++i) idx[i] = 2 * rank + i;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 6, idx, PETSC_COPY_VALUES, &is));
  PetscCall(MatCreateSubMatrices(A, 1, &is, &is, MAT_INITIAL_MATRIX, &sub));
  PetscCall(MatShift(sub[0], TestScalar(0.35, 0.2)));
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPSetOperators(ksp, sub[0], sub[0]));
  PetscCall(KSPGetPC(ksp, &subpc));
  PetscCall(PCSetType(subpc, PCLU));
  PetscCall(MatCreateVecs(sub[0], &x, &y));
  for (PetscInt j = 0; j < 6; ++j) {
    PetscCall(VecSet(x, 0.0));
    PetscCall(VecSetValue(x, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    PetscCall(KSPSolve(ksp, x, y));
    PetscCall(VecGetArrayRead(y, &values));
    for (PetscInt i = 0; i < 6; ++i) column[i] = Weight(rank, idx[i]) * values[i];
    PetscCall(MatSetValues(B, 6, idx, 1, idx + j, column, ADD_VALUES));
    PetscCall(VecRestoreArrayRead(y, &values));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroySubMatrices(1, &sub));
  PetscCall(ISDestroy(&is));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, empty_rank0 && !rank ? 0 : 1, 8, empty_rank0 ? 1 : 2, 1, NULL, 1, NULL, &Z));
  PetscCall(MatGetOwnershipRange(Z, &start, &end));
  for (PetscInt i = start; i < end; ++i)
    for (PetscInt j = empty_rank0 ? 1 : 0; j < 2; ++j) PetscCall(MatSetValue(Z, i, j - (empty_rank0 ? 1 : 0), Weight(j, i) * Mode(j, i), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Z, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Z, MAT_FINAL_ASSEMBLY));
  PetscCall(MatHermitianTranspose(Z, MAT_INITIAL_MATRIX, &R));
  PetscCall(MatMatMult(A, Z, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AZ));
  PetscCall(MatMatMult(R, AZ, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &E));
  PetscCall(MatConvert(R, MATDENSE, MAT_INPLACE_MATRIX, &R));
  PetscCall(MatDuplicate(R, MAT_DO_NOT_COPY_VALUES, &QE));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPSetOperators(ksp, E, E));
  PetscCall(KSPGetPC(ksp, &subpc));
  PetscCall(PCSetType(subpc, PCLU));
  PetscCall(PCFactorSetMatSolverType(subpc, MATSOLVERMUMPS));
  PetscCall(KSPMatSolve(ksp, R, QE));
  PetscCall(MatMatMult(Z, QE, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Q));
  /* Deflated: Q + B (I - A Q). Reversed: B + Q (I - A B). */
  PetscCall(MatMatMult(A, reversed ? B : Q, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AB));
  PetscCall(MatScale(AB, -1.0));
  PetscCall(MatShift(AB, 1.0));
  PetscCall(MatMatMult(reversed ? Q : B, AB, MAT_INITIAL_MATRIX, PETSC_DETERMINE, T));
  PetscCall(MatConvert(B, MATDENSE, MAT_INPLACE_MATRIX, &B));
  PetscCall(MatAXPY(*T, 1.0, reversed ? B : Q, DIFFERENT_NONZERO_PATTERN));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&AZ));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&AB));
  PetscCall(MatDestroy(&QE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckVector(Vec actual, Vec expected)
{
  PetscReal error, norm;

  PetscFunctionBeginUser;
  PetscCall(VecNorm(expected, NORM_INFINITY, &norm));
  PetscCall(VecAXPY(expected, -1.0, actual));
  PetscCall(VecNorm(expected, NORM_INFINITY, &error));
  PetscCheck(error <= 1000.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, norm), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Weighted HPDDM error %g exceeds tolerance", (double)error);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckApplications(PC pc, Mat B)
{
  Vec      x, y, expected;
  PetscInt N, m, start, end;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(B, &N, NULL));
  PetscCall(MatGetLocalSize(B, &m, NULL));
  PetscCall(MatCreateVecs(B, &x, &y));
  PetscCall(VecDuplicate(y, &expected));
  PetscCall(VecGetOwnershipRange(x, &start, &end));
  for (PetscInt j = 0; j < N; ++j) {
    PetscCall(VecSet(x, 0.0));
    if (start <= j && j < end) PetscCall(VecSetValue(x, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    PetscCall(PCApply(pc, x, y));
    PetscCall(MatMult(B, x, expected));
    PetscCall(CheckVector(y, expected));
#if !PetscDefined(USE_COMPLEX)
    PetscCall(PCApplyTranspose(pc, x, y));
    PetscCall(MatMultTranspose(B, x, expected));
    PetscCall(CheckVector(y, expected));
#endif
  }
  {
    Mat X, Y;

    PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, N, 3, NULL, &X));
    PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
    for (PetscInt j = 0; j < 3; ++j) {
      Vec          column;
      PetscScalar *values;

      PetscCall(MatDenseGetColumnVecWrite(X, j, &column));
      PetscCall(VecGetArray(column, &values));
      for (PetscInt i = 0; i < m; ++i) values[i] = TestScalar(0.1 * (start + i + 1) * (j + 1), 0.2 * (j - start - i));
      PetscCall(VecRestoreArray(column, &values));
      PetscCall(MatDenseRestoreColumnVecWrite(X, j, &column));
    }
    for (PetscInt transpose = 0; transpose < (PetscDefined(USE_COMPLEX) ? 1 : 2); ++transpose) {
      if (transpose) PetscCall(PCMatApplyTranspose(pc, X, Y));
      else PetscCall(PCMatApply(pc, X, Y));
      for (PetscInt j = 0; j < 3; ++j) {
        Vec xc, yc;

        PetscCall(MatDenseGetColumnVecRead(X, j, &xc));
        PetscCall(MatDenseGetColumnVecRead(Y, j, &yc));
        if (transpose) PetscCall(MatMultTranspose(B, xc, expected));
        else PetscCall(MatMult(B, xc, expected));
        PetscCall(CheckVector(yc, expected));
        PetscCall(MatDenseRestoreColumnVecRead(Y, j, &yc));
        PetscCall(MatDenseRestoreColumnVecRead(X, j, &xc));
      }
    }
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&Y));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PC                          pc;
  Mat                         A, U, T;
  IS                          is;
  Vec                         D;
  PetscMPIInt                 size;
  PetscBool                   empty_rank0 = PETSC_FALSE;
  PCHPDDMCoarseCorrectionType correction;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank0", &empty_rank0, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Run with two processes");
  PetscCall(CreateOperator(&A));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCHPDDM));
  for (PetscInt pass = 0; pass < 3; ++pass) {
    if (pass != 1) {
      if (pass) PetscCall(PCReset(pc));
      PetscCall(CreateDeflation(empty_rank0, &is, &U, &D));
      PetscCall(PCHPDDMSetDeflationMat(pc, is, U));
      PetscCall(PCHPDDMSetDeflationMatScaling(pc, D));
      if (!pass) {
        PetscCall(PCHPDDMSetDeflationMatScaling(pc, NULL));
        PetscCall(PCHPDDMSetDeflationMatScaling(pc, D));
      }
      PetscCall(VecDestroy(&D)); /* PC retains the vector, independently of the caller. */
      PetscCall(ISDestroy(&is));
      PetscCall(MatDestroy(&U));
    } else PetscCall(MatScale(A, 1.1)); /* rebuild coarse and fine operators with the same PoU */
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetModifySubMatrices(pc, ModifySubmatrices, NULL));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PCSetUp(pc));
    PetscCall(PCHPDDMGetCoarseCorrectionType(pc, &correction));
    PetscCall(CreateReference(A, empty_rank0, (PetscBool)(correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED_REVERSED), &T));
    PetscCall(CheckApplications(pc, T));
    PetscCall(MatDestroy(&T));
  }
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    requires: hpddm slepc mumps defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -pc_hpddm_define_subdomains true -pc_hpddm_coarse_correction {{deflated deflated_reversed}} -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_coarse_mat_type aij -empty_rank0 {{false true}}
    output_file: output/empty.out

TEST*/
