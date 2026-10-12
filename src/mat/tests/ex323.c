static char help[] = "Tests that MAT_KEEP_NONZERO_PATTERN and MAT_IGNORE_ZERO_ENTRIES survive a new preallocation of a MATAIJ matrix.\n\n";

#include <petscmat.h>

/* Fills a tridiagonal matrix; with -coo through COO assembly, otherwise through MatSetValues(), which also adds explicit zeros
   two columns right of the diagonal that MAT_IGNORE_ZERO_ENTRIES must drop */
static PetscErrorCode FillTridiagonal(Mat A, PetscBool coo)
{
  PetscInt rstart, rend, N, n = 0;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetSize(A, &N, NULL));
  if (coo) {
    PetscInt    *coo_i, *coo_j;
    PetscScalar *v;

    PetscCall(PetscMalloc3(3 * (rend - rstart), &coo_i, 3 * (rend - rstart), &coo_j, 3 * (rend - rstart), &v));
    for (PetscInt i = rstart; i < rend; i++) {
      for (PetscInt j = PetscMax(i - 1, 0); j <= PetscMin(i + 1, N - 1); j++) {
        coo_i[n] = i;
        coo_j[n] = j;
        v[n++]   = i == j ? 2.0 : -1.0;
      }
    }
    PetscCall(MatSetPreallocationCOO(A, n, coo_i, coo_j));
    PetscCall(MatSetValuesCOO(A, v, INSERT_VALUES));
    PetscCall(PetscFree3(coo_i, coo_j, v));
  } else {
    PetscCall(MatMPIAIJSetPreallocation(A, 4, NULL, 4, NULL));
    PetscCall(MatSeqAIJSetPreallocation(A, 4, NULL));
    for (PetscInt i = rstart; i < rend; i++) {
      for (PetscInt j = PetscMax(i - 1, 0); j <= PetscMin(i + 1, N - 1); j++) PetscCall(MatSetValue(A, i, j, i == j ? 2.0 : -1.0, ADD_VALUES));
      if (i + 2 < N) PetscCall(MatSetValue(A, i, i + 2, 0.0, ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A;
  PetscInt    N   = 8, rstart, rend, row;
  PetscBool   coo = PETSC_FALSE;
  MatInfo     info;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-coo", &coo, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  /* MatSetOption() for these options requires a preallocated matrix */
  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 0, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
  PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));

  /* The new preallocation must keep both options */
  PetscCall(FillTridiagonal(A, coo));
  PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
  PetscCheck((PetscInt)info.nz_used == 3 * N - 2, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MAT_IGNORE_ZERO_ENTRIES lost: %" PetscInt_FMT " nonzeros, expected %" PetscInt_FMT, (PetscInt)info.nz_used, 3 * N - 2);

  /* Zero the first row owned by each process; MAT_KEEP_NONZERO_PATTERN keeps its entries in the nonzero structure */
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  row = rstart;
  PetscCall(MatZeroRows(A, rend > rstart ? 1 : 0, &row, 1.0, NULL, NULL));
  PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
  PetscCheck((PetscInt)info.nz_used == 3 * N - 2, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MAT_KEEP_NONZERO_PATTERN lost: %" PetscInt_FMT " nonzeros, expected %" PetscInt_FMT, (PetscInt)info.nz_used, 3 * N - 2);

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: {{1 2}}
      output_file: output/empty.out

      test:
         suffix: setvalues

      test:
         suffix: coo
         args: -coo

TEST*/
