/*
  Defines the matrix-matrix product C = A^T * B of an MPIAIJ matrix A and an MPIDENSE matrix B
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>
#include <petsc/private/sfimpl.h>

/* As in MatMultTranspose_MPIAIJ(): the local block of C is A_d^T B_loc, and A_o^T B_loc, computed into workC, is added to the owning
   processes with a reduction over strided PetscSFs built from the MatMult() scatter, so device data stays on the device */
typedef struct {
  Mat      workC; /* A_o^T times a batch of columns of B_loc */
  PetscSF  sf[2]; /* reductions of ncols[0] = workC->cmap->n columns and of the ncols[1] remainder columns of the last batch */
  PetscInt ncols[2];
  PetscInt clda; /* leading dimension of C used by the reductions */
} MPIAIJ_MPIDense_AtB;

static PetscErrorCode MatProductCtxDestroy_MPIAIJ_MPIDense_AtB(PetscCtxRt ctx)
{
  MPIAIJ_MPIDense_AtB *data = *(MPIAIJ_MPIDense_AtB **)ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&data->workC));
  PetscCall(PetscSFDestroy(&data->sf[0]));
  PetscCall(PetscSFDestroy(&data->sf[1]));
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat, Mat, Mat);

PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIDense(Mat A, Mat B, PetscReal fill, Mat C)
{
  Mat_MPIAIJ          *aij = (Mat_MPIAIJ *)A->data;
  MPIAIJ_MPIDense_AtB *data;
  VecScatter           ctx = aij->Mvctx;
  PetscInt             nz = aij->B->cmap->n, BN = B->cmap->N, Bbn, numBb, ncols;
  Mat                  bloc, cloc;
  PetscBool            cisdense;

  PetscFunctionBegin;
  MatCheckProduct(C, 4);
  PetscCheck(C->product->data == NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Extra product struct not empty");

  /* create output dense matrix C = A^T*B */
  PetscCall(MatSetSizes(C, A->cmap->n, B->cmap->n, A->cmap->N, BN));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)C, MATMPIDENSE, &cisdense));
  if (!cisdense) {
    PetscCall(MatSetType(C, ((PetscObject)B)->type_name));
    PetscCall(MatSetVecType(C, B->defaultvectype));
  }
  PetscCall(MatSetUp(C));
  bloc = ((Mat_MPIDense *)B->data)->A;
  cloc = ((Mat_MPIDense *)C->data)->A;

  PetscCall(PetscNew(&data));
  PetscCall(MatDenseGetLDA(C, &data->clda));
  PetscCall(MatMPIDenseGetBatchSize_Private(nz, A->rmap->n, B, C, &Bbn, &numBb));
  ncols = Bbn > 0 ? Bbn : BN;
  PetscCheck(ctx->vscat.bs <= 1 || data->clda % ctx->vscat.bs == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Leading dimension %" PetscInt_FMT " of the dense matrix must be a multiple of the block size %" PetscInt_FMT, data->clda, ctx->vscat.bs);
  if (ncols > 0) {
    data->ncols[0] = ncols;
    PetscCall(PetscSFCreateStridedSF(ctx, ncols, data->clda, nz, &data->sf[0]));
    if (numBb > 0 && BN % Bbn != 0) {
      data->ncols[1] = BN % Bbn;
      PetscCall(PetscSFCreateStridedSF(ctx, BN % Bbn, data->clda, nz, &data->sf[1]));
    }
    PetscCall(MatCreate(PETSC_COMM_SELF, &data->workC));
    PetscCall(MatSetSizes(data->workC, nz, ncols, nz, ncols));
    PetscCall(MatSetType(data->workC, ((PetscObject)cloc)->type_name));
    PetscCall(MatSetUp(data->workC));
  }
  PetscCall(MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  /* the local block of B is left alone: it may hold the product of which B is the result */
  PetscCall(MatProductClear(cloc));
  if (nz > 0 && ncols > 0) {
    Mat Bb, workC1;

    PetscCall(MatDenseGetSubMatrix(bloc, PETSC_DECIDE, PETSC_DECIDE, 0, ncols, &Bb));
    PetscCall(MatProductCreateWithMat(aij->B, Bb, NULL, data->workC));
    PetscCall(MatProductSetType(data->workC, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(data->workC));
    PetscCall(MatProductSymbolic(data->workC));
    PetscCall(MatDenseRestoreSubMatrix(bloc, &Bb));
    if (numBb > 0 && BN % Bbn != 0) {
      /* the last batch is smaller, set up its product on the sub-matrices the numeric uses */
      PetscCall(MatDenseGetSubMatrix(bloc, PETSC_DECIDE, PETSC_DECIDE, 0, BN % Bbn, &Bb));
      PetscCall(MatDenseGetSubMatrix(data->workC, PETSC_DECIDE, PETSC_DECIDE, 0, BN % Bbn, &workC1));
      PetscCall(MatProductCreateWithMat(aij->B, Bb, NULL, workC1));
      PetscCall(MatProductSetType(workC1, MATPRODUCT_AtB));
      PetscCall(MatProductSetFromOptions(workC1));
      PetscCall(MatProductSymbolic(workC1));
      PetscCall(MatDenseRestoreSubMatrix(data->workC, &workC1));
      PetscCall(MatDenseRestoreSubMatrix(bloc, &Bb));
    }
  }
  if (ncols > 0) {
    PetscCall(MatProductCreateWithMat(aij->A, bloc, NULL, cloc));
    PetscCall(MatProductSetType(cloc, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(cloc));
    PetscCall(MatProductSymbolic(cloc));
  }
  C->product->data    = data;
  C->product->destroy = MatProductCtxDestroy_MPIAIJ_MPIDense_AtB;

  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIDense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat A, Mat B, Mat C)
{
  Mat_MPIAIJ          *aij  = (Mat_MPIAIJ *)A->data;
  Mat                  bloc = ((Mat_MPIDense *)B->data)->A, cloc = ((Mat_MPIDense *)C->data)->A;
  MPIAIJ_MPIDense_AtB *data;
  PetscInt             BN = B->cmap->N, nz = aij->B->cmap->n, n, clda;
  PetscBool            clear = PETSC_FALSE, flg;

  PetscFunctionBegin;
  MatCheckProduct(C, 3);
  data = (MPIAIJ_MPIDense_AtB *)C->product->data;
  PetscCheck(data != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing product struct");
  if (BN == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(data->workC->rmap->n == nz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of rows of workC %" PetscInt_FMT " not equal to columns of off-diagonal block %" PetscInt_FMT, data->workC->rmap->n, nz);
  PetscCall(MatDenseGetLDA(C, &clda));
  PetscCheck(clda == data->clda, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot reuse an output matrix with lda %" PetscInt_FMT " != %" PetscInt_FMT, clda, data->clda);
  if (PetscDefined(HAVE_CUPM)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)C, MATMPIDENSE, &flg));
    if (flg) PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &flg));
    clear = (PetscBool)!flg; /* if either A or C is a device Mat, make sure MatProductClear() is called */
  }
  /* A_d^T B_loc sets every entry of the local block of C */
  PetscCall(MatMPIAIJ_MPIDenseProductNumeric_Private(aij->A, bloc, cloc, MATPRODUCT_AtB, clear));
  n = data->workC->cmap->n;
  for (PetscInt i = 0; i < BN; i += n) {
    PetscInt           cols = PetscMin(n, BN - i), k = cols == data->ncols[0] ? 0 : 1;
    PetscSF            sf    = data->sf[k];
    Mat                workC = data->workC, Bb;
    const PetscScalar *warray;
    PetscScalar       *carray;
    PetscMemType       wmtype, cmtype;

    PetscCheck(sf != NULL && cols == data->ncols[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "No reduction set up for %" PetscInt_FMT " columns", cols);
    if (cols != n) PetscCall(MatDenseGetSubMatrix(data->workC, PETSC_DECIDE, PETSC_DECIDE, 0, cols, &workC));
    /* A_o^T times a batch of columns of B_loc, for rows of C owned by other processes */
    if (nz > 0) {
      PetscCall(MatDenseGetSubMatrix(bloc, PETSC_DECIDE, PETSC_DECIDE, i, i + cols, &Bb));
      PetscCall(MatMPIAIJ_MPIDenseProductNumeric_Private(aij->B, Bb, workC, MATPRODUCT_AtB, clear));
      PetscCall(MatDenseRestoreSubMatrix(bloc, &Bb));
    }
    /* every process takes part in the reduction, even one with no off-diagonal columns */
    PetscCall(MatDenseGetArrayReadAndMemType(workC, &warray, &wmtype));
    PetscCall(MatDenseGetArrayAndMemType(cloc, &carray, &cmtype));
    PetscCall(PetscSFReduceWithMemTypeBegin(sf, sf->vscat.unit, wmtype, warray, cmtype, carray + i * clda, MPIU_SUM));
    PetscCall(PetscSFReduceEnd(sf, sf->vscat.unit, warray, carray + i * clda, MPIU_SUM));
    PetscCall(MatDenseRestoreArrayAndMemType(cloc, &carray));
    PetscCall(MatDenseRestoreArrayReadAndMemType(workC, &warray));
    if (cols != n) PetscCall(MatDenseRestoreSubMatrix(data->workC, &workC));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
