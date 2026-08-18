const char help[] = "Demonstration of elastic net regularization (https://en.wikipedia.org/wiki/Elastic_net_regularization) using TAO";

#include <petsctao.h>

static PetscErrorCode FillMatDeterministic(Mat M, PetscReal shift)
{
  const PetscScalar weights[] = {2.0, -0.3, 0.2, -0.1};
  PetscInt          rstart, rend, N;
  PetscInt         *cols;
  PetscScalar      *vals;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(M, &rstart, &rend));
  PetscCall(MatGetSize(M, NULL, &N));
  PetscCall(PetscMalloc2(N, &cols, N, &vals));
  for (PetscInt j = 0; j < N; j++) cols[j] = j;
  for (PetscInt i = rstart; i < rend; i++) {
    PetscCall(PetscArrayzero(vals, N));
    for (PetscInt j = 0; j < 4; j++) vals[(i + j) % N] += weights[j];
    vals[i % N] += shift;
    PetscCall(MatSetValues(M, 1, &i, N, cols, vals, INSERT_VALUES));
  }
  PetscCall(PetscFree2(cols, vals));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillVecDeterministic(Vec v, PetscReal shift)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(v, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(VecSetValue(v, i, shift + 0.1 * (i % 11 - 5), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Check that the objects TAO actually computes with live in device memory.
static PetscErrorCode CheckDeviceMemType(Tao tao, Vec x)
{
  const PetscScalar *array;
  PetscMemType       memtype;
  MPI_Comm           comm;
  Mat                H;
  MatType            H_type;
  TaoTerm            sum;
  Vec                params, p;
  PetscInt           n_terms;
  PetscBool          is_dense;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));

  PetscCall(VecGetArrayReadAndMemType(x, &array, &memtype));
  PetscCall(VecRestoreArrayReadAndMemType(x, &array));
  PetscCheck(PetscMemTypeDevice(memtype), comm, PETSC_ERR_ARG_WRONG, "Solution vector is in host memory");

  PetscCall(TaoGetTerm(tao, NULL, &sum, &params, NULL));
  PetscCall(TaoTermSumGetNumberTerms(sum, &n_terms));
  for (PetscInt i = 0; i < n_terms; i++) {
    PetscCall(VecNestGetTaoTermSumParameters(params, i, &p));
    if (!p) continue; // a term may legitimately carry no parameter vector
    PetscCall(VecGetArrayReadAndMemType(p, &array, &memtype));
    PetscCall(VecRestoreArrayReadAndMemType(p, &array));
    PetscCheck(PetscMemTypeDevice(memtype), comm, PETSC_ERR_ARG_WRONG, "Parameter vector of term %" PetscInt_FMT " is in host memory", i);
  }

  PetscCall(TaoGetHessian(tao, &H, NULL, NULL, NULL));
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)H, &is_dense, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCall(MatGetType(H, &H_type));
  PetscCheck(is_dense, comm, PETSC_ERR_ARG_WRONG, "-check_memtype expects an assembled dense Hessian; got %s", H_type);
  PetscCall(MatDenseGetArrayReadAndMemType(H, &array, &memtype));
  PetscCall(MatDenseRestoreArrayReadAndMemType(H, &array));
  PetscCheck(PetscMemTypeDevice(memtype), comm, PETSC_ERR_ARG_WRONG, "Hessian is in host memory");

  PetscCall(PetscPrintf(comm, "Device memtype check passed\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /*
    This example demonstrates the solution of an elastic net regularized least squares problem

    (1/2) || Ax - b ||_W^2 + lambda_2 (1/2) || x ||_2^2 + lambda_1 || Dx - y ||_1
   */

  MPI_Comm  comm;
  Mat       A;                   // data matrix
  Mat       D;                   // dictionary matrix
  Mat       W;                   // weight matrix
  Vec       w;                   // observation vector
  Vec       b;                   // observation vector
  Vec       y;                   // dictionary vector
  Vec       x;                   // solution vector
  PetscInt  m             = 100; // data size
  PetscInt  n             = 20;  // model size
  PetscInt  k             = 10;  // dictionary size
  PetscBool set_prefix    = PETSC_TRUE;
  PetscBool set_name      = PETSC_FALSE;
  PetscBool check_eps     = PETSC_FALSE;
  PetscBool check_memtype = PETSC_FALSE;
  TaoTerm   data_term;
  TaoTerm   l2_reg_term;
  TaoTerm   l1_reg_term;
  TaoTerm   full_objective;
  PetscReal lambda_1 = 0.1;
  PetscReal lambda_2 = 0.1;
  Tao       tao;
  VecType   vec_type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsBoundedInt("-m", "data size", "", m, &m, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-n", "model size", "", n, &n, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-k", "dictionary size", "", k, &k, NULL, 0));
  PetscCall(PetscOptionsBool("-set_term_prefix", "Set prefix to terms", NULL, set_prefix, &set_prefix, NULL));
  PetscCall(PetscOptionsBool("-set_term_name", "Set name to terms", NULL, set_name, &set_name, NULL));
  PetscCall(PetscOptionsBool("-check_l1_eps", "Check epsilon of L1 term", NULL, check_eps, &check_eps, NULL));
  PetscCall(PetscOptionsBool("-check_memtype", "Check that solution, parameters and Hessian are in device memory", NULL, check_memtype, &check_memtype, NULL));
  PetscOptionsEnd();

  PetscCall(TaoCreate(comm, &tao));

  // create the model data, A, W and b.
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(A, MATDENSE));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)A, "Amat_"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(FillMatDeterministic(A, 0.0));
  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(FillVecDeterministic(b, 0.0));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(FillVecDeterministic(w, 0.0));
  PetscCall(VecAbs(w));
  PetscCall(VecShift(w, 1.0));
  PetscCall(MatCreateDiagonal(w, &W));
  PetscCall(VecDestroy(&w));

  // create the dictionary data, D and y
  PetscCall(MatCreate(comm, &D));
  PetscCall(MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, k, n));
  PetscCall(MatSetType(D, MATDENSE));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)D, "Dmat_"));
  PetscCall(MatSetFromOptions(D));
  PetscCall(MatSetUp(D));
  PetscCall(FillMatDeterministic(D, 0.5));
  PetscCall(MatCreateVecs(D, NULL, &y));
  PetscCall(FillVecDeterministic(y, 0.25));

  // the model term,  (1/2) || Ax - b ||_W^2
  PetscCall(TaoTermCreateQuadratic(W, &data_term));
  if (set_prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_term, "data_"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)b, "bvec_"));
  }
  if (set_name) PetscCall(PetscObjectSetName((PetscObject)data_term, "Data TaoTerm"));
  PetscCall(TaoAddTerm(tao, "data_", 1.0, data_term, b, A));
  PetscCall(TaoTermDestroy(&data_term));

  // the L2 term,  (1/2) lambda_2 || x ||_2^2. Its solution space is the column space of A
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &l2_reg_term));
  PetscCall(MatGetVecType(A, &vec_type));
  PetscCall(TaoTermSetSolutionVecType(l2_reg_term, vec_type));
  PetscCall(TaoTermSetParametersVecType(l2_reg_term, vec_type));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l2_reg_term, "ridge_"));
  if (set_name) PetscCall(PetscObjectSetName((PetscObject)l2_reg_term, "Ridge TaoTerm"));
  PetscCall(TaoAddTerm(tao, "ridge_", lambda_2, l2_reg_term, NULL, NULL)); // Note: no parameter vector, no map matrix needed
  PetscCall(TaoTermDestroy(&l2_reg_term));

  // the L1 term,  lambda_1 || Dx - y ||_1.  Its solution space is the row space of D
  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, k, 0.0, &l1_reg_term));
  PetscCall(MatGetVecType(D, &vec_type));
  PetscCall(TaoTermSetSolutionVecType(l1_reg_term, vec_type));
  PetscCall(TaoTermSetParametersVecType(l1_reg_term, vec_type));
  if (set_prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l1_reg_term, "lasso_"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)y, "yvec_"));
  }
  if (set_name) PetscCall(PetscObjectSetName((PetscObject)l1_reg_term, "Lasso TaoTerm"));
  PetscCall(TaoAddTerm(tao, "lasso_", lambda_1, l1_reg_term, y, D));
  PetscCall(TaoTermDestroy(&l1_reg_term));

  PetscCall(TaoGetTerm(tao, NULL, &full_objective, NULL, NULL));
  PetscCall(TaoTermCreateSolutionVec(full_objective, &x));
  PetscCall(FillVecDeterministic(x, 0.1));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  {
    PetscReal scale_get;
    TaoTerm   get_term;
    Vec       get_vec, p2, p1;
    Mat       get_mat;

    PetscCall(TaoGetTerm(tao, &scale_get, &get_term, &get_vec, &get_mat));
    PetscCall(VecNestGetTaoTermSumParameters(get_vec, 0, &p1));
    PetscCall(VecNestGetTaoTermSumParameters(get_vec, 1, &p2));
    PetscCheck(p1 == b, PETSC_COMM_SELF, PETSC_ERR_COR, "First parameter vector is not same as what was set");
    PetscCheck(p2 == NULL, PETSC_COMM_SELF, PETSC_ERR_COR, "Second parameter vector is not none");
  }

  if (check_memtype) PetscCall(CheckDeviceMemType(tao, x));

  if (check_eps) {
    PetscReal scale_get;
    TaoTerm   get_term;
    Vec       get_vec;
    Mat       get_mat;
    PetscInt  n_terms;
    PetscInt  last_index;
    PetscBool is_l1;
    TaoTerm   last_subterm;
    PetscReal epsilon;

    PetscCall(TaoGetTerm(tao, &scale_get, &get_term, &get_vec, &get_mat));
    PetscCall(TaoTermSumGetNumberTerms(get_term, &n_terms));
    last_index = n_terms - 1;
    PetscCall(TaoTermSumGetTerm(get_term, last_index, NULL, NULL, &last_subterm, NULL));
    PetscCall(PetscObjectTypeCompare((PetscObject)last_subterm, TAOTERML1, &is_l1));
    PetscCheck(is_l1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Last term is not L1");
    PetscCall(TaoTermL1GetEpsilon(last_subterm, &epsilon));
    PetscCheck(PetscAbsReal(epsilon - 0.1) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "L1 epsilon is not 0.1, got: %g", (double)epsilon);
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&A));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: 0
    args: -tao_monitor -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -check_l1_eps 1

  testset:
    args: -tao_type nls -tao_view ::ascii_info_detail -lasso_tao_term_l1_epsilon 0.1

    test:
      suffix: 1
      args: -lasso_tao_term_hessian_mat_type aij

    test:
      suffix: sum_hpre_is_not_h
      args: -tao_term_hessian_pre_is_hessian 0

    test:
      suffix: data_hpre_is_not_h
      args: -data_tao_term_hessian_pre_is_hessian 0

    test:
      suffix: ridge_hpre_is_not_h
      args: -ridge_tao_term_hessian_pre_is_hessian 0

    test:
      suffix: lasso_hpre_is_not_h
      args: -lasso_tao_term_hessian_pre_is_hessian 0

    test:
      suffix: hpre_is_not_h
      args: -lasso_tao_term_hessian_pre_is_hessian 0
      args: -ridge_tao_term_hessian_pre_is_hessian 0 -data_tao_term_hessian_pre_is_hessian 0

    test:
      suffix: data_ridge_hpre_is_not_h
      args: -ridge_tao_term_hessian_pre_is_hessian 0 -data_tao_term_hessian_pre_is_hessian 0

  test:
    suffix: no_prefix
    args: -tao_monitor -tao_view -tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 0

  test:
    suffix: no_prefix_yes_name
    args: -tao_monitor -tao_view -tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 0 -set_term_name 1

  test:
    suffix: yes_prefix_yes_name
    args: -tao_monitor -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 1 -set_term_name 1

  test:
    suffix: mask_failure
    args: -tao_monitor -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls
    args: -tao_term_sum_ridge_mask objective -tao_term_sum_lasso_mask gradient
    args: -tao_view ::ascii_info_detail

  test:
    suffix: assembled
    args: -tao_monitor -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -ridge_tao_term_hessian_mat_type constantdiagonal -lasso_tao_term_hessian_mat_type diagonal

  test:
    suffix: snes
    args: -tao_monitor -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type snes

  test:
    suffix: extra_info_view
    args: -tao_type nls -tao_add_terms extra_ -extra_tao_term_type halfl2squared -tao_term_sum_extra_scale 1.0 -tao_view

  test:
    suffix: device_cuda_mpi
    nsize: 2
    requires: cuda
    args: -Amat_mat_type densecuda -Dmat_mat_type densecuda -tao_term_hessian_mat_type densecuda
    args: -tao_type nls -tao_monitor -lasso_tao_term_l1_epsilon 0.1 -tao_nls_pc_type jacobi -check_memtype
    args: -tao_view ::ascii_info_detail

  # Exercises the outer-sum Hessian-shell path: TaoTermComputeHessianMult_Sum
  # dispatches MatMult on the MATSHELL outer Hessian through the per-summand
  # cache + MatMult fallback. The L1 epsilon is kept large enough that the
  # smoothed Hessian is well-conditioned for the unpreconditioned KSP used
  # when the outer Hessian is a MATSHELL.
  test:
    suffix: shell_hessian
    args: -tao_type nls -tao_term_hessian_mat_type shell -lasso_tao_term_l1_epsilon 0.1 -tao_monitor

TEST*/
