#include "../jacobicupm.hpp"

PETSC_INTERN PetscErrorCode PCMatApply_Jacobi_CUDA(Vec diag, Mat X, Mat Y)
{
  PetscFunctionBegin;
  PetscCall(Petsc::pc::cupm::impl::PCJacobi_CUPM<Petsc::device::cupm::DeviceType::CUDA>::MatApply(diag, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
