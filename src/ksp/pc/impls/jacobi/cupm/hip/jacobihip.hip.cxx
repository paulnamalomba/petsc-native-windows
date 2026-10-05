#include "../jacobicupm.hpp"

PETSC_INTERN PetscErrorCode PCMatApply_Jacobi_HIP(Vec diag, Mat X, Mat Y)
{
  PetscFunctionBegin;
  PetscCall(Petsc::pc::cupm::impl::PCJacobi_CUPM<Petsc::device::cupm::DeviceType::HIP>::MatApply(diag, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
