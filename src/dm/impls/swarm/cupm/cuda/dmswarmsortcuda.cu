#include "../dmswarmsortcupm.hpp"

using namespace Petsc::device::cupm;

static constexpr impl::DMSwarmSort_CUPM<DeviceType::CUDA> cupm_sort{};

/* Install the CUDA sort ops; called from DMSwarmSortGetAccess() for a cuda-backend swarm */
PETSC_INTERN PetscErrorCode DMSwarmSortSetUpDevice_CUDA(DMSwarmSort ctx)
{
  PetscFunctionBegin;
  PetscCall(cupm_sort.SetUp(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
