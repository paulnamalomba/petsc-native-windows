#include "../dmswarmsortcupm.hpp"

using namespace Petsc::device::cupm;

static constexpr impl::DMSwarmSort_CUPM<DeviceType::HIP> cupm_sort{};

/* Install the HIP sort ops; called from DMSwarmSortGetAccess() for a hip-memtype bucket */
PETSC_INTERN PetscErrorCode DMSwarmSortSetUpDevice_HIP(DMSwarmSort ctx)
{
  PetscFunctionBegin;
  PetscCall(cupm_sort.SetUp(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
