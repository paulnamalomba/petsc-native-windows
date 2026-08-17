#include "../dmswarmdatabucketcupm.hpp"

using namespace Petsc::device::cupm;

static constexpr impl::DMSwarmDataBucket_CUPM<DeviceType::HIP> cupm_bucket{};

/* Install the HIP storage ops on a DMSwarmDataBucket; called from DMSwarmSetBackend() for DMSWARM_BACKEND_HIP */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetUpDevice_HIP(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  PetscCall(cupm_bucket.SetUp(db));
  PetscFunctionReturn(PETSC_SUCCESS);
}
