#include "../dmswarmdatabucketcupm.hpp"

using namespace Petsc::device::cupm;

static constexpr impl::DMSwarmDataBucket_CUPM<DeviceType::CUDA> cupm_bucket{};

/* Install the CUDA storage ops on a DMSwarmDataBucket; called from DMSwarmSetBackend() for DMSWARM_BACKEND_CUDA */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetUpDevice_CUDA(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  PetscCall(cupm_bucket.SetUp(db));
  PetscFunctionReturn(PETSC_SUCCESS);
}
