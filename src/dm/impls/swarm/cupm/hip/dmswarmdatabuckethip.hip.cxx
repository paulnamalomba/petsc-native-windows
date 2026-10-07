#include "../dmswarmdatabucketcupm.hpp"

#include <../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

using namespace Petsc::device::cupm;

static constexpr impl::DMSwarmDataBucket_CUPM<DeviceType::HIP> cupm_bucket{};

namespace
{

struct EqualsTarget {
  PetscInt target;

  PETSC_HOSTDEVICE_INLINE_DECL bool operator()(PetscInt v) const noexcept { return v == target; }
};

} // namespace

/*
  Device compaction per the compact ops slot contract in data_bucket.h: reproduce the host
  swap-with-last removal order through its closed form. The sent mask and its deterministic
  exclusive prefix sum enumerate the ascending sent slots < m and the descending surviving tail
  points >= m (the scan is the only thrust piece, so it lives in this TU like the sort build);
  a sparse scatter then moves only the filled holes' records, and the size bookkeeping of the
  host loop's per-point DMSwarmDataBucketRemovePoint() shrinks is applied at once by
  DMSwarmDataBucketRemovePoints_Internal(), so L, buffer, allocated and the zeroed tail end
  bit-identical to the host backend with at most one reallocation of each device mirror. The
  scratch arrays are allocated and freed in stream order on the stream all compaction work runs
  on, which avoids the device-wide synchronization of hipFree().
*/
static PetscErrorCode DMSwarmDataBucketCompact_HIP(DMSwarmDataBucket db, PetscInt nsent, const PetscInt *sentlist)
{
  const PetscInt n = db->L, m = n - nsent;
  PetscInt       nholes = 0;
  PetscInt      *mask, *scan, *holes, *srcs;

  PetscFunctionBegin;
  if (!nsent) PetscFunctionReturn(PETSC_SUCCESS);
  /* untouched survivors must be device valid before the device copy becomes authoritative */
  for (PetscInt f = 0; f < db->nfields; ++f) PetscCall(impl::DMSwarmDataBucket_CUPM<DeviceType::HIP>::FieldToDevice(db->field[f]));
  if (m > 0 && db->nfields > 0) {
    PetscCallHIP(hipMallocAsync((void **)&mask, (size_t)n * sizeof(*mask), PetscDefaultHipStream));
    PetscCallHIP(hipMallocAsync((void **)&scan, (size_t)n * sizeof(*scan), PetscDefaultHipStream));
    PetscCallHIP(hipMallocAsync((void **)&holes, (size_t)nsent * sizeof(*holes), PetscDefaultHipStream));
    PetscCallHIP(hipMallocAsync((void **)&srcs, (size_t)nsent * sizeof(*srcs), PetscDefaultHipStream));
    PetscCall(cupm_bucket.CompactMask(db, nsent, sentlist, mask));
    /* the FieldToDevice()/CompactMask() calls above initialized the device, so the default stream exists */
    {
      const auto mask_p = thrust::device_pointer_cast(mask);
      const auto scan_p = thrust::device_pointer_cast(scan);

      PetscCallThrust(THRUST_CALL(thrust::exclusive_scan, PetscDefaultHipStream, mask_p, mask_p + n, scan_p));
    }
    /* nholes = scan[m] = number of sent slots < m = number of surviving tail points >= m */
    PetscCallHIP(hipMemcpyAsync(&nholes, scan + m, sizeof(nholes), hipMemcpyDeviceToHost, PetscDefaultHipStream));
    PetscCallHIP(hipStreamSynchronize(PetscDefaultHipStream));
    PetscCall(cupm_bucket.CompactScatter(db, nsent, nholes, mask, scan, holes, srcs));
    PetscCallHIP(hipFreeAsync(mask, PetscDefaultHipStream));
    PetscCallHIP(hipFreeAsync(scan, PetscDefaultHipStream));
    PetscCallHIP(hipFreeAsync(holes, PetscDefaultHipStream));
    PetscCallHIP(hipFreeAsync(srcs, PetscDefaultHipStream));
  }
  /* the size bookkeeping of the host loop's per-point shrinks, applied at once */
  PetscCall(DMSwarmDataBucketRemovePoints_Internal(db, nsent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Build the device-resident send list of the migration fork in swarm_migrate.c: the indices
  i in [start, end) with vals_d[i] == match, in strictly ascending order (every pointer device
  resident). thrust::copy_if() over an ascending counting iterator is stable and scan based, so
  the order and content are deterministic by construction. The list is allocated here, sized for
  the worst case, and released with DMSwarmSentListDestroy_HIP(); an empty range returns a NULL
  list with *nsent = 0.
*/
PETSC_INTERN PetscErrorCode DMSwarmSentListCreate_HIP(PetscInt start, PetscInt end, PetscInt match, const PetscInt *vals_d, PetscInt *nsent, PetscInt **sentlist_d)
{
  const PetscInt n = end - start;
  PetscInt      *list;

  PetscFunctionBegin;
  *nsent      = 0;
  *sentlist_d = NULL;
  if (n <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallHIP(hipMallocAsync((void **)&list, (size_t)n * sizeof(*list), PetscDefaultHipStream));
  {
    const auto first = thrust::counting_iterator<PetscInt>(start);
    const auto last  = thrust::counting_iterator<PetscInt>(end);
    const auto vals  = thrust::device_pointer_cast(vals_d);
    const auto out   = thrust::device_pointer_cast(list);
    auto       tail  = out;

    PetscCallThrust(tail = THRUST_CALL(thrust::copy_if, PetscDefaultHipStream, first, last, vals + start, out, EqualsTarget{match}));
    *nsent = static_cast<PetscInt>(tail - out);
  }
  *sentlist_d = list;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Release a device send list built by DMSwarmSentListCreate_HIP(), in stream order on the stream
  it was built on; every consumer synchronizes before returning, so the list is no longer in use
*/
PETSC_INTERN PetscErrorCode DMSwarmSentListDestroy_HIP(PetscInt **sentlist_d)
{
  PetscFunctionBegin;
  if (*sentlist_d) PetscCallHIP(hipFreeAsync(*sentlist_d, PetscDefaultHipStream));
  *sentlist_d = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fill [start, end) of a PetscInt field's device mirror with a constant; see DMSwarmDataBucket_CUPM::FieldFillIntRange() */
PETSC_INTERN PetscErrorCode DMSwarmDataFieldFillIntRange_HIP(DMSwarmDataField df, PetscInt start, PetscInt end, PetscInt v)
{
  PetscFunctionBegin;
  PetscCall(cupm_bucket.FieldFillIntRange(df, start, end, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Install the HIP storage ops on a DMSwarmDataBucket; called from DMSwarmSetBackend() for DMSWARM_BACKEND_HIP */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetUpDevice_HIP(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  PetscCall(cupm_bucket.SetUp(db));
  /* the compaction needs a deterministic device prefix sum, so its orchestration lives in this TU */
  db->ops.compact = DMSwarmDataBucketCompact_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}
