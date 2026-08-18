/*
  HIP backend for the DMSwarmSort cell-view tier: builds the cell CSR on the device with
  thrust and mirrors it into the host arrays of the sort context.

  Determinism by construction (user ruling): one thrust::stable_sort_by_key() on the cell
  index with the ascending point-index sequence as values realizes the total order
  (cell_index, point_index) — identical to the host qsort() with its point-index tie-break —
  and the CSR offsets come from thrust::lower_bound() over the sorted keys. No atomics, no
  order-dependent parallelism anywhere.

  Thrust usage is confined to this TU (rocThrust headers only compile under hipcc); the
  include set follows aijhipsparse.hip.cxx.
*/
#include <petsc/private/dmswarmimpl.h>
#include <petscdevice_hip.h>
#include <../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

/*
  Build the CSR on the device and mirror it to the host arrays.

  The generic DMSwarmSortSetup() has already sized ctx->pcell_offsets/ctx->list and set
  ctx->ncells/ctx->npoints. The single D2H mirror per build (offsets + sorted keys + sorted
  point indices) keeps every existing host query (DMSwarmSortGetPointsPerCell() etc.)
  bit-identical: host callers read exactly the arrays the device consumers see, from one
  build, so the two residences cannot drift apart.
*/
static PetscErrorCode DMSwarmSortSetup_HIP(DMSwarmSort ctx, DM sw, PetscInt ncells)
{
  DMSwarmCellDM  celldm;
  PetscMemType   mtype;
  PetscInt      *cellid_d, *keys_h, *pids_h;
  const PetscInt npoints = ctx->npoints;
  const char    *cellid;
  hipStream_t    stream;

  PetscFunctionBegin;
  /* (re)allocate the device CSR arrays when the sizes change; freed by DMSwarmSortDestroy_HIP() */
  if (ctx->ncells_d != ncells) {
    PetscCallHIP(hipFree(ctx->pcell_offsets_d));
    PetscCallHIP(hipMalloc((void **)&ctx->pcell_offsets_d, ((size_t)ncells + 1) * sizeof(PetscInt)));
    ctx->ncells_d = ncells;
  }
  if (ctx->npoints_d != npoints) {
    PetscCallHIP(hipFree(ctx->list_d));
    PetscCallHIP(hipFree(ctx->cell_index_d));
    PetscCallHIP(hipMalloc((void **)&ctx->list_d, (size_t)PetscMax(npoints, 1) * sizeof(PetscInt)));
    PetscCallHIP(hipMalloc((void **)&ctx->cell_index_d, (size_t)PetscMax(npoints, 1) * sizeof(PetscInt)));
    ctx->npoints_d = npoints;
  }
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetFieldReadAndMemType(sw, cellid, NULL, NULL, &mtype, (void **)&cellid_d));
  PetscCheck(PetscMemTypeHIP(mtype), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected the cell id field of a hip-backend swarm in HIP memory, got %s", PetscMemTypeToString(mtype));
  /* the accessor above guarantees the PETSc device is initialized, so the default stream exists */
  stream = PetscDefaultHipStream;
  {
    const auto keys  = thrust::device_pointer_cast(ctx->cell_index_d);
    const auto vals  = thrust::device_pointer_cast(ctx->list_d);
    const auto offs  = thrust::device_pointer_cast(ctx->pcell_offsets_d);
    const auto cells = thrust::device_pointer_cast(cellid_d);

    PetscCallThrust(THRUST_CALL(thrust::copy, stream, cells, cells + npoints, keys));
    PetscCallThrust(THRUST_CALL(thrust::sequence, stream, vals, vals + npoints));
    PetscCallThrust(THRUST_CALL(thrust::stable_sort_by_key, stream, keys, keys + npoints, vals));
    PetscCallThrust(THRUST_CALL(thrust::lower_bound, stream, keys, keys + npoints, thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(ncells + 1), offs));
  }
  PetscCall(DMSwarmRestoreFieldReadAndMemType(sw, cellid, NULL, NULL, &mtype, (void **)&cellid_d));
  /* one D2H mirror per build so host queries stay bit-identical (see the function comment) */
  PetscCall(PetscMalloc2(npoints, &keys_h, npoints, &pids_h));
  PetscCallHIP(hipMemcpyAsync(ctx->pcell_offsets, ctx->pcell_offsets_d, ((size_t)ncells + 1) * sizeof(PetscInt), hipMemcpyDeviceToHost, stream));
  if (npoints) {
    PetscCallHIP(hipMemcpyAsync(keys_h, ctx->cell_index_d, (size_t)npoints * sizeof(PetscInt), hipMemcpyDeviceToHost, stream));
    PetscCallHIP(hipMemcpyAsync(pids_h, ctx->list_d, (size_t)npoints * sizeof(PetscInt), hipMemcpyDeviceToHost, stream));
  }
  PetscCallHIP(hipStreamSynchronize(stream));
  if (npoints) {
    PetscCheck(keys_h[0] >= 0 && keys_h[npoints - 1] < ncells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell index range [%" PetscInt_FMT ", %" PetscInt_FMT "] is not in [0, %" PetscInt_FMT ")", keys_h[0], keys_h[npoints - 1], ncells);
    for (PetscInt p = 0; p < npoints; p++) {
      ctx->list[p].point_index = pids_h[p];
      ctx->list[p].cell_index  = keys_h[p];
    }
  }
  PetscCall(PetscFree2(keys_h, pids_h));
  ctx->memtype = PETSC_MEMTYPE_HIP;
  PetscCall(PetscInfo(sw, "Built the DMSwarmSort cell CSR on the device: %" PetscInt_FMT " points in %" PetscInt_FMT " cells\n", npoints, ncells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Release the device CSR arrays; dispatched from DMSwarmSortDestroy() */
static PetscErrorCode DMSwarmSortDestroy_HIP(DMSwarmSort ctx)
{
  PetscFunctionBegin;
  PetscCallHIP(hipFree(ctx->pcell_offsets_d));
  PetscCallHIP(hipFree(ctx->list_d));
  PetscCallHIP(hipFree(ctx->cell_index_d));
  ctx->pcell_offsets_d = NULL;
  ctx->list_d          = NULL;
  ctx->cell_index_d    = NULL;
  ctx->ncells_d        = -1;
  ctx->npoints_d       = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Install the HIP sort ops; called from DMSwarmSortGetAccess() for a hip-memtype bucket */
PETSC_INTERN PetscErrorCode DMSwarmSortSetUpDevice_HIP(DMSwarmSort ctx)
{
  PetscFunctionBegin;
  ctx->ops->setup   = DMSwarmSortSetup_HIP;
  ctx->ops->destroy = DMSwarmSortDestroy_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}
