#pragma once

#include "../data_bucket.h"

#include <petsc/private/cupmobject.hpp>
#include <petsc/private/deviceimpl.h>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

/*
  Device-backend implementations of the DMSwarmDataBucket storage ops.

  Design notes (Phase 1.1 primitives; since Phase 1.2 the per-field offloadmask is live and
  these ops are dispatched from the coherence chokepoints in data_bucket.c and from the
  DMSwarmGetField*AndMemType() accessors in swarm.c):
  - Device mirrors are allocated lazily. FieldAlloc() only maintains a mirror that already
    exists (df->data_d != NULL); the first FieldToDevice() performs the initial allocation.
    This is correct because every FieldToDevice() issued while offloadmask == PETSC_OFFLOAD_CPU
    uploads the full host array, so a deferred mirror never loses information.
  - Growth is geometric. When a mirror must grow, it is sized to
    max(needed, 1.6 * current capacity) entries, with the capacity tracked in df->capacity_d,
    so repeated single-particle growth does not cost one device reallocation per particle.
    There is no device realloc(): grow = allocate + stream-ordered copy of surviving entries +
    zero the tail + free the old buffer.
  - Shrinks reallocate to the exact new size, but FieldAlloc() is only invoked with a smaller
    size when the host sizing logic of DMSwarmDataBucketSetSizes_Host() decides to shrink, so
    the host hysteresis (new_unused > 2 * new_buffer) applies to the mirror as well.
  - All mirrors are owned by the field (data_d == data_allocated_d); FieldFreeDevice() releases
    them and is dispatched from DMSwarmDataBucketDestroy() before DMSwarmDataFieldDestroy().
*/
template <DeviceType T>
struct DMSwarmDataBucket_CUPM : CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode FieldAlloc(DMSwarmDataField, PetscInt) noexcept;
  static PetscErrorCode FieldAllocDevice(DMSwarmDataField) noexcept;
  static PetscErrorCode FieldToDevice(DMSwarmDataField) noexcept;
  static PetscErrorCode FieldToHost(DMSwarmDataField) noexcept;
  static PetscErrorCode FieldZeroRange(DMSwarmDataField, PetscInt, PetscInt) noexcept;
  static PetscErrorCode FieldFreeDevice(DMSwarmDataField) noexcept;
  static PetscErrorCode SetSizes(DMSwarmDataBucket, PetscInt, PetscInt) noexcept;
  static PetscErrorCode SetUp(DMSwarmDataBucket) noexcept;

private:
  static PetscErrorCode EnsureDeviceCapacity_(DMSwarmDataField, PetscInt, PetscInt) noexcept;
};

/*
  Bring the device mirror of df in line with a host array that changed size from old_L to
  new_L entries, mimicking what DMSwarmDataFieldSetSize() did to the host array. Must only be
  called when a mirror exists (df->data_d != NULL), except for the initial allocation path of
  FieldToDevice() which calls it with old_L = 0 so that no stale device contents are copied.
  The device-to-device copies here are not host-device traffic, so they are deliberately not
  accounted in the -log_view CpuToGpu/GpuToCpu totals, matching what PetscCUPMMemcpyAsync()
  (and hence the Vec resize paths) does for cupmMemcpyDeviceToDevice.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::EnsureDeviceCapacity_(DMSwarmDataField df, PetscInt old_L, PetscInt new_L) noexcept
{
  const std::size_t as = df->atomic_size;
  cupmStream_t      stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&stream));
  if (new_L > df->capacity_d) {
    /* grow: no device realloc() exists, so allocate-copy-zero-free, with geometric growth */
    const PetscInt surviving    = PetscMin(old_L, new_L);
    const PetscInt new_capacity = PetscMax(new_L, static_cast<PetscInt>(1.6 * static_cast<double>(df->capacity_d)));
    char          *tmp          = nullptr;

    PetscCall(PetscCUPMMallocAsync(&tmp, static_cast<std::size_t>(new_capacity) * as, stream));
    if (surviving > 0) PetscCall(PetscCUPMMemcpyAsync(tmp, static_cast<const char *>(df->data_d), static_cast<std::size_t>(surviving) * as, cupmMemcpyDeviceToDevice, stream));
    PetscCall(PetscCUPMMemsetAsync(tmp + static_cast<std::size_t>(surviving) * as, 0, static_cast<std::size_t>(new_capacity - surviving) * as, stream, true));
    PetscCallCUPM(cupmFreeAsync(df->data_allocated_d, stream));
    df->data_d = df->data_allocated_d = tmp;
    df->capacity_d                    = new_capacity;
  } else if (new_L < old_L) {
    /* shrink: reallocate to the exact new size, mirroring the host realloc in DMSwarmDataFieldSetSize() */
    char *tmp = nullptr;

    PetscCall(PetscCUPMMallocAsync(&tmp, static_cast<std::size_t>(new_L) * as, stream));
    if (new_L > 0) PetscCall(PetscCUPMMemcpyAsync(tmp, static_cast<const char *>(df->data_d), static_cast<std::size_t>(new_L) * as, cupmMemcpyDeviceToDevice, stream));
    PetscCallCUPM(cupmFreeAsync(df->data_allocated_d, stream));
    df->data_d = df->data_allocated_d = tmp;
    df->capacity_d                    = new_L;
  } else if (new_L > old_L) {
    /* grow within the existing capacity: zero the newly exposed entries like the host grow does */
    PetscCall(PetscCUPMMemsetAsync(static_cast<char *>(df->data_d) + static_cast<std::size_t>(old_L) * as, 0, static_cast<std::size_t>(new_L - old_L) * as, stream, true));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldAlloc(DMSwarmDataField df, PetscInt new_L) noexcept
{
  const PetscInt old_L = df->L;

  PetscFunctionBegin;
  /* host array first, bit-identical to the host backend */
  PetscCall(DMSwarmDataFieldSetSize(df, new_L));
  /* mirrors are lazily created by FieldToDevice(); only maintain one that already exists */
  if (df->data_d) PetscCall(EnsureDeviceCapacity_(df, old_L, new_L));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldAllocDevice(DMSwarmDataField df) noexcept
{
  PetscFunctionBegin;
  /*
    Size the mirror for the write-only access path without transferring any entries. An
    undersized mirror implies the device copy is stale (the lazy-allocation invariant keeps
    capacity_d >= L whenever the device copy is valid), so no device contents survive. The
    synchronize orders the stream-ordered allocation and zeroing before the caller's writes,
    matching the guarantee FieldToDevice() gives the read paths.
  */
  if (df->capacity_d < df->L) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(EnsureDeviceCapacity_(df, 0, df->L));
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldToDevice(DMSwarmDataField df) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  if (PetscOffloadDevice(df->offloadmask)) PetscFunctionReturn(PETSC_SUCCESS); /* device copy already valid */
  PetscCall(GetHandles_(&dctx, &stream));
  /* lazy first-touch allocation; old_L = 0 since any prior device contents are stale */
  if (df->capacity_d < df->L) PetscCall(EnsureDeviceCapacity_(df, 0, df->L));
  /* PetscCUPMMemcpyAsync() reports the transferred bytes via PetscLogCpuToGpu() */
  PetscCall(PetscLogEventBegin(DMSWARM_CopyToGPU, 0, 0, 0, 0));
  PetscCall(PetscCUPMMemcpyAsync(static_cast<char *>(df->data_d), static_cast<const char *>(df->data), static_cast<std::size_t>(df->L) * df->atomic_size, cupmMemcpyHostToDevice, stream));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscLogEventEnd(DMSWARM_CopyToGPU, 0, 0, 0, 0));
  df->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldToHost(DMSwarmDataField df) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  /* nothing to do if the host copy is already valid, or if no device copy exists at all */
  if (PetscOffloadHost(df->offloadmask) || PetscOffloadUnallocated(df->offloadmask) || !df->data_d) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(GetHandles_(&dctx, &stream));
  /* PetscCUPMMemcpyAsync() reports the transferred bytes via PetscLogGpuToCpu() */
  PetscCall(PetscLogEventBegin(DMSWARM_CopyFromGPU, 0, 0, 0, 0));
  PetscCall(PetscCUPMMemcpyAsync(static_cast<char *>(df->data), static_cast<const char *>(df->data_d), static_cast<std::size_t>(df->L) * df->atomic_size, cupmMemcpyDeviceToHost, stream));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscLogEventEnd(DMSWARM_CopyFromGPU, 0, 0, 0, 0));
  df->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldZeroRange(DMSwarmDataField df, PetscInt start, PetscInt end) noexcept
{
  PetscFunctionBegin;
  /* host range first: argument checking + bit-identical host behavior */
  PetscCall(DMSwarmDataFieldZeroBlock(df, start, end));
  if (df->data_d) {
    const PetscInt e = PetscMin(end, df->capacity_d);

    if (e > start) {
      cupmStream_t stream;

      PetscCall(GetHandles_(&stream));
      PetscCall(PetscCUPMMemsetAsync(static_cast<char *>(df->data_d) + static_cast<std::size_t>(start) * df->atomic_size, 0, static_cast<std::size_t>(e - start) * df->atomic_size, stream, true));
    }
  }
  /* both residences were zeroed over the range, so the offload mask is unchanged */
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldFreeDevice(DMSwarmDataField df) noexcept
{
  PetscFunctionBegin;
  if (df->data_allocated_d) {
    cupmStream_t stream;

    PetscCall(GetHandles_(&stream));
    PetscCallCUPM(cupmFreeAsync(df->data_allocated_d, stream));
  }
  df->data_d      = nullptr;
  df->capacity_d  = 0;
  df->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::SetSizes(DMSwarmDataBucket db, PetscInt L, PetscInt buffer) noexcept
{
  PetscFunctionBegin;
  /*
    The host sizing algorithm dispatches through db->ops.fieldalloc/fieldzerorange; for a device
    bucket those are FieldAlloc()/FieldZeroRange() above, so the host arrays receive bit-identical
    treatment while any existing device mirrors are maintained alongside. Fields stay host
    authoritative (offloadmask PETSC_OFFLOAD_CPU) until an explicit FieldToDevice().
  */
  PetscCall(DMSwarmDataBucketSetSizes_Host(db, L, buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::SetUp(DMSwarmDataBucket db) noexcept
{
  PetscFunctionBegin;
  db->ops.fieldalloc       = FieldAlloc;
  db->ops.fieldallocdevice = FieldAllocDevice;
  db->ops.fieldtodevice    = FieldToDevice;
  db->ops.fieldtohost      = FieldToHost;
  db->ops.fieldzerorange   = FieldZeroRange;
  db->ops.fieldfreedevice  = FieldFreeDevice;
  db->ops.setsizes         = SetSizes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
