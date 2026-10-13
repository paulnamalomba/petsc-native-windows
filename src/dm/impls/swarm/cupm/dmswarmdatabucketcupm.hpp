#pragma once

#include "../data_bucket.h"

#include <petsc/private/cupmobject.hpp>
#include <petsc/private/deviceimpl.h>

#include "../src/sys/objects/device/impls/cupm/kernels.hpp"
#include "../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

namespace kernels
{

/* Field descriptors passed to the pack/unpack kernels by value; buckets with more fields are handled in chunks of this many */
constexpr int DMSWARM_PACK_NFIELDS_MAX = 16;

/*
  One chunk of the field table for the wire-record gather/scatter kernels: per field the device
  base of its entries, its atomic size, and its byte offset inside a record. record_size is the
  full record (the sum over ALL fields, not just this chunk), so every launch strides the buffer
  identically and chunks write disjoint byte ranges of each record.
*/
struct DMSwarmPackCtx {
  int         nfields;                          /* fields in this chunk */
  std::size_t record_size;                      /* bytes per wire record, summed over all fields of the bucket */
  char       *data[DMSWARM_PACK_NFIELDS_MAX];   /* device base of each field's entries */
  std::size_t asize[DMSWARM_PACK_NFIELDS_MAX];  /* atomic size of each field */
  std::size_t offset[DMSWARM_PACK_NFIELDS_MAX]; /* byte offset of each field inside a record */
};

/*
  Copy one field's atomic_size bytes between a field entry and a wire record. A record is the
  concatenation of every field's atomic_size bytes, so neither the per-record base nor the
  in-record field offsets carry any alignment guarantee; a typed load or store wider than the
  actual alignment of its address is undefined behavior on the device. The copy therefore widens
  to 8-, 4- or 2-byte chunks only when both addresses and the byte count share that alignment,
  and falls back to a plain byte loop otherwise. The chunk width never changes the bytes written,
  so the result is independent of the (address-dependent) branch taken.
*/
PETSC_DEVICE_INLINE_DECL static void CopyRecordBytes(char *PETSC_RESTRICT dst, const char *PETSC_RESTRICT src, std::size_t n) noexcept
{
  const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src) | static_cast<std::uintptr_t>(n);

  if (!(a & 7)) {
    for (std::size_t i = 0; i < n; i += 8) *reinterpret_cast<std::uint64_t *>(dst + i) = *reinterpret_cast<const std::uint64_t *>(src + i);
  } else if (!(a & 3)) {
    for (std::size_t i = 0; i < n; i += 4) *reinterpret_cast<std::uint32_t *>(dst + i) = *reinterpret_cast<const std::uint32_t *>(src + i);
  } else if (!(a & 1)) {
    for (std::size_t i = 0; i < n; i += 2) *reinterpret_cast<std::uint16_t *>(dst + i) = *reinterpret_cast<const std::uint16_t *>(src + i);
  } else {
    for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
  }
}

/*
  One thread per point, matching the host loop of DMSwarmDataBucketPackRecords_Host(): record p of
  buf receives the bytes of point list[p], fields laid out at the host-computed offsets. Threads
  write disjoint records and read without ordering dependencies, so the kernel is deterministic by
  construction; the gather order is exactly the order of list[].
*/
PETSC_KERNEL_DECL static void pack_records(const DMSwarmPackCtx ctx, const PetscInt npoints, const PetscInt *PETSC_RESTRICT list, char *PETSC_RESTRICT buf)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(npoints, [&](PetscInt p) {
    const std::size_t rec = static_cast<std::size_t>(p) * ctx.record_size;

    for (int f = 0; f < ctx.nfields; ++f) CopyRecordBytes(buf + rec + ctx.offset[f], ctx.data[f] + static_cast<std::size_t>(list[p]) * ctx.asize[f], ctx.asize[f]);
  });
}

/*
  One thread per point, matching the host loop of DMSwarmDataBucketUnpackRecords_Host(): record p
  of buf is scattered into point start + p. Destination indices are fixed and disjoint, so the
  kernel is deterministic by construction.
*/
PETSC_KERNEL_DECL static void unpack_records(const DMSwarmPackCtx ctx, const PetscInt start, const PetscInt npoints, const char *PETSC_RESTRICT buf)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(npoints, [&](PetscInt p) {
    const std::size_t rec = static_cast<std::size_t>(p) * ctx.record_size;

    for (int f = 0; f < ctx.nfields; ++f) CopyRecordBytes(ctx.data[f] + static_cast<std::size_t>(start + p) * ctx.asize[f], buf + rec + ctx.offset[f], ctx.asize[f]);
  });
}

/* Raise the sent flag of every listed point; the list entries are unique, so the writes are disjoint and no atomics are needed */
PETSC_KERNEL_DECL static void compact_fill_mask(const PetscInt nsent, const PetscInt *PETSC_RESTRICT sentlist, PetscInt *PETSC_RESTRICT mask)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(nsent, [&](PetscInt i) { mask[sentlist[i]] = 1; });
}

/*
  Enumerate the closed form of the host swap-with-last removal loop from the sent mask and its
  exclusive prefix sum scan (scan[i] = number of sent points with index < i). With n points, k of
  them sent and m = n - k survivors, the sent slots < m taken in ascending order (hole number
  scan[i]) are filled by the surviving points >= m taken in descending original index order
  (descending number (n - 1 - i) - (k - scan[i]), the count of survivors above i). Every thread
  writes at most one distinct slot of holes[] or srcs[], so the kernel is deterministic by
  construction with no atomics.
*/
PETSC_KERNEL_DECL static void compact_build_pairs(const PetscInt n, const PetscInt m, const PetscInt k, const PetscInt *PETSC_RESTRICT mask, const PetscInt *PETSC_RESTRICT scan, PetscInt *PETSC_RESTRICT holes, PetscInt *PETSC_RESTRICT srcs)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(n, [&](PetscInt i) {
    if (i < m) {
      if (mask[i]) holes[scan[i]] = i;
    } else if (!mask[i]) srcs[(n - 1 - i) - (k - scan[i])] = i;
  });
}

/* One thread per filled hole: move the whole record of point srcs[j] into slot holes[j]. Sources are all >= m and destinations all < m, so reads and writes never overlap */
PETSC_KERNEL_DECL static void compact_move_records(const DMSwarmPackCtx ctx, const PetscInt nholes, const PetscInt *PETSC_RESTRICT holes, const PetscInt *PETSC_RESTRICT srcs)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(nholes, [&](PetscInt j) {
    for (int f = 0; f < ctx.nfields; ++f) CopyRecordBytes(ctx.data[f] + static_cast<std::size_t>(holes[j]) * ctx.asize[f], ctx.data[f] + static_cast<std::size_t>(srcs[j]) * ctx.asize[f], ctx.asize[f]);
  });
}

/* Store a constant into a[0..n); the writes are disjoint, so the kernel is deterministic by construction */
PETSC_KERNEL_DECL static void fill_int_range(const PetscInt n, const PetscInt v, PetscInt *PETSC_RESTRICT a)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(n, [&](PetscInt i) { a[i] = v; });
}

/* Selection predicate of SentListCreate() */
struct EqualsTarget {
  PetscInt target;

  PETSC_HOSTDEVICE_INLINE_DECL bool operator()(PetscInt v) const noexcept { return v == target; }
};

} // namespace kernels

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
  - Shrinks keep the capacity until less than half of it is in use, and only then reallocate to
    the exact new size. FieldAlloc() sees a smaller size whenever the host sizing logic of
    DMSwarmDataBucketSetSizes_Host() shrinks the host array (new_unused > 2 * new_buffer), which
    with the default buffer happens in almost every migration; following each of those shrinks
    made a migration reallocate and copy every mirror twice, once for the removed points and once
    for the received ones. Entries kept beyond the host allocation are zeroed again before they
    are exposed, by the grow-within-capacity path of EnsureDeviceCapacity_().
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
  static PetscErrorCode PackRecords(DMSwarmDataBucket, PetscInt, const PetscInt *, void *) noexcept;
  static PetscErrorCode UnpackRecords(DMSwarmDataBucket, PetscInt, PetscInt, const void *) noexcept;
  static PetscErrorCode Compact(DMSwarmDataBucket, PetscInt, const PetscInt *) noexcept;
  static PetscErrorCode FieldFillIntRange(DMSwarmDataField, PetscInt, PetscInt, PetscInt) noexcept;
  static PetscErrorCode SentListCreate(DMSwarmDataBucket, PetscInt, PetscInt, PetscInt, const PetscInt *, PetscInt *, PetscInt **) noexcept;
  static PetscErrorCode SentListDestroy(DMSwarmDataBucket, PetscInt **) noexcept;
  static PetscErrorCode SetUp(DMSwarmDataBucket) noexcept;

private:
  static PetscErrorCode EnsureDeviceCapacity_(DMSwarmDataField, PetscInt, PetscInt) noexcept;
  static PetscErrorCode LaunchRecordKernels_(DMSwarmDataBucket, PetscBool, PetscInt, PetscInt, const PetscInt *, void *) noexcept;
  static PetscErrorCode CompactMask_(DMSwarmDataBucket, PetscInt, const PetscInt *, PetscInt *) noexcept;
  static PetscErrorCode CompactScatter_(DMSwarmDataBucket, PetscInt, PetscInt, const PetscInt *, const PetscInt *, PetscInt *, PetscInt *) noexcept;
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
  } else if (new_L < old_L && 2 * new_L < df->capacity_d) {
    /* shrink to the exact new size, but only once less than half the capacity is in use, see the design notes */
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
  /*
    A device-valid mask with no mirror is the zero-length degenerate: when L was 0 nothing was
    allocated, yet the mask may have been marked device valid (vacuously, no bytes existed). If
    the bucket has since grown, the host array holds the only valid bytes (the growth zeroed
    them and no device write can have happened without allocating), so fall through and treat
    the field as host authoritative rather than trusting the stale mask.
  */
  if (PetscOffloadDevice(df->offloadmask) && df->data_d) PetscFunctionReturn(PETSC_SUCCESS); /* device copy already valid */
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

/*
  Launch the wire-record gather (pack) or scatter (unpack) kernels over all fields of db. The
  in-record field offsets are precomputed here on the host and passed to the kernels by value in
  chunks of DMSWARM_PACK_NFIELDS_MAX fields; every chunk strides the buffer by the full record
  size and touches only its own fields' byte ranges, so the launches are independent. Callers
  must have made every field device valid. The kernels move device bytes only, so like every
  device-to-device copy they are deliberately not accounted in the -log_view CpuToGpu/GpuToCpu
  totals; the synchronize lets callers consume the results from any stream.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::LaunchRecordKernels_(DMSwarmDataBucket db, PetscBool pack, PetscInt start, PetscInt npoints, const PetscInt *list, void *buf) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  std::size_t        record_size = 0, offset = 0;

  PetscFunctionBegin;
  for (PetscInt f = 0; f < db->nfields; ++f) record_size += db->field[f]->atomic_size;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  for (PetscInt f0 = 0; f0 < db->nfields; f0 += kernels::DMSWARM_PACK_NFIELDS_MAX) {
    kernels::DMSwarmPackCtx ctx;
    const int               nf = static_cast<int>(PetscMin(static_cast<PetscInt>(kernels::DMSWARM_PACK_NFIELDS_MAX), db->nfields - f0));

    ctx.nfields     = nf;
    ctx.record_size = record_size;
    for (int i = 0; i < nf; ++i) {
      const DMSwarmDataField df = db->field[f0 + i];

      ctx.data[i]   = static_cast<char *>(df->data_d);
      ctx.asize[i]  = df->atomic_size;
      ctx.offset[i] = offset;
      offset += df->atomic_size;
    }
    if (pack) PetscCall(PetscCUPMLaunchKernel1D(npoints, 0, stream, kernels::pack_records, ctx, npoints, list, static_cast<char *>(buf)));
    else PetscCall(PetscCUPMLaunchKernel1D(npoints, 0, stream, kernels::unpack_records, ctx, start, npoints, static_cast<const char *>(buf)));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Gather the wire records of the points in the device-resident index list into a contiguous
  device buffer, one thread per point, in list order. Every field is synced to the device first:
  uploading a CPU-resident field is genuine host-to-device traffic and is byte-accounted inside
  FieldToDevice(). The gather only reads the field entries, so the offload masks are unchanged.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::PackRecords(DMSwarmDataBucket db, PetscInt npoints, const PetscInt *list, void *buf) noexcept
{
  PetscFunctionBegin;
  if (!npoints || !db->nfields) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt f = 0; f < db->nfields; ++f) PetscCall(FieldToDevice(db->field[f]));
  PetscCall(LaunchRecordKernels_(db, PETSC_TRUE, 0, npoints, list, buf));
  PetscCall(PetscInfo(nullptr, "Packed %" PetscInt_FMT " wire records on the device\n", npoints));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Scatter npoints contiguous wire records from a device buffer into the bucket points
  [start, start + npoints), one thread per point. Every field is synced to the device first so
  that the untouched points are valid there before the device copy becomes authoritative (the
  same reason DMSwarmDataFieldInsertPoint() syncs to the host before a host write); afterwards
  the device copy alone is valid, so every field mask becomes PETSC_OFFLOAD_GPU.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::UnpackRecords(DMSwarmDataBucket db, PetscInt start, PetscInt npoints, const void *buf) noexcept
{
  PetscFunctionBegin;
  if (!npoints || !db->nfields) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(start >= 0 && start + npoints <= db->field[0]->L, PETSC_COMM_SELF, PETSC_ERR_USER, "records [%" PetscInt_FMT ", %" PetscInt_FMT ") must lie in the allocated range [0, %" PetscInt_FMT ")", start, start + npoints, db->field[0]->L);
  for (PetscInt f = 0; f < db->nfields; ++f) PetscCall(FieldToDevice(db->field[f]));
  PetscCall(LaunchRecordKernels_(db, PETSC_FALSE, start, npoints, nullptr, const_cast<void *>(buf)));
  for (PetscInt f = 0; f < db->nfields; ++f) db->field[f]->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(PetscInfo(nullptr, "Unpacked %" PetscInt_FMT " wire records on the device\n", npoints));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  First phase of the device compaction (see the compact ops slot in data_bucket.h): zero the
  sent mask over all db->L points and raise the flags of the sentlist entries. Compact() runs
  the deterministic exclusive prefix sum over the mask between the two phases, then finishes
  with CompactScatter_().
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::CompactMask_(DMSwarmDataBucket db, PetscInt nsent, const PetscInt *sentlist, PetscInt *mask) noexcept
{
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&stream));
  PetscCall(PetscCUPMMemsetAsync(mask, 0, static_cast<std::size_t>(db->L), stream, true));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscCUPMLaunchKernel1D(nsent, 0, stream, kernels::compact_fill_mask, nsent, sentlist, mask));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Second phase of the device compaction: from the sent mask and its exclusive prefix sum,
  enumerate the nholes filled holes (ascending sent slots < m paired with descending surviving
  tail points >= m, the closed form of the host removal loop) and move only those points'
  records, all fields, in chunked launches like the pack/unpack kernels. Afterwards the device
  copy alone is authoritative, so every field mask becomes PETSC_OFFLOAD_GPU; the caller must
  have synced every field to the device beforehand. The record moves are device-to-device and
  so deliberately not accounted in the CpuToGpu/GpuToCpu totals.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::CompactScatter_(DMSwarmDataBucket db, PetscInt nsent, PetscInt nholes, const PetscInt *mask, const PetscInt *scan, PetscInt *holes, PetscInt *srcs) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  const PetscInt     n = db->L, m = n - nsent;
  std::size_t        offset = 0;

  PetscFunctionBegin;
  if (!nholes) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscCUPMLaunchKernel1D(n, 0, stream, kernels::compact_build_pairs, n, m, nsent, mask, scan, holes, srcs));
  for (PetscInt f0 = 0; f0 < db->nfields; f0 += kernels::DMSWARM_PACK_NFIELDS_MAX) {
    kernels::DMSwarmPackCtx ctx;
    const int               nf = static_cast<int>(PetscMin(static_cast<PetscInt>(kernels::DMSWARM_PACK_NFIELDS_MAX), db->nfields - f0));

    ctx.nfields     = nf;
    ctx.record_size = 0;
    for (int i = 0; i < nf; ++i) {
      const DMSwarmDataField df = db->field[f0 + i];

      ctx.data[i]   = static_cast<char *>(df->data_d);
      ctx.asize[i]  = df->atomic_size;
      ctx.offset[i] = offset;
      offset += df->atomic_size;
    }
    PetscCall(PetscCUPMLaunchKernel1D(nholes, 0, stream, kernels::compact_move_records, ctx, nholes, holes, srcs));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt f = 0; f < db->nfields; ++f) db->field[f]->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(PetscInfo(nullptr, "Compacted %" PetscInt_FMT " sent points on the device: filled %" PetscInt_FMT " holes\n", nsent, nholes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Device compaction per the compact ops slot contract in data_bucket.h: reproduce the host
  swap-with-last removal order through its closed form. The sent mask and its deterministic
  exclusive prefix sum enumerate the ascending sent slots < m and the descending surviving tail
  points >= m; a sparse scatter then moves only the filled holes' records, and the size
  bookkeeping of the host loop's per-point DMSwarmDataBucketRemovePoint() shrinks is applied at
  once by DMSwarmDataBucketRemovePoints_Internal(), so L, buffer, allocated and the zeroed tail
  end bit-identical to the host backend with at most one reallocation of each device mirror. The
  scratch arrays are allocated and freed in stream order on the stream all compaction work runs
  on, which avoids the device-wide synchronization of a plain free.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::Compact(DMSwarmDataBucket db, PetscInt nsent, const PetscInt *sentlist) noexcept
{
  const PetscInt n = db->L, m = n - nsent;
  PetscInt       nholes = 0;
  PetscInt      *mask, *scan, *holes, *srcs;
  cupmStream_t   stream;

  PetscFunctionBegin;
  if (!nsent) PetscFunctionReturn(PETSC_SUCCESS);
  /* untouched survivors must be device valid before the device copy becomes authoritative */
  for (PetscInt f = 0; f < db->nfields; ++f) PetscCall(FieldToDevice(db->field[f]));
  if (m > 0 && db->nfields > 0) {
    PetscCall(GetHandles_(&stream));
    PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&mask), static_cast<std::size_t>(n) * sizeof(*mask), stream));
    PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&scan), static_cast<std::size_t>(n) * sizeof(*scan), stream));
    PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&holes), static_cast<std::size_t>(nsent) * sizeof(*holes), stream));
    PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&srcs), static_cast<std::size_t>(nsent) * sizeof(*srcs), stream));
    PetscCall(CompactMask_(db, nsent, sentlist, mask));
    {
      const auto mask_p = thrust::device_pointer_cast(mask);
      const auto scan_p = thrust::device_pointer_cast(scan);

      PetscCallThrust(THRUST_CALL(thrust::exclusive_scan, stream, mask_p, mask_p + n, scan_p));
    }
    /* nholes = scan[m] = number of sent slots < m = number of surviving tail points >= m */
    PetscCallCUPM(cupmMemcpyAsync(&nholes, scan + m, sizeof(nholes), cupmMemcpyDeviceToHost, stream));
    PetscCallCUPM(cupmStreamSynchronize(stream));
    PetscCall(CompactScatter_(db, nsent, nholes, mask, scan, holes, srcs));
    PetscCallCUPM(cupmFreeAsync(mask, stream));
    PetscCallCUPM(cupmFreeAsync(scan, stream));
    PetscCallCUPM(cupmFreeAsync(holes, stream));
    PetscCallCUPM(cupmFreeAsync(srcs, stream));
  }
  /* the size bookkeeping of the host loop's per-point shrinks, applied at once */
  PetscCall(DMSwarmDataBucketRemovePoints_Internal(db, nsent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Fill the entries [start, end) of a PetscInt-typed field with a constant on the device. Used by
  the device migration fork in swarm_migrate.c for the per-point rank stores and for resetting
  the cell ids of freshly received points, which as host loops would force a device-host round
  trip of the field every migration cycle. The field is synced to the device first so the
  untouched entries stay valid when the device copy becomes authoritative; afterwards only the
  device copy is valid, so the offload mask becomes PETSC_OFFLOAD_GPU.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::FieldFillIntRange(DMSwarmDataField df, PetscInt start, PetscInt end, PetscInt v) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheck(df->atomic_size == sizeof(PetscInt), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Field \"%s\" has atomic size %zu, not the size of PetscInt", df->name, df->atomic_size);
  PetscCheck(start >= 0 && start <= end && end <= df->L, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Fill range [%" PetscInt_FMT ", %" PetscInt_FMT ") must lie in [0, %" PetscInt_FMT "]", start, end, df->L);
  if (start == end) PetscFunctionReturn(PETSC_SUCCESS); /* nothing written, so the offload mask must stay unchanged */
  PetscCall(FieldToDevice(df));
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscCUPMLaunchKernel1D(end - start, 0, stream, kernels::fill_int_range, end - start, v, static_cast<PetscInt *>(df->data_d) + start));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscDeviceContextSynchronize(dctx));
  df->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Build the device-resident send list of the migration fork in swarm_migrate.c: the indices
  i in [start, end) with vals[i] == match, in strictly ascending order (every pointer device
  resident). thrust::copy_if() over an ascending counting iterator is stable and scan based, so
  the order and content are deterministic by construction. The list is allocated here, sized for
  the worst case, and released with SentListDestroy(); an empty range returns a NULL list with
  *nsent = 0.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::SentListCreate(DMSwarmDataBucket, PetscInt start, PetscInt end, PetscInt match, const PetscInt *vals, PetscInt *nsent, PetscInt **sentlist) noexcept
{
  const PetscInt n = end - start;
  PetscInt      *list;
  cupmStream_t   stream;

  PetscFunctionBegin;
  *nsent    = 0;
  *sentlist = nullptr;
  if (n <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(GetHandles_(&stream));
  PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&list), static_cast<std::size_t>(n) * sizeof(*list), stream));
  {
    const auto first  = thrust::counting_iterator<PetscInt>(start);
    const auto last   = thrust::counting_iterator<PetscInt>(end);
    const auto vals_p = thrust::device_pointer_cast(vals);
    const auto out    = thrust::device_pointer_cast(list);
    auto       tail   = out;

    PetscCallThrust(tail = THRUST_CALL(thrust::copy_if, stream, first, last, vals_p + start, out, kernels::EqualsTarget{match}));
    *nsent = static_cast<PetscInt>(tail - out);
  }
  *sentlist = list;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Release a device send list built by SentListCreate(), in stream order on the stream it was
  built on; every consumer synchronizes before returning, so the list is no longer in use
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::SentListDestroy(DMSwarmDataBucket, PetscInt **sentlist) noexcept
{
  cupmStream_t stream;

  PetscFunctionBegin;
  if (*sentlist) {
    PetscCall(GetHandles_(&stream));
    PetscCallCUPM(cupmFreeAsync(*sentlist, stream));
  }
  *sentlist = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmDataBucket_CUPM<T>::SetUp(DMSwarmDataBucket db) noexcept
{
  PetscFunctionBegin;
  db->ops.fieldalloc        = FieldAlloc;
  db->ops.fieldallocdevice  = FieldAllocDevice;
  db->ops.fieldtodevice     = FieldToDevice;
  db->ops.fieldtohost       = FieldToHost;
  db->ops.fieldzerorange    = FieldZeroRange;
  db->ops.fieldfreedevice   = FieldFreeDevice;
  db->ops.fieldfillintrange = FieldFillIntRange;
  db->ops.setsizes          = SetSizes;
  db->ops.packrecords       = PackRecords;
  db->ops.unpackrecords     = UnpackRecords;
  db->ops.compact           = Compact;
  db->ops.sentlistcreate    = SentListCreate;
  db->ops.sentlistdestroy   = SentListDestroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
