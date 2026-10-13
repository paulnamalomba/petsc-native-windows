#pragma once

#include <petsc/private/dmpleximpl.h>

#include <petsc/private/cupmobject.hpp>
#include <petsc/private/deviceimpl.h>

#include "../src/sys/objects/device/impls/cupm/kernels.hpp"

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

using ::Petsc::device::cupm::kernels::util::mul_rn;

/*
  Parameter block for the location kernel: the location cache flattened to device pointers and
  plain values, passed to the kernel by value.
*/
struct DMPlexLocateCtx {
  PetscInt         cdim;                     /* Coordinate dimension (== topological dimension) */
  PetscInt         cStart, cEnd;             /* Covered cell range */
  PetscInt         uniformCt;                /* DMPolytopeType of every cell, or -1 when celltype[] is used */
  const PetscInt  *celltype;                 /* [cEnd-cStart] DMPolytopeType per cell, or NULL when uniform */
  const PetscReal *v0;                       /* [(cEnd-cStart) * cdim] affine origin per simplex cell */
  const PetscReal *invJ;                     /* [(cEnd-cStart) * cdim * cdim] affine inverse Jacobian per simplex cell */
  PetscInt         maxCorners;               /* Vertices per cell slot in cellCoords */
  const PetscReal *cellCoords;               /* [(cEnd-cStart) * maxCorners * cdim] closure-order vertex coordinates per tensor cell */
  const PetscBool *leafCell;                 /* [cEnd-cStart] cells the brute-force scan skips, or NULL */
  PetscReal        gmin[3], gmax[3];         /* Global bounding box */
  PetscBool        useHash;                  /* Scan grid hash bins instead of all cells, mirroring DM_Plex->useHashLocation */
  PetscReal        lower[3], upper[3], h[3]; /* Grid hash geometry */
  PetscInt         n[3];                     /* Grid hash subbox counts */
  PetscInt         binStart, binEnd;         /* Chart of the grid hash cell section */
  const PetscInt  *binOffset;                /* [binEnd-binStart+1] CSR offsets into boxCells */
  const PetscInt  *boxCells;                 /* Candidate cells per bin, in host scan order */
};

/*
  The containment tests below are operation-for-operation device ports of the host tests in
  plexgeometry.c (DMPlexLocatePoint_Simplex_1D/2D/3D_Internal(), the linear
  DMPlexLocatePoint_Quad_2D/Hex_3D_Linear_Internal()) so that every marginal decision matches the
  host bitwise. Floating-point contraction is prevented in each of them: the host library is built
  at -O0 for a target without fused multiply-add, while device compilers contract by default, and
  a contracted multiply-add can flip a containment decision on the cell boundary. Location decides
  which cell, and hence which rank, owns a particle, so unlike the projection kernels it must
  match the host exactly: every product therefore goes through mul_rn(), which no device compiler
  fuses with an addition.
*/

PETSC_DEVICE_INLINE_DECL static PetscBool ContainsPoint_Segment(const PetscReal point[], const PetscReal v0[], const PetscReal invJ[]) noexcept
{
  const PetscReal eps = PETSC_SQRT_MACHINE_EPSILON;
  const PetscReal xi  = mul_rn(invJ[0], point[0] - v0[0]);

  return (xi >= -eps) && (xi <= 2. + eps) ? PETSC_TRUE : PETSC_FALSE;
}

PETSC_DEVICE_INLINE_DECL static PetscBool ContainsPoint_Triangle(const PetscReal point[], const PetscReal v0[], const PetscReal invJ[]) noexcept
{
  const PetscReal eps   = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal       xi[2] = {0., 0.};

  for (PetscInt i = 0; i < 2; ++i) {
    for (PetscInt j = 0; j < 2; ++j) xi[i] += mul_rn(invJ[i * 2 + j], point[j] - v0[j]);
  }
  return (xi[0] >= -eps) && (xi[1] >= -eps) && (xi[0] + xi[1] <= 2.0 + eps) ? PETSC_TRUE : PETSC_FALSE;
}

PETSC_DEVICE_INLINE_DECL static PetscBool ContainsPoint_Tetrahedron(const PetscReal point[], const PetscReal v0[], const PetscReal invJ[]) noexcept
{
  const PetscReal eps = PETSC_SQRT_MACHINE_EPSILON;
  const PetscReal x   = point[0];
  const PetscReal y   = point[1];
  const PetscReal z   = point[2];
  PetscReal       xi, eta, zeta;

  xi   = mul_rn(invJ[0 * 3 + 0], x - v0[0]) + mul_rn(invJ[0 * 3 + 1], y - v0[1]) + mul_rn(invJ[0 * 3 + 2], z - v0[2]);
  eta  = mul_rn(invJ[1 * 3 + 0], x - v0[0]) + mul_rn(invJ[1 * 3 + 1], y - v0[1]) + mul_rn(invJ[1 * 3 + 2], z - v0[2]);
  zeta = mul_rn(invJ[2 * 3 + 0], x - v0[0]) + mul_rn(invJ[2 * 3 + 1], y - v0[1]) + mul_rn(invJ[2 * 3 + 2], z - v0[2]);

  return (xi >= -eps) && (eta >= -eps) && (zeta >= -eps) && (xi + eta + zeta <= 2.0 + eps) ? PETSC_TRUE : PETSC_FALSE;
}

// The ray-casting, or even-odd algorithm, identical to DMPlexLocatePoint_Quad_2D_Linear_Internal()
PETSC_DEVICE_INLINE_DECL static PetscBool ContainsPoint_Quadrilateral(const PetscReal point[], const PetscReal coords[]) noexcept
{
  const PetscInt  faces[8]  = {0, 1, 1, 2, 2, 3, 3, 0};
  const PetscReal x         = point[0];
  const PetscReal y         = point[1];
  PetscInt        crossings = 0;

  for (PetscInt f = 0; f < 4; ++f) {
    const PetscReal x_i = coords[faces[2 * f + 0] * 2 + 0];
    const PetscReal y_i = coords[faces[2 * f + 0] * 2 + 1];
    const PetscReal x_j = coords[faces[2 * f + 1] * 2 + 0];
    const PetscReal y_j = coords[faces[2 * f + 1] * 2 + 1];

    if ((x == x_j) && (y == y_j)) {
      // point is a corner
      crossings = 1;
      break;
    }
    if ((y_j > y) != (y_i > y)) {
      const PetscReal slope = mul_rn(x - x_j, y_i - y_j) - mul_rn(x_i - x_j, y - y_j);

      if (slope == 0) {
        // point is a corner
        crossings = 1;
        break;
      }
      if ((slope < 0) != (y_i < y_j)) ++crossings;
    }
  }
  return crossings % 2 ? PETSC_TRUE : PETSC_FALSE;
}

// The six half-space tests of DMPlexLocatePoint_Hex_3D_Linear_Internal()
PETSC_DEVICE_INLINE_DECL static PetscBool ContainsPoint_Hexahedron(const PetscReal point[], const PetscReal coords[]) noexcept
{
  const PetscInt faces[24] = {0, 3, 2, 1, 5, 4, 7, 6, 3, 0, 4, 5, 1, 2, 6, 7, 3, 5, 6, 2, 0, 1, 7, 4};

  for (PetscInt f = 0; f < 6; ++f) {
    PetscReal v_i[3], v_j[3], normal[3], pp[3], dot;

    v_i[0]    = coords[faces[f * 4 + 3] * 3 + 0] - coords[faces[f * 4 + 0] * 3 + 0];
    v_i[1]    = coords[faces[f * 4 + 3] * 3 + 1] - coords[faces[f * 4 + 0] * 3 + 1];
    v_i[2]    = coords[faces[f * 4 + 3] * 3 + 2] - coords[faces[f * 4 + 0] * 3 + 2];
    v_j[0]    = coords[faces[f * 4 + 1] * 3 + 0] - coords[faces[f * 4 + 0] * 3 + 0];
    v_j[1]    = coords[faces[f * 4 + 1] * 3 + 1] - coords[faces[f * 4 + 0] * 3 + 1];
    v_j[2]    = coords[faces[f * 4 + 1] * 3 + 2] - coords[faces[f * 4 + 0] * 3 + 2];
    normal[0] = mul_rn(v_i[1], v_j[2]) - mul_rn(v_i[2], v_j[1]);
    normal[1] = mul_rn(v_i[2], v_j[0]) - mul_rn(v_i[0], v_j[2]);
    normal[2] = mul_rn(v_i[0], v_j[1]) - mul_rn(v_i[1], v_j[0]);
    pp[0]     = coords[faces[f * 4 + 0] * 3 + 0] - point[0];
    pp[1]     = coords[faces[f * 4 + 0] * 3 + 1] - point[1];
    pp[2]     = coords[faces[f * 4 + 0] * 3 + 2] - point[2];
    dot       = mul_rn(normal[0], pp[0]) + mul_rn(normal[1], pp[1]) + mul_rn(normal[2], pp[2]);

    if (dot < 0.0) return PETSC_FALSE;
  }
  return PETSC_TRUE;
}

// Per-cell-type dispatch in the order of DMPlexLocatePoint_Internal(); the cache admits no other type
PETSC_DEVICE_INLINE_DECL static PetscBool CellContainsPoint(const DMPlexLocateCtx &ctx, PetscInt c, const PetscReal point[]) noexcept
{
  const PetscInt cell = c - ctx.cStart;
  const PetscInt ct   = ctx.celltype ? ctx.celltype[cell] : ctx.uniformCt;

  switch (ct) {
  case DM_POLYTOPE_SEGMENT:
    return ContainsPoint_Segment(point, ctx.v0 + cell * ctx.cdim, ctx.invJ + cell * ctx.cdim * ctx.cdim);
  case DM_POLYTOPE_TRIANGLE:
    return ContainsPoint_Triangle(point, ctx.v0 + cell * ctx.cdim, ctx.invJ + cell * ctx.cdim * ctx.cdim);
  case DM_POLYTOPE_QUADRILATERAL:
    return ContainsPoint_Quadrilateral(point, ctx.cellCoords + cell * ctx.maxCorners * ctx.cdim);
  case DM_POLYTOPE_TETRAHEDRON:
    return ContainsPoint_Tetrahedron(point, ctx.v0 + cell * ctx.cdim, ctx.invJ + cell * ctx.cdim * ctx.cdim);
  case DM_POLYTOPE_HEXAHEDRON:
    return ContainsPoint_Hexahedron(point, ctx.cellCoords + cell * ctx.maxCorners * ctx.cdim);
  default:
    break;
  }
  return PETSC_FALSE;
}

// The upper-snap bin query of PetscGridHashGetEnclosingBoxQuery() (which has no lower snap)
PETSC_DEVICE_INLINE_DECL static PetscBool EnclosingBin(const DMPlexLocateCtx &ctx, const PetscReal point[], PetscInt *bin) noexcept
{
  const PetscInt dim     = ctx.cdim;
  PetscInt       dbin[3] = {-1, -1, -1}, b;

  for (PetscInt d = 0; d < dim; ++d) {
    PetscInt dbox = (PetscInt)PetscFloorReal((point[d] - ctx.lower[d]) / ctx.h[d]);

    if (dbox == ctx.n[d] && PetscAbsReal(point[d] - ctx.upper[d]) < 1.0e-9) dbox = ctx.n[d] - 1;
    if (dbox < 0 || dbox >= ctx.n[d]) return PETSC_FALSE;
    dbin[d] = dbox;
  }
  b = dbin[dim - 1];
  for (PetscInt d = dim - 2; d >= 0; --d) b = b * ctx.n[d] + dbin[d];
  // It is possible for a box to overlap no grid cells
  if (b < ctx.binStart || b >= ctx.binEnd) return PETSC_FALSE;
  *bin = b;
  return PETSC_TRUE;
}

/*
  One thread per point; each thread reproduces the per-point decision sequence of
  DMLocatePoints_Plex() exactly: bounding-box reject, then the incoming cell id as initial
  guess, then either the grid hash bin scan or the brute-force scan in host candidate order,
  first containing cell wins. No thread reads another thread's result, so the kernel is
  deterministic by construction.
*/
PETSC_KERNEL_DECL static void locate_points(const DMPlexLocateCtx ctx, const PetscInt npoints, const PetscInt bs, const PetscReal *PETSC_RESTRICT coords, PetscInt *PETSC_RESTRICT cellids)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(npoints, [&](PetscInt p) {
    const PetscReal *point   = coords + p * bs;
    const PetscInt   seed    = cellids[p];
    PetscInt         found   = DMLOCATEPOINT_POINT_NOT_FOUND;
    PetscBool        outside = PETSC_FALSE;

    for (PetscInt d = 0; d < ctx.cdim; ++d) {
      if (point[d] < ctx.gmin[d] || point[d] > ctx.gmax[d]) {
        outside = PETSC_TRUE;
        break;
      }
    }
    if (!outside) {
      /* The host uses any incoming cell id as initial guess; ids outside the covered range would be host errors and are ignored here */
      if (seed != DMLOCATEPOINT_POINT_NOT_FOUND && seed >= ctx.cStart && seed < ctx.cEnd && CellContainsPoint(ctx, seed, point)) found = seed;
      if (found == DMLOCATEPOINT_POINT_NOT_FOUND) {
        if (ctx.useHash) {
          PetscInt bin;

          if (ctx.binOffset && EnclosingBin(ctx, point, &bin)) {
            for (PetscInt i = ctx.binOffset[bin - ctx.binStart]; i < ctx.binOffset[bin - ctx.binStart + 1]; ++i) {
              const PetscInt c = ctx.boxCells[i];

              if (CellContainsPoint(ctx, c, point)) {
                found = c;
                break;
              }
            }
          }
        } else {
          for (PetscInt c = ctx.cStart; c < ctx.cEnd; ++c) {
            if (ctx.leafCell && ctx.leafCell[c - ctx.cStart]) continue;
            if (CellContainsPoint(ctx, c, point)) {
              found = c;
              break;
            }
          }
        }
      }
    }
    cellids[p] = found;
  });
}

} // namespace kernels

/*
  Device-mirror management for the DMPlexLocationCache (struct in dmpleximpl.h).

  The cache is entirely host-built; the device side is a plain one-shot mirror. Upload()
  allocates one device buffer per present host array and copies it host-to-device (the copies
  are byte-accounted in -log_view by PetscCUPMMemcpyAsync() itself), then synchronizes so the
  caller may launch kernels on any stream. The mirrors are immutable: coordinate changes destroy
  the whole cache (see DMPlexInvalidateStaleLocationCache_Internal()) and the next device
  request re-uploads a freshly built cache. FreeDevice() releases the mirrors; it is installed
  as cache->freedevice by the C shim so that DMPlexLocationCacheDestroy_Internal() can free them
  without referencing device symbols.

  LocatePoints() launches the point-location kernel above over an uploaded cache; results are
  bitwise identical to DMLocatePoints_Plex() by construction, see the kernel comments.
*/
template <DeviceType T>
struct DMPlexLocationCache_CUPM : CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode Upload(DMPlexLocationCache) noexcept;
  static PetscErrorCode FreeDevice(DMPlexLocationCache) noexcept;
  static PetscErrorCode LocatePoints(DMPlexLocationCache, PetscBool, PetscInt, PetscInt, const PetscReal[], PetscInt[]) noexcept;

private:
  template <typename A>
  static PetscErrorCode UploadArray_(const A *, PetscInt, void **, cupmStream_t) noexcept;
};

template <DeviceType T>
template <typename A>
inline PetscErrorCode DMPlexLocationCache_CUPM<T>::UploadArray_(const A *host, PetscInt count, void **device, cupmStream_t stream) noexcept
{
  A *d = nullptr;

  PetscFunctionBegin;
  if (host && count > 0) {
    PetscCall(PetscCUPMMallocAsync(&d, count, stream));
    PetscCall(PetscCUPMMemcpyAsync(d, host, count, cupmMemcpyHostToDevice, stream));
  }
  *device = d;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMPlexLocationCache_CUPM<T>::Upload(DMPlexLocationCache cache) noexcept
{
  const PetscInt     Ncell = cache->cEnd - cache->cStart;
  const PetscInt     cdim = cache->cdim, nbins = cache->binEnd - cache->binStart;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  if (cache->onDevice) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(UploadArray_(cache->celltype, Ncell, &cache->celltype_d, stream));
  PetscCall(UploadArray_(cache->v0, Ncell * cdim, &cache->v0_d, stream));
  PetscCall(UploadArray_(cache->invJ, Ncell * cdim * cdim, &cache->invJ_d, stream));
  PetscCall(UploadArray_(cache->cellCoords, Ncell * cache->maxCorners * cdim, &cache->cellCoords_d, stream));
  PetscCall(UploadArray_(cache->leafCell, Ncell, &cache->leafCell_d, stream));
  PetscCall(UploadArray_(cache->binOffset, nbins + 1, &cache->binOffset_d, stream));
  PetscCall(UploadArray_(cache->boxCells, cache->binOffset ? cache->binOffset[nbins] : 0, &cache->boxCells_d, stream));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  cache->onDevice = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMPlexLocationCache_CUPM<T>::FreeDevice(DMPlexLocationCache cache) noexcept
{
  void        *arrays[7];
  cupmStream_t stream;

  PetscFunctionBegin;
  arrays[0] = cache->celltype_d;
  arrays[1] = cache->v0_d;
  arrays[2] = cache->invJ_d;
  arrays[3] = cache->cellCoords_d;
  arrays[4] = cache->leafCell_d;
  arrays[5] = cache->binOffset_d;
  arrays[6] = cache->boxCells_d;
  PetscCall(GetHandles_(&stream));
  for (std::size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(arrays); ++i) {
    if (arrays[i]) PetscCallCUPM(cupmFreeAsync(arrays[i], stream));
  }
  cache->celltype_d = cache->v0_d = cache->invJ_d = cache->cellCoords_d = cache->leafCell_d = cache->binOffset_d = cache->boxCells_d = nullptr;
  cache->onDevice                                                                                                                    = PETSC_FALSE;
  cache->freedevice                                                                                                                  = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMPlexLocationCache_CUPM<T>::LocatePoints(DMPlexLocationCache cache, PetscBool useHash, PetscInt npoints, PetscInt bs, const PetscReal coords[], PetscInt cellids[]) noexcept
{
  kernels::DMPlexLocateCtx ctx;
  PetscDeviceContext       dctx;
  cupmStream_t             stream;

  PetscFunctionBegin;
  ctx.cdim       = cache->cdim;
  ctx.cStart     = cache->cStart;
  ctx.cEnd       = cache->cEnd;
  ctx.uniformCt  = cache->celltype ? -1 : (PetscInt)cache->uniformCt;
  ctx.celltype   = static_cast<const PetscInt *>(cache->celltype_d);
  ctx.v0         = static_cast<const PetscReal *>(cache->v0_d);
  ctx.invJ       = static_cast<const PetscReal *>(cache->invJ_d);
  ctx.maxCorners = cache->maxCorners;
  ctx.cellCoords = static_cast<const PetscReal *>(cache->cellCoords_d);
  ctx.leafCell   = static_cast<const PetscBool *>(cache->leafCell_d);
  ctx.useHash    = useHash;
  for (PetscInt d = 0; d < 3; ++d) {
    ctx.gmin[d]  = cache->gmin[d];
    ctx.gmax[d]  = cache->gmax[d];
    ctx.lower[d] = cache->lower[d];
    ctx.upper[d] = cache->upper[d];
    ctx.h[d]     = cache->h[d];
    ctx.n[d]     = cache->n[d];
  }
  ctx.binStart  = cache->binStart;
  ctx.binEnd    = cache->binEnd;
  ctx.binOffset = static_cast<const PetscInt *>(cache->binOffset_d);
  ctx.boxCells  = static_cast<const PetscInt *>(cache->boxCells_d);
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscCUPMLaunchKernel1D(npoints, 0, stream, kernels::locate_points, ctx, npoints, bs, coords, cellids));
  PetscCall(PetscLogGpuTimeEnd());
  /* Synchronize so callers may consume the results from any stream */
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
