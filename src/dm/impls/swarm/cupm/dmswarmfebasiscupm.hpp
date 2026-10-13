#pragma once

#include <petsc/private/cupmobject.hpp>
#include <petsc/private/deviceimpl.h>

#include "../src/sys/objects/device/impls/cupm/kernels.hpp"
#include "../swarmfebasis.h"

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

/* Parameter block of a DMSwarmFEBasis (swarmfebasis.h), with its device mirrors, passed to kernels by value */
struct DMSwarmFEBasisCtx {
  PetscInt                dim, pdim, fdim, fpdim;
  PetscBool               tensor;
  DMSwarmFEBasisPrimeType ptype;
  const PetscInt         *pkd;
  const PetscReal        *coef;
  const PetscReal        *invV;
};

/*
  The evaluation below computes PetscFECreateTabulation() (B only) for the bases DMSwarmFEBasisGet_Internal() accepts,
  with the same steps: the point loop of PetscDTPKDEvalJet() (jet 0) followed by its scaling, or the coordinate basis of
  PetscSpaceEvaluate_Polynomial() for degree 1; the products of PetscSpaceEvaluate_Tensor() in factor order; and the
  contraction of TensorContract_Private() with invV. The values agree with the host to rounding: the device compiler
  may contract multiply-adds, which the host library (built for a target without fused multiply-add) does not.
*/
PETSC_DEVICE_INLINE_DECL static void DMSwarmFEBasisFactorPrime(const DMSwarmFEBasisCtx &ctx, const PetscReal x[], PetscReal p[]) noexcept
{
  const PetscInt dim = ctx.fdim;

  switch (ctx.ptype) {
  case DMSWARM_FEBASIS_CONSTANT:
    p[0] = 1.;
    break;
  case DMSWARM_FEBASIS_LINEAR:
    p[0] = 1.;
    for (PetscInt d = 0; d < dim; ++d) p[1 + d] = x[d];
    break;
  case DMSWARM_FEBASIS_PKD:
    for (PetscInt i = 0; i < ctx.fpdim; ++i) {
      const PetscInt d = ctx.pkd[3 * i + 0], m1 = ctx.pkd[3 * i + 1], m2 = ctx.pkd[3 * i + 2];
      PetscReal      thetanm1, thetanm1x, thetanm2;

      if (m1 < 0) {
        p[i] = 1.;
        continue;
      }
      thetanm1x = dim - (d + 1) + 2. * x[d];
      for (PetscInt e = d + 1; e < dim; ++e) thetanm1x += x[e];
      thetanm1x *= 0.5;
      thetanm1 = (2. - (dim - (d + 1)));
      for (PetscInt e = d + 1; e < dim; ++e) thetanm1 -= x[e];
      thetanm1 *= 0.5;
      thetanm2 = thetanm1 * thetanm1;
      p[i]     = (ctx.coef[4 * i + 0] * thetanm1 + ctx.coef[4 * i + 1] * thetanm1x) * p[m1];
      if (m2 >= 0) p[i] -= ctx.coef[4 * i + 2] * thetanm2 * p[m2];
    }
    for (PetscInt i = 0; i < ctx.fpdim; ++i) p[i] *= ctx.coef[4 * i + 3];
    break;
  }
}

/* B[pdim]: the nodal basis functions at the reference point xi[dim] */
PETSC_DEVICE_INLINE_DECL static void DMSwarmFEBasisTabulate(const DMSwarmFEBasisCtx &ctx, const PetscReal xi[], PetscReal B[]) noexcept
{
  PetscReal prime[DMSWARM_FEBASIS_MAX_PDIM];

  if (ctx.tensor) {
    PetscReal f[3][DMSWARM_FEBASIS_MAX_FACTOR_PDIM];

    for (PetscInt s = 0; s < ctx.dim; ++s) DMSwarmFEBasisFactorPrime(ctx, &xi[s], f[s]);
    /* Factor s varies with stride fpdim^s in the prime index */
    for (PetscInt i = 0; i < ctx.pdim; ++i) {
      PetscInt  r = i;
      PetscReal v = 1.;

      for (PetscInt s = 0; s < ctx.dim; ++s, r /= ctx.fpdim) v *= f[s][r % ctx.fpdim];
      prime[i] = v;
    }
  } else DMSwarmFEBasisFactorPrime(ctx, xi, prime);
  if (!ctx.invV) {
    for (PetscInt n = 0; n < ctx.pdim; ++n) B[n] = prime[n];
    return;
  }
  for (PetscInt n = 0; n < ctx.pdim; ++n) {
    PetscReal c = 0.;

    for (PetscInt l = 0; l < ctx.pdim; ++l) c += ctx.invV[l * ctx.pdim + n] * prime[l];
    B[n] = c;
  }
}

/* One thread per point: B[p * pdim + n] for the reference points xi[p * dim + d] */
PETSC_KERNEL_DECL static void tabulate_points(const DMSwarmFEBasisCtx ctx, const PetscInt npoints, const PetscReal *PETSC_RESTRICT xi, PetscReal *PETSC_RESTRICT B)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(npoints, [&](PetscInt p) { DMSwarmFEBasisTabulate(ctx, &xi[p * ctx.dim], &B[p * ctx.pdim]); });
}

/* Parameter block of the particle mass matrix values kernel */
struct DMSwarmMassCtx {
  DMSwarmFEBasisCtx basis;
  PetscInt          numF;      /* Closure size of every cell, equal to basis.pdim */
  const PetscInt   *list;      /* [npoints] particle of each sort CSR slot */
  const PetscInt   *cell;      /* [npoints] cell of each sort CSR slot */
  const PetscReal  *coords;    /* [Np * dim] particle coordinates */
  const PetscReal  *v0, *invJ; /* Cell geometry, see DMSwarmCellGeometry */
  const PetscReal  *detJ;
  PetscBool         useDelta; /* Delta function particles: weight 1 instead of detJ */
};

/*
  One thread per sort CSR slot q, writing the numF values of the particle in that slot at q * numF, the position
  DMSwarmComputeMassMatrixCOO_Private() gives them on the host: CoordinatesRealToRef() with the reference origin -1,
  the tabulation above, and the elemMat entry 0.0 += B * w of DMSwarmFillMassMatrix().
*/
PETSC_KERNEL_DECL static void mass_values(const DMSwarmMassCtx ctx, const PetscInt npoints, PetscScalar *PETSC_RESTRICT vals)
{
#if !PetscDefined(USE_COMPLEX)
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(npoints, [&](PetscInt q) {
    const PetscInt   dim = ctx.basis.dim, c = ctx.cell[q];
    const PetscReal *x  = &ctx.coords[ctx.list[q] * dim];
    const PetscReal *v0 = &ctx.v0[c * dim], *invJ = &ctx.invJ[c * dim * dim];
    const PetscReal  w = ctx.useDelta ? 1.0 : ctx.detJ[c];
    PetscReal        xi[3], B[DMSWARM_FEBASIS_MAX_PDIM];

    for (PetscInt d = 0; d < dim; ++d) {
      xi[d] = -1.0;
      for (PetscInt e = 0; e < dim; ++e) xi[d] += invJ[d * dim + e] * (x[e] - v0[e]);
    }
    DMSwarmFEBasisTabulate(ctx.basis, xi, B);
    for (PetscInt k = 0; k < ctx.numF; ++k) {
      PetscScalar v = 0.0;

      v += B[k] * w;
      vals[q * ctx.numF + k] = v;
    }
  });
#endif
}

/* Parameter block of the Gram kernel of the particle mass matrix square */
struct DMSwarmMassSquareCtx {
  PetscInt           numF;          /* Basis values per particle, the leading dimension of B */
  PetscInt           ld;            /* Leading dimension of the staged rows in shared memory, numF rounded up to odd */
  PetscInt           ncells;        /* Cells of the sort CSR */
  const PetscInt    *pcell_offsets; /* [ncells + 1] sort CSR slot range of each cell */
  const PetscCount  *sq_offsets;    /* [ncells + 1] COO offset of the n_c x n_c block of each cell */
  const PetscCount  *tile_offsets;  /* [ncells + 1] first tile of each cell, ceil(n_c / DMSWARM_MASS_SQUARE_TILE)^2 tiles per cell */
  PetscCount         ntiles;        /* Total number of tiles */
  PetscCount         sq_total;      /* Total number of block entries, sq_offsets[ncells] */
  const PetscScalar *B;             /* [npoints * numF] basis values of the particle in each sort CSR slot */
};

/*
  The n_c x n_c block of a cell with n_c particles is the Gram matrix of their basis values, entry (a, b) the sum over
  k of B[a, k] B[b, k], stored row-major at sq_offsets[c]. The blocks are cut into DMSWARM_MASS_SQUARE_TILE x
  DMSWARM_MASS_SQUARE_TILE tiles so that the work of a cell grows with n_c^2 across thread blocks rather than inside
  one, whatever the particle distribution. A thread block of 256 threads computes one tile: it stages the basis values
  of the tile's rows and columns in shared memory (each a contiguous range of B, as the slots of a cell are), then each
  thread accumulates a 2 x 2 register tile, with consecutive threads writing consecutive entries of a row. Every entry
  is summed in increasing k from 0.0, the loop order of the reference BLAS gemm that
  DMSwarmComputeMassMatrixSquare_Private() calls; the values agree with the host to rounding.
*/
PETSC_KERNEL_DECL static void mass_square_values(const DMSwarmMassSquareCtx ctx, PetscScalar *PETSC_RESTRICT vals)
{
#if !PetscDefined(USE_COMPLEX)
  constexpr PetscInt            T = DMSWARM_MASS_SQUARE_TILE, H = T / 2;
  extern __shared__ PetscScalar mass_square_shared[];
  PetscScalar                  *rows = mass_square_shared, *cols = mass_square_shared + T * ctx.ld;
  const PetscInt                tx = threadIdx.x % H, ty = threadIdx.x / H;

  for (PetscCount t = blockIdx.x; t < ctx.ntiles; t += gridDim.x) {
    PetscInt   lo = 0, hi = ctx.ncells, start, n, nt, a0, b0, na, nb;
    PetscCount local;

    /* the cell whose tile range holds t: the last c with tile_offsets[c] <= t */
    while (hi - lo > 1) {
      const PetscInt mid = lo + (hi - lo) / 2;

      if (ctx.tile_offsets[mid] <= t) lo = mid;
      else hi = mid;
    }
    start = ctx.pcell_offsets[lo];
    n     = ctx.pcell_offsets[lo + 1] - start;
    nt    = (n + T - 1) / T;
    local = t - ctx.tile_offsets[lo];
    a0    = (PetscInt)(local / nt) * T;
    b0    = (PetscInt)(local % nt) * T;
    na    = PetscMin(T, n - a0);
    nb    = PetscMin(T, n - b0);
    __syncthreads(); /* the previous tile is done with the shared memory */
    for (PetscInt i = threadIdx.x; i < T * ctx.numF; i += blockDim.x) {
      const PetscInt r = i / ctx.numF, k = i % ctx.numF;

      rows[r * ctx.ld + k] = r < na ? ctx.B[(PetscCount)(start + a0 + r) * ctx.numF + k] : 0.0;
      cols[r * ctx.ld + k] = r < nb ? ctx.B[(PetscCount)(start + b0 + r) * ctx.numF + k] : 0.0;
    }
    __syncthreads();
    {
      PetscScalar g[2][2] = {
        {0.0, 0.0},
        {0.0, 0.0}
      };

      for (PetscInt k = 0; k < ctx.numF; ++k) {
        for (PetscInt i = 0; i < 2; ++i) {
          for (PetscInt j = 0; j < 2; ++j) g[i][j] += rows[(ty + i * H) * ctx.ld + k] * cols[(tx + j * H) * ctx.ld + k];
        }
      }
      for (PetscInt i = 0; i < 2; ++i) {
        for (PetscInt j = 0; j < 2; ++j) {
          const PetscInt a = ty + i * H, b = tx + j * H;

          if (a < na && b < nb) vals[ctx.sq_offsets[lo] + (PetscCount)(a0 + a) * n + b0 + b] = g[i][j];
        }
      }
    }
  }
#endif
}

} // namespace kernels

/*
  Device mirrors of a DMSwarmFEBasis. The basis is immutable once built (it is cached on its discretization), so the
  mirrors are uploaded once and freed with the basis through basis->freedevice, installed by the C shim.
*/
template <DeviceType T>
struct DMSwarmFEBasis_CUPM : CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode             Upload(DMSwarmFEBasis) noexcept;
  static PetscErrorCode             FreeDevice(DMSwarmFEBasis) noexcept;
  static kernels::DMSwarmFEBasisCtx GetCtx(DMSwarmFEBasis) noexcept;
  static PetscErrorCode             Tabulate(DMSwarmFEBasis, PetscInt, const PetscReal[], PetscReal[]) noexcept;
  static PetscErrorCode             UploadGeometry(DMSwarmCellGeometry) noexcept;
  static PetscErrorCode             FreeGeometryDevice(DMSwarmCellGeometry) noexcept;
  static PetscErrorCode             MassSetValuesCOO(Mat, DMSwarmFEBasis, DMSwarmCellGeometry, PetscInt, const PetscInt[], const PetscInt[], const PetscReal[], PetscInt, PetscBool) noexcept;
  static PetscErrorCode MassSquareSetValuesCOO(Mat, DMSwarmFEBasis, DMSwarmCellGeometry, PetscInt, const PetscInt[], const PetscInt[], const PetscReal[], PetscInt, PetscBool, PetscInt, const PetscInt[], const PetscCount[], const PetscCount[]) noexcept;

private:
  template <typename A>
  static PetscErrorCode          UploadArray_(const A *, PetscInt, void **, cupmStream_t) noexcept;
  static kernels::DMSwarmMassCtx GetMassCtx_(DMSwarmFEBasis, DMSwarmCellGeometry, const PetscInt[], const PetscInt[], const PetscReal[], PetscInt, PetscBool) noexcept;
  static PetscErrorCode          MassSquareGram_(const kernels::DMSwarmMassSquareCtx &, PetscScalar[], cupmStream_t) noexcept;
};

template <DeviceType T>
template <typename A>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::UploadArray_(const A *host, PetscInt count, void **device, cupmStream_t stream) noexcept
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
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::Upload(DMSwarmFEBasis basis) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(UploadArray_(basis->pkd, 3 * basis->fpdim, &basis->pkd_d, stream));
  PetscCall(UploadArray_(basis->coef, 4 * basis->fpdim, &basis->coef_d, stream));
  PetscCall(UploadArray_(basis->invV, basis->pdim * basis->pdim, &basis->invV_d, stream));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::FreeDevice(DMSwarmFEBasis basis) noexcept
{
  void        *arrays[3] = {basis->pkd_d, basis->coef_d, basis->invV_d};
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&stream));
  for (std::size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(arrays); ++i) {
    if (arrays[i]) PetscCallCUPM(cupmFreeAsync(arrays[i], stream));
  }
  basis->pkd_d = basis->coef_d = basis->invV_d = nullptr;
  basis->freedevice                            = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline kernels::DMSwarmFEBasisCtx DMSwarmFEBasis_CUPM<T>::GetCtx(DMSwarmFEBasis basis) noexcept
{
  kernels::DMSwarmFEBasisCtx ctx;

  ctx.dim    = basis->dim;
  ctx.pdim   = basis->pdim;
  ctx.fdim   = basis->fdim;
  ctx.fpdim  = basis->fpdim;
  ctx.tensor = basis->tensor;
  ctx.ptype  = basis->ptype;
  ctx.pkd    = static_cast<const PetscInt *>(basis->pkd_d);
  ctx.coef   = static_cast<const PetscReal *>(basis->coef_d);
  ctx.invV   = static_cast<const PetscReal *>(basis->invV_d);
  return ctx;
}

template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::Tabulate(DMSwarmFEBasis basis, PetscInt npoints, const PetscReal xi[], PetscReal B[]) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscCUPMLaunchKernel1D(npoints, 0, stream, kernels::tabulate_points, GetCtx(basis), npoints, xi, B));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::UploadGeometry(DMSwarmCellGeometry geom) noexcept
{
  const PetscInt     Ncell = geom->cEnd - geom->cStart, dim = geom->dim;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(UploadArray_(geom->v0, Ncell * dim, &geom->v0_d, stream));
  PetscCall(UploadArray_(geom->invJ, Ncell * dim * dim, &geom->invJ_d, stream));
  PetscCall(UploadArray_(geom->detJ, Ncell, &geom->detJ_d, stream));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::FreeGeometryDevice(DMSwarmCellGeometry geom) noexcept
{
  void        *arrays[3] = {geom->v0_d, geom->invJ_d, geom->detJ_d};
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&stream));
  for (std::size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(arrays); ++i) {
    if (arrays[i]) PetscCallCUPM(cupmFreeAsync(arrays[i], stream));
  }
  geom->v0_d = geom->invJ_d = geom->detJ_d = nullptr;
  geom->freedevice                         = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline kernels::DMSwarmMassCtx DMSwarmFEBasis_CUPM<T>::GetMassCtx_(DMSwarmFEBasis basis, DMSwarmCellGeometry geom, const PetscInt list[], const PetscInt cell[], const PetscReal coords[], PetscInt numF, PetscBool useDelta) noexcept
{
  kernels::DMSwarmMassCtx ctx;

  ctx.basis    = GetCtx(basis);
  ctx.numF     = numF;
  ctx.list     = list;
  ctx.cell     = cell;
  ctx.coords   = coords;
  ctx.v0       = static_cast<const PetscReal *>(geom->v0_d);
  ctx.invJ     = static_cast<const PetscReal *>(geom->invJ_d);
  ctx.detJ     = static_cast<const PetscReal *>(geom->detJ_d);
  ctx.useDelta = useDelta;
  return ctx;
}

/* Compute the COO values of the particle mass matrix on the device and pass them to MatSetValuesCOO() without a host copy */
template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::MassSetValuesCOO(Mat mass, DMSwarmFEBasis basis, DMSwarmCellGeometry geom, PetscInt npoints, const PetscInt list[], const PetscInt cell[], const PetscReal coords[], PetscInt numF, PetscBool useDelta) noexcept
{
  const kernels::DMSwarmMassCtx ctx = GetMassCtx_(basis, geom, list, cell, coords, numF, useDelta);
  PetscDeviceContext            dctx;
  cupmStream_t                  stream;
  PetscScalar                  *vals = nullptr;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  if (npoints > 0) {
    PetscCall(PetscCUPMMallocAsync(&vals, npoints * numF, stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(PetscCUPMLaunchKernel1D(npoints, 0, stream, kernels::mass_values, ctx, npoints, vals));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscCall(MatSetValuesCOO(mass, vals, INSERT_VALUES));
  if (vals) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCallCUPM(cupmFreeAsync(vals, stream));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The per-cell Gram blocks of the particle mass matrix square, the one place the dense multiply happens. The inputs are
  laid out as a batched gemm (op T, N) takes them: the basis values of cell c are the numF x n_c column-major matrix
  at B + pcell_offsets[c] * numF (leading dimension numF), and its block is the n_c x n_c matrix at vals +
  sq_offsets[c] (leading dimension n_c; symmetric, so row- and column-major agree), so a vendor batched gemm can
  replace the kernel here without changing the callers.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::MassSquareGram_(const kernels::DMSwarmMassSquareCtx &ctx, PetscScalar vals[], cupmStream_t stream) noexcept
{
  constexpr PetscCount maxBlocks = 1 << 20; /* the kernel strides over the tiles, so this only bounds the grid */
  const std::size_t    shared    = 2 * DMSWARM_MASS_SQUARE_TILE * ctx.ld * sizeof(PetscScalar);

  PetscFunctionBegin;
  if (!ctx.ntiles) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPM(cupmLaunchKernel(kernels::mass_square_values, (unsigned int)PetscMin(ctx.ntiles, maxBlocks), (unsigned int)(DMSWARM_MASS_SQUARE_TILE * DMSWARM_MASS_SQUARE_TILE / 4), shared, stream, ctx, vals));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * (PetscLogDouble)ctx.numF * (PetscLogDouble)ctx.sq_total));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Compute the COO values of the particle mass matrix square on the device and pass them to MatSetValuesCOO() without a
  host copy: the basis values of every sort CSR slot (the particle mass matrix values above), then the Gram blocks. The
  host arrays sq_offsets and tile_offsets describe the blocks, see DMSwarmMassSquareCtx.
*/
template <DeviceType T>
inline PetscErrorCode DMSwarmFEBasis_CUPM<T>::MassSquareSetValuesCOO(Mat mass, DMSwarmFEBasis basis, DMSwarmCellGeometry geom, PetscInt npoints, const PetscInt list[], const PetscInt cell[], const PetscReal coords[], PetscInt numF, PetscBool useDelta, PetscInt ncells, const PetscInt pcell_offsets[], const PetscCount sq_offsets[], const PetscCount tile_offsets[]) noexcept
{
  const kernels::DMSwarmMassCtx ctx = GetMassCtx_(basis, geom, list, cell, coords, numF, useDelta);
  kernels::DMSwarmMassSquareCtx sctx;
  PetscDeviceContext            dctx;
  cupmStream_t                  stream;
  PetscScalar                  *B = nullptr, *vals = nullptr;
  PetscCount                   *sq_d = nullptr, *tile_d = nullptr;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  sctx.numF          = numF;
  sctx.ld            = numF | 1;
  sctx.ncells        = ncells;
  sctx.pcell_offsets = pcell_offsets;
  sctx.ntiles        = tile_offsets[ncells];
  sctx.sq_total      = sq_offsets[ncells];
  if (npoints > 0) {
    PetscCall(PetscCUPMMallocAsync(&B, (std::size_t)npoints * numF, stream));
    PetscCall(PetscCUPMMallocAsync(&vals, (std::size_t)sctx.sq_total, stream));
    PetscCall(PetscCUPMMallocAsync(&sq_d, (std::size_t)ncells + 1, stream));
    PetscCall(PetscCUPMMallocAsync(&tile_d, (std::size_t)ncells + 1, stream));
    PetscCall(PetscCUPMMemcpyAsync(sq_d, sq_offsets, (std::size_t)ncells + 1, cupmMemcpyHostToDevice, stream));
    PetscCall(PetscCUPMMemcpyAsync(tile_d, tile_offsets, (std::size_t)ncells + 1, cupmMemcpyHostToDevice, stream));
    sctx.sq_offsets   = sq_d;
    sctx.tile_offsets = tile_d;
    sctx.B            = B;
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(PetscCUPMLaunchKernel1D(npoints, 0, stream, kernels::mass_values, ctx, npoints, B));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(MassSquareGram_(sctx, vals, stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscCall(MatSetValuesCOO(mass, vals, INSERT_VALUES));
  if (vals) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCallCUPM(cupmFreeAsync(B, stream));
    PetscCallCUPM(cupmFreeAsync(vals, stream));
    PetscCallCUPM(cupmFreeAsync(sq_d, stream));
    PetscCallCUPM(cupmFreeAsync(tile_d, stream));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
