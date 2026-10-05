#pragma once

#include <petsc/private/matdensecupmimpl.h>

namespace Petsc
{
namespace pc
{
namespace cupm
{
namespace impl
{

template <device::cupm::DeviceType T>
class PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL PCJacobi_CUPM : device::cupm::impl::CUPMObject<T> {
public:
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode MatApply(Vec diag, Mat X, Mat Y) noexcept
  {
    cupmBlasHandle_t   handle;
    cupmBlasInt_t      m, n, ldx, ldy;
    PetscInt           rows, cols, ldaX, ldaY;
    const PetscScalar *x, *d;
    PetscScalar       *y;

    PetscFunctionBegin;
    PetscCall(MatGetLocalSize(X, &rows, nullptr));
    PetscCall(MatGetSize(X, nullptr, &cols));
    if (!rows || !cols) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscCUPMBlasIntCast(rows, &m));
    PetscCall(PetscCUPMBlasIntCast(cols, &n));
    PetscCall(GetHandles_(&handle));
    PetscCall(mat::cupm::MatDenseCUPMGetArrayRead<T>(X, &x));
    PetscCall(mat::cupm::MatDenseCUPMGetArrayWrite<T>(Y, &y));
    PetscCall(VecGetArrayReadAndMemType(diag, &d, nullptr));
    PetscCall(MatDenseGetLDA(X, &ldaX));
    PetscCall(MatDenseGetLDA(Y, &ldaY));
    PetscCall(PetscCUPMBlasIntCast(ldaX, &ldx));
    PetscCall(PetscCUPMBlasIntCast(ldaY, &ldy));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXdgmm(handle, CUPMBLAS_SIDE_LEFT, m, n, cupmScalarPtrCast(x), ldx, cupmScalarPtrCast(d), 1, cupmScalarPtrCast(y), ldy));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(1.0 * rows * cols));
    PetscCall(VecRestoreArrayReadAndMemType(diag, &d));
    PetscCall(mat::cupm::MatDenseCUPMRestoreArrayWrite<T>(Y, &y));
    PetscCall(mat::cupm::MatDenseCUPMRestoreArrayRead<T>(X, &x));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

} // namespace impl
} // namespace cupm
} // namespace pc
} // namespace Petsc
