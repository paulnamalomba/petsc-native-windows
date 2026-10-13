#include "../dmswarmfebasiscupm.hpp"

using namespace Petsc::device::cupm;

static constexpr impl::DMSwarmFEBasis_CUPM<DeviceType::HIP> cupm_febasis{};

/* Free the HIP mirrors of a basis; installed as basis->freedevice by the upload below */
static PetscErrorCode DMSwarmFEBasisFreeDevice_HIP(DMSwarmFEBasis basis)
{
  PetscFunctionBegin;
  PetscCall(cupm_febasis.FreeDevice(basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Upload the host arrays of a basis to HIP mirrors; called from DMSwarmFEBasisGetDevice_Internal() */
PETSC_INTERN PetscErrorCode DMSwarmFEBasisUpload_HIP(DMSwarmFEBasis basis)
{
  PetscFunctionBegin;
  PetscCall(cupm_febasis.Upload(basis));
  basis->freedevice = DMSwarmFEBasisFreeDevice_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMSwarmFEBasisTabulate_Internal_HIP - Evaluate the nodal basis functions at reference points on the device,
  reproducing PetscFECreateTabulation() (B only) to rounding

  Input Parameters:
+ basis   - A supported basis with device mirrors, see DMSwarmFEBasisGetDevice_Internal()
. npoints - The number of points
- xi_d    - [npoints * dim] reference coordinates in device memory

  Output Parameter:
. B_d - [npoints * pdim] basis function values in device memory
*/
PETSC_INTERN PetscErrorCode DMSwarmFEBasisTabulate_Internal_HIP(DMSwarmFEBasis basis, PetscInt npoints, const PetscReal xi_d[], PetscReal B_d[])
{
  PetscFunctionBegin;
  PetscCheck(basis->supported && basis->freedevice, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Basis is not uploaded to the device");
  if (npoints > 0) PetscCall(cupm_febasis.Tabulate(basis, npoints, xi_d, B_d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Free the HIP mirrors of a cell geometry table; installed as geom->freedevice by the upload below */
static PetscErrorCode DMSwarmCellGeometryFreeDevice_HIP(DMSwarmCellGeometry geom)
{
  PetscFunctionBegin;
  PetscCall(cupm_febasis.FreeGeometryDevice(geom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Upload the host arrays of a cell geometry table to HIP mirrors; called from DMSwarmCellGeometryGetDevice_Internal() */
PETSC_INTERN PetscErrorCode DMSwarmCellGeometryUpload_HIP(DMSwarmCellGeometry geom)
{
  PetscFunctionBegin;
  PetscCall(cupm_febasis.UploadGeometry(geom));
  geom->freedevice = DMSwarmCellGeometryFreeDevice_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMSwarmMassMatrixSetValuesCOO_Internal_HIP - Set the values of a COO-preallocated particle mass matrix from the device

  Input Parameters:
+ mass     - The (seq|mpi)aijhipsparse matrix, preallocated by DMSwarmComputeMassMatrixCOO_Private()
. basis    - The device basis of the field, see DMSwarmFEBasisGetDevice_Internal()
. geom     - The device cell geometry, see DMSwarmCellGeometryGetDevice_Internal()
. npoints  - The number of sort CSR slots, the local number of particles
. list_d   - [npoints] particle of each sort CSR slot, in device memory
. cell_d   - [npoints] cell of each sort CSR slot, in device memory
. coords_d - [Np * dim] particle coordinates in device memory
. numF     - The closure size of every cell, the basis dimension
- useDelta - Use weight 1 instead of the cell Jacobian determinant

  Note:
  The values agree with those of the host COO assembly to rounding, in the same COO order.
*/
PETSC_INTERN PetscErrorCode DMSwarmMassMatrixSetValuesCOO_Internal_HIP(Mat mass, DMSwarmFEBasis basis, DMSwarmCellGeometry geom, PetscInt npoints, const PetscInt list_d[], const PetscInt cell_d[], const PetscReal coords_d[], PetscInt numF, PetscBool useDelta)
{
  PetscFunctionBegin;
  PetscCall(cupm_febasis.MassSetValuesCOO(mass, basis, geom, npoints, list_d, cell_d, coords_d, numF, useDelta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMSwarmMassMatrixSquareSetValuesCOO_Internal_HIP - Set the values of a COO-preallocated particle mass matrix square from the device

  Input Parameters:
+ mass            - The (seq|mpi)aijhipsparse matrix, preallocated by DMSwarmComputeMassMatrixSquareCOO_Private()
. basis           - The device basis of the field, see DMSwarmFEBasisGetDevice_Internal()
. geom            - The device cell geometry, see DMSwarmCellGeometryGetDevice_Internal()
. npoints         - The number of sort CSR slots, the local number of particles
. list_d          - [npoints] particle of each sort CSR slot, in device memory
. cell_d          - [npoints] cell of each sort CSR slot, in device memory
. coords_d        - [Np * dim] particle coordinates in device memory
. numF            - The closure size of every cell, the basis dimension
. useDelta        - Use weight 1 instead of the cell Jacobian determinant
. ncells          - The number of cells of the sort CSR
. pcell_offsets_d - [ncells + 1] sort CSR slot range of each cell, in device memory
. sq_offsets      - [ncells + 1] COO offset of the block of each cell, in host memory
- tile_offsets    - [ncells + 1] first DMSWARM_MASS_SQUARE_TILE tile of each cell, in host memory

  Note:
  The values agree with those of the host COO assembly to rounding, in the same COO order.
*/
PETSC_INTERN PetscErrorCode DMSwarmMassMatrixSquareSetValuesCOO_Internal_HIP(Mat mass, DMSwarmFEBasis basis, DMSwarmCellGeometry geom, PetscInt npoints, const PetscInt list_d[], const PetscInt cell_d[], const PetscReal coords_d[], PetscInt numF, PetscBool useDelta, PetscInt ncells, const PetscInt pcell_offsets_d[], const PetscCount sq_offsets[], const PetscCount tile_offsets[])
{
  PetscFunctionBegin;
  PetscCall(cupm_febasis.MassSquareSetValuesCOO(mass, basis, geom, npoints, list_d, cell_d, coords_d, numF, useDelta, ncells, pcell_offsets_d, sq_offsets, tile_offsets));
  PetscFunctionReturn(PETSC_SUCCESS);
}
