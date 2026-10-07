#include "../plexlocatecachecupm.hpp"

using namespace Petsc::device::cupm;

static constexpr impl::DMPlexLocationCache_CUPM<DeviceType::HIP> cupm_cache{};

/* Free the HIP mirrors of a location cache; installed as cache->freedevice by the upload below */
PETSC_INTERN PetscErrorCode DMPlexLocationCacheFreeDevice_HIP(DMPlexLocationCache cache)
{
  PetscFunctionBegin;
  PetscCall(cupm_cache.FreeDevice(cache));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Upload the host arrays of a location cache to HIP mirrors; called from DMPlexGetLocationCacheDevice_Internal() */
PETSC_INTERN PetscErrorCode DMPlexLocationCacheUpload_HIP(DMPlexLocationCache cache)
{
  PetscFunctionBegin;
  PetscCall(cupm_cache.Upload(cache));
  cache->freedevice = DMPlexLocationCacheFreeDevice_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexLocatePointsInPlace_Internal_HIP - Locate points on the device, reproducing the host
  location decisions of DMLocatePoints_Plex() bitwise

  Input Parameters:
+ dm       - The DMPLEX, whose location cache must be usable (see DMPlexGetLocationCacheDevice_Internal())
. npoints  - The number of points
. bs       - The coordinate block size, at least the coordinate dimension of dm
- coords_d - [npoints * bs] point coordinates in device memory

  Input/Output Parameter:
. cellids_d - [npoints] cell ids in device memory; on input the initial guesses (DMLOCATEPOINT_POINT_NOT_FOUND
              for none), on output the located cell or DMLOCATEPOINT_POINT_NOT_FOUND

  Notes:
  Whether the grid hash or the brute-force scan is used follows DM_Plex->useHashLocation, exactly
  like the host path. The work is logged in the DMPlexLocatePoints event of DMLocatePoints_Plex(),
  so -log_view compares host and device point location under one name.
*/
PETSC_INTERN PetscErrorCode DMPlexLocatePointsInPlace_Internal_HIP(DM dm, PetscInt npoints, PetscInt bs, const PetscReal coords_d[], PetscInt cellids_d[])
{
  DM_Plex            *mesh = (DM_Plex *)dm->data;
  DMPlexLocationCache cache;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_LocatePoints, 0, 0, 0, 0));
  PetscCall(DMPlexGetLocationCacheDevice_Internal(dm, &cache));
  PetscCheck(bs >= cache->cdim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Block size for point coordinates %" PetscInt_FMT " must be the mesh coordinate dimension %" PetscInt_FMT, bs, cache->cdim);
  if (npoints > 0) PetscCall(cupm_cache.LocatePoints(cache, mesh->useHashLocation, npoints, bs, coords_d, cellids_d));
  PetscCall(PetscInfo(dm, "[DMPlexLocatePointsInPlace_Internal_HIP] npoints %" PetscInt_FMT " located on device (%s)\n", npoints, mesh->useHashLocation ? "hash" : "brute-force"));
  PetscCall(PetscLogEventEnd(DMPLEX_LocatePoints, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
