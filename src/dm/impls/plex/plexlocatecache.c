#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/

/*
  The flattened point-location geometry cache (struct in dmpleximpl.h) collects, in plain
  arrays, everything DMLocatePoints_Plex() derives from PETSc objects: per-simplex-cell affine
  geometry, per-tensor-cell vertex coordinates, the bounding box, and the grid hash flattened
  to CSR form. It is built on the host and mirrored to the device on first request, so that
  device location kernels can run without touching PETSc objects. The cache is invalidated,
  together with DM_Plex->lbox, whenever the coordinates change, see
  DMPlexInvalidateStaleLocationCache_Internal().
*/

/*
  DMPlexLocationCacheDestroy_Internal - Destroy a location cache, including any device mirrors

  Input Parameter:
. cache - The cache, or NULL; set to NULL on return
*/
PetscErrorCode DMPlexLocationCacheDestroy_Internal(DMPlexLocationCache *cache)
{
  PetscFunctionBegin;
  if (!*cache) PetscFunctionReturn(PETSC_SUCCESS);
  if ((*cache)->freedevice) PetscCall((*(*cache)->freedevice)(*cache));
  PetscCall(PetscFree((*cache)->celltype));
  PetscCall(PetscFree2((*cache)->v0, (*cache)->invJ));
  PetscCall(PetscFree((*cache)->cellCoords));
  PetscCall(PetscFree((*cache)->leafCell));
  PetscCall(PetscFree((*cache)->binOffset));
  PetscCall(PetscFree((*cache)->boxCells));
  PetscCall(PetscFree(*cache));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Build the host side of the location cache. Configurations the flat arrays cannot represent
  (complex scalars, embedded manifolds, coordinate degree > 1, cell types the host location in
  DMPlexLocatePoint_Internal() does not support) are recorded as cache->unavailable instead of
  raising an error, so that callers can fall back to the host location path.
*/
static PetscErrorCode DMPlexLocationCacheBuild_Private(DM dm, DMPlexLocationCache cache)
{
  DM_Plex  *mesh = (DM_Plex *)dm->data;
  Vec       coordinates;
  PetscInt  dim, cdim, degree, cStart, cEnd, Ncell;
  PetscBool hasSimplex = PETSC_FALSE, hasTensor = PETSC_FALSE;

  PetscFunctionBegin;
  cache->generation = ++mesh->loccachegen;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  cache->dim    = dim;
  cache->cdim   = cdim;
  cache->cStart = cStart;
  cache->cEnd   = cEnd;
  Ncell         = cEnd - cStart;
  if (PetscDefined(USE_COMPLEX) || cdim != dim) {
    PetscCall(PetscInfo(dm, "Location cache unavailable: complex scalars or embedded manifold (dim %" PetscInt_FMT ", cdim %" PetscInt_FMT ")\n", dim, cdim));
    cache->unavailable = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMGetCoordinateDegree_Internal(dm, &degree));
  if (degree != 1) {
    PetscCall(PetscInfo(dm, "Location cache unavailable: coordinate space degree %" PetscInt_FMT " > 1\n", degree));
    cache->unavailable = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  if (!coordinates) {
    PetscCall(PetscInfo(dm, "Location cache unavailable: DM has no local coordinates\n"));
    cache->unavailable = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /*
   DMGetBoundingBox() is collective; it must run before the per-cell classification below, whose
   unavailable early return can differ between ranks
  */
  PetscCall(DMGetBoundingBox(dm, cache->gmin, cache->gmax));
  /* Classify cells: simplices get affine v0/invJ, tensor cells get vertex coordinates */
  cache->uniform   = PETSC_TRUE;
  cache->uniformCt = DM_POLYTOPE_UNKNOWN;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DMPolytopeType ct;

    PetscCall(DMPlexGetCellType(dm, c, &ct));
    switch (ct) {
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_TRIANGLE:
    case DM_POLYTOPE_TETRAHEDRON:
      hasSimplex = PETSC_TRUE;
      break;
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_HEXAHEDRON:
      hasTensor         = PETSC_TRUE;
      cache->maxCorners = PetscMax(cache->maxCorners, DMPolytopeTypeGetNumVertices(ct));
      break;
    default:
      PetscCall(PetscInfo(dm, "Location cache unavailable: unsupported cell type %s for cell %" PetscInt_FMT "\n", DMPolytopeTypes[ct], c));
      cache->unavailable = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (c == cStart) cache->uniformCt = ct;
    else if (ct != cache->uniformCt) cache->uniform = PETSC_FALSE;
  }
  if (!cache->uniform) {
    PetscCall(PetscMalloc1(Ncell, &cache->celltype));
    for (PetscInt c = cStart; c < cEnd; ++c) {
      DMPolytopeType ct;

      PetscCall(DMPlexGetCellType(dm, c, &ct));
      cache->celltype[c - cStart] = (PetscInt)ct;
    }
  }
  if (hasSimplex) PetscCall(PetscCalloc2(Ncell * cdim, &cache->v0, Ncell * cdim * cdim, &cache->invJ));
  if (hasTensor) PetscCall(PetscCalloc1(Ncell * cache->maxCorners * cdim, &cache->cellCoords));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt cell = c - cStart;
    DMPolytopeType ct;

    PetscCall(DMPlexGetCellType(dm, c, &ct));
    switch (ct) {
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_TRIANGLE:
    case DM_POLYTOPE_TETRAHEDRON: {
      PetscReal J[9], detJ;

      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, &cache->v0[cell * cdim], J, &cache->invJ[cell * cdim * cdim], &detJ));
    } break;
    default: {
      const PetscScalar *array;
      PetscScalar       *coords = NULL;
      PetscInt           numCoords;
      PetscBool          isDG;

      PetscCall(DMPlexGetCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
      PetscCheck(numCoords == DMPolytopeTypeGetNumVertices(ct) * cdim, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " of type %s has %" PetscInt_FMT " coordinates, not %" PetscInt_FMT, c, DMPolytopeTypes[ct], numCoords, DMPolytopeTypeGetNumVertices(ct) * cdim);
      for (PetscInt i = 0; i < numCoords; ++i) cache->cellCoords[cell * cache->maxCorners * cdim + i] = PetscRealPart(coords[i]);
      PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDG, &numCoords, &array, &coords));
    }
    }
  }
  /* Record which cells are point SF leaves through the exact predicate the host brute-force scan skips them with */
  {
    PetscSF         sf;
    const PetscInt *leaves;
    PetscInt        Nl      = 0;
    PetscBool       anyLeaf = PETSC_FALSE;

    PetscCall(DMGetPointSF(dm, &sf));
    if (sf) PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
    Nl = PetscMax(Nl, 0);
    if (Nl > 0 && Ncell > 0) {
      PetscCall(PetscCalloc1(Ncell, &cache->leafCell));
      for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt idx;

        PetscCall(PetscFindInt(c, Nl, leaves, &idx));
        if (idx >= 0) {
          cache->leafCell[c - cStart] = PETSC_TRUE;
          anyLeaf                     = PETSC_TRUE;
        }
      }
      if (!anyLeaf) PetscCall(PetscFree(cache->leafCell));
    }
  }
  /* Flatten the grid hash into CSR form; build it through the existing internal function if not yet present */
  if (Ncell > 0) {
    PetscGridHash   lbox;
    const PetscInt *cells;
    PetscInt        nbins;

    if (!mesh->lbox) PetscCall(DMPlexComputeGridHash_Internal(dm, &mesh->lbox));
    lbox = mesh->lbox;
    for (PetscInt d = 0; d < 3; ++d) {
      cache->lower[d] = lbox->lower[d];
      cache->upper[d] = lbox->upper[d];
      cache->h[d]     = lbox->h[d];
      cache->n[d]     = lbox->n[d];
    }
    PetscCall(PetscSectionGetChart(lbox->cellSection, &cache->binStart, &cache->binEnd));
    nbins = cache->binEnd - cache->binStart;
    PetscCall(PetscMalloc1(nbins + 1, &cache->binOffset));
    cache->binOffset[0] = 0;
    for (PetscInt b = 0; b < nbins; ++b) {
      PetscInt dof;

      PetscCall(PetscSectionGetDof(lbox->cellSection, cache->binStart + b, &dof));
      cache->binOffset[b + 1] = cache->binOffset[b] + dof;
    }
    PetscCall(PetscMalloc1(cache->binOffset[nbins], &cache->boxCells));
    PetscCall(ISGetIndices(lbox->cells, &cells));
    for (PetscInt b = 0; b < nbins; ++b) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(lbox->cellSection, cache->binStart + b, &dof));
      PetscCall(PetscSectionGetOffset(lbox->cellSection, cache->binStart + b, &off));
      for (PetscInt i = 0; i < dof; ++i) cache->boxCells[cache->binOffset[b] + i] = cells[off + i];
    }
    PetscCall(ISRestoreIndices(lbox->cells, &cells));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexInvalidateStaleLocationCache_Internal - Destroy the grid hash and the location cache if the coordinates changed since they were built

  Input Parameter:
. dm - The DMPLEX

  Notes:
  This is the single invalidation choke point for all coordinate-derived location structures.
  Every coordinate mutation either replaces the local coordinate Vec (DMSetCoordinates() and
  DMSetCoordinatesLocal() and their cellwise variants destroy it, so it is lazily recreated with
  a new PetscObjectId) or writes into it in place (DMPlexRemapGeometry(), or any write access to
  the Vec returned by DMGetCoordinatesLocal(), bumps its PetscObjectState). Comparing the
  (id, state) pairs of the local coordinate and cellwise-coordinate Vecs against the values
  recorded when the structures were built therefore catches every mutation path, including the
  in-place ones no setter-based hook can see. The same idiom is used for subpointIS, see
  DM_Plex->subpointState.
*/
PetscErrorCode DMPlexInvalidateStaleLocationCache_Internal(DM dm)
{
  DM_Plex         *mesh     = (DM_Plex *)dm->data;
  PetscObjectId    id[2]    = {0, 0};
  PetscObjectState state[2] = {0, 0};

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMGetCellCoordinatesLocalSetUp(dm));
  for (PetscInt i = 0; i < 2; ++i) {
    Vec xl = dm->coordinates[i].xl;

    if (xl) {
      PetscCall(PetscObjectGetId((PetscObject)xl, &id[i]));
      PetscCall(PetscObjectStateGet((PetscObject)xl, &state[i]));
    }
  }
  if (id[0] != mesh->coordId[0] || state[0] != mesh->coordState[0] || id[1] != mesh->coordId[1] || state[1] != mesh->coordState[1]) {
    if (mesh->lbox || mesh->loccache) PetscCall(PetscInfo(dm, "Coordinates changed: invalidating grid hash and location cache\n"));
    PetscCall(PetscGridHashDestroy(&mesh->lbox));
    PetscCall(DMPlexLocationCacheDestroy_Internal(&mesh->loccache));
    for (PetscInt i = 0; i < 2; ++i) {
      mesh->coordId[i]    = id[i];
      mesh->coordState[i] = state[i];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexGetLocationCache_Internal - Return the host-built location cache, building it if necessary

  Input Parameter:
. dm - The DMPLEX

  Output Parameter:
. cache - The location cache; check cache->unavailable before using its arrays

  Note:
  The returned cache is owned by the DM. It is invalidated when the coordinates change, so do
  not hold on to it across operations that may modify the coordinates.
*/
PetscErrorCode DMPlexGetLocationCache_Internal(DM dm, DMPlexLocationCache *cache)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscCall(DMPlexInvalidateStaleLocationCache_Internal(dm));
  if (!mesh->loccache) {
    PetscCall(PetscNew(&mesh->loccache));
    PetscCall(DMPlexLocationCacheBuild_Private(dm, mesh->loccache));
  }
  *cache = mesh->loccache;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexGetLocationCacheDevice_Internal - Return the location cache with valid device mirrors, uploading the host arrays on first request

  Input Parameter:
. dm - The DMPLEX

  Output Parameter:
. cache - The location cache with its device pointers valid
*/
PetscErrorCode DMPlexGetLocationCacheDevice_Internal(DM dm, DMPlexLocationCache *cache)
{
  PetscFunctionBegin;
  PetscCall(DMPlexGetLocationCache_Internal(dm, cache));
  PetscCheck(!(*cache)->unavailable, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Point-location geometry cache unavailable for this DM configuration; use the host location path");
#if PetscDefined(HAVE_CUDA)
  if (!(*cache)->onDevice) PetscCall(DMPlexLocationCacheUpload_CUDA(*cache));
#elif PetscDefined(HAVE_HIP)
  if (!(*cache)->onDevice) PetscCall(DMPlexLocationCacheUpload_HIP(*cache));
#else
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Device location cache requires PETSc configured with CUDA or HIP");
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
