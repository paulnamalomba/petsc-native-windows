static char help[] = "Tests the domain decomposition transform DMPLEXTRANSFORMDD\n\n";

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscsf.h>
#include <petscbt.h>

typedef enum {
  LABEL_SLABS,
  LABEL_RANKS
} LabelType;
static const char *const LabelTypes[] = {"slabs", "ranks", "LabelType", "LABEL_", NULL};

typedef struct {
  LabelType labelType;   // How subdomains are marked
  PetscInt  numSlabs;    // Number of slabs in x, for LABEL_SLABS
  PetscReal slabOverlap; // Distance by which consecutive slabs overlap, for LABEL_SLABS
  PetscInt  numLocalSub; // Number of subdomains on each process, for LABEL_RANKS
  PetscBool labelHalo;   // Also mark the ghost cells with the subdomain of their owner, for LABEL_RANKS
  PetscInt  grow;        // Number of layers of adjacent cells added to each subdomain
  PetscBool fvAdjacency; // Grow subdomains through facets instead of vertices
  PetscBool checkReferenceCells;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->labelType           = LABEL_SLABS;
  options->numSlabs            = 2;
  options->slabOverlap         = 0.;
  options->numLocalSub         = 1;
  options->labelHalo           = PETSC_FALSE;
  options->grow                = 0;
  options->fvAdjacency         = PETSC_FALSE;
  options->checkReferenceCells = PETSC_FALSE;
  PetscOptionsBegin(comm, "", "Domain decomposition transform test options", "DMPLEX");
  PetscCall(PetscOptionsEnum("-label_type", "How subdomains are marked", __FILE__, LabelTypes, (PetscEnum)options->labelType, (PetscEnum *)&options->labelType, NULL));
  PetscCall(PetscOptionsBoundedInt("-num_slabs", "Number of slabs in x", __FILE__, options->numSlabs, &options->numSlabs, NULL, 1));
  PetscCall(PetscOptionsReal("-slab_overlap", "Distance by which consecutive slabs overlap", __FILE__, options->slabOverlap, &options->slabOverlap, NULL));
  PetscCall(PetscOptionsBoundedInt("-num_local_sub", "Number of subdomains on each process", __FILE__, options->numLocalSub, &options->numLocalSub, NULL, 1));
  PetscCall(PetscOptionsBool("-label_halo", "Mark the ghost cells with the subdomain of their owner", __FILE__, options->labelHalo, &options->labelHalo, NULL));
  PetscCall(PetscOptionsBoundedInt("-grow", "Number of layers of adjacent cells added to each subdomain", __FILE__, options->grow, &options->grow, NULL, 0));
  PetscCall(PetscOptionsBool("-fv_adjacency", "Grow subdomains through facets instead of vertices", __FILE__, options->fvAdjacency, &options->fvAdjacency, NULL));
  PetscCall(PetscOptionsBool("-check_reference_cells", "Build the reference-cell coordinate cache", __FILE__, options->checkReferenceCells, &options->checkReferenceCells, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Mark the subdomain cells in the label "subdomain".

  LABEL_SLABS: slab j holds the cells whose centroid x lies in [j/n - overlap, (j+1)/n + overlap), so a cell belongs to
  several slabs when the overlap is positive. The marking depends only on the geometry, so it is the same on every
  process, ghost cells included.

  LABEL_RANKS: each process splits its owned cells into consecutive chunks, chunk j of process r being subdomain
  r * k + j. Ghost cells are marked with the subdomain of their owner only with -label_halo.
*/
static PetscErrorCode CreateSubdomainLabel(DM dm, AppCtx *user, PetscInt *numSubdomains)
{
  DMLabel     label;
  PetscSF     sf;
  PetscMPIInt rank, size;
  PetscInt    cStart, cEnd, pStart, pEnd;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMCreateLabel(dm, "subdomain"));
  PetscCall(DMGetLabel(dm, "subdomain", &label));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  if (user->labelType == LABEL_SLABS) {
    const PetscInt  n = user->numSlabs;
    const PetscReal h = user->slabOverlap;

    for (PetscInt c = cStart; c < cEnd; ++c) {
      PetscReal centroid[3];

      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      for (PetscInt j = 0; j < n; ++j)
        if (centroid[0] >= (PetscReal)j / n - h && centroid[0] < (PetscReal)(j + 1) / n + h) PetscCall(DMLabelSetValue(label, c, j));
    }
    *numSubdomains = n;
  } else {
    const PetscInt     k = user->numLocalSub;
    const PetscInt    *leaves;
    const PetscSFNode *remotes;
    PetscInt          *owned, *values;
    PetscInt           Nl, No = 0;

    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, &remotes));
    PetscCall(PetscMalloc2(pEnd - pStart, &owned, pEnd - pStart, &values));
    for (PetscInt p = 0; p < pEnd - pStart; ++p) {
      owned[p]  = 1;
      values[p] = -1;
    }
    for (PetscInt l = 0; l < PetscMax(Nl, 0); ++l) owned[(leaves ? leaves[l] : l) - pStart] = 0;
    for (PetscInt c = cStart; c < cEnd; ++c) No += owned[c - pStart];
    for (PetscInt c = cStart, i = 0; c < cEnd; ++c) {
      if (!owned[c - pStart]) continue;
      values[c - pStart] = rank * k + (i++ * k) / No;
    }
    if (user->labelHalo && Nl >= 0) {
      PetscCall(PetscSFBcastBegin(sf, MPIU_INT, values, values, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf, MPIU_INT, values, values, MPI_REPLACE));
    }
    for (PetscInt c = cStart; c < cEnd; ++c)
      if (values[c - pStart] >= 0) PetscCall(DMLabelSetValue(label, c, values[c - pStart]));
    PetscCall(PetscFree2(owned, values));
    *numSubdomains = size * k;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Check the relation between each point of the transformed mesh and its source point: the cell type, the vertex
  coordinates, and the cone, whose points must come from the cone of the source point, with the same orientation, in
  the same subdomain.
*/
static PetscErrorCode CheckSourcePoints(DMPlexTransform tr, DM dm, DM rdm)
{
  DMLabel      label, rlabel;
  Vec          coords, rcoords;
  PetscSection cs, rcs;
  PetscInt     pStart, pEnd, vStart, vEnd, cStart, cEnd, cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "subdomain", &label));
  PetscCall(DMGetLabel(rdm, "subdomain", &rlabel));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(DMGetCoordinatesLocal(rdm, &rcoords));
  PetscCall(DMGetCoordinateSection(dm, &cs));
  PetscCall(DMGetCoordinateSection(rdm, &rcs));
  PetscCall(DMPlexGetChart(rdm, &pStart, &pEnd));
  PetscCall(DMPlexGetDepthStratum(rdm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
  for (PetscInt q = pStart; q < pEnd; ++q) {
    const PetscInt *cone, *ornt, *rcone, *rornt;
    DMPolytopeType  ct, rct;
    PetscInt        p, r, v, coneSize;

    PetscCall(DMPlexTransformGetSourcePoint(tr, q, &ct, &rct, &p, &r));
    PetscCheck(ct == rct, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has type %s but its source %" PetscInt_FMT " has type %s", q, DMPolytopeTypes[rct], p, DMPolytopeTypes[ct]);
    PetscCall(DMLabelGetValue(rlabel, q, &v));
    PetscCheck(v >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has no subdomain", q);
    PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
    PetscCall(DMPlexGetCone(dm, p, &cone));
    PetscCall(DMPlexGetConeOrientation(dm, p, &ornt));
    PetscCall(DMPlexGetCone(rdm, q, &rcone));
    PetscCall(DMPlexGetConeOrientation(rdm, q, &rornt));
    for (PetscInt c = 0; c < coneSize; ++c) {
      PetscInt cp, cv;

      PetscCall(DMPlexTransformGetSourcePoint(tr, rcone[c], NULL, NULL, &cp, NULL));
      PetscCall(DMLabelGetValue(rlabel, rcone[c], &cv));
      PetscCheck(cp == cone[c] && rornt[c] == ornt[c], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cone point %" PetscInt_FMT " of point %" PetscInt_FMT " comes from (%" PetscInt_FMT ", %" PetscInt_FMT "), not (%" PetscInt_FMT ", %" PetscInt_FMT ")", c, q, cp, rornt[c], cone[c], ornt[c]);
      PetscCheck(cv == v, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cone point %" PetscInt_FMT " of point %" PetscInt_FMT " is in subdomain %" PetscInt_FMT ", not %" PetscInt_FMT, c, q, cv, v);
    }
    if (q >= vStart && q < vEnd) {
      const PetscScalar *x, *rx;
      PetscInt           off, roff;

      PetscCall(PetscSectionGetOffset(cs, p, &off));
      PetscCall(PetscSectionGetOffset(rcs, q, &roff));
      PetscCall(VecGetArrayRead(coords, &x));
      PetscCall(VecGetArrayRead(rcoords, &rx));
      for (PetscInt d = 0; d < cdim; ++d) PetscCheck(x[off + d] == rx[roff + d], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Vertex %" PetscInt_FMT " does not have the coordinates of its source %" PetscInt_FMT, q, p);
      PetscCall(VecRestoreArrayRead(coords, &x));
      PetscCall(VecRestoreArrayRead(rcoords, &rx));
    }
    if (q >= cStart && q < cEnd) {
      const PetscScalar *array, *rarray;
      PetscScalar       *x = NULL, *rx = NULL;
      PetscBool          isDG, risDG;
      PetscInt           Nc, rNc;

      PetscCall(DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &x));
      PetscCall(DMPlexGetCellCoordinates(rdm, q, &risDG, &rNc, &rarray, &rx));
      PetscCheck(isDG == risDG && Nc == rNc, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " does not have the coordinate layout of its source %" PetscInt_FMT, q, p);
      for (PetscInt d = 0; d < Nc; ++d) PetscCheck(x[d] == rx[d], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " does not have the coordinates of its source %" PetscInt_FMT, q, p);
      PetscCall(DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &x));
      PetscCall(DMPlexRestoreCellCoordinates(rdm, q, &risDG, &rNc, &rarray, &rx));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Check that every owned point of the transformed mesh is in the closure of an owned cell, as for a distributed mesh.
*/
static PetscErrorCode CheckOwnership(DM rdm)
{
  PetscSF         sf;
  PetscBT         ghost, covered;
  const PetscInt *leaves;
  PetscInt        Nl, pStart, pEnd, cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetChart(rdm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
  PetscCall(DMGetPointSF(rdm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
  PetscCall(PetscBTCreate(pEnd - pStart, &ghost));
  PetscCall(PetscBTCreate(pEnd - pStart, &covered));
  for (PetscInt l = 0; l < PetscMax(Nl, 0); ++l) PetscCall(PetscBTSet(ghost, (leaves ? leaves[l] : l) - pStart));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  Ncl;

    if (PetscBTLookup(ghost, c - pStart)) continue;
    PetscCall(DMPlexGetTransitiveClosure(rdm, c, PETSC_TRUE, &Ncl, &closure));
    for (PetscInt cl = 0; cl < Ncl; ++cl) PetscCall(PetscBTSet(covered, closure[2 * cl] - pStart));
    PetscCall(DMPlexRestoreTransitiveClosure(rdm, c, PETSC_TRUE, &Ncl, &closure));
  }
  for (PetscInt p = pStart; p < pEnd; ++p) PetscCheck(PetscBTLookup(ghost, p - pStart) || PetscBTLookup(covered, p - pStart), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Owned point %" PetscInt_FMT " is not in the closure of an owned cell", p);
  PetscCall(PetscBTDestroy(&ghost));
  PetscCall(PetscBTDestroy(&covered));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Print, for each subdomain, the global number of points of each depth and the total cell volume. Only owned points
  are counted, so these numbers do not depend on the parallel layout when the point SF of the transformed mesh is
  correct. The number of shared points, which depends on the layout, is printed only if viewShared is true.
*/
static PetscErrorCode ViewSubdomains(DM rdm, PetscInt numSubdomains, PetscBool viewShared)
{
  MPI_Comm        comm = PetscObjectComm((PetscObject)rdm);
  DMLabel         label;
  PetscSF         sf;
  const PetscInt *leaves;
  PetscInt       *owned, *counts;
  PetscInt        Nl, NlGlobal, pStart, pEnd, cStart, cEnd, depth;
  PetscReal      *volumes;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(rdm, "subdomain", &label));
  PetscCall(DMPlexGetDepth(rdm, &depth));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &depth, 1, MPIU_INT, MPI_MAX, comm));
  if (depth < 0) PetscCall(DMGetDimension(rdm, &depth));
  PetscCall(DMPlexGetChart(rdm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
  PetscCall(DMGetPointSF(rdm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
  Nl = PetscMax(Nl, 0);
  PetscCall(PetscMalloc1(pEnd - pStart, &owned));
  PetscCall(PetscCalloc2(numSubdomains * (depth + 1), &counts, numSubdomains, &volumes));
  for (PetscInt p = 0; p < pEnd - pStart; ++p) owned[p] = 1;
  for (PetscInt l = 0; l < Nl; ++l) owned[(leaves ? leaves[l] : l) - pStart] = 0;
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt v, d;

    if (!owned[p - pStart]) continue;
    PetscCall(DMLabelGetValue(label, p, &v));
    PetscCall(DMPlexGetPointDepth(rdm, p, &d));
    ++counts[v * (depth + 1) + d];
    if (p >= cStart && p < cEnd) {
      PetscReal vol;

      PetscCall(DMPlexComputeCellGeometryFVM(rdm, p, &vol, NULL, NULL));
      volumes[v] += vol;
    }
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, counts, numSubdomains * (depth + 1), MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, volumes, numSubdomains, MPIU_REAL, MPIU_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(&Nl, &NlGlobal, 1, MPIU_INT, MPI_SUM, comm));
  for (PetscInt s = 0; s < numSubdomains; ++s) {
    PetscCall(PetscPrintf(comm, "Subdomain %" PetscInt_FMT ": points per depth", s));
    for (PetscInt d = 0; d <= depth; ++d) PetscCall(PetscPrintf(comm, " %" PetscInt_FMT, counts[s * (depth + 1) + d]));
    PetscCall(PetscPrintf(comm, ", volume %g\n", (double)volumes[s]));
  }
  if (viewShared) PetscCall(PetscPrintf(comm, "Shared points: %" PetscInt_FMT "\n", NlGlobal));
  PetscCall(PetscFree(owned));
  PetscCall(PetscFree2(counts, volumes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM              dm, rdm;
  DMPlexTransform tr;
  DMLabel         label;
  AppCtx          user;
  PetscInt        numSubdomains;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  if (user.fvAdjacency) PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  PetscCall(CreateSubdomainLabel(dm, &user, &numSubdomains));
  PetscCall(DMGetLabel(dm, "subdomain", &label));
  PetscCall(DMPlexLabelAddOverlap(dm, label, user.grow));

  PetscCall(DMPlexTransformCreate(PETSC_COMM_WORLD, &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "DD"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "dd_"));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXTRANSFORMDD));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetActive(tr, label));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
  PetscCall(DMPlexTransformApply(tr, dm, &rdm));
  if (user.checkReferenceCells) PetscCall(DMPlexTransformGetCellVertices(tr, DM_POLYTOPE_TRIANGLE, NULL, NULL));
  PetscCall(PetscObjectSetName((PetscObject)rdm, "Decomposed Mesh"));
  PetscCall(DMViewFromOptions(rdm, NULL, "-rdm_view"));

  PetscCall(DMPlexCheck(rdm));
  PetscCall(CheckSourcePoints(tr, dm, rdm));
  PetscCall(CheckOwnership(rdm));
  PetscCall(ViewSubdomains(rdm, numSubdomains, user.labelType == LABEL_RANKS ? PETSC_TRUE : PETSC_FALSE));

  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMDestroy(&rdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Overlapping slabs marked by geometry. The owned counts are the same on any number of processes, which checks that the
  # point SF joins the replicas of each subdomain across processes without duplicating any.
  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,3 -petscpartitioner_type simple -num_slabs 3 -slab_overlap 0.1
    output_file: output/ex107_slabs.out

    test:
      suffix: slabs
      nsize: {{1 2 3}}
      args: -dd_dm_plex_transform_dd_ignore_halo {{0 1}}

    test:
      suffix: slabs_overlap
      nsize: {{1 2}}
      args: -dm_distribute_overlap 1 -dd_dm_plex_transform_dd_ignore_halo {{0 1}}

  # The same slabs, produced from disjoint slabs by growing each one by a layer of cells sharing a vertex or a facet
  test:
    suffix: slabs_grow
    nsize: {{1 2 3}}
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,3 -petscpartitioner_type simple -num_slabs 3 -dm_distribute_overlap 1 \
          -grow 1 -fv_adjacency {{0 1}} -dd_dm_plex_transform_dd_ignore_halo {{0 1}}
    output_file: output/ex107_slabs.out

  test:
    suffix: hex_slabs
    nsize: {{1 2}}
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,2,2 -petscpartitioner_type simple -num_slabs 2 -slab_overlap 0.2
    output_file: output/ex107_hex_slabs.out

  test:
    suffix: tet_slabs
    nsize: {{1 2}}
    args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -dm_refine 1 -petscpartitioner_type simple -num_slabs 2 -slab_overlap 0.25
    output_file: output/ex107_tet_slabs.out

  # Disjoint tetrahedral slabs grown by a layer of cells sharing a vertex, or a facet, which adds fewer cells
  testset:
    nsize: {{1 2}}
    args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -dm_refine 1 -petscpartitioner_type simple -num_slabs 2 \
          -dm_distribute_overlap 1 -grow 1

    test:
      suffix: tet_grow_vertex
      output_file: output/ex107_tet_grow_vertex.out

    test:
      suffix: tet_grow_facet
      args: -fv_adjacency
      output_file: output/ex107_tet_grow_facet.out

  # Subdomains aligned with the partition, two on each process. Without a halo, no point is shared. With the subdomains
  # of the owners marked on the ghost cells, the replicas of these cells are shared with their owners.
  testset:
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -petscpartitioner_type simple -dm_distribute_overlap 1 -label_type ranks -num_local_sub 2

    test:
      suffix: ranks
      args: -label_halo 0 -dd_dm_plex_transform_view

    test:
      suffix: ranks_halo
      args: -label_halo 1

  # Growing rank subdomains into the mesh overlap creates replicas whose source cells are ghost cells on a different rank.
  test:
    suffix: ranks_grow_halo
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,1 -petscpartitioner_type simple -dm_distribute_overlap 1 \
          -label_type ranks -num_local_sub 1 -label_halo 0 -grow 1 -dd_dm_plex_transform_dd_ignore_halo 0
    output_file: output/ex107_ranks_grow_halo.out

  # Preserve localized coordinates on periodic meshes.
  test:
    suffix: periodic_ranks
    nsize: 1
    args: -dm_coord_space 0 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -label_type ranks -num_local_sub 1
    output_file: output/ex107_periodic_ranks.out

  test:
    suffix: periodic_slabs
    nsize: {{1 2}}
    args: -dm_coord_space 0 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -num_slabs 2 -slab_overlap 1 -petscpartitioner_type simple

  test:
    suffix: periodic_replicas
    nsize: {{1 2}}
    args: -dm_coord_space 0 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -num_slabs 3 -slab_overlap 10 -petscpartitioner_type simple -check_reference_cells

  test:
    suffix: slabs_empty
    nsize: {{1 2}}
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_box_lower 2,2 -dm_plex_box_upper 3,3 -petscpartitioner_type simple

  # One process owns no cells
  test:
    suffix: ranks_empty
    nsize: 3
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -petscpartitioner_type shell -petscpartitioner_shell_sizes 4,0,4 \
          -petscpartitioner_shell_points 0,1,2,3,4,5,6,7 -dm_distribute_overlap 1 -label_type ranks -label_halo 1

TEST*/
