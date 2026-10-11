static const char help[] = "Tests for adaptive refinement and coarsening";

#include <petscdmplex.h>
#include <petscdmplextransform.h>

typedef struct {
  PetscBool metric;  /* Flag to use metric adaptation, instead of tagging */
  PetscInt  iter;    /* Number of adaptation generations */
  PetscInt *refcell; /* A cell to be refined on each process */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscMPIInt size;
  PetscInt    n;

  PetscFunctionBeginUser;
  options->metric = PETSC_FALSE;
  options->iter   = 1;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscCalloc1(size, &options->refcell));
  n = size;

  PetscOptionsBegin(comm, "", "Parallel Mesh Adaptation Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL));
  PetscCall(PetscOptionsInt("-adapt_iter", "Number of adaptation generations", "ex41.c", options->iter, &options->iter, NULL));
  PetscCall(PetscOptionsIntArray("-refcell", "The cell to be refined", "ex41.c", options->refcell, &n, NULL));
  if (n) PetscCheck(n == size, comm, PETSC_ERR_ARG_SIZ, "Only gave %" PetscInt_FMT " cells to refine, must give one for all %d processes", n, size);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Mark the cells whose centroid lies in the box lo_0,hi_0,lo_1,hi_1,... given by the option -<prefix>_box_<gen> */
static PetscErrorCode MarkBox(DM dm, const char prefix[], PetscInt gen, PetscInt value, DMLabel adaptLabel)
{
  char      name[PETSC_MAX_OPTION_NAME];
  PetscReal box[6];
  PetscInt  dim, n = 6, cStart, cEnd;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(name, sizeof(name), "-%s_box_%" PetscInt_FMT, prefix, gen));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, name, box, &n, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCheck(n == 2 * dim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Option %s needs %" PetscInt_FMT " values, not %" PetscInt_FMT, name, 2 * dim, n);
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3];
    PetscBool inside = PETSC_TRUE;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    for (PetscInt d = 0; d < dim; ++d) inside = (PetscBool)(inside && centroid[d] > box[2 * d] && centroid[d] < box[2 * d + 1]);
    if (inside) PetscCall(DMLabelSetValue(adaptLabel, c, value));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, PetscInt gen, DMLabel *adaptLabel)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  if (ctx->refcell[rank] >= 0) PetscCall(DMLabelSetValue(*adaptLabel, ctx->refcell[rank], DM_ADAPT_REFINE));
  PetscCall(MarkBox(dm, "coarsen", gen, DM_ADAPT_COARSEN, *adaptLabel));
  PetscCall(MarkBox(dm, "refine", gen, DM_ADAPT_REFINE, *adaptLabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConstructRefineTree(DM dm)
{
  DMPlexTransform tr;
  DM              odm;
  PetscInt        cStart, cEnd;

  PetscFunctionBegin;
  PetscCall(DMPlexGetTransform(dm, &tr));
  PetscCall(DMGetCoarseDM(dm, &odm));
  if (!tr || !odm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, dim, pNew = 0;

    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "Cell %" PetscInt_FMT " produced new cells", c));
    PetscCall(DMPlexGetCellType(odm, c, &ct));
    dim = DMPolytopeTypeGetDim(ct);
    PetscCall(DMPlexTransformCellTransform(tr, ct, c, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
    for (PetscInt n = 0; n < Nct; ++n) {
      if (DMPolytopeTypeGetDim(rct[n]) != dim) continue;
      for (PetscInt r = 0; r < rsize[n]; ++r) {
        PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], c, r, &pNew));
        PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), " %" PetscInt_FMT, pNew));
      }
    }
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "\n"));
  }
  PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject)dm), PETSC_STDOUT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM              dm, dma = NULL, cdm;
  DMPlexTransform tr;
  DMLabel         adaptLabel;
  AppCtx          ctx;
  PetscBool       save;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  for (PetscInt i = 0; i < ctx.iter; ++i) {
    PetscCall(CreateAdaptLabel(dm, &ctx, i, &adaptLabel));
    PetscCall(DMAdaptLabel(dm, adaptLabel, &dma));
    PetscCall(PetscObjectSetName((PetscObject)dma, "Adapted Mesh"));
    /* Coarsening needs the transforms of all the generations, and the meshes they were applied to. A coarsening links
       the meshes it makes, and may return an ancestor, so only a mesh made by a refinement is linked here */
    PetscCall(DMPlexGetSaveTransform(dm, &save));
    PetscCall(DMPlexSetSaveTransform(dma, save));
    PetscCall(DMPlexGetTransform(dma, &tr));
    PetscCall(DMGetCoarseDM(dma, &cdm));
    if (tr && !cdm) PetscCall(DMSetCoarseDM(dma, dm));
    PetscCall(DMLabelDestroy(&adaptLabel));
    PetscCall(DMPlexCheck(dma));
    PetscCall(DMViewFromOptions(dma, NULL, "-adapt_dm_view"));
    if (i < ctx.iter - 1) {
      PetscCall(DMDestroy(&dm));
      dm = dma;
    }
  }
  PetscCall(ConstructRefineTree(dma));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dma));
  PetscCall(PetscFree(ctx.refcell));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_adaptor cellrefiner -dm_plex_transform_type refine_sbr

    test:
      suffix: 0
      requires: triangle
      args: -dm_view -adapt_dm_view

    test:
      suffix: 1
      requires: triangle
      args: -dm_coord_space 0 -refcell 2 -dm_view ::ascii_info_detail -adapt_dm_view ::ascii_info_detail

    test:
      suffix: 1_save
      requires: triangle
      args: -refcell 2 -dm_plex_save_transform -dm_view -adapt_dm_view

    test:
      suffix: 2
      requires: triangle
      nsize: 2
      args: -refcell 2,-1 -petscpartitioner_type simple -dm_view -adapt_dm_view

    # Refine one tetrahedron of two, the example of Fig. 6 of Plaza & Carey (2000), and validate
    # the subdivision generator against the enumeration of Table 4 of the paper
    test:
      suffix: 3
      args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -refcell 0 \
            -dm_plex_transform_sbr_validate -dm_view -adapt_dm_view

    # Keep refining the first cell of each generation, verifying mesh validity each time
    test:
      suffix: 4
      args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -refcell 0 -adapt_iter 4 -adapt_dm_view

    # One tetrahedron per process, exercising the conformity closure across the shared face
    test:
      suffix: 5
      nsize: 2
      args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -refcell 0,-1 -adapt_iter 2 \
            -petscpartitioner_type simple -dm_distribute -dm_view -adapt_dm_view

    # Coarsening needs the meshes linked by DMSetCoarseDM(); the filter drops their views from the view of each generation
    # Coarsening undoes one generation of requested refinement per call, ending at the initial mesh
    test:
      suffix: coarsen_0
      filter: grep "^  Number of"
      args: -dm_plex_shape doublet -dm_plex_simplex 1 -refcell -1 -dm_plex_save_transform -adapt_iter 4 \
            -refine_box_0 -1,1,-1,2 -refine_box_1 -1,1,-1,2 -coarsen_box_2 -1,1,-1,2 -coarsen_box_3 -1,1,-1,2 \
            -adapt_dm_view

    # Coarsening the left triangle keeps the closure bisection of the right triangle,
    # whose lower child was refined in the second generation
    test:
      suffix: coarsen_1
      filter: grep "^  Number of"
      args: -dm_plex_shape doublet -dm_plex_simplex 1 -refcell -1 -dm_plex_save_transform -adapt_iter 3 \
            -refine_box_0 -1,0,-1,2 -refine_box_1 0,1,-1,0.5 -coarsen_box_2 -1,0,-1,2 -adapt_dm_view

    # Coarsen one side and refine the other in the same call
    test:
      suffix: coarsen_2
      filter: grep "^  Number of"
      args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -refcell -1 -dm_plex_save_transform -adapt_iter 4 \
            -refine_box_0 -2,2,-2,2,-2,2 -refine_box_1 -2,0,-2,2,-2,2 \
            -coarsen_box_2 -2,0,-2,2,-2,2 -refine_box_2 0,2,-2,2,-2,2 -coarsen_box_3 -2,2,-2,2,-2,2 -adapt_dm_view

    # One tetrahedron per process
    test:
      suffix: coarsen_3
      filter: grep "^  Number of"
      nsize: 2
      args: -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -refcell -1,-1 -dm_plex_save_transform -adapt_iter 3 \
            -refine_box_0 -2,2,-2,2,-2,2 -refine_box_1 -2,0,-2,2,-2,2 \
            -coarsen_box_2 -2,2,-2,2,-2,2 -petscpartitioner_type simple -dm_distribute -adapt_dm_view

    # Edges that cross the periodic boundary are measured between the nearest images of their vertices,
    # so refining the corner of a periodic mesh gives the same mesh sizes as refining an interior square
    test:
      suffix: periodic
      filter: grep "^  Number of"
      args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -dm_plex_box_bd periodic,periodic -dm_sparse_localize 0 \
            -dm_plex_option_phases sim_ -sim_dm_refine 1 -sim_dm_plex_transform_type refine_tosimplex -refcell -1 -adapt_iter 3 \
            -refine_box_0 0.75,1,0.75,1 -refine_box_1 0.75,1,0.75,1 -refine_box_2 0.75,1,0.75,1 -adapt_dm_view

TEST*/
