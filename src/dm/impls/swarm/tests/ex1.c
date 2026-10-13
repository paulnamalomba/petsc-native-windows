static const char help[] = "Test initialization and migration with swarm.\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>

typedef struct {
  PetscReal L[3]; /* Dimensions of the mesh bounding box */
  PetscBool setClosurePermutation;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Swarm configuration options", "DMSWARM");
  options->setClosurePermutation = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-set_closure_permutation", "Set the closure permutation to tensor ordering", NULL, options->setClosurePermutation, &options->setClosurePermutation, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscReal low[3], high[3];
  PetscInt  cdim;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  if (user->setClosurePermutation) {
    DM cdm;

    // -- Set tensor permutation
    PetscCall(DMGetCoordinateDM(*dm, &cdm));
    PetscCall(DMPlexSetClosurePermutationTensor(*dm, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
  }
  PetscCall(DMGetCoordinateDim(*dm, &cdim));
  PetscCall(DMGetBoundingBox(*dm, low, high));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim: %" PetscInt_FMT "\n", cdim));
  for (PetscInt d = 0; d < cdim; ++d) user->L[d] = high[d] - low[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This function initializes all particles on rank 0.
  They are sent to other ranks to test migration across non nearest neighbors
*/
static PetscErrorCode CreateSwarm(DM dm, DM *sw, AppCtx *user)
{
  DMSwarmCellDM celldm;
  PetscInt      particleInitSize = 10;
  PetscReal    *coords, upper[3], lower[3];
  PetscInt     *swarm_cellid, Np, dim, Nfc;
  PetscMPIInt   rank, size;
  MPI_Comm      comm;
  const char  **coordFields, *cellid;

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetBoundingBox(dm, lower, upper));
  PetscCall(DMCreate(PETSC_COMM_WORLD, sw));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  /* Options must be consumed before DMSwarmSetLocalSizes() triggers DMSetUp(), so that
     -dm_swarm_backend can still be honored (it is a hard error after setup) */
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmSetLocalSizes(*sw, rank == 0 ? particleInitSize : 0, 0));
  PetscCall(DMSwarmGetCellDMActive(*sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmGetField(*sw, coordFields[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(*sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmGetLocalSize(*sw, &Np));
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < dim; ++d) coords[p * dim + d] = 0.5 * (upper[d] - lower[d]);
    coords[p * dim + 1] = (upper[1] - lower[1]) / particleInitSize * p + lower[1];
    swarm_cellid[p]     = 0;
  }
  PetscCall(DMSwarmRestoreField(*sw, coordFields[0], NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(*sw, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Configure the swarm on rank 0 with all particles
  located there, then migrate where they need to be
*/
static PetscErrorCode CheckMigrate(DM sw)
{
  Vec       preMigrate, postMigrate, tmp;
  PetscInt  preSize, postSize;
  PetscReal prenorm, postnorm;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(VecDuplicate(tmp, &preMigrate));
  PetscCall(VecCopy(tmp, preMigrate));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(VecDuplicate(tmp, &postMigrate));
  PetscCall(VecCopy(tmp, postMigrate));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(VecGetSize(preMigrate, &preSize));
  PetscCall(VecGetSize(postMigrate, &postSize));
  PetscCheck(preSize == postSize, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Particles either lost or duplicated. Pre migrate global size %" PetscInt_FMT " != Post migrate size %" PetscInt_FMT, preSize, postSize);
  PetscCall(VecNorm(preMigrate, NORM_2, &prenorm));
  PetscCall(VecNorm(postMigrate, NORM_2, &postnorm));
  PetscCheck(PetscAbsReal(prenorm - postnorm) < 100. * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_COR, "Particle coordinates corrupted in migrate with abs(norm(pre) - norm(post)) = %.16g", PetscAbsReal(prenorm - postnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Migrate check passes.\n"));
  PetscCall(VecDestroy(&preMigrate));
  PetscCall(VecDestroy(&postMigrate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Checks added points persist on migrate
*/
static PetscErrorCode CheckPointInsertion(DM sw)
{
  PetscInt    Np_pre, Np_post;
  PetscMPIInt rank, size;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscPrintf(comm, "Basic point insertion check...\n"));
  PetscCall(DMSwarmGetSize(sw, &Np_pre));
  if (rank == 0) PetscCall(DMSwarmAddPoint(sw));
  PetscCall(DMSwarmGetSize(sw, &Np_post));
  PetscCheck(Np_post == (Np_pre + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Particle insertion failed. Global size pre insertion: %" PetscInt_FMT " global size post insertion: %" PetscInt_FMT, Np_pre, Np_post);
  PetscCall(CheckMigrate(sw));
  PetscCall(PetscPrintf(comm, "Basic point insertion check passes.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Checks tie breaking works properly when a particle
  is located at a shared boundary. The higher rank should
  receive the particle while the lower rank deletes it.

  TODO: Currently only works for 2 procs.
*/
static PetscErrorCode CheckPointInsertion_Boundary(DM sw)
{
  PetscInt    Np_loc_pre, Np_loc_post, dim;
  PetscMPIInt rank, size;
  PetscReal   lbox_low[3], lbox_high[3], gbox_low[3], gbox_high[3];
  MPI_Comm    comm;
  DM          cdm;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscPrintf(comm, "Rank boundary point insertion check...\n"));
  PetscCall(DMSwarmGetCellDM(sw, &cdm));
  PetscCall(DMGetDimension(cdm, &dim));
  PetscCall(DMGetBoundingBox(cdm, gbox_low, gbox_high));
  if (rank == 0) {
    PetscReal *coords;
    PetscInt   adjacentdim = 0, Np;

    PetscCall(DMGetLocalBoundingBox(cdm, lbox_low, lbox_high));
    // find a face that belongs to the neighbor.
    for (PetscInt d = 0; d < dim; ++d) {
      if ((gbox_high[d] - lbox_high[d]) != 0.) adjacentdim = d;
    }
    PetscCall(DMSwarmAddPoint(sw));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    for (PetscInt d = 0; d < dim; ++d) coords[(Np - 1) * dim + d] = 0.5 * (lbox_high[d] - lbox_low[d]);
    coords[(Np - 1) * dim + adjacentdim] = lbox_high[adjacentdim];
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  }
  PetscCall(DMSwarmGetLocalSize(sw, &Np_loc_pre));
  PetscCall(CheckMigrate(sw));
  PetscCall(DMSwarmGetLocalSize(sw, &Np_loc_post));
  if (rank == 0) PetscCheck(Np_loc_pre == (Np_loc_post + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Migration tie breaking failed on rank %d. Particle on boundary not sent.", rank);
  if (rank == 1) PetscCheck(Np_loc_pre == (Np_loc_post - 1), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Migration tie breaking failed on rank %d. Particle on boundary not received.", rank);
  PetscCall(PetscPrintf(comm, "Rank boundary point insertion check passes.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       dm, sw;
  MPI_Comm comm;
  AppCtx   user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateSwarm(dm, &sw, &user));
  PetscCall(CheckMigrate(sw));
  PetscCall(CheckPointInsertion(sw));
  PetscCall(CheckPointInsertion_Boundary(sw));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not handle complex or quad
  build:
    requires: !complex double
  # swarm_migrate_hash and swarm_migrate_scan test swarm migration against point location types
  # with a distributed mesh where ranks overlap by 1. Points in the shared boundary should
  # be sent to the process which has the highest rank that has that portion of the domain.
  test:
    suffix: swarm_migrate_vec_hip_scan
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location false
  # Same migration exercised with device-resident particle storage; migration itself still
  # runs on the host through the sync-on-access machinery, so the output is unchanged
  test:
    suffix: swarm_migrate_backend_hip_scan
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location false\
          -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_vec_hip_scan.out
  # Same device migration with the particle payload staged through pinned host buffers instead
  # of gpu-aware MPI; the transport moves identical bytes, so the expected output is shared
  test:
    suffix: swarm_migrate_backend_hip_scan_no_gpu_aware
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location false\
          -dm_swarm_backend hip -use_gpu_aware_mpi 0
    output_file: output/ex1_swarm_migrate_vec_hip_scan.out
  test:
    suffix: swarm_migrate_vec_hip_hash
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location true
  # Same migration with device-resident particle storage; the cell ids are computed by the
  # device grid-hash locate kernel, bitwise identical to the host path
  test:
    suffix: swarm_migrate_backend_hip_hash
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location true\
          -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_vec_hip_hash.out
  test:
    suffix: swarm_migrate_vec_hip_hash_tensor_permutation
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location true\
          -set_closure_permutation
  # Device-resident twin of the tensor-permutation hash variant, see swarm_migrate_backend_hip_hash
  test:
    suffix: swarm_migrate_backend_hip_hash_tensor_permutation
    nsize: 2
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type hip -dm_plex_hash_location true\
          -set_closure_permutation -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_vec_hip_hash_tensor_permutation.out
  test:
    suffix: swarm_migrate_hash
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location true
    filter: grep -v marker | grep -v atomic | grep -v usage
  test:
    suffix: swarm_migrate_hash_tensor_permutation
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location true -set_closure_permutation
    filter: grep -v marker | grep -v atomic | grep -v usage
  test:
    suffix: swarm_migrate_scan
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false
    filter: grep -v marker | grep -v atomic | grep -v usage
  test:
    suffix: swarm_migrate_scan_tensor_permutation
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -set_closure_permutation
    filter: grep -v marker | grep -v atomic | grep -v usage
  # Simplex migration: the host variants pin the baseline and the hip-backend twins share their
  # expected output, making device-host parity of the simplex (v0/invJ barycentric) locate path
  # a committed regression
  test:
    suffix: swarm_migrate_tri
    nsize: 2
    requires: triangle
    args: -dm_plex_dim 2 -dm_plex_simplex 1 -dm_distribute_overlap 1 -dm_plex_box_faces 4,16\
          -dm_plex_box_lower 0.,0. -dm_plex_box_upper 1.,10. -dm_plex_box_bd none,none\
          -dm_plex_hash_location false
  test:
    suffix: swarm_migrate_backend_hip_tri
    nsize: 2
    requires: triangle hip
    args: -dm_plex_dim 2 -dm_plex_simplex 1 -dm_distribute_overlap 1 -dm_plex_box_faces 4,16\
          -dm_plex_box_lower 0.,0. -dm_plex_box_upper 1.,10. -dm_plex_box_bd none,none\
          -dm_plex_hash_location false -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_tri.out
  # The RCM reordering makes the contiguous-chunk (simple) partition of the ctetgen mesh clean
  # enough for the rank-boundary tie-breaking check
  test:
    suffix: swarm_migrate_tet
    nsize: 2
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_simplex 1 -dm_plex_reorder rcm -dm_distribute_overlap 1\
          -dm_plex_box_faces 2,2,8 -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10.\
          -dm_plex_box_bd none,none,none -dm_plex_hash_location false
  test:
    suffix: swarm_migrate_backend_hip_tet
    nsize: 2
    requires: ctetgen hip
    args: -dm_plex_dim 3 -dm_plex_simplex 1 -dm_plex_reorder rcm -dm_distribute_overlap 1\
          -dm_plex_box_faces 2,2,8 -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10.\
          -dm_plex_box_bd none,none,none -dm_plex_hash_location false -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_tet.out
  # Degree-2 coordinate space: the point-location geometry cache reports itself unavailable, so
  # the hip backend silently falls back to host point location with identical output
  test:
    suffix: swarm_migrate_coord_deg2
    nsize: 2
    args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10\
          -dm_plex_box_lower 0.,0. -dm_plex_box_upper 1.,10. -dm_plex_box_bd none,none\
          -dm_plex_hash_location false -dm_coord_petscspace_degree 2
  test:
    suffix: swarm_migrate_backend_hip_coord_deg2
    nsize: 2
    requires: hip
    args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10\
          -dm_plex_box_lower 0.,0. -dm_plex_box_upper 1.,10. -dm_plex_box_bd none,none\
          -dm_plex_hash_location false -dm_coord_petscspace_degree 2 -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_coord_deg2.out
  # Four-rank migration: the 8 z-slabs split into two clean slabs per rank under the simple
  # partitioner, and the particle line at the domain z midpoint sits on the rank 1/rank 2
  # ownership plane, one neighbour hop from rank 0 where every point starts. Points a receiving
  # rank cannot locate are dropped, so all particles must be locatable within one hop of their
  # origin (a four-band partition of the particle spread dimension would lose points).
  test:
    suffix: swarm_migrate_scan_np4
    nsize: 4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false
  # Device-resident twins of the four-rank migration under every payload transport: the default
  # (gpu-aware where supported), forced host staging, and forced gpu-aware MPI. Every transport
  # moves identical bytes, so all three share the host baseline's expected output.
  test:
    suffix: swarm_migrate_backend_hip_scan_np4
    nsize: 4
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_swarm_backend hip
    output_file: output/ex1_swarm_migrate_scan_np4.out
  test:
    suffix: swarm_migrate_backend_hip_scan_np4_no_gpu_aware
    nsize: 4
    requires: hip
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_swarm_backend hip -use_gpu_aware_mpi 0
    output_file: output/ex1_swarm_migrate_scan_np4.out
  test:
    suffix: swarm_migrate_backend_hip_scan_np4_gpu_aware
    nsize: 4
    requires: hip defined(PETSC_HAVE_MPI_GPU_AWARE)
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_swarm_backend hip -use_gpu_aware_mpi 1
    output_file: output/ex1_swarm_migrate_scan_np4.out

  # CUDA twins of the hip tests above: the same configurations under the cuda backend, sharing their expected outputs
  test:
    suffix: swarm_migrate_backend_cuda_scan
    nsize: 2
    requires: cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type cuda -dm_plex_hash_location false\
          -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_vec_hip_scan.out
  test:
    suffix: swarm_migrate_backend_cuda_scan_no_gpu_aware
    nsize: 2
    requires: cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type cuda -dm_plex_hash_location false\
          -dm_swarm_backend cuda -use_gpu_aware_mpi 0
    output_file: output/ex1_swarm_migrate_vec_hip_scan.out
  test:
    suffix: swarm_migrate_backend_cuda_hash
    nsize: 2
    requires: cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type cuda -dm_plex_hash_location true\
          -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_vec_hip_hash.out
  test:
    suffix: swarm_migrate_backend_cuda_hash_tensor_permutation
    nsize: 2
    requires: cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_vec_type cuda -dm_plex_hash_location true\
          -set_closure_permutation -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_vec_hip_hash_tensor_permutation.out
  test:
    suffix: swarm_migrate_backend_cuda_tri
    nsize: 2
    requires: triangle cuda
    args: -dm_plex_dim 2 -dm_plex_simplex 1 -dm_distribute_overlap 1 -dm_plex_box_faces 4,16\
          -dm_plex_box_lower 0.,0. -dm_plex_box_upper 1.,10. -dm_plex_box_bd none,none\
          -dm_plex_hash_location false -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_tri.out
  test:
    suffix: swarm_migrate_backend_cuda_tet
    nsize: 2
    requires: ctetgen cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 1 -dm_plex_reorder rcm -dm_distribute_overlap 1\
          -dm_plex_box_faces 2,2,8 -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10.\
          -dm_plex_box_bd none,none,none -dm_plex_hash_location false -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_tet.out
  test:
    suffix: swarm_migrate_backend_cuda_coord_deg2
    nsize: 2
    requires: cuda
    args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10\
          -dm_plex_box_lower 0.,0. -dm_plex_box_upper 1.,10. -dm_plex_box_bd none,none\
          -dm_plex_hash_location false -dm_coord_petscspace_degree 2 -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_coord_deg2.out
  test:
    suffix: swarm_migrate_backend_cuda_scan_np4
    nsize: 4
    requires: cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_swarm_backend cuda
    output_file: output/ex1_swarm_migrate_scan_np4.out
  test:
    suffix: swarm_migrate_backend_cuda_scan_np4_no_gpu_aware
    nsize: 4
    requires: cuda
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_swarm_backend cuda -use_gpu_aware_mpi 0
    output_file: output/ex1_swarm_migrate_scan_np4.out
  test:
    suffix: swarm_migrate_backend_cuda_scan_np4_gpu_aware
    nsize: 4
    requires: cuda defined(PETSC_HAVE_MPI_GPU_AWARE)
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,8\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -dm_swarm_backend cuda -use_gpu_aware_mpi 1
    output_file: output/ex1_swarm_migrate_scan_np4.out
TEST*/
