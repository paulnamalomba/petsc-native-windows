static const char help[] = "Times particle migration with swarm for comparing storage backends and point location algorithms.\n";

/*
  Particle migration benchmark.

  Every migration runs inside its own logging stage, "Swarm Migration", so that -log_view (and its
  CSV format, -log_view :migrate.csv:ascii_csv, which is convenient for plotting) reports the cost of
  migration separately from the setup and from the particle push. With -report_timings the example
  also reads the logged events of that stage back with PetscLogEventGetPerfInfo() and prints a
  short table, the maximum over all ranks of the count and time of each migration event.

  The configurations to compare are selected purely by options:
    -dm_swarm_backend host|hip           where the particle data lives and migration runs
    -dm_plex_hash_location true|false    grid hashing or brute force search in point location

  Particles are laid out at the Gauss points of every cell and pushed back and forth along the
  fixed field v = min(u - x0, D) pointing to the upper corner u of the box, x <- x + dt v on even
  steps and x <- x - dt v on odd steps; the minimum is taken per component with the cap D set by
  -push_distance (no cap by default). A particle only ever moves between its initial position and a
  point on the way to the corner, so no particle leaves the domain, and the particles just below
  any plane that splits the box cross it, so there is migration between ranks. Every step after
  the first moves the same particles across the same cell and rank boundaries, so the migration
  workload is the same every step. Migration only exchanges points with neighboring ranks, so
  dt min(L, D) must stay below the extent of the subdomains (L is the box size). With a cap, the
  motion is the same everywhere in a box elongated across the ranks, which keeps the work per rank
  fixed when the box grows with the rank count.

  The push is done with VecAXPY() on the field vectors, so under the hip backend it runs on the
  device and no particle data moves between host and device during the timed migrations. The
  device is synchronized and the ranks are lined up with a barrier before each migration, and the
  device is synchronized again before leaving the stage, so the stage measures exactly the
  migration. The first -warmup migrations, which build the point location and communication
  caches, are run outside the stage.

  The report also gives the number of particles that change rank in each migration. Each particle
  gets a unique DMSwarmField_pid at creation, so after a push away from the initial positions the
  particles not on the rank that created them are exactly those every migration moves. They are
  counted after the timed migrations, with one more untimed push and migration if the last push
  returned the particles to their initial positions.

  With -check the particle data is verified after every migration (every -check_every migrations):
  every id occurs exactly once over all ranks, the coordinates are where the pushes so far have
  taken the particle's initial position (kept in an extra field), the velocity is the one computed
  from that position at creation, and the cell id is the cell that host point location finds for
  the coordinates. Corruption of the records exchanged between ranks or of the device copies of
  the fields shows up in one of these. The extra field and the host-side checks change the
  timings, so -check is for validating a configuration, not for timing it.
*/

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscdevice.h>
#include <petscbt.h>
#include <petscsf.h>

typedef struct {
  PetscInt    order;         /* Quadrature order of the Gauss point layout, which sets the particles per cell */
  PetscInt    nsteps;        /* Number of timed migrations */
  PetscInt    warmup;        /* Number of untimed migrations before the timed ones */
  PetscReal   dt;            /* Push step */
  PetscReal   pushDistance;  /* Cap on each component of the push field */
  PetscBool   reportTimings; /* Print the timings of the migration stage */
  PetscBool   check;         /* Verify the particle data after migrations */
  PetscInt    checkEvery;    /* Verify after every this many migrations */
  PetscInt    step;          /* Number of pushes so far, odd steps push back */
  DMSwarmPID *pidStart;      /* The particles created on rank r have the ids [pidStart[r], pidStart[r + 1]) */
  PetscReal   upper[3];      /* Upper corner of the box, the push target */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->order         = 2;
  options->nsteps        = 10;
  options->warmup        = 1;
  options->dt            = 0.1;
  options->pushDistance  = PETSC_MAX_REAL;
  options->reportTimings = PETSC_FALSE;
  options->check         = PETSC_FALSE;
  options->checkEvery    = 1;
  options->step          = 0;
  PetscOptionsBegin(comm, "", "Swarm migration benchmark options", "DMSWARM");
  PetscCall(PetscOptionsBoundedInt("-particle_order", "Quadrature order of the Gauss point layout of the particles in each cell", NULL, options->order, &options->order, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-nsteps", "Number of timed migrations", NULL, options->nsteps, &options->nsteps, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-warmup", "Number of untimed migrations before the timed ones", NULL, options->warmup, &options->warmup, NULL, 0));
  PetscCall(PetscOptionsReal("-push_distance", "Cap on each component of the push field, a particle moves at most dt times this per coordinate", NULL, options->pushDistance, &options->pushDistance, NULL));
  PetscCall(PetscOptionsReal("-dt", "Push step, a particle moves dt times its distance to the upper corner of the box", NULL, options->dt, &options->dt, NULL));
  PetscCall(PetscOptionsBool("-report_timings", "Print the timings of the migration stage", NULL, options->reportTimings, &options->reportTimings, NULL));
  PetscCall(PetscOptionsBool("-check", "Verify the particle data (ids, coordinates, velocity, cell) after migrations", NULL, options->check, &options->check, NULL));
  PetscCall(PetscOptionsBoundedInt("-check_every", "Verify after every this many migrations", NULL, options->checkEvery, &options->checkEvery, NULL, 1));
  PetscOptionsEnd();
  PetscCheck(options->pushDistance > 0., comm, PETSC_ERR_ARG_OUTOFRANGE, "The push distance %g must be positive", (double)options->pushDistance);
  PetscCheck(options->dt >= 0. && options->dt < 1., comm, PETSC_ERR_ARG_OUTOFRANGE, "The push step %g must be in [0, 1) to keep the particles in the box", (double)options->dt);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Particles at the Gauss points of every cell, each with the velocity v = min(u - x0, D) pointing to the upper corner of the box */
static PetscErrorCode CreateSwarm(DM dm, DM *sw, AppCtx *user)
{
  PetscReal  *coords, *vel, upper[3];
  DMSwarmPID *pid;
  PetscInt    dim, Np, *sizes;
  PetscMPIInt rank, size;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetBoundingBox(dm, NULL, upper));
  PetscCall(DMCreate(PETSC_COMM_WORLD, sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  if (user->check) PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "x0", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  /* Options must be consumed before DMSwarmInsertPointsUsingCellDM() triggers DMSetUp(), so that
     -dm_swarm_backend can still be honored (it is a hard error after setup) */
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmInsertPointsUsingCellDM(*sw, DMSWARMPIC_LAYOUT_GAUSS, user->order));
  PetscCall(DMSwarmGetLocalSize(*sw, &Np));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(*sw, "velocity", NULL, NULL, (void **)&vel));
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < dim; ++d) vel[p * dim + d] = PetscMin(upper[d] - coords[p * dim + d], user->pushDistance);
  }
  PetscCall(DMSwarmRestoreField(*sw, "velocity", NULL, NULL, (void **)&vel));
  if (user->check) {
    PetscReal *x0;

    PetscCall(DMSwarmGetField(*sw, "x0", NULL, NULL, (void **)&x0));
    PetscCall(PetscArraycpy(x0, coords, Np * dim));
    PetscCall(DMSwarmRestoreField(*sw, "x0", NULL, NULL, (void **)&x0));
  }
  for (PetscInt d = 0; d < dim; ++d) user->upper[d] = upper[d];
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  /* unique ids, numbered consecutively by creating rank */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscMalloc1(size, &sizes));
  PetscCall(PetscMalloc1(size + 1, &user->pidStart));
  PetscCallMPI(MPI_Allgather(&Np, 1, MPIU_INT, sizes, 1, MPIU_INT, PETSC_COMM_WORLD));
  user->pidStart[0] = 0;
  for (PetscMPIInt r = 0; r < size; ++r) user->pidStart[r + 1] = user->pidStart[r] + sizes[r];
  PetscCall(PetscFree(sizes));
  PetscCall(DMSwarmGetField(*sw, DMSwarmField_pid, NULL, NULL, (void **)&pid));
  for (PetscInt p = 0; p < Np; ++p) pid[p] = user->pidStart[rank] + p;
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmField_pid, NULL, NULL, (void **)&pid));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x <- x + a v, done by the Vec implementation of the backend, so on the device under the hip backend */
static PetscErrorCode PushParticles(DM sw, PetscReal a)
{
  Vec x, v;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &x));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(VecAXPY(x, a, v));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Verify the particle data after a migration: every particle id occurs exactly once over all ranks,
  the coordinates are where the pushes so far have taken the particle's initial position x0 (a field
  filled at creation), the velocity is the one computed from x0 at creation, and the cell id is the
  cell that host point location on the cell DM finds for the coordinates. Any corruption of the
  records exchanged between ranks, or of the device copies of the fields, shows up in one of these.
*/
static PetscErrorCode CheckParticles(DM sw, AppCtx *user)
{
  DM                 dm;
  DMSwarmCellDM      celldm;
  Vec                pos;
  PetscSF            sf = NULL;
  const PetscSFNode *cells;
  const PetscReal   *coords, *x0, *vel;
  const PetscInt    *cellid;
  const DMSwarmPID  *pid;
  DMSwarmPID        *allpid = NULL;
  PetscMPIInt       *counts = NULL, *displs = NULL, np, rank, size;
  const char        *cellidName;
  PetscInt           dim, Np, N, cStart, cEnd, bad = 0, badtot, nprint = 0;
  const PetscReal    a   = user->step % 2 ? user->dt : 0.; /* an odd number of pushes so far: away from the initial positions */
  const PetscReal    tol = 1e-12;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetSize(sw, &N));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellidName));
  /* the cells containing the current coordinates, located on the host independently of the migration */
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dim, Np * dim, coords, &pos));
  PetscCall(DMLocatePoints(dm, pos, DM_POINTLOCATION_NONE, &sf));
  PetscCall(VecDestroy(&pos));
  PetscCall(PetscSFGetGraph(sf, NULL, NULL, NULL, &cells));
  PetscCall(DMSwarmGetField(sw, "x0", NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, cellidName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pid));
  for (PetscInt p = 0; p < Np; ++p) {
    PetscBool ok = (PetscBool)(pid[p] >= 0 && pid[p] < N && cellid[p] >= cStart && cellid[p] < cEnd && cells[p].index == cellid[p]);

    for (PetscInt d = 0; d < dim; ++d) {
      const PetscReal expected = x0[p * dim + d] + a * vel[p * dim + d];

      if (PetscAbsReal(coords[p * dim + d] - expected) > tol * (1. + PetscAbsReal(expected))) ok = PETSC_FALSE;
      if (vel[p * dim + d] != PetscMin(user->upper[d] - x0[p * dim + d], user->pushDistance)) ok = PETSC_FALSE;
    }
    if (!ok) {
      ++bad;
      if (nprint++ < 3) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] corrupted particle %" PetscInt_FMT ": pid %" DMSwarmPID_FMT ", cell %" PetscInt_FMT " (located %" PetscInt_FMT ")\n", rank, p, pid[p], cellid[p], cells[p].index));
        for (PetscInt d = 0; d < dim; ++d)
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "      d = %" PetscInt_FMT ": x %.17g, x0 %.17g, v %.17g, expected x %.17g\n", d, (double)coords[p * dim + d], (double)x0[p * dim + d], (double)vel[p * dim + d], (double)(x0[p * dim + d] + a * vel[p * dim + d])));
      }
    }
  }
  /* every id exactly once over all ranks: gather them and mark */
  PetscCall(PetscMPIIntCast(Np, &np));
  if (rank == 0) PetscCall(PetscMalloc3(size, &counts, size + 1, &displs, N, &allpid));
  PetscCallMPI(MPI_Gather(&np, 1, MPI_INT, counts, 1, MPI_INT, 0, PETSC_COMM_WORLD));
  if (rank == 0) {
    displs[0] = 0;
    for (PetscMPIInt r = 0; r < size; ++r) displs[r + 1] = displs[r] + counts[r];
  }
  PetscCallMPI(MPI_Gatherv(pid, np, MPIU_DMSWARM_PID, allpid, counts, displs, MPIU_DMSWARM_PID, 0, PETSC_COMM_WORLD));
  if (rank == 0) {
    PetscBT seen;

    PetscCall(PetscBTCreate(N, &seen));
    for (PetscInt i = 0; i < displs[size]; ++i) {
      if (allpid[i] < 0 || allpid[i] >= N || PetscBTLookupSet(seen, (PetscInt)allpid[i])) ++bad;
    }
    PetscCall(PetscBTDestroy(&seen));
    PetscCall(PetscFree3(counts, displs, allpid));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pid));
  PetscCall(DMSwarmRestoreField(sw, cellidName, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "x0", NULL, NULL, (void **)&x0));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(PetscSFDestroy(&sf));
  PetscCallMPI(MPIU_Allreduce(&bad, &badtot, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCheck(!badtot, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%" PetscInt_FMT " corrupted particles after push %" PetscInt_FMT, badtot, user->step);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Push and migrate nsteps times, checking that no particle is lost. The push direction alternates
  with the step count kept in user, so it continues across calls. When timed, each migration is
  logged in the stage, isolated from the push by synchronizing the device and the ranks before it
  and the device after it.
*/
static PetscErrorCode PushAndMigrate(DM sw, AppCtx *user, PetscInt nsteps, PetscBool timed, PetscLogStage stage)
{
  PetscDeviceContext dctx;
  PetscInt           N0, N;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(DMSwarmGetSize(sw, &N0));
  for (PetscInt step = 0; step < nsteps; ++step) {
    PetscCall(PushParticles(sw, user->step++ % 2 ? -user->dt : user->dt));
    if (timed) {
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscLogStagePush(stage));
    }
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    if (timed) {
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(PetscLogStagePop());
    }
    PetscCall(DMSwarmGetSize(sw, &N));
    PetscCheck(N == N0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Migration %" PetscInt_FMT ": %" PetscInt_FMT " of %" PetscInt_FMT " particles were lost", step, N0 - N, N0);
    if (user->check && user->step % user->checkEvery == 0) PetscCall(CheckParticles(sw, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Count the particles that are not on the rank that created them. Called after a push away from
  the initial positions, these are the particles that every migration moves between ranks.
*/
static PetscErrorCode CountMovedParticles(DM sw, AppCtx *user, PetscInt *nmoved)
{
  const DMSwarmPID *pid;
  PetscInt          Np, n = 0;
  PetscMPIInt       rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pid));
  for (PetscInt p = 0; p < Np; ++p) n += pid[p] < user->pidStart[rank] || pid[p] >= user->pidStart[rank + 1];
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, NULL, NULL, (void **)&pid));
  PetscCallMPI(MPIU_Allreduce(&n, nmoved, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Report the count and time of the migration events logged in the stage, the maximum over all
  ranks as in -log_view. As in -log_view, event times are inclusive: the later events are parts of
  DMSwarmMigrate. The configuration is printed first so that the output of several runs can be told
  apart when collecting timings for plots.
*/
static PetscErrorCode ReportTimings(DM sw, AppCtx *user, PetscLogStage stage, PetscInt nmoved)
{
  const char        *events[] = {"DMSwarmMigrate", "DMPlexLocatePoints", "DMSwarmDETSetup", "DMSwarmDESendCnt", "DMSwarmDEPack",    "DMSwarmDExBegin",
                                 "DMSwarmDExEnd",  "DMSwarmAddPnts",     "DMSwarmRmvPnts",  "DMSwarmSort",      "DMSwarmCopyToGPU", "DMSwarmCopyFromGPU"};
  PetscEventPerfInfo info;
  DMSwarmBackendType backend;
  PetscLogDouble     time;
  PetscInt           N;
  PetscMPIInt        size;
  int                count;
  PetscBool          hash = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(DMSwarmGetBackend(sw, &backend));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_hash_location", &hash, NULL));
  PetscCall(DMSwarmGetSize(sw, &N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Migration timings in stage \"Swarm Migration\", maximum over %d ranks\n", size));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  backend %s, point location %s, %" PetscInt_FMT " particles, %" PetscInt_FMT " migrations\n", DMSwarmBackendTypeNames[backend], hash ? "grid hash" : "brute force", N, user->nsteps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %" PetscInt_FMT " particles change rank in each migration\n", nmoved));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %-20s %8s %14s %14s\n", "Event", "Count", "Time (s)", "Time/call (s)"));
  PetscCall(PetscLogStageGetPerfInfo(stage, &info));
  PetscCallMPI(MPIU_Allreduce(&info.time, &time, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %-20s %8" PetscInt_FMT " %14.6e %14.6e\n", "Stage total", user->nsteps, time, user->nsteps ? time / user->nsteps : 0.));
  for (size_t e = 0; e < PETSC_STATIC_ARRAY_LENGTH(events); ++e) {
    PetscLogEvent event;

    PetscCall(PetscLogEventGetId(events[e], &event));
    PetscCall(PetscLogEventGetPerfInfo(stage, event, &info));
    PetscCallMPI(MPIU_Allreduce(&info.count, &count, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&info.time, &time, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %-20s %8d %14.6e %14.6e\n", events[e], count, time, count ? time / count : 0.));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM            dm, sw;
  AppCtx        user;
  PetscLogStage stage;
  PetscInt      N, nmoved;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  /* The timings are read back from the default log handler, which -log_view would otherwise start */
  if (user.reportTimings) PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogStageRegister("Swarm Migration", &stage));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CreateSwarm(dm, &sw, &user));
  PetscCall(DMSwarmGetSize(sw, &N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " particles\n", N));
  PetscCall(PushAndMigrate(sw, &user, user.warmup, PETSC_FALSE, stage));
  PetscCall(PushAndMigrate(sw, &user, user.nsteps, PETSC_TRUE, stage));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Particle count preserved by %" PetscInt_FMT " warm-up and %" PetscInt_FMT " timed migrations\n", user.warmup, user.nsteps));
  if (user.check) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Particle data verified after every %" PetscInt_FMT " migrations\n", user.checkEvery));
  if (user.reportTimings) {
    /* after an even number of pushes the particles are back at their initial positions, so push away once more (untimed) */
    if (user.step % 2 == 0) PetscCall(PushAndMigrate(sw, &user, 1, PETSC_FALSE, stage));
    PetscCall(CountMovedParticles(sw, &user, &nmoved));
    PetscCall(ReportTimings(sw, &user, stage, nmoved));
  }
  PetscCall(PetscFree(user.pidStart));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not handle complex or quad
  build:
    requires: !complex double
  # Particle migration for the host and hip storage backends, each with brute force and grid hash
  # point location. Without -report_timings the output does not depend on the configuration, so
  # all four share one expected output; the particle count is checked after every migration.
  testset:
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 -petscpartitioner_type simple\
          -particle_order 2 -nsteps 4 -dt 0.1
    output_file: output/ex15_migrate.out
    test:
      suffix: migrate_scan
      args: -dm_plex_hash_location false
    test:
      suffix: migrate_hash
      args: -dm_plex_hash_location true
    test:
      suffix: migrate_backend_hip_scan
      requires: hip
      args: -dm_plex_hash_location false -dm_swarm_backend hip
    test:
      suffix: migrate_backend_hip_hash
      requires: hip
      args: -dm_plex_hash_location true -dm_swarm_backend hip
  # The timing report of each configuration. The times are masked when comparing outputs, so these
  # check the event counts: host and device point location are both logged as DMPlexLocatePoints,
  # and the hip backend moves no particle data between host and device (DMSwarmCopyToGPU).
  testset:
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 -petscpartitioner_type simple\
          -particle_order 2 -nsteps 4 -dt 0.1 -report_timings
    test:
      suffix: timings_scan
      args: -dm_plex_hash_location false
    test:
      suffix: timings_hash
      args: -dm_plex_hash_location true
    test:
      suffix: timings_backend_hip_scan
      requires: hip
      args: -dm_plex_hash_location false -dm_swarm_backend hip
    test:
      suffix: timings_backend_hip_hash
      requires: hip
      args: -dm_plex_hash_location true -dm_swarm_backend hip
  # The same migrations with the particle data verified after every one: every id once over all
  # ranks, coordinates and velocity consistent with the initial position carried in an extra field,
  # and the cell id confirmed by host point location
  testset:
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 -petscpartitioner_type simple\
          -particle_order 2 -nsteps 4 -dt 0.1 -dm_plex_hash_location true -check
    output_file: output/ex15_check.out
    test:
      suffix: check_hash
    test:
      suffix: check_backend_hip_hash
      requires: hip
      args: -dm_swarm_backend hip
  # Four ranks on a box elongated in z, split into one slab per rank, with the push capped so that
  # every particle moves at most 0.1 per coordinate and step: the fixed-load rank sweep setup, in
  # which each rank exchanges points with its slab neighbors only
  testset:
    nsize: 4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,16 -dm_plex_box_upper 1,1,4\
          -petscpartitioner_type simple -particle_order 1 -nsteps 4 -dt 0.2 -push_distance 0.5\
          -dm_plex_hash_location true
    output_file: output/ex15_rank_sweep_np4.out
    test:
      suffix: rank_sweep_np4
    test:
      suffix: rank_sweep_np4_backend_hip
      requires: hip
      args: -dm_swarm_backend hip
TEST*/
