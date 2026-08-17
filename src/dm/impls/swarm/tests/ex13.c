static char help[] = "Tests DMSwarm data bucket operations: field registration for PETSc datatypes, user structs and user datatypes, point add/copy/remove, resizing, and Vec views sharing field storage.\n\n";

#include <petscdm.h>
#include <petscdmswarm.h>

typedef struct {
  PetscInt  id;
  PetscReal w;
} Payload;

static const char *FieldNames[] = {"phi", "vel", "tag", "payload", "csum"};

static const char *TypeLabel(PetscDataType type)
{
  if (type == PETSC_REAL) return "real";
  if (type == PETSC_INT) return "int";
  if (type == PETSC_STRUCT) return "struct";
  return "user";
}

/* Fill points [pstart, pend) of every field with values derived from a per-point seed */
static PetscErrorCode FillRange(DM sw, PetscInt pstart, PetscInt pend, PetscInt base)
{
  Payload    *payload;
  PetscReal  *phi, *vel;
  PetscInt   *tag;
  PetscInt64 *csum;
  PetscInt    p, seed;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetField(sw, "phi", NULL, NULL, (void **)&phi));
  PetscCall(DMSwarmGetField(sw, "vel", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "tag", NULL, NULL, (void **)&tag));
  PetscCall(DMSwarmGetField(sw, "payload", NULL, NULL, (void **)&payload));
  PetscCall(DMSwarmGetField(sw, "csum", NULL, NULL, (void **)&csum));
  for (p = pstart; p < pend; ++p) {
    seed            = base + (p - pstart);
    phi[p]          = seed + 0.5;
    vel[2 * p + 0]  = seed + 0.1;
    vel[2 * p + 1]  = seed + 0.2;
    tag[p]          = -seed;
    payload[p].id   = 2 * seed + 1;
    payload[p].w    = seed / 4.0;
    csum[2 * p + 0] = 10 * seed;
    csum[2 * p + 1] = 10 * seed + 1;
  }
  PetscCall(DMSwarmRestoreField(sw, "phi", NULL, NULL, (void **)&phi));
  PetscCall(DMSwarmRestoreField(sw, "vel", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "tag", NULL, NULL, (void **)&tag));
  PetscCall(DMSwarmRestoreField(sw, "payload", NULL, NULL, (void **)&payload));
  PetscCall(DMSwarmRestoreField(sw, "csum", NULL, NULL, (void **)&csum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print every field of every local point so any storage corruption is visible in the output */
static PetscErrorCode PrintState(DM sw, const char *stage)
{
  MPI_Comm    comm;
  Payload    *payload;
  PetscReal  *phi, *vel;
  PetscInt   *tag;
  PetscInt64 *csum;
  PetscInt    p, nlocal, nglobal;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMSwarmGetLocalSize(sw, &nlocal));
  PetscCall(DMSwarmGetSize(sw, &nglobal));
  PetscCall(PetscPrintf(comm, "%s: global size %" PetscInt_FMT "\n", stage, nglobal));
  PetscCall(DMSwarmGetField(sw, "phi", NULL, NULL, (void **)&phi));
  PetscCall(DMSwarmGetField(sw, "vel", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "tag", NULL, NULL, (void **)&tag));
  PetscCall(DMSwarmGetField(sw, "payload", NULL, NULL, (void **)&payload));
  PetscCall(DMSwarmGetField(sw, "csum", NULL, NULL, (void **)&csum));
  PetscCall(PetscSynchronizedPrintf(comm, "  [%d] nlocal %" PetscInt_FMT "\n", rank, nlocal));
  for (p = 0; p < nlocal; ++p) {
    PetscCall(PetscSynchronizedPrintf(comm, "  [%d] %2" PetscInt_FMT ": phi %7.2f vel (%7.2f, %7.2f) tag %5" PetscInt_FMT " payload (%4" PetscInt_FMT ", %6.2f) csum (%5" PetscInt64_FMT ", %5" PetscInt64_FMT ")\n", rank, p, (double)phi[p], (double)vel[2 * p + 0], (double)vel[2 * p + 1], tag[p],
                                      payload[p].id, (double)payload[p].w, csum[2 * p + 0], csum[2 * p + 1]));
  }
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  PetscCall(DMSwarmRestoreField(sw, "phi", NULL, NULL, (void **)&phi));
  PetscCall(DMSwarmRestoreField(sw, "vel", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "tag", NULL, NULL, (void **)&tag));
  PetscCall(DMSwarmRestoreField(sw, "payload", NULL, NULL, (void **)&payload));
  PetscCall(DMSwarmRestoreField(sw, "csum", NULL, NULL, (void **)&csum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM            sw;
  Vec           v;
  MPI_Comm      comm;
  PetscReal     nrm;
  PetscDataType type;
  PetscInt      nlocal, bs, f;
  PetscMPIInt   rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMCreate(comm, &sw));
  PetscCall(DMSetType(sw, DMSWARM));
  PetscCall(DMSetFromOptions(sw));
  PetscCall(PetscObjectSetName((PetscObject)sw, "Particles"));
  PetscCall(DMSwarmInitializeFieldRegister(sw));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "phi", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "vel", 2, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "tag", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterUserStructField(sw, "payload", sizeof(Payload)));
  PetscCall(DMSwarmRegisterUserDatatypeField(sw, "csum", sizeof(PetscInt64), 2));
  PetscCall(DMSwarmFinalizeFieldRegister(sw));
  PetscCall(DMSwarmSetLocalSizes(sw, 3 + rank, 4));

  for (f = 0; f < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(FieldNames); ++f) {
    PetscCall(DMSwarmGetFieldInfo(sw, FieldNames[f], &bs, &type));
    PetscCall(PetscPrintf(comm, "field %-7s blocksize %" PetscInt_FMT " type %s\n", FieldNames[f], bs, TypeLabel(type)));
  }

  PetscCall(DMSwarmGetLocalSize(sw, &nlocal));
  PetscCall(FillRange(sw, 0, nlocal, rank * 100));
  PetscCall(PrintState(sw, "After initial fill"));

  /* New points must arrive zero-initialized */
  PetscCall(DMSwarmAddPoint(sw));
  PetscCall(DMSwarmAddNPoints(sw, 2));
  PetscCall(PrintState(sw, "After DMSwarmAddPoint() + DMSwarmAddNPoints(2)"));
  PetscCall(FillRange(sw, nlocal, nlocal + 3, rank * 100 + 50));
  PetscCall(PrintState(sw, "After filling the added points"));

  /* DMSwarmRemovePointAtIndex() moves the last point into the removed slot */
  PetscCall(DMSwarmGetLocalSize(sw, &nlocal));
  PetscCall(DMSwarmCopyPoint(sw, 0, nlocal - 1));
  PetscCall(DMSwarmRemovePointAtIndex(sw, 1));
  PetscCall(DMSwarmRemovePoint(sw));
  PetscCall(PrintState(sw, "After copy 0 -> last, remove index 1, remove last"));

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "phi", &v));
  PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecScale(v, 2.0));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "phi", &v));
  PetscCall(PrintState(sw, "After scaling phi by 2 through the global Vec"));

  PetscCall(DMSwarmCreateLocalVectorFromField(sw, "vel", &v));
  PetscCall(VecNorm(v, NORM_1, &nrm));
  PetscCall(PetscSynchronizedPrintf(comm, "  [%d] ||vel||_1 %.2f\n", rank, (double)nrm));
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  PetscCall(VecScale(v, 0.5));
  PetscCall(DMSwarmDestroyLocalVectorFromField(sw, "vel", &v));
  PetscCall(PrintState(sw, "After scaling vel by 0.5 through the local Vec"));

  /* Shrinking must preserve the surviving points */
  PetscCall(DMSwarmSetLocalSizes(sw, 2, 4));
  PetscCall(PrintState(sw, "After shrinking to 2 local points"));

  PetscCall(DMDestroy(&sw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: double !complex

  test:
    suffix: 1

  # The host backend must be a no-op: identical output to the default
  test:
    suffix: 1_host
    args: -dm_swarm_backend host
    output_file: output/ex13_1.out

  test:
    suffix: 2
    nsize: 2

TEST*/
