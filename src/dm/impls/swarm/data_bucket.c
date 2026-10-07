#include "../src/dm/impls/swarm/data_bucket.h"

/* string helpers */
PetscErrorCode DMSwarmDataFieldStringInList(const char name[], const PetscInt N, const DMSwarmDataField gfield[], PetscBool *val)
{
  PetscFunctionBegin;
  *val = PETSC_FALSE;
  for (PetscInt i = 0; i < N; ++i) {
    PetscBool flg;
    PetscCall(PetscStrcmp(name, gfield[i]->name, &flg));
    if (flg) {
      *val = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldStringFindInList(const char name[], const PetscInt N, const DMSwarmDataField gfield[], PetscInt *index)
{
  PetscFunctionBegin;
  *index = -1;
  for (PetscInt i = 0; i < N; ++i) {
    PetscBool flg;
    PetscCall(PetscStrcmp(name, gfield[i]->name, &flg));
    if (flg) {
      *index = i;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldCreate(const char registration_function[], const char name[], const size_t size, const PetscInt L, DMSwarmDataField *DF)
{
  DMSwarmDataField df;

  PetscFunctionBegin;
  PetscCall(PetscNew(&df));
  PetscCall(PetscStrallocpy(registration_function, &df->registration_function));
  PetscCall(PetscStrallocpy(name, &df->name));
  df->atomic_size = size;
  df->L           = L;
  df->bs          = 1;
  /* allocate something so we don't have to reallocate */
  PetscCall(PetscMalloc(size * L, &df->data));
  PetscCall(PetscMemzero(df->data, size * L));
  df->data_d           = NULL;
  df->data_allocated_d = NULL;
  df->capacity_d       = 0;
  df->offloadmask      = PETSC_OFFLOAD_CPU;
  df->bucket           = NULL;
  *DF                  = df;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldDestroy(DMSwarmDataField *DF)
{
  DMSwarmDataField df = *DF;

  PetscFunctionBegin;
  /* device mirrors are released through the bucket-level ops.fieldfreedevice dispatch, since freeing them requires the device backend */
  PetscCheck(!df->data_allocated_d, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Device mirror of field \"%s\" was not released; dispatch ops.fieldfreedevice before DMSwarmDataFieldDestroy()", df->name);
  PetscCall(PetscFree(df->registration_function));
  PetscCall(PetscFree(df->name));
  PetscCall(PetscFree(df->data));
  PetscCall(PetscFree(df));
  *DF = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Host storage backend: device transfers are no-ops since the host array is the single residence */
static PetscErrorCode DMSwarmDataFieldToDevice_Host(DMSwarmDataField df)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmDataFieldToHost_Host(DMSwarmDataField df)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmDataFieldFreeDevice_Host(DMSwarmDataField df)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmDataFieldAllocDevice_Host(DMSwarmDataField df)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Host wire-record pack/unpack: thin loops over the per-point packed-array helpers migration uses
  today, so the record layout and the insertion semantics are identical by construction. Records
  are packed in the order of list[] and unpacked to ascending destination indices.
*/
static PetscErrorCode DMSwarmDataBucketPackRecords_Host(DMSwarmDataBucket db, PetscInt npoints, const PetscInt list[], void *buf)
{
  size_t record_size = 0;

  PetscFunctionBegin;
  for (PetscInt f = 0; f < db->nfields; ++f) record_size += db->field[f]->atomic_size;
  for (PetscInt p = 0; p < npoints; ++p) PetscCall(DMSwarmDataBucketFillPackedArray(db, list[p], (void *)((char *)buf + (size_t)p * record_size)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmDataBucketUnpackRecords_Host(DMSwarmDataBucket db, PetscInt start, PetscInt npoints, const void *buf)
{
  size_t record_size = 0;

  PetscFunctionBegin;
  for (PetscInt f = 0; f < db->nfields; ++f) record_size += db->field[f]->atomic_size;
  for (PetscInt p = 0; p < npoints; ++p) PetscCall(DMSwarmDataBucketInsertPackedArray(db, start + p, (void *)((char *)buf + (size_t)p * record_size)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Host compaction: a thin wrapper around the historical migration removal loop. The loop scans
  the slots in ascending order and, on a sent slot, swaps the current last point in and re-checks
  the same slot, so the sent status must be swapped alongside; the mask array mirrors exactly
  what the removal loops in swarm_migrate.c read from the swapped rank/cellid field entries.
  The resulting survivor order and the size bookkeeping (one DMSwarmDataBucketRemovePoint()
  shrink per removed point) are bit-identical to those loops by construction.
*/
static PetscErrorCode DMSwarmDataBucketCompact_Host(DMSwarmDataBucket db, PetscInt nsent, const PetscInt sentlist[])
{
  PetscBool *mask;
  PetscInt   npoints = db->L;

  PetscFunctionBegin;
  if (!nsent) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCalloc1(npoints, &mask));
  for (PetscInt i = 0; i < nsent; ++i) mask[sentlist[i]] = PETSC_TRUE;
  for (PetscInt p = 0; p < npoints; ++p) {
    if (mask[p]) {
      PetscCall(DMSwarmDataBucketRemovePointAtIndex(db, p));
      npoints--;
      mask[p] = mask[npoints]; /* the swapped-in point's sent status; re-check this slot */
      p--;
    }
  }
  PetscCall(PetscFree(mask));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Make the host array of df valid before its bytes are read or written on the host. Only acts
  when the device copy is the sole valid residence (PETSC_OFFLOAD_GPU); the D2H transfer is
  dispatched through the owning bucket's ops so this file stays device-agnostic. A field
  created outside a bucket has no back-pointer and is always host valid.
*/
static PetscErrorCode DMSwarmDataFieldSyncToHost_Private(const DMSwarmDataField df)
{
  PetscFunctionBegin;
  if (df->bucket && df->offloadmask == PETSC_OFFLOAD_GPU) PetscCall((*df->bucket->ops.fieldtohost)(df));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* data bucket */
PetscErrorCode DMSwarmDataBucketCreate(DMSwarmDataBucket *DB)
{
  DMSwarmDataBucket db;

  PetscFunctionBegin;
  PetscCall(PetscNew(&db));

  db->finalised = PETSC_FALSE;
  /* create empty spaces for fields */
  db->L         = -1;
  db->buffer    = 1;
  db->allocated = 1;
  db->nfields   = 0;
  PetscCall(PetscMalloc1(1, &db->field));
  db->memtype              = PETSC_MEMTYPE_HOST;
  db->ops.fieldalloc       = DMSwarmDataFieldSetSize;
  db->ops.fieldallocdevice = DMSwarmDataFieldAllocDevice_Host;
  db->ops.fieldtodevice    = DMSwarmDataFieldToDevice_Host;
  db->ops.fieldtohost      = DMSwarmDataFieldToHost_Host;
  db->ops.fieldzerorange   = DMSwarmDataFieldZeroBlock;
  db->ops.fieldfreedevice  = DMSwarmDataFieldFreeDevice_Host;
  db->ops.setsizes         = DMSwarmDataBucketSetSizes_Host;
  db->ops.packrecords      = DMSwarmDataBucketPackRecords_Host;
  db->ops.unpackrecords    = DMSwarmDataBucketUnpackRecords_Host;
  db->ops.compact          = DMSwarmDataBucketCompact_Host;
  *DB                      = db;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketDestroy(DMSwarmDataBucket *DB)
{
  DMSwarmDataBucket db = *DB;
  PetscInt          f;

  PetscFunctionBegin;
  /* release fields; device mirrors first, since only the backend ops can free them */
  for (f = 0; f < db->nfields; ++f) {
    PetscCall((*db->ops.fieldfreedevice)(db->field[f]));
    PetscCall(DMSwarmDataFieldDestroy(&db->field[f]));
  }
  /* this will catch the initially allocated objects in the event that no fields are registered */
  if (db->field != NULL) PetscCall(PetscFree(db->field));
  PetscCall(PetscFree(db));
  *DB = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketQueryForActiveFields(DMSwarmDataBucket db, PetscBool *any_active_fields)
{
  PetscInt f;

  PetscFunctionBegin;
  *any_active_fields = PETSC_FALSE;
  for (f = 0; f < db->nfields; ++f) {
    if (db->field[f]->active) {
      *any_active_fields = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketRegisterField(DMSwarmDataBucket db, const char registration_function[], const char field_name[], size_t atomic_size, DMSwarmDataField *_gfield)
{
  PetscBool        val;
  DMSwarmDataField fp;

  PetscFunctionBegin;
  /* check we haven't finalised the registration of fields */
  /*
   if (db->finalised==PETSC_TRUE) {
   printf("ERROR: DMSwarmDataBucketFinalize() has been called. Cannot register more fields\n");
   ERROR();
   }
  */
  /* check for repeated name */
  PetscCall(DMSwarmDataFieldStringInList(field_name, db->nfields, (const DMSwarmDataField *)db->field, &val));
  PetscCheck(val != PETSC_TRUE, PETSC_COMM_SELF, PETSC_ERR_USER, "Field %s already exists. Cannot add same field twice", field_name);
  /* create new space for data */
  PetscCall(PetscRealloc(sizeof(DMSwarmDataField) * (db->nfields + 1), &db->field));
  /* add field */
  PetscCall(DMSwarmDataFieldCreate(registration_function, field_name, atomic_size, db->allocated, &fp));
  fp->bucket             = db;
  db->field[db->nfields] = fp;
  db->nfields++;
  if (_gfield != NULL) *_gfield = fp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 #define DMSwarmDataBucketRegisterField(db,name,size,k) {\
 char *location;\
 asprintf(&location,"Registered by %s() at line %d within file %s", __FUNCTION__, __LINE__, __FILE__);\
 _DMSwarmDataBucketRegisterField( (db), location, (name), (size), (k));\
 ierr = PetscFree(location);\
 }
 */

/*@
  DMSwarmDataBucketGetDMSwarmDataFieldIdByName - Return the index of a `DMSwarmDataField` within a `DMSwarmDataBucket` given its name.

  Not Collective

  Input Parameters:
+ db   - the `DMSwarmDataBucket`
- name - the field name

  Output Parameter:
. idx - the index of the field within the bucket

  Level: developer

  Note:
  Errors if no field with the given name is registered; use `DMSwarmDataBucketQueryDMSwarmDataFieldByName()` first when the name may be absent.

.seealso: `DMSwarmDataBucket`, `DMSwarmDataField`, `DMSwarmDataBucketGetDMSwarmDataFieldByName()`, `DMSwarmDataBucketQueryDMSwarmDataFieldByName()`
@*/
PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldIdByName(DMSwarmDataBucket db, const char name[], PetscInt *idx)
{
  PetscBool found;

  PetscFunctionBegin;
  *idx = -1;
  PetscCall(DMSwarmDataFieldStringInList(name, db->nfields, (const DMSwarmDataField *)db->field, &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot find DMSwarmDataField with name %s", name);
  PetscCall(DMSwarmDataFieldStringFindInList(name, db->nfields, (const DMSwarmDataField *)db->field, idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmDataBucketGetDMSwarmDataFieldByName - Return the `DMSwarmDataField` handle registered in a `DMSwarmDataBucket` under a given name.

  Not Collective

  Input Parameters:
+ db   - the `DMSwarmDataBucket`
- name - the field name

  Output Parameter:
. gfield - the `DMSwarmDataField` handle

  Level: developer

  Note:
  Errors if no field with the given name is registered; use `DMSwarmDataBucketQueryDMSwarmDataFieldByName()` first when the name may be absent.

.seealso: `DMSwarmDataBucket`, `DMSwarmDataField`, `DMSwarmDataBucketGetDMSwarmDataFieldIdByName()`, `DMSwarmDataBucketQueryDMSwarmDataFieldByName()`
@*/
PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldByName(DMSwarmDataBucket db, const char name[], DMSwarmDataField *gfield)
{
  PetscInt  idx;
  PetscBool found;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataFieldStringInList(name, db->nfields, (const DMSwarmDataField *)db->field, &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot find DMSwarmDataField with name %s", name);
  PetscCall(DMSwarmDataFieldStringFindInList(name, db->nfields, (const DMSwarmDataField *)db->field, &idx));
  *gfield = db->field[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmDataBucketQueryDMSwarmDataFieldByName - Test whether a `DMSwarmDataBucket` contains a `DMSwarmDataField` with the given name.

  Not Collective

  Input Parameters:
+ db   - the `DMSwarmDataBucket`
- name - the field name to look up

  Output Parameter:
. found - `PETSC_TRUE` if a field with the given name is registered, otherwise `PETSC_FALSE`

  Level: developer

.seealso: `DMSwarmDataBucket`, `DMSwarmDataField`, `DMSwarmDataBucketGetDMSwarmDataFieldByName()`, `DMSwarmDataBucketGetDMSwarmDataFieldIdByName()`
@*/
PetscErrorCode DMSwarmDataBucketQueryDMSwarmDataFieldByName(DMSwarmDataBucket db, const char name[], PetscBool *found)
{
  PetscFunctionBegin;
  *found = PETSC_FALSE;
  PetscCall(DMSwarmDataFieldStringInList(name, db->nfields, (const DMSwarmDataField *)db->field, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketFinalize(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  db->finalised = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldGetNumEntries(DMSwarmDataField df, PetscInt *sum)
{
  PetscFunctionBegin;
  *sum = df->L;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldSetBlockSize(DMSwarmDataField df, PetscInt blocksize)
{
  PetscFunctionBegin;
  df->bs = blocksize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldSetSize(DMSwarmDataField df, const PetscInt new_L)
{
  PetscFunctionBegin;
  PetscCheck(new_L >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot set size of DMSwarmDataField to be < 0");
  if (new_L == df->L) PetscFunctionReturn(PETSC_SUCCESS);
  if (new_L > df->L) {
    PetscCall(PetscRealloc(df->atomic_size * (new_L), &df->data));
    /* init new contents */
    PetscCall(PetscMemzero(((char *)df->data) + df->L * df->atomic_size, (new_L - df->L) * df->atomic_size));
  } else {
    /* reallocate pointer list, add +1 in case new_L = 0 */
    PetscCall(PetscRealloc(df->atomic_size * (new_L + 1), &df->data));
  }
  df->L = new_L;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldZeroBlock(DMSwarmDataField df, const PetscInt start, const PetscInt end)
{
  PetscFunctionBegin;
  PetscCheck(start <= end, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot zero a block of entries if start(%" PetscInt_FMT ") > end(%" PetscInt_FMT ")", start, end);
  PetscCheck(start >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot zero a block of entries if start(%" PetscInt_FMT ") < 0", start);
  PetscCheck(end <= df->L, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot zero a block of entries if end(%" PetscInt_FMT ") >= array size(%" PetscInt_FMT ")", end, df->L);
  PetscCall(PetscMemzero(((char *)df->data) + start * df->atomic_size, (end - start) * df->atomic_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketSetSizes_Host(DMSwarmDataBucket db, PetscInt L, PetscInt buffer)
{
  PetscInt current_allocated, current_used, new_used, new_unused, new_buffer, new_allocated, f, end;

  PetscFunctionBegin;
  current_allocated = db->allocated;
  current_used      = PetscMax(db->L, 0);
  new_used          = L;
  new_unused        = current_allocated - new_used;
  new_buffer        = db->buffer;
  if (buffer >= 0) { /* update the buffer value */
    new_buffer = buffer;
  }
  new_allocated = new_used + new_buffer;
  /* action */
  if (new_allocated > current_allocated) {
    /* increase size to new_used + new_buffer and zero new space */
    for (f = 0; f < db->nfields; f++) {
      PetscCall((*db->ops.fieldalloc)(db->field[f], new_allocated));
      PetscCall((*db->ops.fieldzerorange)(db->field[f], current_allocated, new_allocated));
    }
    db->L         = new_used;
    db->buffer    = new_buffer;
    db->allocated = new_used + new_buffer;
  } else {
    if (new_unused > 2 * new_buffer) {
      /* shrink array to new_used + new_buffer */
      for (f = 0; f < db->nfields; ++f) PetscCall((*db->ops.fieldalloc)(db->field[f], new_allocated));
      db->L         = new_used;
      db->buffer    = new_buffer;
      db->allocated = new_used + new_buffer;
    } else {
      db->L      = new_used;
      db->buffer = new_buffer;
    }
  }
  /* if we shrunk, zero old entries from new_used to current_used or end of array */
  end = PetscMin(current_used, new_allocated);
  if (end > new_used) {
    for (f = 0; f < db->nfields; ++f) PetscCall((*db->ops.fieldzerorange)(db->field[f], new_used, end));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 A negative buffer value will simply be ignored and the old buffer value will be used.
 */
PetscErrorCode DMSwarmDataBucketSetSizes(DMSwarmDataBucket db, const PetscInt L, const PetscInt buffer)
{
  PetscBool any_active_fields;

  PetscFunctionBegin;
  PetscCheck(db->finalised != PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_USER, "You must call DMSwarmDataBucketFinalize() before DMSwarmDataBucketSetSizes()");
  PetscCall(DMSwarmDataBucketQueryForActiveFields(db, &any_active_fields));
  PetscCheck(!any_active_fields, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot safely re-size as at least one DMSwarmDataField is currently being accessed");
  /*
    No sync to host is needed for device-authoritative fields: the sizing logic dispatches
    ops.fieldalloc/ops.fieldzerorange, whose device implementations resize the mirror by
    copying its surviving entries and zero both residences over the same ranges, so a
    PETSC_OFFLOAD_GPU field stays valid on the device while the host realloc only moves
    bytes that were stale to begin with.
  */
  PetscCall((*db->ops.setsizes)(db, L, buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketSetInitialSizes(DMSwarmDataBucket db, const PetscInt L, const PetscInt buffer)
{
  PetscInt f;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketSetSizes(db, L, buffer));
  /* dispatch through the bucket ops so any device mirror is zeroed alongside the host array */
  for (f = 0; f < db->nfields; ++f) PetscCall((*db->ops.fieldzerorange)(db->field[f], 0, db->allocated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketGetSizes(DMSwarmDataBucket db, PetscInt *L, PetscInt *buffer, PetscInt *allocated)
{
  PetscFunctionBegin;
  if (L) *L = db->L;
  if (buffer) *buffer = db->buffer;
  if (allocated) *allocated = db->allocated;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketGetGlobalSizes(MPI_Comm comm, DMSwarmDataBucket db, PetscInt *L, PetscInt *buffer, PetscInt *allocated)
{
  PetscFunctionBegin;
  if (L) PetscCallMPI(MPIU_Allreduce(&db->L, L, 1, MPIU_INT, MPI_SUM, comm));
  if (buffer) PetscCallMPI(MPIU_Allreduce(&db->buffer, buffer, 1, MPIU_INT, MPI_SUM, comm));
  if (allocated) PetscCallMPI(MPIU_Allreduce(&db->allocated, allocated, 1, MPIU_INT, MPI_SUM, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFields(DMSwarmDataBucket db, PetscInt *L, DMSwarmDataField *fields[])
{
  PetscFunctionBegin;
  if (L) *L = db->nfields;
  if (fields) *fields = db->field;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Take the exclusive access lock of a field without any coherence action; the device accessors in swarm.c pair this with a residence update dispatched through the bucket ops */
PetscErrorCode DMSwarmDataFieldAcquireAccess(const DMSwarmDataField gfield)
{
  PetscFunctionBegin;
  PetscCheck(!gfield->active, PETSC_COMM_SELF, PETSC_ERR_USER, "Field \"%s\" is already active. You must call DMSwarmDataFieldRestoreAccess()", gfield->name);
  gfield->active = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldReleaseAccess(DMSwarmDataField gfield)
{
  PetscFunctionBegin;
  PetscCheck(gfield->active != PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_USER, "Field \"%s\" is not active. You must call DMSwarmDataFieldGetAccess()", gfield->name);
  gfield->active = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldGetAccess(const DMSwarmDataField gfield)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmDataFieldAcquireAccess(gfield));
  /* access through this lock is host read-write, so the host bytes must be valid */
  PetscCall(DMSwarmDataFieldSyncToHost_Private(gfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldAccessPoint(const DMSwarmDataField gfield, const PetscInt pid, void **ctx_p)
{
  PetscFunctionBegin;
  *ctx_p = NULL;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* debug mode */
  /* check point is valid */
  PetscCheck(pid >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be >= 0");
  PetscCheck(pid < gfield->L, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be < %" PetscInt_FMT, gfield->L);
  PetscCheck(gfield->active != PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_USER, "Field \"%s\" is not active. You must call DMSwarmDataFieldGetAccess() before point data can be retrivied", gfield->name);
#endif
  *ctx_p = DMSWARM_DATAFIELD_point_access(gfield->data, pid, gfield->atomic_size);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldAccessPointOffset(const DMSwarmDataField gfield, const size_t offset, const PetscInt pid, void **ctx_p)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* debug mode */
  /* check point is valid */
  /* PetscCheck(offset >= 0,PETSC_COMM_SELF,PETSC_ERR_USER,"offset must be >= 0");*/
  /* Note compiler realizes this can never happen with an unsigned PetscInt */
  PetscCheck(offset < gfield->atomic_size, PETSC_COMM_SELF, PETSC_ERR_USER, "offset must be < %zu", gfield->atomic_size);
  /* check point is valid */
  PetscCheck(pid >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be >= 0");
  PetscCheck(pid < gfield->L, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be < %" PetscInt_FMT, gfield->L);
  PetscCheck(gfield->active != PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_USER, "Field \"%s\" is not active. You must call DMSwarmDataFieldGetAccess() before point data can be retrivied", gfield->name);
#endif
  *ctx_p = DMSWARM_DATAFIELD_point_access_offset(gfield->data, pid, gfield->atomic_size, offset);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldRestoreAccess(DMSwarmDataField gfield)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmDataFieldReleaseAccess(gfield));
  /* the caller may have written the host array, so any device copy is now stale */
  gfield->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldVerifyAccess(const DMSwarmDataField gfield, const size_t size)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  PetscCheck(gfield->atomic_size == size, PETSC_COMM_SELF, PETSC_ERR_USER, "Field \"%s\" must be mapped to %zu bytes, your intended structure is %zu bytes in length.", gfield->name, gfield->atomic_size, size);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataFieldGetAtomicSize(const DMSwarmDataField gfield, size_t *size)
{
  PetscFunctionBegin;
  if (size) *size = gfield->atomic_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmDataFieldGetEntries - Return a pointer to the raw contiguous storage backing a `DMSwarmDataField`.

  Not Collective

  Input Parameter:
. gfield - the `DMSwarmDataField`

  Output Parameter:
. data - pointer to the raw entries; must be released with `DMSwarmDataFieldRestoreEntries()`

  Level: developer

  Note:
  The array has `L * bs` entries of `atomic_size` bytes, where `L` is the number of particles and `bs` is the block size of the field. Access is host read-write; stale host entries are refreshed from the device first, and `DMSwarmDataFieldRestoreEntries()` marks the host copy authoritative.

.seealso: `DMSwarmDataField`, `DMSwarmDataFieldRestoreEntries()`, `DMSwarmDataFieldGetNumEntries()`, `DMSwarmDataFieldGetAtomicSize()`
@*/
PetscErrorCode DMSwarmDataFieldGetEntries(const DMSwarmDataField gfield, void **data)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmDataFieldSyncToHost_Private(gfield));
  if (data) *data = gfield->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmDataFieldRestoreEntries - Release a pointer obtained from `DMSwarmDataFieldGetEntries()`, clearing the caller's handle to `NULL`.

  Not Collective

  Input Parameter:
. gfield - the `DMSwarmDataField`

  Output Parameter:
. data - pointer that will be set to `NULL`

  Level: developer

  Note:
  Since the access is host read-write, the host copy of the entries is marked authoritative and any device copy becomes stale.

.seealso: `DMSwarmDataField`, `DMSwarmDataFieldGetEntries()`
@*/
PetscErrorCode DMSwarmDataFieldRestoreEntries(const DMSwarmDataField gfield, void **data)
{
  PetscFunctionBegin;
  /* the caller may have written the host array, so any device copy is now stale */
  gfield->offloadmask = PETSC_OFFLOAD_CPU;
  if (data) *data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = x */
PetscErrorCode DMSwarmDataBucketCopyPoint(const DMSwarmDataBucket xb, const PetscInt pid_x, const DMSwarmDataBucket yb, const PetscInt pid_y)
{
  PetscInt f;

  PetscFunctionBegin;
  for (f = 0; f < xb->nfields; ++f) {
    void *dest;
    void *src;

    PetscCall(DMSwarmDataFieldGetAccess(xb->field[f]));
    if (xb != yb) PetscCall(DMSwarmDataFieldGetAccess(yb->field[f]));
    PetscCall(DMSwarmDataFieldAccessPoint(xb->field[f], pid_x, &src));
    PetscCall(DMSwarmDataFieldAccessPoint(yb->field[f], pid_y, &dest));
    PetscCall(PetscMemcpy(dest, src, xb->field[f]->atomic_size));
    PetscCall(DMSwarmDataFieldRestoreAccess(xb->field[f]));
    if (xb != yb) PetscCall(DMSwarmDataFieldRestoreAccess(yb->field[f]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketCreateFromSubset(DMSwarmDataBucket DBIn, const PetscInt N, const PetscInt list[], DMSwarmDataBucket *DB)
{
  PetscInt          nfields;
  DMSwarmDataField *fields;
  PetscInt          f, L, buffer, allocated, p;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketCreate(DB));
  /* copy contents of DBIn */
  PetscCall(DMSwarmDataBucketGetDMSwarmDataFields(DBIn, &nfields, &fields));
  PetscCall(DMSwarmDataBucketGetSizes(DBIn, &L, &buffer, &allocated));
  for (f = 0; f < nfields; ++f) PetscCall(DMSwarmDataBucketRegisterField(*DB, "DMSwarmDataBucketCreateFromSubset", fields[f]->name, fields[f]->atomic_size, NULL));
  PetscCall(DMSwarmDataBucketFinalize(*DB));
  PetscCall(DMSwarmDataBucketSetSizes(*DB, L, buffer));
  for (f = 0; f < nfields; ++f) {
    DMSwarmDataField gfield;

    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(*DB, fields[f]->name, &gfield));
    PetscCall(DMSwarmDataFieldSetBlockSize(gfield, fields[f]->bs));
    gfield->petsc_type = fields[f]->petsc_type;
  }
  /* now copy the desired guys from DBIn => DB */
  for (p = 0; p < N; ++p) PetscCall(DMSwarmDataBucketCopyPoint(DBIn, list[p], *DB, list[p]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* insert into an existing location */
PetscErrorCode DMSwarmDataFieldInsertPoint(const DMSwarmDataField field, const PetscInt index, const void *data)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be >= 0");
  PetscCheck(index < field->L, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be < %" PetscInt_FMT, field->L);
#endif
  /* a single point is written, so the untouched host entries must be valid before the host copy is marked authoritative */
  PetscCall(DMSwarmDataFieldSyncToHost_Private(field));
  PetscCall(PetscMemcpy(DMSWARM_DATAFIELD_point_access(field->data, index, field->atomic_size), data, field->atomic_size));
  field->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* remove data at index - replace with last point */
PetscErrorCode DMSwarmDataBucketRemovePointAtIndex(const DMSwarmDataBucket db, const PetscInt index)
{
  PetscInt  f;
  PetscBool any_active_fields;

  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be >= 0");
  PetscCheck(index < db->allocated, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be < %" PetscInt_FMT, db->L + db->buffer);
#endif
  PetscCall(DMSwarmDataBucketQueryForActiveFields(db, &any_active_fields));
  PetscCheck(!any_active_fields, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot safely remove point as at least one DMSwarmDataField is currently being accessed");
  if (index >= db->L) { /* this point is not in the list - no need to error, but I will anyway */
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "You should not be trying to remove point at index=%" PetscInt_FMT " since it's < db->L = %" PetscInt_FMT, index, db->L);
  }
  if (index != db->L - 1) { /* not last point in list */
    for (f = 0; f < db->nfields; ++f) {
      DMSwarmDataField field = db->field[f];

      /* copy then remove */
      PetscCall(DMSwarmDataFieldCopyPoint(db->L - 1, field, index, field));
      /* DMSwarmDataFieldZeroPoint(field,index); */
    }
  }
  /* decrement size */
  /* this will zero out an crap at the end of the list */
  PetscCall(DMSwarmDataBucketRemovePoint(db));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Remove the nsent points listed in sentlist (strictly ascending indices in [0, L)) at once.
  Every backend reproduces the survivor order of the historical ascending removal loop
  (swap-with-last and re-check, see DMSwarmDataBucketCompact_Host()) exactly: with
  m = L - nsent, survivors with original index < m keep their slots, the sent slots < m taken
  ascending are filled by the survivors with original index >= m taken in descending original
  order, and points with original index >= m vanish. For a device backend sentlist is a device
  pointer, so the ascending-order check can only be applied to a host-resident list here.
*/
PetscErrorCode DMSwarmDataBucketCompact(DMSwarmDataBucket db, PetscInt nsent, const PetscInt sentlist[])
{
  PetscBool any_active_fields;

  PetscFunctionBegin;
  PetscCheck(db->finalised != PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_USER, "You must call DMSwarmDataBucketFinalize() before DMSwarmDataBucketCompact()");
  PetscCall(DMSwarmDataBucketQueryForActiveFields(db, &any_active_fields));
  PetscCheck(!any_active_fields, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot safely compact as at least one DMSwarmDataField is currently being accessed");
  PetscCheck(nsent >= 0 && nsent <= db->L, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of points to remove %" PetscInt_FMT " must lie in [0, %" PetscInt_FMT "]", nsent, db->L);
  if (PetscMemTypeHost(db->memtype)) {
    for (PetscInt i = 0; i < nsent; ++i) {
      PetscCheck(sentlist[i] >= 0 && sentlist[i] < db->L, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "sentlist[%" PetscInt_FMT "] = %" PetscInt_FMT " must lie in [0, %" PetscInt_FMT ")", i, sentlist[i], db->L);
      PetscCheck(i == 0 || sentlist[i] > sentlist[i - 1], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "sentlist must be strictly ascending, sentlist[%" PetscInt_FMT "] = %" PetscInt_FMT " follows %" PetscInt_FMT, i, sentlist[i], sentlist[i - 1]);
    }
  }
  PetscCall((*db->ops.compact)(db, nsent, sentlist));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* copy x into y */
PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt pid_x, const DMSwarmDataField field_x, const PetscInt pid_y, const DMSwarmDataField field_y)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheck(pid_x >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "(IN) index must be >= 0");
  PetscCheck(pid_x < field_x->L, PETSC_COMM_SELF, PETSC_ERR_USER, "(IN) index must be < %" PetscInt_FMT, field_x->L);
  PetscCheck(pid_y >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "(OUT) index must be >= 0");
  PetscCheck(pid_y < field_y->L, PETSC_COMM_SELF, PETSC_ERR_USER, "(OUT) index must be < %" PetscInt_FMT, field_y->L);
  PetscCheck(field_y->atomic_size == field_x->atomic_size, PETSC_COMM_SELF, PETSC_ERR_USER, "atomic size must match");
#endif
  PetscCall(DMSwarmDataFieldSyncToHost_Private(field_x));
  /* a single point of y is written, so its untouched host entries must be valid before the host copy is marked authoritative */
  if (field_y != field_x) PetscCall(DMSwarmDataFieldSyncToHost_Private(field_y));
  PetscCall(PetscMemcpy(DMSWARM_DATAFIELD_point_access(field_y->data, pid_y, field_y->atomic_size), DMSWARM_DATAFIELD_point_access(field_x->data, pid_x, field_x->atomic_size), field_y->atomic_size));
  field_y->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* zero only the datafield at this point */
PetscErrorCode DMSwarmDataFieldZeroPoint(const DMSwarmDataField field, const PetscInt index)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be >= 0");
  PetscCheck(index < field->L, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be < %" PetscInt_FMT, field->L);
#endif
  /* a single point is zeroed, so the untouched host entries must be valid before the host copy is marked authoritative */
  PetscCall(DMSwarmDataFieldSyncToHost_Private(field));
  PetscCall(PetscMemzero(DMSWARM_DATAFIELD_point_access(field->data, index, field->atomic_size), field->atomic_size));
  field->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* zero ALL data for this point */
PetscErrorCode DMSwarmDataBucketZeroPoint(const DMSwarmDataBucket db, const PetscInt index)
{
  PetscInt f;

  PetscFunctionBegin;
  /* check point is valid */
  PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be >= 0");
  PetscCheck(index < db->allocated, PETSC_COMM_SELF, PETSC_ERR_USER, "index must be < %" PetscInt_FMT, db->allocated);
  for (f = 0; f < db->nfields; ++f) {
    DMSwarmDataField field = db->field[f];
    PetscCall(DMSwarmDataFieldZeroPoint(field, index));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* increment */
PetscErrorCode DMSwarmDataBucketAddPoint(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketSetSizes(db, PetscMax(db->L, 0) + 1, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* decrement */
PetscErrorCode DMSwarmDataBucketRemovePoint(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  PetscCheck(db->L > 0, PetscObjectComm((PetscObject)db), PETSC_ERR_ARG_WRONG, "Swarm has no points to be removed");
  PetscCall(DMSwarmDataBucketSetSizes(db, db->L - 1, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Remove the last n points, leaving the bucket exactly as n successive calls of
  DMSwarmDataBucketRemovePoint() would. Every one of those calls keeps the leading entries,
  shrinks the arrays to L + buffer once the unused space exceeds twice the buffer, and zeros
  the vacated slot (when the buffer is nonzero), so only the size arithmetic of
  DMSwarmDataBucketSetSizes_Host() is replayed here: the arrays are then reallocated at most once,
  to the final allocated size, and the vacated slots that survive it are zeroed at once. The
  values of L, buffer and allocated and the array contents are identical to the loop, but a
  device backend, which has no realloc(), copies each field at most once instead of on every
  other removal. Should the replay reach the growth branch, which a consistent bucket never
  does, the loop itself is run.
*/
PetscErrorCode DMSwarmDataBucketRemovePoints_Internal(DMSwarmDataBucket db, PetscInt n)
{
  PetscInt  L, allocated, end;
  PetscBool any_active_fields;

  PetscFunctionBegin;
  PetscCheck(n >= 0 && n <= db->L, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of points to remove %" PetscInt_FMT " must lie in [0, %" PetscInt_FMT "]", n, db->L);
  PetscCheck(db->finalised != PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_USER, "You must call DMSwarmDataBucketFinalize() before DMSwarmDataBucketRemovePoints_Internal()");
  PetscCall(DMSwarmDataBucketQueryForActiveFields(db, &any_active_fields));
  PetscCheck(!any_active_fields, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot safely re-size as at least one DMSwarmDataField is currently being accessed");
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  allocated = db->allocated;
  for (L = db->L - 1; L >= db->L - n; --L) {
    if (L + db->buffer > allocated) break;
    if (allocated - L > 2 * db->buffer) allocated = L + db->buffer;
  }
  if (L >= db->L - n) {
    for (PetscInt i = 0; i < n; ++i) PetscCall(DMSwarmDataBucketRemovePoint(db));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  L   = db->L - n;
  end = PetscMin(db->L, allocated);
  if (allocated != db->allocated) {
    for (PetscInt f = 0; f < db->nfields; ++f) PetscCall((*db->ops.fieldalloc)(db->field[f], allocated));
  }
  if (db->buffer > 0 && end > L) {
    for (PetscInt f = 0; f < db->nfields; ++f) PetscCall((*db->ops.fieldzerorange)(db->field[f], L, end));
  }
  db->L         = L;
  db->allocated = allocated;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*  Should be redone to user PetscViewer */
static PetscErrorCode DMSwarmDataBucketView_stdout(MPI_Comm comm, DMSwarmDataBucket db)
{
  PetscInt f;
  double   memory_usage_total = 0.0;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm, "DMSwarmDataBucketView: \n"));
  PetscCall(PetscPrintf(comm, "  L                  = %" PetscInt_FMT " \n", db->L));
  PetscCall(PetscPrintf(comm, "  buffer             = %" PetscInt_FMT " \n", db->buffer));
  PetscCall(PetscPrintf(comm, "  allocated          = %" PetscInt_FMT " \n", db->allocated));
  PetscCall(PetscPrintf(comm, "  nfields registered = %" PetscInt_FMT " \n", db->nfields));
  /* device-backend introspection only: host bucket output must stay unchanged */
  if (PetscMemTypeDevice(db->memtype)) PetscCall(PetscPrintf(comm, "  memtype            = %s \n", PetscMemTypeToString(db->memtype)));

  for (f = 0; f < db->nfields; ++f) {
    double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
    memory_usage_total += memory_usage_f;
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &memory_usage_total, 1, MPI_DOUBLE, MPI_SUM, comm));

  for (f = 0; f < db->nfields; ++f) {
    double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
    PetscCall(PetscPrintf(comm, "    [%3" PetscInt_FMT "] %15s : Mem. usage       = %1.2e (MB) [rank0]\n", f, db->field[f]->name, memory_usage_f));
    PetscCall(PetscPrintf(comm, "                            blocksize        = %" PetscInt_FMT " \n", db->field[f]->bs));
    if (db->field[f]->bs != 1) {
      PetscCall(PetscPrintf(comm, "                            atomic size      = %zu [full block, bs=%" PetscInt_FMT "]\n", db->field[f]->atomic_size, db->field[f]->bs));
      PetscCall(PetscPrintf(comm, "                            atomic size/item = %zu \n", (size_t)(db->field[f]->atomic_size / db->field[f]->bs)));
    } else {
      PetscCall(PetscPrintf(comm, "                            atomic size      = %zu \n", db->field[f]->atomic_size));
    }
    /* repeat the field name here so the state survives the "grep -v usage" test filters */
    if (PetscMemTypeDevice(db->memtype)) PetscCall(PetscPrintf(comm, "    [%3" PetscInt_FMT "] %15s : offload          = %s \n", f, db->field[f]->name, PetscOffloadMaskToString(db->field[f]->offloadmask)));
  }
  PetscCall(PetscPrintf(comm, "  Total mem. usage                           = %1.2e (MB) (collective)\n", memory_usage_total));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmDataBucketView_Seq(MPI_Comm comm, DMSwarmDataBucket db, const char filename[], DMSwarmDataBucketViewType type)
{
  PetscFunctionBegin;
  switch (type) {
  case DATABUCKET_VIEW_STDOUT:
    PetscCall(DMSwarmDataBucketView_stdout(PETSC_COMM_SELF, db));
    break;
  case DATABUCKET_VIEW_ASCII:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for ascii output");
  case DATABUCKET_VIEW_BINARY:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for binary output");
  case DATABUCKET_VIEW_HDF5:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for HDF5 output");
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unknown viewer method requested");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmDataBucketView_MPI(MPI_Comm comm, DMSwarmDataBucket db, const char filename[], DMSwarmDataBucketViewType type)
{
  PetscFunctionBegin;
  switch (type) {
  case DATABUCKET_VIEW_STDOUT:
    PetscCall(DMSwarmDataBucketView_stdout(comm, db));
    break;
  case DATABUCKET_VIEW_ASCII:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for ascii output");
  case DATABUCKET_VIEW_BINARY:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for binary output");
  case DATABUCKET_VIEW_HDF5:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for HDF5 output");
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unknown viewer method requested");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketView(MPI_Comm comm, DMSwarmDataBucket db, const char filename[], DMSwarmDataBucketViewType type)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    PetscCall(DMSwarmDataBucketView_Seq(comm, db, filename, type));
  } else {
    PetscCall(DMSwarmDataBucketView_MPI(comm, db, filename, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketDuplicateFields(DMSwarmDataBucket dbA, DMSwarmDataBucket *dbB)
{
  DMSwarmDataBucket db2;
  PetscInt          f;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketCreate(&db2));
  /* copy contents from dbA into db2 */
  for (f = 0; f < dbA->nfields; ++f) {
    DMSwarmDataField field;
    size_t           atomic_size;
    char            *name;

    field       = dbA->field[f];
    atomic_size = field->atomic_size;
    name        = field->name;
    PetscCall(DMSwarmDataBucketRegisterField(db2, "DMSwarmDataBucketDuplicateFields", name, atomic_size, NULL));
  }
  PetscCall(DMSwarmDataBucketFinalize(db2));
  PetscCall(DMSwarmDataBucketSetInitialSizes(db2, 0, 1000));
  *dbB = db2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Insert points from db2 into db1
 db1 <<== db2
 */
PetscErrorCode DMSwarmDataBucketInsertValues(DMSwarmDataBucket db1, DMSwarmDataBucket db2)
{
  PetscInt n_mp_points1, n_mp_points2;
  PetscInt n_mp_points1_new;

  PetscFunctionBegin;
  PetscCall(DMSwarmDataBucketGetSizes(db1, &n_mp_points1, NULL, NULL));
  PetscCall(DMSwarmDataBucketGetSizes(db2, &n_mp_points2, NULL, NULL));
  n_mp_points1_new = n_mp_points1 + n_mp_points2;
  PetscCall(DMSwarmDataBucketSetSizes(db1, n_mp_points1_new, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (PetscInt p = 0; p < n_mp_points2; ++p) {
    /* db1 <<== db2 */
    PetscCall(DMSwarmDataBucketCopyPoint(db2, p, db1, n_mp_points1 + p));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* helpers for parallel send/recv */
PetscErrorCode DMSwarmDataBucketCreatePackedArray(DMSwarmDataBucket db, size_t *bytes, void **buf)
{
  size_t sizeof_marker_contents;
  void  *buffer;

  PetscFunctionBegin;
  sizeof_marker_contents = 0;
  for (PetscInt f = 0; f < db->nfields; ++f) {
    DMSwarmDataField df = db->field[f];
    sizeof_marker_contents += df->atomic_size;
  }
  PetscCall(PetscMalloc(sizeof_marker_contents, &buffer));
  PetscCall(PetscMemzero(buffer, sizeof_marker_contents));
  if (bytes) *bytes = sizeof_marker_contents;
  if (buf) *buf = buffer;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketDestroyPackedArray(DMSwarmDataBucket db, void **buf)
{
  PetscFunctionBegin;
  if (buf) {
    PetscCall(PetscFree(*buf));
    *buf = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketFillPackedArray(DMSwarmDataBucket db, const PetscInt index, void *buf)
{
  void  *data, *data_p;
  size_t asize, offset;

  PetscFunctionBegin;
  offset = 0;
  for (PetscInt f = 0; f < db->nfields; ++f) {
    DMSwarmDataField df = db->field[f];

    /* read-only host access: refresh stale host entries, leave the offload mask untouched */
    PetscCall(DMSwarmDataFieldSyncToHost_Private(df));
    asize  = df->atomic_size;
    data   = df->data;
    data_p = (void *)((char *)data + index * asize);
    PetscCall(PetscMemcpy((void *)((char *)buf + offset), data_p, asize));
    offset = offset + asize;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmDataBucketInsertPackedArray(DMSwarmDataBucket db, const PetscInt idx, void *data)
{
  void  *data_p;
  size_t offset;

  PetscFunctionBegin;
  offset = 0;
  for (PetscInt f = 0; f < db->nfields; ++f) {
    DMSwarmDataField df = db->field[f];

    data_p = (void *)((char *)data + offset);
    PetscCall(DMSwarmDataFieldInsertPoint(df, idx, data_p));
    offset = offset + df->atomic_size;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
