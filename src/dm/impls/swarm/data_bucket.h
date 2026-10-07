#pragma once

#include <petsc/private/dmswarmimpl.h> /*I   "petscdmswarm.h"   I*/

#define DMSWARM_DATA_BUCKET_BUFFER_DEFAULT -1
#define DMSWARM_DATAFIELD_POINT_ACCESS_GUARD

/* Logging flag */
#define DMSWARM_DATA_BUCKET_LOG

typedef enum {
  DATABUCKET_VIEW_STDOUT = 0,
  DATABUCKET_VIEW_ASCII,
  DATABUCKET_VIEW_BINARY,
  DATABUCKET_VIEW_HDF5
} DMSwarmDataBucketViewType;

struct _p_DMSwarmDataField {
  char             *registration_function;
  PetscInt          L, bs;
  PetscBool         active;
  size_t            atomic_size;
  char             *name; /* what are they called */
  void             *data; /* the data - an array of structs */
  PetscDataType     petsc_type;
  void             *data_d;           /* device mirror of data */
  void             *data_allocated_d; /* non-NULL iff we own data_d */
  PetscInt          capacity_d;       /* entries allocated in data_d; may exceed L under geometric growth */
  PetscOffloadMask  offloadmask;      /* which of data/data_d holds valid entries */
  DMSwarmDataBucket bucket;           /* owning bucket, so field-level entry points can dispatch its ops; NULL for a field created outside a bucket */
};

/* Backend dispatch for every operation that touches particle storage; host implementations are the historical code paths */
typedef struct _DMSwarmDataBucketOps {
  PetscErrorCode (*fieldalloc)(DMSwarmDataField, PetscInt); /* resize field storage to a new capacity */
  PetscErrorCode (*fieldallocdevice)(DMSwarmDataField);     /* size the device mirror without transferring entries (write-only device access) */
  PetscErrorCode (*fieldtodevice)(DMSwarmDataField);        /* H2D */
  PetscErrorCode (*fieldtohost)(DMSwarmDataField);          /* D2H */
  PetscErrorCode (*fieldzerorange)(DMSwarmDataField, PetscInt, PetscInt);
  PetscErrorCode (*fieldfreedevice)(DMSwarmDataField); /* release the device mirror of a field; must be dispatched before DMSwarmDataFieldDestroy() */
  PetscErrorCode (*setsizes)(DMSwarmDataBucket, PetscInt, PetscInt);
  /*
    Migration wire records: one record is the concatenation of every field's atomic_size bytes for
    one point, in field registration order (the DMSwarmDataBucketFillPackedArray() layout). For the
    host backend the index list and buffer are host pointers; for a device backend both live on the
    device. packrecords gathers the records of the points in list[] (in list order) into a
    contiguous buffer; unpackrecords scatters n contiguous records into the points [start, start+n).
  */
  PetscErrorCode (*packrecords)(DMSwarmDataBucket, PetscInt, const PetscInt *, void *);
  PetscErrorCode (*unpackrecords)(DMSwarmDataBucket, PetscInt, PetscInt, const void *);
  /*
    Remove the nsent points whose indices are listed (strictly ascending, in [0, L)) in sentlist,
    reproducing exactly the survivor order and size bookkeeping of the historical migration loop
    (ascending scan calling DMSwarmDataBucketRemovePointAtIndex() with swap-with-last and
    re-check). For the host backend sentlist is a host pointer; for a device backend it lives on
    the device. Call through DMSwarmDataBucketCompact(), which performs the argument checking.
  */
  PetscErrorCode (*compact)(DMSwarmDataBucket, PetscInt, const PetscInt *);
} DMSwarmDataBucketOps;

struct _p_DMSwarmDataBucket {
  PetscInt             L;         /* number in use */
  PetscInt             buffer;    /* memory buffer used for re-allocation */
  PetscInt             allocated; /* number allocated, this will equal datafield->L */
  PetscBool            finalised; /* DEPRECATED */
  PetscInt             nfields;   /* how many fields of this type */
  DMSwarmDataField    *field;     /* the data */
  PetscMemType         memtype;   /* where particle data natively lives */
  DMSwarmDataBucketOps ops;       /* backend dispatch */
};

#define DMSWARM_DATAFIELD_point_access(data, index, atomic_size)                (void *)((char *)(data) + (index) * (atomic_size))
#define DMSWARM_DATAFIELD_point_access_offset(data, index, atomic_size, offset) (void *)((char *)(data) + (index) * (atomic_size) + (offset))

PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringInList(const char[], const PetscInt, const DMSwarmDataField[], PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringFindInList(const char[], const PetscInt, const DMSwarmDataField[], PetscInt *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldCreate(const char[], const char[], const size_t, const PetscInt, DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldDestroy(DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreate(DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroy(DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketQueryForActiveFields(DMSwarmDataBucket, PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRegisterField(DMSwarmDataBucket, const char[], const char[], size_t, DMSwarmDataField *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetNumEntries(DMSwarmDataField, PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetBlockSize(DMSwarmDataField, PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetSize(DMSwarmDataField, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroBlock(DMSwarmDataField, const PetscInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAcquireAccess(const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldReleaseAccess(DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAccess(const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPoint(const DMSwarmDataField, const PetscInt, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPointOffset(const DMSwarmDataField, const size_t, const PetscInt, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldRestoreAccess(DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldVerifyAccess(const DMSwarmDataField, const size_t);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAtomicSize(const DMSwarmDataField, size_t *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldInsertPoint(const DMSwarmDataField, const PetscInt, const void *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt, const DMSwarmDataField, const PetscInt, const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroPoint(const DMSwarmDataField, const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketFinalize(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetInitialSizes(DMSwarmDataBucket, const PetscInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetSizes(DMSwarmDataBucket, const PetscInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetSizes_Host(DMSwarmDataBucket, PetscInt, PetscInt);
/* Defined in src/dm/impls/swarm/cupm/hip/; declared unconditionally, call only when PetscDefined(HAVE_HIP) */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetUpDevice_HIP(DMSwarmDataBucket);
/*
  Device helpers for the migration fork in swarm_migrate.c (same build/call rule as above).
  DMSwarmDataFieldFillIntRange_HIP() fills [start, end) of a PetscInt field's device mirror with a
  constant and leaves the field device authoritative. DMSwarmSentListCreate_HIP() enumerates, in
  strictly ascending order, the indices in [start, end) whose entry of a device-resident PetscInt
  array equals a match value, into a freshly allocated device list released with
  DMSwarmSentListDestroy_HIP().
*/
PETSC_INTERN PetscErrorCode DMSwarmDataFieldFillIntRange_HIP(DMSwarmDataField, PetscInt, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmSentListCreate_HIP(PetscInt, PetscInt, PetscInt, const PetscInt *, PetscInt *, PetscInt **);
PETSC_INTERN PetscErrorCode DMSwarmSentListDestroy_HIP(PetscInt **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetSizes(DMSwarmDataBucket, PetscInt *, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetGlobalSizes(MPI_Comm, DMSwarmDataBucket, PetscInt *, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFields(DMSwarmDataBucket, PetscInt *, DMSwarmDataField *[]);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketCopyPoint(const DMSwarmDataBucket, const PetscInt, const DMSwarmDataBucket, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreateFromSubset(DMSwarmDataBucket, const PetscInt, const PetscInt[], DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketZeroPoint(const DMSwarmDataBucket, const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketView(MPI_Comm, DMSwarmDataBucket, const char[], DMSwarmDataBucketViewType);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketAddPoint(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePoint(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePointAtIndex(const DMSwarmDataBucket, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePoints_Internal(DMSwarmDataBucket, PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCompact(DMSwarmDataBucket, PetscInt, const PetscInt[]);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketDuplicateFields(DMSwarmDataBucket, DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertValues(DMSwarmDataBucket, DMSwarmDataBucket);

/* helpers for parallel send/recv */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreatePackedArray(DMSwarmDataBucket, size_t *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroyPackedArray(DMSwarmDataBucket, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketFillPackedArray(DMSwarmDataBucket, const PetscInt, void *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertPackedArray(DMSwarmDataBucket, const PetscInt, void *);
