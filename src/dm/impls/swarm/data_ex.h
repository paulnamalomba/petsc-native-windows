#pragma once

#include <petscvec.h>
#include <petscmat.h>
#include <petscdevicetypes.h>

typedef enum {
  DEOBJECT_INITIALIZED = 0,
  DEOBJECT_FINALIZED,
  DEOBJECT_STATE_UNKNOWN
} DMSwarmDEObjectState;

typedef struct _p_DMSwarmDataEx *DMSwarmDataEx;
struct _p_DMSwarmDataEx {
  PetscInt     instance;
  MPI_Comm     comm;
  PetscMPIInt  rank;
  PetscMPIInt  n_neighbour_procs;
  PetscMPIInt *neighbour_procs;         /* [n_neighbour_procs] */
  PetscInt    *messages_to_be_sent;     /* [n_neighbour_procs] */
  PetscInt    *message_offsets;         /* [n_neighbour_procs] */
  PetscInt    *messages_to_be_received; /* [n_neighbour_procs] */
  size_t       unit_message_size;
  void        *send_message;
  PetscInt     send_message_length;
  void        *recv_message;
  PetscInt     recv_message_length;
  /*
    Device-buffer support: with DMSwarmDataExSetMemType() set to a device memory type before
    DMSwarmDataExPackInitialize(), the payload is packed into send_message_d (reserved through
    DMSwarmDataExPackDataDevice(), filled by the caller, e.g. with the bucket packrecords op) and
    received into recv_message_d. When gpu-aware MPI is enabled (use_gpu_aware_mpi) the device
    pointers go straight to MPI; otherwise the packed buffer is staged through the host copies in
    one logged transfer each way. Message sizes, tags and neighbor topology are identical to the
    host path in every mode; only the residency of the payload buffers changes. Since exchangers
    are cached (on the swarm) and the memory type persists across pack cycles, EVERY pack site
    must declare its payload residency with DMSwarmDataExSetMemType() before its
    DMSwarmDataExPackInitialize(); a host-path DMSwarmDataExPackData() on an exchanger left in
    device mode by an earlier cycle is refused, and vice versa.
  */
  PetscMemType         memtype;           /* requested payload residency for the next pack cycle */
  PetscBool            message_on_device; /* the current cycle's payload buffers were allocated in device mode */
  PetscBool            gpu_aware_cycle;   /* snapshot of use_gpu_aware_mpi taken by DMSwarmDataExBegin() */
  void                *send_message_d;
  void                *recv_message_d;
  PetscMPIInt         *send_tags, *recv_tags;
  PetscInt             total_pack_cnt;
  PetscInt            *pack_cnt; /* [n_neighbour_procs] */
  DMSwarmDEObjectState topology_status;
  DMSwarmDEObjectState message_lengths_status;
  DMSwarmDEObjectState packer_status;
  DMSwarmDEObjectState communication_status;
  MPI_Status          *_stats;
  MPI_Request         *_requests;
};

/* OBJECT_STATUS */
/* #define OBJECT_INITIALIZED    0 */
/* #define OBJECT_FINALIZED      1 */
/* #define OBJECT_STATE_UNKNOWN  2 */

extern const char *status_names[];

PETSC_INTERN PetscErrorCode DMSwarmDataExCreate(MPI_Comm, const PetscInt, DMSwarmDataEx *);
PETSC_INTERN PetscErrorCode DMSwarmDataExView(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExDestroy(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyInitialize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyAddNeighbour(DMSwarmDataEx, const PetscMPIInt);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyFinalize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExInitializeSendCount(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExAddToSendCount(DMSwarmDataEx, const PetscMPIInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataExFinalizeSendCount(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExSetMemType(DMSwarmDataEx, PetscMemType);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackInitialize(DMSwarmDataEx, size_t);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackData(DMSwarmDataEx, PetscMPIInt, PetscInt, void *);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackDataDevice(DMSwarmDataEx, PetscMPIInt, PetscInt, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackFinalize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExBegin(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExEnd(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetSendData(DMSwarmDataEx, PetscInt *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetRecvData(DMSwarmDataEx, PetscInt *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetRecvDataDevice(DMSwarmDataEx, PetscInt *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyGetNeighbours(DMSwarmDataEx, PetscMPIInt *, PetscMPIInt *[]);
