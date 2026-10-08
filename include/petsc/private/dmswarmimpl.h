#pragma once

#include <petscvec.h>     /*I "petscvec.h" I*/
#include <petscmat.h>     /*I      "petscmat.h"          I*/
#include <petscdmswarm.h> /*I      "petscdmswarm.h"    I*/
#include <petsc/private/dmimpl.h>

PETSC_SINGLE_LIBRARY_INTERN PetscBool  SwarmProjcite;
PETSC_SINGLE_LIBRARY_INTERN const char SwarmProjCitation[];

PETSC_EXTERN PetscLogEvent DMSWARM_Migrate;
PETSC_EXTERN PetscLogEvent DMSWARM_SetSizes;
PETSC_EXTERN PetscLogEvent DMSWARM_AddPoints;
PETSC_EXTERN PetscLogEvent DMSWARM_RemovePoints;
PETSC_EXTERN PetscLogEvent DMSWARM_Sort;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerTopologySetup;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerBegin;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerEnd;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerSendCount;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerPack;
PETSC_EXTERN PetscLogEvent DMSWARM_CopyToGPU;
PETSC_EXTERN PetscLogEvent DMSWARM_CopyFromGPU;
PETSC_EXTERN PetscLogEvent DMSWARM_MassMatrixSquare;

typedef struct _DMSwarmCellDMOps *DMSwarmCellDMOps;
struct _DMSwarmCellDMOps {
  PetscErrorCode (*create)(DMSwarmCellDM);
  PetscErrorCode (*destroy)(DMSwarmCellDM);
  PetscErrorCode (*setfromoptions)(PetscOptionItems *, DMSwarmCellDM);
  PetscErrorCode (*setup)(DMSwarmCellDM);
  PetscErrorCode (*view)(DMSwarmCellDM, PetscViewer);
};

struct _p_DMSwarmCellDM {
  PETSCHEADER(struct _DMSwarmCellDMOps);
  DM          dm;          // The cell DM
  PetscInt    Nf;          // The number of DM fields
  char      **dmFields;    // Swarm fields defining this DM
  PetscInt    Nfc;         // The number of DM coordinate fields
  char      **coordFields; // Swarm field for coordinates on this DM
  char       *cellid;      // Swarm field with cell ids for particles
  DMSwarmSort sort;        // Ordering of particles by enclosing cell
};

/*
 Error checking to ensure the swarm type is correct and that a cell DM has been set
*/
#define DMSWARMPICVALID(obj) \
  do { \
    DM_Swarm *_swarm = (DM_Swarm *)(obj)->data; \
    PetscCheck(_swarm->swarm_type == DMSWARM_PIC, PetscObjectComm((PetscObject)(obj)), PETSC_ERR_SUP, "Valid only for DMSwarm-PIC. You must call DMSwarmSetType(dm,DMSWARM_PIC)"); \
    PetscCheck(_swarm->activeCellDM, PetscObjectComm((PetscObject)(obj)), PETSC_ERR_SUP, "Valid only for DMSwarmPIC if the cell DM is set. You must call DMSwarmSetCellDM() or DMSwarmAddCellDM()"); \
  } while (0)

typedef struct {
  DMSwarmDataBucket db;
  PetscInt          refct;
  PetscBool         field_registration_initialized;
  PetscBool         field_registration_finalized;
  /* DMSwarmProjectMethod *swarm_project;*/ /* swarm, geometry, result */

  /* PetscInt overlap; */
  /* PetscErrorCode (*update_overlap)(void); */

  PetscObjectList cellDMs;
  const char     *activeCellDM;

  struct _p_DMSwarmDataEx *de;           /* cached point exchanger, reused across migration/collection calls */
  PetscObjectId            de_layout_id; /* id of the cell DM whose neighbor topology is cached in de, 0 if none */

  PetscBool          issetup;
  DMSwarmType        swarm_type;
  DMSwarmMigrateType migrate_type;
  DMSwarmCollectType collect_type;
  DMSwarmRemapType   remap_type;
  DMSwarmBackendType backend;

  PetscBool migrate_error_on_missing_point;
  PetscBool mass_coo; /* assemble the particle mass matrix, and its square when its values are computed on the device, with MatSetValuesCOO() instead of MatSetValues() */

  PetscBool collect_view_active;
  PetscInt  collect_view_reset_nlocal;

  /* Support for PIC */
  PetscInt Ns; /* The number of particle species */

  PetscSimplePointFn *coordFunc; /* Function to set particle coordinates */
  PetscSimplePointFn *velFunc;   /* Function to set particle velocities */

  /* Debugging */
  PetscInt printCoords;
  PetscInt printWeights;
} DM_Swarm;

typedef struct {
  PetscInt point_index;
  PetscInt cell_index;
} SwarmPoint;

typedef struct _DMSwarmSortOps *DMSwarmSortOps;
struct _DMSwarmSortOps {
  PetscErrorCode (*create)(DMSwarmSort);
  PetscErrorCode (*destroy)(DMSwarmSort);
  PetscErrorCode (*setfromoptions)(PetscOptionItems *, DMSwarmSort);
  PetscErrorCode (*setup)(DMSwarmSort, DM, PetscInt); /* (sort, swarm, ncells) builds the cell CSR; backend chosen in DMSwarmSortGetAccess() from the bucket memtype */
  PetscErrorCode (*view)(DMSwarmSort, PetscViewer);
};

struct _p_DMSwarmSort {
  PETSCHEADER(struct _DMSwarmSortOps);
  PetscBool   isvalid;
  PetscInt    ncells, npoints;
  PetscInt   *pcell_offsets;
  SwarmPoint *list;
  /* Device-resident CSR, built by the device setup op alongside the host arrays above (which
     always hold a bitwise-identical mirror). The device arrays are valid only while isvalid is
     PETSC_TRUE and memtype is a device memtype; they keep their allocation across rebuilds. */
  PetscMemType memtype;             /* memory space the last setup ran in; PETSC_MEMTYPE_HOST for the host path */
  PetscInt    *pcell_offsets_d;     /* [ncells+1] device mirror of pcell_offsets */
  PetscInt    *list_d;              /* [npoints] sorted point indices, the device analog of list[].point_index */
  PetscInt    *cell_index_d;        /* [npoints] ascending cell index per CSR slot, the device analog of list[].cell_index */
  PetscInt     ncells_d, npoints_d; /* allocated sizes of the device arrays, -1 when unallocated */
};

PETSC_INTERN PetscErrorCode DMSwarmSortSetUpDevice_HIP(DMSwarmSort);

PETSC_INTERN PetscErrorCode DMSwarmMigrate_Push_Basic(DM, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMigrate_CellDMScatter(DM, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMigrate_CellDMExact(DM, PetscBool);

PETSC_INTERN PetscErrorCode DMGlobalToLocalBegin_Swarm(DM, Vec, InsertMode, Vec);
PETSC_INTERN PetscErrorCode DMGlobalToLocalEnd_Swarm(DM, Vec, InsertMode, Vec);
PETSC_INTERN PetscErrorCode DMLocalToGlobalBegin_Swarm(DM, Vec, InsertMode, Vec);
PETSC_INTERN PetscErrorCode DMLocalToGlobalEnd_Swarm(DM, Vec, InsertMode, Vec);
PETSC_INTERN PetscErrorCode DMProjectFieldLocal_Swarm(DM, PetscReal, Vec, PetscPointFn **, InsertMode, Vec);
