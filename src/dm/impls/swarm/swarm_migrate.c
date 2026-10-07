#include <petscsf.h>
#include <petscdmswarm.h>
#include <petscdmda.h>
#include <petsc/private/dmswarmimpl.h> /*I   "petscdmswarm.h"   I*/
#include <petsc/private/dmpleximpl.h>
#include "../src/dm/impls/swarm/data_bucket.h"
#include "../src/dm/impls/swarm/data_ex.h"

/*
 Returns the data exchanger cached on the swarm, creating it on first use. Reusing the exchanger
 (and thus its duplicated communicator) across calls avoids an MPI_Comm_dup() per migration. All
 exchanges are completed before each call returns, so no messages can cross between calls. Callers
 must either rebuild the neighbor topology (resetting swarm->de_layout_id to 0) or reuse it only
 when swarm->de_layout_id proves it was built from the same cell DM.
*/
static PetscErrorCode DMSwarmGetDataExchanger_Private(DM dm, DMSwarmDataEx *de)
{
  DM_Swarm *swarm = (DM_Swarm *)dm->data;

  PetscFunctionBegin;
  if (!swarm->de) PetscCall(DMSwarmDataExCreate(PetscObjectComm((PetscObject)dm), 0, &swarm->de));
  *de = swarm->de;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 User loads desired location (MPI rank) into field DMSwarm_rank

 It should be storing the rank information as MPIInt not Int
*/
PetscErrorCode DMSwarmMigrate_Push_Basic(DM dm, PetscBool remove_sent_points)
{
  DM_Swarm     *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx de;
  PetscInt      p, npoints, *rankval, n_points_recv;
  PetscMPIInt   rank, nrank;
  void         *point_buffer, *recv_points;
  size_t        sizeof_dmswarm_point;
  PetscBool     debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmGetDataExchanger_Private(dm, &de));
  swarm->de_layout_id = 0;
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < npoints; ++p) {
    PetscCall(PetscMPIIntCast(rankval[p], &nrank));
    if (nrank != rank) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, nrank));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < npoints; p++) {
    PetscCall(PetscMPIIntCast(rankval[p], &nrank));
    if (nrank != rank) PetscCall(DMSwarmDataExAddToSendCount(de, nrank, 1));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  /*
   DMSwarmMigrate_Push_Basic(), DMSwarmMigrate_GlobalToLocal_Basic() and the DMSwarmCollect_*
   functions run entirely on the host on every backend; device-backend buckets work through the
   sync-on-access coherence of the field accessors. Only DMSwarmMigrate_DMNeighborScatter() has
   a device fork. Since the cached exchanger keeps its payload memory type across cycles, every
   pack site declares its residency before packing (see data_ex.h).
  */
  PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HOST));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (p = 0; p < npoints; p++) {
    PetscCall(PetscMPIIntCast(rankval[p], &nrank));
    if (nrank != rank) {
      /* copy point into buffer */
      PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
      /* insert point buffer into DMSwarmDataExchanger */
      PetscCall(DMSwarmDataExPackData(de, nrank, 1, point_buffer));
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));

  if (remove_sent_points) {
    DMSwarmDataField gfield;

    PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &gfield));
    PetscCall(DMSwarmDataFieldGetAccess(gfield));
    PetscCall(DMSwarmDataFieldGetEntries(gfield, (void **)&rankval));

    /* remove points which left processor */
    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
    for (p = 0; p < npoints; p++) {
      PetscCall(PetscMPIIntCast(rankval[p], &nrank));
      if (nrank != rank) {
        /* kill point */
        PetscCall(DMSwarmDataFieldRestoreAccess(gfield));

        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));

        PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetAccess(gfield));
        PetscCall(DMSwarmDataFieldGetEntries(gfield, (void **)&rankval));
        p--; /* check replacement point */
      }
    }
    PetscCall(DMSwarmDataFieldRestoreEntries(gfield, (void **)&rankval));
    PetscCall(DMSwarmDataFieldRestoreAccess(gfield));
    PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
  }
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  if (debug) PetscCall(DMSwarmDataExView(de));
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 use_device selects the device fork of the scatter: the caller passes the located_on_device
 result of the first (pre-exchange) point location, which is MPI_MIN-allreduced there and hence
 collectively consistent. On the device fork the send list, the packed wire records, the
 removal compaction and the unpack of received records all act on the device-resident particle
 payload; message counts, tags and neighbor topology are computed by the same host code as the
 host fork, so the exchanged bytes are identical. Any fallback configuration keeps the entire
 legacy host path.
*/
static PetscErrorCode DMSwarmMigrate_DMNeighborScatter(DM dm, DM dmcell, PetscBool remove_sent_points, PetscBool use_device, PetscInt *npoints_prior_migration)
{
  DM_Swarm          *swarm = (DM_Swarm *)dm->data;
  DMSwarmCellDM      celldm;
  DMSwarmDataEx      de;
  PetscInt           r, p, npoints, *p_cellid, n_points_recv;
  PetscMPIInt        rank, _rank;
  const PetscMPIInt *neighbourranks;
  void              *point_buffer, *recv_points;
  size_t             sizeof_dmswarm_point;
  PetscInt           nneighbors;
  PetscMPIInt        mynneigh, *myneigh;
  const char        *cellid;
  PetscObjectId      layout_id;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmGetCellDMActive(dm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  if (!use_device) PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&p_cellid));
  PetscCall(DMSwarmGetDataExchanger_Private(dm, &de));
  /*
   The symmetrized neighbor topology and message tags built by DMSwarmDataExTopologyFinalize() are a
   deterministic function of the ranks returned by DMGetNeighbors(), which are fixed for a given cell
   DM. The cached topology is therefore reused unless the cell DM changed or another exchange rebuilt
   the topology, both of which reset swarm->de_layout_id. The check is collectively consistent since
   every exchange over the swarm is collective with the same cell DM on every rank.
  */
  PetscCall(PetscObjectGetId((PetscObject)dmcell, &layout_id));
  if (swarm->de_layout_id != layout_id) {
    PetscCall(DMGetNeighbors(dmcell, &nneighbors, &neighbourranks));
    PetscCall(DMSwarmDataExTopologyInitialize(de));
    for (r = 0; r < nneighbors; r++) {
      _rank = neighbourranks[r];
      if ((_rank != rank) && (_rank > 0)) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, _rank));
    }
    PetscCall(DMSwarmDataExTopologyFinalize(de));
    swarm->de_layout_id = layout_id;
  }
  PetscCall(DMSwarmDataExTopologyGetNeighbours(de, &mynneigh, &myneigh));
#if PetscDefined(HAVE_HIP)
  if (use_device) {
    /*
     Device migration fork: the particle payload stays in device memory end to end. The send
     list (the strictly ascending indices of the unlocated points) is built on the device from
     the cell id field, the wire records are gathered by the bucket packrecords op straight into
     the exchanger's device send buffer, removal replays the host swap-with-last survivor order
     through DMSwarmDataBucketCompact(), and the received records are scattered into the bucket
     tail by the unpackrecords op. Only the payload residency differs from the host fork below;
     counts, tags, topology and the per-neighbour segment bytes are identical.
    */
    PetscInt    *cellids_d, *sentlist_d = NULL;
    PetscInt     nsent       = 0;
    PetscMemType mti         = PETSC_MEMTYPE_HOST;
    size_t       record_size = 0;
    void        *dest_d, *recv_d;

    PetscCall(DMSwarmGetFieldReadAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
    PetscCheck(PetscMemTypeDevice(mti), PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "hip-backend swarm returned a host pointer for its cell id field");
    PetscCall(DMSwarmSentListCreate_HIP(0, npoints, DMLOCATEPOINT_POINT_NOT_FOUND, cellids_d, &nsent, &sentlist_d));
    PetscCall(DMSwarmRestoreFieldReadAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
    /* every unlocated point is sent to every neighbour, so the count is nsent per neighbour exactly as in the host loops */
    PetscCall(DMSwarmDataExInitializeSendCount(de));
    for (r = 0; r < mynneigh; r++) PetscCall(DMSwarmDataExAddToSendCount(de, myneigh[r], nsent));
    PetscCall(DMSwarmDataExFinalizeSendCount(de));
    for (PetscInt f = 0; f < swarm->db->nfields; f++) record_size += swarm->db->field[f]->atomic_size;
    PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HIP));
    PetscCall(DMSwarmDataExPackInitialize(de, record_size));
    for (r = 0; r < mynneigh; r++) {
      /* reserving this neighbour's segment and gathering the same ascending list into it reproduces the host segment bytes */
      PetscCall(DMSwarmDataExPackDataDevice(de, myneigh[r], nsent, &dest_d));
      PetscCall((*swarm->db->ops.packrecords)(swarm->db, nsent, sentlist_d, dest_d));
    }
    PetscCall(DMSwarmDataExPackFinalize(de));
    PetscCall(PetscInfo(dm, "DMSwarmMigrate: packed %" PetscInt_FMT " points for %d neighbours on the device\n", nsent, mynneigh));
    if (remove_sent_points) {
      /* packing precedes removal exactly as in the host fork, which packs from the original layout before removing */
      PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
      PetscCall(DMSwarmDataBucketCompact(swarm->db, nsent, sentlist_d));
      PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
      PetscCall(PetscInfo(dm, "DMSwarmMigrate: compacted away %" PetscInt_FMT " sent points on the device\n", nsent));
    }
    PetscCall(DMSwarmSentListDestroy_HIP(&sentlist_d));
    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, npoints_prior_migration, NULL, NULL));
    PetscCall(DMSwarmDataExBegin(de));
    PetscCall(DMSwarmDataExEnd(de));
    PetscCall(DMSwarmDataExGetRecvDataDevice(de, &n_points_recv, &recv_d));
    PetscCall(DMSwarmDataBucketSetSizes(swarm->db, *npoints_prior_migration + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
    PetscCall((*swarm->db->ops.unpackrecords)(swarm->db, *npoints_prior_migration, n_points_recv, recv_d));
    PetscCall(PetscInfo(dm, "DMSwarmMigrate: unpacked %" PetscInt_FMT " received points on the device\n", n_points_recv));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < npoints; p++) {
    if (p_cellid[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r = 0; r < mynneigh; r++) {
        _rank = myneigh[r];
        PetscCall(DMSwarmDataExAddToSendCount(de, _rank, 1));
      }
    }
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HOST));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (p = 0; p < npoints; p++) {
    if (p_cellid[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
      for (r = 0; r < mynneigh; r++) {
        _rank = myneigh[r];
        /* copy point into buffer */
        PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
        /* insert point buffer into DMSwarmDataExchanger */
        PetscCall(DMSwarmDataExPackData(de, _rank, 1, point_buffer));
      }
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&p_cellid));
  if (remove_sent_points) {
    DMSwarmDataField PField;

    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, cellid, &PField));
    PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
    PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&p_cellid));
    /* remove points which left processor */
    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
    for (p = 0; p < npoints; p++) {
      if (p_cellid[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
        /* kill point */
        PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));
        PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL)); /* you need to update npoints as the list size decreases! */
        PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&p_cellid));     /* update date point increase realloc performed */
        p--;                                                                   /* check replacement point */
      }
    }
    PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
  }
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, npoints_prior_migration, NULL, NULL));
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Attempt the point-location step of DMSwarmMigrate_CellDMScatter() on the device. For a hip-backend
 swarm whose cell DM is a DMPlex with a usable point-location geometry cache, the cell ids of the
 points [pStart, pStart + npoints) are computed in place in the cell id field's device mirror from
 the coordinate field's device mirror, reproducing the host location decisions bitwise, see
 DMPlexLocatePointsInPlace_Internal_HIP(). When useSeeds is PETSC_FALSE the incoming cell ids
 (freshly received points carry the sender's local cell numbers) are reset first by a device fill
 of the cell id range, matching the host path which locates received points through a fresh cell
 SF while keeping the cell id field off the host. On return *located indicates whether
 the device path ran; otherwise (host backend, non-DMPlex cell DM, unusable cache, multiple
 coordinate fields) the caller must run the host location path unchanged. The decision is made
 collectively consistent because the host path performs collective operations.
*/
static PetscErrorCode DMSwarmMigrate_LocatePointsDevice_Private(DM dm, DM dmcell, PetscInt pStart, PetscInt npoints, PetscBool useSeeds, PetscBool *located)
{
  PetscFunctionBegin;
  *located = PETSC_FALSE;
#if PetscDefined(HAVE_HIP)
  if (((DM_Swarm *)dm->data)->backend == DMSWARM_BACKEND_HIP) {
    DMSwarmCellDM celldm;
    PetscInt      Nfc, ok = 0;
    const char  **coordFields, *cellid;
    PetscBool     isplex;

    PetscCall(PetscObjectTypeCompare((PetscObject)dmcell, DMPLEX, &isplex));
    PetscCall(DMSwarmGetCellDMActive(dm, &celldm));
    PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
    if (isplex && Nfc == 1) {
      DMPlexLocationCache cache;

      PetscCall(DMPlexGetLocationCache_Internal(dmcell, &cache));
      if (!cache->unavailable) ok = 1;
      else PetscCall(PetscInfo(dm, "DMSwarmMigrate: using host point location, the cell DM cannot use the location cache\n"));
    } else PetscCall(PetscInfo(dm, "DMSwarmMigrate: using host point location, the cell DM is not a DMPlex with a single coordinate field\n"));
    /* The host location path performs collective operations, so every rank must take the same branch */
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &ok, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm)));
    if (ok) {
      PetscInt     bs;
      PetscMemType mtc = PETSC_MEMTYPE_HOST, mti = PETSC_MEMTYPE_HOST;
      PetscReal   *coords_d;
      PetscInt    *cellids_d;

      PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
      if (!useSeeds && npoints) {
        DMSwarmDataField cfield;

        /* reset the seeds in the device mirror directly; a host loop here would round-trip the whole cell id field through the host every migration cycle */
        PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(((DM_Swarm *)dm->data)->db, cellid, &cfield));
        PetscCall(DMSwarmDataFieldFillIntRange_HIP(cfield, pStart, pStart + npoints, DMLOCATEPOINT_POINT_NOT_FOUND));
      }
      PetscCall(DMSwarmGetFieldReadAndMemType(dm, coordFields[0], &bs, NULL, &mtc, (void **)&coords_d));
      PetscCall(DMSwarmGetFieldAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
      PetscCheck(PetscMemTypeDevice(mtc) && PetscMemTypeDevice(mti), PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "hip-backend swarm returned host pointers for its fields");
      PetscCall(DMPlexLocatePointsInPlace_Internal_HIP(dmcell, npoints, bs, coords_d + pStart * bs, cellids_d + pStart));
      PetscCall(DMSwarmRestoreFieldAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
      PetscCall(DMSwarmRestoreFieldReadAndMemType(dm, coordFields[0], &bs, NULL, &mtc, (void **)&coords_d));
      PetscCall(PetscInfo(dm, "DMSwarmMigrate: located %" PetscInt_FMT " points on the device\n", npoints));
      *located = PETSC_TRUE;
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmMigrate_CellDMScatter(DM dm, PetscBool remove_sent_points)
{
  DM_Swarm          *swarm = (DM_Swarm *)dm->data;
  DMSwarmCellDM      celldm;
  PetscInt           p, npoints, npointsg = 0, npoints2, npoints2g, *rankval, *p_cellid, npoints_prior_migration, Nfc;
  PetscSF            sfcell = NULL;
  const PetscSFNode *LA_sfcell;
  DM                 dmcell;
  Vec                pos;
  PetscBool          error_check       = swarm->migrate_error_on_missing_point;
  PetscBool          located_on_device = PETSC_FALSE;
  const char       **coordFields;
  PetscMPIInt        size, rank;
  const char        *cellid;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(dm, &dmcell));
  PetscCheck(dmcell, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only valid if cell DM provided");
  PetscCall(DMSwarmGetCellDMActive(dm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmMigrate_LocatePointsDevice_Private(dm, dmcell, 0, npoints, PETSC_TRUE, &located_on_device));
  if (!located_on_device) {
    /* Seed the point location with the current cell ids */
    PetscInt     npoints_curr, range = 0;
    PetscSFNode *sf_cells;

    PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL));
    PetscCall(PetscMalloc1(npoints_curr, &sf_cells));

    PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&p_cellid));
    for (p = 0; p < npoints_curr; p++) {
      sf_cells[p].rank  = 0;
      sf_cells[p].index = p_cellid[p];
      if (p_cellid[p] > range) range = p_cellid[p];
    }
    PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&p_cellid));

    /* PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm),&sfcell)); */
    PetscCall(PetscSFCreate(PETSC_COMM_SELF, &sfcell));
    PetscCall(PetscSFSetGraph(sfcell, range, npoints_curr, NULL, PETSC_OWN_POINTER, sf_cells, PETSC_OWN_POINTER));

    PetscCall(DMSwarmCreateLocalVectorFromFields(dm, Nfc, coordFields, &pos));
    PetscCall(DMLocatePoints(dmcell, pos, DM_POINTLOCATION_NONE, &sfcell));
    PetscCall(DMSwarmDestroyLocalVectorFromFields(dm, Nfc, coordFields, &pos));
  }

  if (error_check) PetscCall(DMSwarmGetSize(dm, &npointsg));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  if (!located_on_device) {
    PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&p_cellid));
    PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
    for (p = 0; p < npoints; p++) {
      p_cellid[p] = LA_sfcell[p].index;
      rankval[p]  = rank;
    }
    PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&p_cellid));
    PetscCall(PetscSFDestroy(&sfcell));
    PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  }
#if PetscDefined(HAVE_HIP)
  if (located_on_device) {
    DMSwarmDataField rfield;

    /* Cell ids were written in place in the cell id field's device mirror; the owning rank is recorded by a device fill so the rank field never round-trips through the host */
    PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &rfield));
    PetscCall(DMSwarmDataFieldFillIntRange_HIP(rfield, 0, npoints, (PetscInt)rank));
  }
#endif

  if (size > 1) {
    PetscCall(DMSwarmMigrate_DMNeighborScatter(dm, dmcell, remove_sent_points, located_on_device, &npoints_prior_migration));
  } else {
    DMSwarmDataField PField;
    PetscInt         npoints_curr;

    /* remove points which the domain */
    if (!located_on_device) {
      PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, cellid, &PField));
      PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&p_cellid));

      PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL));
      if (remove_sent_points) {
        PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
        for (p = 0; p < npoints_curr; p++) {
          if (p_cellid[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
            /* kill point */
            PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));
            PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL)); /* you need to update npoints as the list size decreases! */
            PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&p_cellid));          /* update date point in case realloc performed */
            p--;                                                                        /* check replacement point */
          }
        }
        PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
      }
    }
#if PetscDefined(HAVE_HIP)
    if (located_on_device && remove_sent_points) {
      /* device fork of the loop above: the compaction replays the host swap-with-last survivor order, see DMSwarmDataBucketCompact() */
      PetscInt    *cellids_d, *sentlist_d = NULL;
      PetscInt     nsent = 0;
      PetscMemType mti   = PETSC_MEMTYPE_HOST;

      PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
      PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints_curr, NULL, NULL));
      PetscCall(DMSwarmGetFieldReadAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
      PetscCall(DMSwarmSentListCreate_HIP(0, npoints_curr, DMLOCATEPOINT_POINT_NOT_FOUND, cellids_d, &nsent, &sentlist_d));
      PetscCall(DMSwarmRestoreFieldReadAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
      PetscCall(DMSwarmDataBucketCompact(swarm->db, nsent, sentlist_d));
      PetscCall(DMSwarmSentListDestroy_HIP(&sentlist_d));
      PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
      PetscCall(PetscInfo(dm, "DMSwarmMigrate: compacted away %" PetscInt_FMT " sent points on the device\n", nsent));
    }
#endif
    PetscCall(DMSwarmGetLocalSize(dm, &npoints_prior_migration));
  }

  /* locate points newly received */
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL));

  { /* perform two point locations: (i) on the initial points set prior to communication; and (ii) on the new (received) points */
    Vec              npos;
    IS               nis;
    PetscInt         npoints_from_neighbours, bs;
    DMSwarmDataField PField;

    npoints_from_neighbours = npoints2 - npoints_prior_migration;

    PetscCall(DMSwarmMigrate_LocatePointsDevice_Private(dm, dmcell, npoints_prior_migration, npoints_from_neighbours, PETSC_FALSE, &located_on_device));
    if (!located_on_device) {
      PetscCall(DMSwarmCreateLocalVectorFromFields(dm, Nfc, coordFields, &pos));
      PetscCall(VecGetBlockSize(pos, &bs));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, npoints_from_neighbours * bs, npoints_prior_migration * bs, 1, &nis));
/*
  Device VecGetSubVector to zero sized subvector triggers
  debug error with mismatching memory types due to the device
  pointer being host memtype without anything to point to in
  Vec"Type"GetArrays(...), and we still need to pass something to
  DMLocatePoints to avoid deadlock
*/
#if PetscDefined(HAVE_DEVICE)
      if (npoints_from_neighbours > 0) {
        PetscCall(VecGetSubVector(pos, nis, &npos));
        PetscCall(DMLocatePoints(dmcell, npos, DM_POINTLOCATION_NONE, &sfcell));
        PetscCall(VecRestoreSubVector(pos, nis, &npos));
      } else {
        PetscCall(VecCreate(PETSC_COMM_SELF, &npos));
        PetscCall(VecSetSizes(npos, 0, PETSC_DETERMINE));
        PetscCall(VecSetBlockSize(npos, bs));
        PetscCall(VecSetType(npos, dm->vectype));
        PetscCall(DMLocatePoints(dmcell, npos, DM_POINTLOCATION_NONE, &sfcell));
        PetscCall(VecDestroy(&npos));
      }
#else
      PetscCall(VecGetSubVector(pos, nis, &npos));
      PetscCall(DMLocatePoints(dmcell, npos, DM_POINTLOCATION_NONE, &sfcell));
      PetscCall(VecRestoreSubVector(pos, nis, &npos));
#endif
      PetscCall(ISDestroy(&nis));
      PetscCall(DMSwarmDestroyLocalVectorFromFields(dm, Nfc, coordFields, &pos));

      PetscCall(PetscSFGetGraph(sfcell, NULL, NULL, NULL, &LA_sfcell));
    }
    if (!located_on_device) {
      PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
      PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&p_cellid));
      for (p = 0; p < npoints_from_neighbours; p++) {
        rankval[npoints_prior_migration + p]  = rank;
        p_cellid[npoints_prior_migration + p] = LA_sfcell[p].index;
      }
      PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&p_cellid));
      PetscCall(PetscSFDestroy(&sfcell));
      PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
    }
#if PetscDefined(HAVE_HIP)
    if (located_on_device) {
      DMSwarmDataField rfield;

      /* Cell ids of the received points were written in place in the cell id field's device mirror; the owning rank is recorded by a device fill so the rank field never round-trips through the host */
      PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, DMSwarmField_rank, &rfield));
      PetscCall(DMSwarmDataFieldFillIntRange_HIP(rfield, npoints_prior_migration, npoints_prior_migration + npoints_from_neighbours, (PetscInt)rank));
    }
#endif

    /* remove points which left processor */
    if (!located_on_device) {
      PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, cellid, &PField));
      PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&p_cellid));

      PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL));
      if (remove_sent_points) {
        PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
        for (p = npoints_prior_migration; p < npoints2; p++) {
          if (p_cellid[p] == DMLOCATEPOINT_POINT_NOT_FOUND) {
            /* kill point */
            PetscCall(DMSwarmDataBucketRemovePointAtIndex(swarm->db, p));
            PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL)); /* you need to update npoints as the list size decreases! */
            PetscCall(DMSwarmDataFieldGetEntries(PField, (void **)&p_cellid));      /* update date point in case realloc performed */
            p--;                                                                    /* check replacement point */
          }
        }
        PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
      }
    }
#if PetscDefined(HAVE_HIP)
    if (located_on_device && remove_sent_points) {
      /*
       Device fork of the loop above, restricted to the received tail [npoints_prior_migration,
       npoints2): the send list entries are global point indices, and the compaction of a
       tail-restricted list replays the host loop exactly (no removals can occur below the tail).
      */
      PetscInt    *cellids_d, *sentlist_d = NULL;
      PetscInt     nsent = 0;
      PetscMemType mti   = PETSC_MEMTYPE_HOST;

      PetscCall(PetscLogEventBegin(DMSWARM_RemovePoints, 0, 0, 0, 0));
      PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints2, NULL, NULL));
      PetscCall(DMSwarmGetFieldReadAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
      PetscCall(DMSwarmSentListCreate_HIP(npoints_prior_migration, npoints2, DMLOCATEPOINT_POINT_NOT_FOUND, cellids_d, &nsent, &sentlist_d));
      PetscCall(DMSwarmRestoreFieldReadAndMemType(dm, cellid, NULL, NULL, &mti, (void **)&cellids_d));
      PetscCall(DMSwarmDataBucketCompact(swarm->db, nsent, sentlist_d));
      PetscCall(DMSwarmSentListDestroy_HIP(&sentlist_d));
      PetscCall(PetscLogEventEnd(DMSWARM_RemovePoints, 0, 0, 0, 0));
      PetscCall(PetscInfo(dm, "DMSwarmMigrate: compacted away %" PetscInt_FMT " unlocated received points on the device\n", nsent));
    }
#endif
  }

  /* check for error on removed points */
  if (error_check) {
    PetscCall(DMSwarmGetSize(dm, &npoints2g));
    PetscCheck(npointsg == npoints2g, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Points from the DMSwarm must remain constant during migration (initial %" PetscInt_FMT " - final %" PetscInt_FMT ")", npointsg, npoints2g);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmMigrate_CellDMExact(DM dm, PetscBool remove_sent_points)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Redundant as this assumes points can only be sent to a single rank
*/
PetscErrorCode DMSwarmMigrate_GlobalToLocal_Basic(DM dm, PetscInt *globalsize)
{
  DM_Swarm     *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx de;
  PetscInt      p, npoints, *rankval, n_points_recv;
  PetscMPIInt   rank, nrank, negrank;
  void         *point_buffer, *recv_points;
  size_t        sizeof_dmswarm_point;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  *globalsize = npoints;
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmGetDataExchanger_Private(dm, &de));
  swarm->de_layout_id = 0;
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < npoints; p++) {
    PetscCall(PetscMPIIntCast(rankval[p], &negrank));
    if (negrank < 0) {
      nrank = -negrank - 1;
      PetscCall(DMSwarmDataExTopologyAddNeighbour(de, nrank));
    }
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < npoints; p++) {
    PetscCall(PetscMPIIntCast(rankval[p], &negrank));
    if (negrank < 0) {
      nrank = -negrank - 1;
      PetscCall(DMSwarmDataExAddToSendCount(de, nrank, 1));
    }
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  /* host-only path on every backend; the cached exchanger may have been left in device mode by the migration fork (see data_ex.h) */
  PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HOST));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (p = 0; p < npoints; p++) {
    PetscCall(PetscMPIIntCast(rankval[p], &negrank));
    if (negrank < 0) {
      nrank      = -negrank - 1;
      rankval[p] = nrank;
      /* copy point into buffer */
      PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
      /* insert point buffer into DMSwarmDataExchanger */
      PetscCall(DMSwarmDataExPackData(de, nrank, 1, point_buffer));
      rankval[p] = negrank;
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  PetscCall(DMSwarmDataExView(de));
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscMPIInt owner_rank;
  PetscReal   min[3], max[3];
} CollectBBox;

PETSC_EXTERN PetscErrorCode DMSwarmCollect_DMDABoundingBox(DM dm, PetscInt *globalsize)
{
  DM_Swarm          *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx      de;
  PetscInt           p, pk, npoints, *rankval, n_points_recv, n_bbox_recv, dim, neighbour_cells;
  PetscMPIInt        rank, nrank;
  void              *point_buffer, *recv_points;
  size_t             sizeof_dmswarm_point, sizeof_bbox_ctx;
  PetscBool          isdmda;
  CollectBBox       *bbox, *recv_bbox;
  const PetscMPIInt *dmneighborranks;
  DM                 dmcell;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  PetscCall(DMSwarmGetCellDM(dm, &dmcell));
  PetscCheck(dmcell, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only valid if cell DM provided");
  isdmda = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)dmcell, DMDA, &isdmda));
  PetscCheck(isdmda, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only DMDA support for CollectBoundingBox");

  PetscCall(DMGetDimension(dm, &dim));
  sizeof_bbox_ctx = sizeof(CollectBBox);
  PetscCall(PetscMalloc1(1, &bbox));
  bbox->owner_rank = rank;

  /* compute the bounding box based on the overlapping / stenctil size */
  {
    Vec lcoor;

    PetscCall(DMGetCoordinatesLocal(dmcell, &lcoor));
    if (dim >= 1) {
      PetscCall(VecStrideMin(lcoor, 0, NULL, &bbox->min[0]));
      PetscCall(VecStrideMax(lcoor, 0, NULL, &bbox->max[0]));
    }
    if (dim >= 2) {
      PetscCall(VecStrideMin(lcoor, 1, NULL, &bbox->min[1]));
      PetscCall(VecStrideMax(lcoor, 1, NULL, &bbox->max[1]));
    }
    if (dim == 3) {
      PetscCall(VecStrideMin(lcoor, 2, NULL, &bbox->min[2]));
      PetscCall(VecStrideMax(lcoor, 2, NULL, &bbox->max[2]));
    }
  }
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  *globalsize = npoints;
  PetscCall(DMSwarmGetField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmGetDataExchanger_Private(dm, &de));
  swarm->de_layout_id = 0;
  /* use DMDA neighbours */
  PetscCall(DMDAGetNeighbors(dmcell, &dmneighborranks));
  if (dim == 1) {
    neighbour_cells = 3;
  } else if (dim == 2) {
    neighbour_cells = 9;
  } else {
    neighbour_cells = 27;
  }
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (p = 0; p < neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, dmneighborranks[p]));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (p = 0; p < neighbour_cells; p++) {
    if ((dmneighborranks[p] >= 0) && (dmneighborranks[p] != rank)) PetscCall(DMSwarmDataExAddToSendCount(de, dmneighborranks[p], 1));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  /* send bounding boxes */
  /* host-only path on every backend; the cached exchanger may have been left in device mode by the migration fork (see data_ex.h) */
  PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HOST));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_bbox_ctx));
  for (p = 0; p < neighbour_cells; p++) {
    nrank = dmneighborranks[p];
    if ((nrank >= 0) && (nrank != rank)) {
      /* insert bbox buffer into DMSwarmDataExchanger */
      PetscCall(DMSwarmDataExPackData(de, nrank, 1, bbox));
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  /* recv bounding boxes */
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_bbox_recv, (void **)&recv_bbox));
  /*  Wrong, should not be using PETSC_COMM_WORLD */
  for (p = 0; p < n_bbox_recv; p++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[rank %d]: box from %d : range[%+1.4e,%+1.4e]x[%+1.4e,%+1.4e]\n", rank, recv_bbox[p].owner_rank, (double)recv_bbox[p].min[0], (double)recv_bbox[p].max[0], (double)recv_bbox[p].min[1],
                                      (double)recv_bbox[p].max[1]));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));
  /* of course this is stupid as this "generic" function should have a better way to know what the coordinates are called */
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (pk = 0; pk < n_bbox_recv; pk++) {
    PetscReal *array_x, *array_y;

    PetscCall(DMSwarmGetField(dm, "coorx", NULL, NULL, (void **)&array_x));
    PetscCall(DMSwarmGetField(dm, "coory", NULL, NULL, (void **)&array_y));
    for (p = 0; p < npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) PetscCall(DMSwarmDataExAddToSendCount(de, recv_bbox[pk].owner_rank, 1));
      }
    }
    PetscCall(DMSwarmRestoreField(dm, "coory", NULL, NULL, (void **)&array_y));
    PetscCall(DMSwarmRestoreField(dm, "coorx", NULL, NULL, (void **)&array_x));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HOST));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (pk = 0; pk < n_bbox_recv; pk++) {
    PetscReal *array_x, *array_y;

    PetscCall(DMSwarmGetField(dm, "coorx", NULL, NULL, (void **)&array_x));
    PetscCall(DMSwarmGetField(dm, "coory", NULL, NULL, (void **)&array_y));
    for (p = 0; p < npoints; p++) {
      if ((array_x[p] >= recv_bbox[pk].min[0]) && (array_x[p] <= recv_bbox[pk].max[0])) {
        if ((array_y[p] >= recv_bbox[pk].min[1]) && (array_y[p] <= recv_bbox[pk].max[1])) {
          /* copy point into buffer */
          PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, p, point_buffer));
          /* insert point buffer into DMSwarmDataExchanger */
          PetscCall(DMSwarmDataExPackData(de, recv_bbox[pk].owner_rank, 1, point_buffer));
        }
      }
    }
    PetscCall(DMSwarmRestoreField(dm, "coory", NULL, NULL, (void **)&array_y));
    PetscCall(DMSwarmRestoreField(dm, "coorx", NULL, NULL, (void **)&array_x));
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  PetscCall(DMSwarmRestoreField(dm, DMSwarmField_rank, NULL, NULL, (void **)&rankval));
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(PetscFree(bbox));
  PetscCall(DMSwarmDataExView(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* General collection when no order, or neighbour information is provided */
/*
 User provides context and collect() method
 Broadcast user context

 for each context / rank {
   collect(swarm,context,n,list)
 }
*/
PETSC_EXTERN PetscErrorCode DMSwarmCollect_General(DM dm, PetscErrorCode (*collect)(DM, void *, PetscInt *, PetscInt **), size_t ctx_size, PetscCtx ctx, PetscInt *globalsize)
{
  DM_Swarm     *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataEx de;
  PetscInt      p, npoints, n_points_recv;
  PetscMPIInt   size, rank, len;
  void         *point_buffer, *recv_points;
  void         *ctxlist;
  PetscInt     *n2collect, **collectlist;
  size_t        sizeof_dmswarm_point;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  *globalsize = npoints;
  /* Broadcast user context */
  PetscCall(PetscMPIIntCast(ctx_size, &len));
  PetscCall(PetscMalloc(ctx_size * size, &ctxlist));
  PetscCallMPI(MPI_Allgather(ctx, len, MPI_CHAR, ctxlist, len, MPI_CHAR, PetscObjectComm((PetscObject)dm)));
  PetscCall(PetscMalloc1(size, &n2collect));
  PetscCall(PetscMalloc1(size, &collectlist));
  for (PetscMPIInt r = 0; r < size; r++) {
    PetscInt  _n2collect;
    PetscInt *_collectlist;
    void     *_ctx_r;

    _n2collect   = 0;
    _collectlist = NULL;
    if (r != rank) { /* don't collect data from yourself */
      _ctx_r = (void *)((char *)ctxlist + r * ctx_size);
      PetscCall(collect(dm, _ctx_r, &_n2collect, &_collectlist));
    }
    n2collect[r]   = _n2collect;
    collectlist[r] = _collectlist;
  }
  PetscCall(DMSwarmGetDataExchanger_Private(dm, &de));
  swarm->de_layout_id = 0;
  /* Define topology */
  PetscCall(DMSwarmDataExTopologyInitialize(de));
  for (PetscMPIInt r = 0; r < size; r++) {
    if (n2collect[r] > 0) PetscCall(DMSwarmDataExTopologyAddNeighbour(de, r));
  }
  PetscCall(DMSwarmDataExTopologyFinalize(de));
  /* Define send counts */
  PetscCall(DMSwarmDataExInitializeSendCount(de));
  for (PetscMPIInt r = 0; r < size; r++) {
    if (n2collect[r] > 0) PetscCall(DMSwarmDataExAddToSendCount(de, r, n2collect[r]));
  }
  PetscCall(DMSwarmDataExFinalizeSendCount(de));
  /* Pack data */
  PetscCall(DMSwarmDataBucketCreatePackedArray(swarm->db, &sizeof_dmswarm_point, &point_buffer));
  /* host-only path on every backend; the cached exchanger may have been left in device mode by the migration fork (see data_ex.h) */
  PetscCall(DMSwarmDataExSetMemType(de, PETSC_MEMTYPE_HOST));
  PetscCall(DMSwarmDataExPackInitialize(de, sizeof_dmswarm_point));
  for (PetscMPIInt r = 0; r < size; r++) {
    for (p = 0; p < n2collect[r]; p++) {
      PetscCall(DMSwarmDataBucketFillPackedArray(swarm->db, collectlist[r][p], point_buffer));
      /* insert point buffer into the data exchanger */
      PetscCall(DMSwarmDataExPackData(de, r, 1, point_buffer));
    }
  }
  PetscCall(DMSwarmDataExPackFinalize(de));
  /* Scatter */
  PetscCall(DMSwarmDataExBegin(de));
  PetscCall(DMSwarmDataExEnd(de));
  /* Collect data in DMSwarm container */
  PetscCall(DMSwarmDataExGetRecvData(de, &n_points_recv, &recv_points));
  PetscCall(DMSwarmDataBucketGetSizes(swarm->db, &npoints, NULL, NULL));
  PetscCall(DMSwarmDataBucketSetSizes(swarm->db, npoints + n_points_recv, DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_points_recv; p++) {
    void *data_p = (void *)((char *)recv_points + p * sizeof_dmswarm_point);

    PetscCall(DMSwarmDataBucketInsertPackedArray(swarm->db, npoints + p, data_p));
  }
  /* Release memory */
  for (PetscMPIInt r = 0; r < size; r++) PetscCall(PetscFree(collectlist[r]));
  PetscCall(PetscFree(collectlist));
  PetscCall(PetscFree(n2collect));
  PetscCall(PetscFree(ctxlist));
  PetscCall(DMSwarmDataBucketDestroyPackedArray(swarm->db, &point_buffer));
  PetscCall(DMSwarmDataExView(de));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmGetMigrateType - Get the style of point migration

  Logically Collective

  Input Parameter:
. dm - the `DMSWARM`

  Output Parameter:
. mtype - The migration type, see `DMSwarmMigrateType`

  Level: intermediate

.seealso: `DM`, `DMSWARM`, `DMSwarmMigrateType`, `DMSwarmMigrate()`
@*/
PetscErrorCode DMSwarmGetMigrateType(DM dm, DMSwarmMigrateType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(mtype, 2);
  *mtype = ((DM_Swarm *)dm->data)->migrate_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSetMigrateType - Set the style of point migration

  Logically Collective

  Input Parameters:
+ dm    - the `DMSWARM`
- mtype - The migration type, see `DMSwarmMigrateType`

  Level: intermediate

.seealso: `DM`, `DMSWARM`, `DMSwarmMigrateType`, `DMSwarmGetMigrateType()`, `DMSwarmMigrate()`
@*/
PetscErrorCode DMSwarmSetMigrateType(DM dm, DMSwarmMigrateType mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, mtype, 2);
  ((DM_Swarm *)dm->data)->migrate_type = mtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}
