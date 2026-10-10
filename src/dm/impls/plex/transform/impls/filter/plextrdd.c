#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

/*
  The domain decomposition transformation produces the union of the subdomains of a mesh.

  Each stratum of the active label marks the cells of one subdomain, and a cell may belong to several subdomains. A
  point of the original mesh produces one replica for each subdomain whose cells contain the point in their closure,
  and points outside every subdomain produce nothing. Replica r of point p belongs to the r-th smallest subdomain value
  containing p, so the cone of a replica takes, from each cone point, the replica of the same subdomain.

  The refine type of a point in m subdomains is m * DM_NUM_POLYTOPES + ct. Since the replica numbers in a cone depend
  on the subdomains of each cone point, the cone descriptions are stored for each point rather than for each refine type.
*/

static PetscErrorCode DMPlexTransformView_DD(DMPlexTransform tr, PetscViewer viewer)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;
  PetscBool           isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Domain decomposition transformation %s\n", name ? name : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ignore halo: %s\n", dd->ignoreHalo ? "YES" : "NO"));
  } else {
    const char *type;

    PetscCall(PetscObjectGetType((PetscObject)viewer, &type));
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_DD(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;
  PetscBool           ignoreHalo, flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlexTransform Domain Decomposition Options");
  PetscCall(PetscOptionsBool("-dm_plex_transform_dd_ignore_halo", "Ignore labeled cells in the halo", "DMPlexTransformDDSetIgnoreHalo", dd->ignoreHalo, &ignoreHalo, &flg));
  if (flg) PetscCall(DMPlexTransformDDSetIgnoreHalo(tr, ignoreHalo));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexTransformDDGetSubdomainPoints_Private - Append (q, s) to seg for each point q in the closure of subdomain s

  Not Collective

  Input Parameters:
+ tr    - The `DMPlexTransform`
. s     - The index of the subdomain value
. value - The subdomain value
. ghost - Bit array marking the leaves of the point `PetscSF`, or `NULL`
. mark  - Work array over the cells, holding the last subdomain index which claimed each cell
. last  - Work array over the points, holding the last subdomain index which claimed each point
. cells - Work array over the cells
- seg   - Buffer of (point, subdomain index) pairs

  Note:
  Subdomain indices must be visited in increasing order.
*/
static PetscErrorCode DMPlexTransformDDGetSubdomainPoints_Private(DMPlexTransform tr, PetscInt s, PetscInt value, PetscBT ghost, PetscInt mark[], PetscInt last[], PetscInt cells[], PetscSegBuffer seg)
{
  DM              dm;
  DMLabel         active;
  IS              pointIS;
  const PetscInt *points;
  PetscInt        Np, Nc = 0, cStart, cEnd, pStart;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCall(DMPlexGetChart(dm, &pStart, NULL));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelGetStratumIS(active, value, &pointIS));
  if (pointIS) {
    PetscCall(ISGetLocalSize(pointIS, &Np));
    PetscCall(ISGetIndices(pointIS, &points));
    for (PetscInt i = 0; i < Np; ++i) {
      const PetscInt c = points[i];

      if (c < cStart || c >= cEnd || mark[c - cStart] == s) continue;
      if (ghost && PetscBTLookup(ghost, c - pStart)) continue;
      mark[c - cStart] = s;
      cells[Nc++]      = c;
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  for (PetscInt i = 0; i < Nc; ++i) {
    PetscInt *closure = NULL;
    PetscInt  Ncl;

    PetscCall(DMPlexGetTransitiveClosure(dm, cells[i], PETSC_TRUE, &Ncl, &closure));
    for (PetscInt cl = 0; cl < Ncl * 2; cl += 2) {
      const PetscInt q = closure[cl];
      PetscInt      *pair;

      if (last[q - pStart] == s) continue;
      last[q - pStart] = s;
      PetscCall(PetscSegBufferGetInts(seg, 2, &pair));
      pair[0] = q;
      pair[1] = s;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cells[i], PETSC_TRUE, &Ncl, &closure));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetUp_DD(DMPlexTransform tr)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;
  DM                  dm;
  DMLabel             active;
  IS                  valueIS;
  PetscBT             ghost = NULL;
  PetscSegBuffer      seg;
  const PetscInt     *values;
  PetscInt           *svalues, *mark, *last, *cells, *pairs, *fill;
  PetscCount          Npairs;
  PetscInt            Nv, pStart, pEnd, cStart, cEnd, maxConeSize, maxM = 0, Ncones = 0;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCheck(active, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_WRONG, "Domain decomposition requires an active label marking the subdomain cells");
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
  if (dd->ignoreHalo) {
    PetscSF         sf;
    const PetscInt *leaves;
    PetscInt        Nl;

    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
    PetscCall(PetscBTCreate(pEnd - pStart, &ghost));
    for (PetscInt l = 0; l < PetscMax(Nl, 0); ++l) PetscCall(PetscBTSet(ghost, (leaves ? leaves[l] : l) - pStart));
  }
  // Collect the subdomain values, which are visited in increasing order so that the values of each point come out sorted
  PetscCall(DMLabelGetValueIS(active, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  PetscCall(PetscMalloc1(Nv, &svalues));
  PetscCall(PetscArraycpy(svalues, values, Nv));
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscCall(PetscSortInt(Nv, svalues));
  PetscCall(PetscMalloc3(cEnd - cStart, &mark, pEnd - pStart, &last, cEnd - cStart, &cells));
  for (PetscInt c = 0; c < cEnd - cStart; ++c) mark[c] = -1;
  for (PetscInt p = 0; p < pEnd - pStart; ++p) last[p] = -1;
  PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 2 * (pEnd - pStart) + 2, &seg));
  for (PetscInt s = 0; s < Nv; ++s) PetscCall(DMPlexTransformDDGetSubdomainPoints_Private(tr, s, svalues[s], ghost, mark, last, cells, seg));
  PetscCall(PetscFree3(mark, last, cells));
  PetscCall(PetscBTDestroy(&ghost));
  PetscCall(PetscSegBufferGetSize(seg, &Npairs));
  PetscCall(PetscSegBufferExtractAlloc(seg, &pairs));
  PetscCall(PetscSegBufferDestroy(&seg));
  Npairs /= 2;
  // Store the sorted subdomain values of each point
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &dd->subSec));
  PetscCall(PetscSectionSetChart(dd->subSec, pStart, pEnd));
  for (PetscCount i = 0; i < Npairs; ++i) PetscCall(PetscSectionAddDof(dd->subSec, pairs[2 * i], 1));
  PetscCall(PetscSectionSetUp(dd->subSec));
  PetscCall(PetscMalloc1(Npairs, &dd->subdomains));
  PetscCall(PetscCalloc1(pEnd - pStart, &fill));
  for (PetscCount i = 0; i < Npairs; ++i) {
    const PetscInt q = pairs[2 * i];
    PetscInt       off;

    PetscCall(PetscSectionGetOffset(dd->subSec, q, &off));
    dd->subdomains[off + fill[q - pStart]++] = svalues[pairs[2 * i + 1]];
  }
  PetscCall(PetscFree(fill));
  PetscCall(PetscFree(pairs));
  PetscCall(PetscFree(svalues));
  // Each point produces one replica per subdomain, and its refine type encodes the number of replicas
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "DD Type", &tr->trType));
  PetscCall(PetscMalloc1(pEnd - pStart, &dd->coneOff));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct;
    PetscInt       m;

    PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
    dd->coneOff[p - pStart] = Ncones;
    if (!m) continue;
    PetscCall(DMPlexGetCellType(dm, p, &ct));
    PetscCall(DMLabelSetValue(tr->trType, p, m * DM_NUM_POLYTOPES + ct));
    Ncones += 4 * m * DMPolytopeTypeGetConeSize(ct);
    maxM = PetscMax(maxM, m);
  }
  // Replica r of p takes, from each cone point, the replica in the same subdomain
  PetscCall(PetscMalloc1(Ncones, &dd->cones));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt *cone;
    PetscInt       *pcone = &dd->cones[dd->coneOff[p - pStart]];
    PetscInt        m, off, coneSize;

    PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
    if (!m) continue;
    PetscCall(PetscSectionGetOffset(dd->subSec, p, &off));
    PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
    PetscCall(DMPlexGetCone(dm, p, &cone));
    for (PetscInt r = 0; r < m; ++r) {
      for (PetscInt c = 0; c < coneSize; ++c) {
        DMPolytopeType cct;
        PetscInt       cm, coff, cr;

        PetscCall(DMPlexGetCellType(dm, cone[c], &cct));
        PetscCall(PetscSectionGetDof(dd->subSec, cone[c], &cm));
        PetscCall(PetscSectionGetOffset(dd->subSec, cone[c], &coff));
        PetscCall(PetscFindInt(dd->subdomains[off + r], cm, &dd->subdomains[coff], &cr));
        PetscCheck(cr >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cone point %" PetscInt_FMT " of point %" PetscInt_FMT " is missing from subdomain %" PetscInt_FMT, cone[c], p, dd->subdomains[off + r]);
        *pcone++ = cct;
        *pcone++ = 1;
        *pcone++ = c;
        *pcone++ = cr;
      }
    }
  }
  PetscCall(PetscCalloc1(PetscMax(maxM * maxConeSize, 1), &dd->ornts));
  PetscCall(PetscMalloc1(maxM + 1, &dd->sizes));
  for (PetscInt m = 0; m <= maxM; ++m) dd->sizes[m] = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_DD(DMPlexTransform tr)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscSectionDestroy(&dd->subSec));
  PetscCall(PetscFree(dd->subdomains));
  PetscCall(PetscFree(dd->coneOff));
  PetscCall(PetscFree(dd->cones));
  PetscCall(PetscFree(dd->ornts));
  PetscCall(PetscFree(dd->sizes));
  PetscCall(PetscFree(dd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_DD(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;
  PetscInt            m  = 0, pStart;

  PetscFunctionBeginHot;
  if (dd->subSec && p >= 0) PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
  if (!m) {
    if (rt) *rt = -1;
    *Nt     = 0;
    *target = NULL;
    *size   = NULL;
    *cone   = NULL;
    *ornt   = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscSectionGetChart(dd->subSec, &pStart, NULL));
  if (rt) *rt = m * DM_NUM_POLYTOPES + source;
  *Nt     = 1;
  *target = &dd->types[source];
  *size   = &dd->sizes[m];
  *cone   = &dd->cones[dd->coneOff[p - pStart]];
  *ornt   = dd->ornts;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Find the lowest rank among the gathered replicas [kStart, kEnd) of a point holding a replica in the subdomain value

  Input Parameters:
. ownedOnly - Whether the replica must be in the closure of an owned cell
*/
static PetscErrorCode DMPlexTransformDDGetLowestOwner_Private(PetscSection mSec, PetscInt kStart, PetscInt kEnd, const PetscInt mranks[], const PetscInt mvalues[], const PetscInt mowned[], const PetscInt mnew[], PetscInt value, PetscBool ownedOnly, PetscSFNode *owner)
{
  PetscFunctionBegin;
  owner->rank  = -1;
  owner->index = -1;
  for (PetscInt k = kStart; k < kEnd; ++k) {
    PetscInt kdof, koff;

    if (owner->rank >= 0 && mranks[k] >= owner->rank) continue;
    PetscCall(PetscSectionGetDof(mSec, k, &kdof));
    PetscCall(PetscSectionGetOffset(mSec, k, &koff));
    for (PetscInt j = koff; j < koff + kdof; ++j) {
      if (mvalues[j] != value || (ownedOnly && !mowned[j])) continue;
      owner->rank  = mranks[k];
      owner->index = mnew[j];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDDGetOwner_Private(PetscSection mSec, PetscInt kStart, PetscInt kEnd, const PetscInt mranks[], const PetscInt mvalues[], const PetscInt mowned[], const PetscInt mnew[], PetscInt value, PetscMPIInt rank, PetscInt localPoint, PetscBool localOwned, PetscSFNode *owner)
{
  PetscFunctionBegin;
  if (localOwned) {
    owner->rank  = rank;
    owner->index = localPoint;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMPlexTransformDDGetLowestOwner_Private(mSec, kStart, kEnd, mranks, mvalues, mowned, mnew, value, PETSC_TRUE, owner));
  if (owner->rank < 0) {
    PetscCall(DMPlexTransformDDGetLowestOwner_Private(mSec, kStart, kEnd, mranks, mvalues, mowned, mnew, value, PETSC_FALSE, owner));
    if (localPoint >= 0 && (owner->rank < 0 || rank < owner->rank)) {
      owner->rank  = rank;
      owner->index = localPoint;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The replicas of point q in subdomain s are owned by the rank owning q, if that rank has q in the closure of one of its
  owned cells in s. Otherwise the lowest rank with q in the closure of one of its owned cells in s owns them. If there
  are no such cells, the lowest rank holding the replica owns it. Each rank marks the replicas in the closure of its
  owned cells, gathers, to the points it owns, the subdomain values, marks, and new point numbers of the replicas on
  other ranks, decides the owner of each replica, and returns it.
*/
static PetscErrorCode DMPlexTransformCreateSF_DD(DMPlexTransform tr, DM rdm)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;
  DM                  dm;
  PetscSF             sf, msf, ssf, sfNew;
  PetscSection        mSec;
  PetscBT             ghost;
  const PetscInt     *leaves, *degree;
  PetscSFNode        *mowner, *owner, *remoteNew, *iremote;
  PetscInt           *newPoints, *counts, *ranks, *owned, *mcounts, *mranks, *mvalues, *mowned, *mnew, *moff, *remoteOffsets, *leavesNew, *perm, *ilocal;
  PetscInt            Nr, Nl, Nm, Ns, Nms, NlNew = 0, pStart, pEnd, cStart, cEnd, pStartNew, pEndNew;
  PetscMPIInt         rank;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(DMGetPointSF(rdm, &sfNew));
  PetscCall(PetscSFGetGraph(sf, &Nr, &Nl, &leaves, NULL));
  if (Nr < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetChart(rdm, &pStartNew, &pEndNew));
  PetscCall(PetscSectionGetStorageSize(dd->subSec, &Ns));
  PetscCall(PetscMalloc4(Ns, &newPoints, pEnd - pStart, &counts, pEnd - pStart, &ranks, Ns, &owner));
  PetscCall(PetscCalloc1(Ns, &owned));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct = DM_POLYTOPE_UNKNOWN;
    PetscInt       m, off;

    PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
    PetscCall(PetscSectionGetOffset(dd->subSec, p, &off));
    counts[p - pStart] = m;
    ranks[p - pStart]  = rank;
    if (m) PetscCall(DMPlexGetCellType(dm, p, &ct));
    for (PetscInt r = 0; r < m; ++r) {
      PetscCall(DMPlexTransformGetTargetPoint(tr, ct, ct, p, r, &newPoints[off + r]));
      owner[off + r].rank  = -1;
      owner[off + r].index = -1;
    }
  }
  // Mark the replicas in the closure of an owned cell in their subdomain
  PetscCall(PetscBTCreate(pEnd - pStart, &ghost));
  for (PetscInt l = 0; l < Nl; ++l) PetscCall(PetscBTSet(ghost, (leaves ? leaves[l] : l) - pStart));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  mc, offc, Ncl;

    if (PetscBTLookup(ghost, c - pStart)) continue;
    PetscCall(PetscSectionGetDof(dd->subSec, c, &mc));
    if (!mc) continue;
    PetscCall(PetscSectionGetOffset(dd->subSec, c, &offc));
    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &Ncl, &closure));
    for (PetscInt cl = 0; cl < Ncl; ++cl) {
      const PetscInt q = closure[2 * cl];
      PetscInt       mq, offq;

      PetscCall(PetscSectionGetDof(dd->subSec, q, &mq));
      PetscCall(PetscSectionGetOffset(dd->subSec, q, &offq));
      for (PetscInt r = 0; r < mc; ++r) {
        PetscInt rq;

        PetscCall(PetscFindInt(dd->subdomains[offc + r], mq, &dd->subdomains[offq], &rq));
        if (rq >= 0) owned[offq + rq] = 1;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &Ncl, &closure));
  }
  PetscCall(PetscBTDestroy(&ghost));
  // Gather the number of replicas of each leaf, and its rank, to the root
  PetscCall(PetscSFComputeDegreeBegin(sf, &degree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &degree));
  PetscCall(PetscMalloc1(Nr + 1, &moff));
  moff[0] = 0;
  for (PetscInt q = 0; q < Nr; ++q) moff[q + 1] = moff[q] + degree[q];
  Nm = moff[Nr];
  PetscCall(PetscMalloc2(Nm, &mcounts, Nm, &mranks));
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, counts, mcounts));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, counts, mcounts));
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, ranks, mranks));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, ranks, mranks));
  // Gather the subdomain value, mark, and new point number of each leaf replica to the root
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &mSec));
  PetscCall(PetscSectionSetChart(mSec, 0, Nm));
  for (PetscInt k = 0; k < Nm; ++k) PetscCall(PetscSectionSetDof(mSec, k, mcounts[k]));
  PetscCall(PetscSectionSetUp(mSec));
  PetscCall(PetscSectionGetStorageSize(mSec, &Nms));
  PetscCall(PetscSFGetMultiSF(sf, &msf));
  PetscCall(PetscSFCreateRemoteOffsets(msf, mSec, dd->subSec, &remoteOffsets));
  PetscCall(PetscSFCreateSectionSF(msf, mSec, remoteOffsets, dd->subSec, &ssf));
  PetscCall(PetscFree(remoteOffsets));
  PetscCall(PetscMalloc4(Nms, &mvalues, Nms, &mowned, Nms, &mnew, Nms, &mowner));
  PetscCall(PetscSFReduceBegin(ssf, MPIU_INT, dd->subdomains, mvalues, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(ssf, MPIU_INT, dd->subdomains, mvalues, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(ssf, MPIU_INT, owned, mowned, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(ssf, MPIU_INT, owned, mowned, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(ssf, MPIU_INT, newPoints, mnew, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(ssf, MPIU_INT, newPoints, mnew, MPI_REPLACE));
  // Decide the owner of each replica of the root points, on this rank and on the leaf ranks
  for (PetscInt q = 0; q < Nr; ++q) {
    PetscInt m, off;

    PetscCall(PetscSectionGetDof(dd->subSec, pStart + q, &m));
    PetscCall(PetscSectionGetOffset(dd->subSec, pStart + q, &off));
    for (PetscInt r = 0; r < m; ++r) {
      PetscCall(DMPlexTransformDDGetOwner_Private(mSec, moff[q], moff[q + 1], mranks, mvalues, mowned, mnew, dd->subdomains[off + r], rank, newPoints[off + r], (PetscBool)owned[off + r], &owner[off + r]));
    }
    for (PetscInt k = moff[q]; k < moff[q + 1]; ++k) {
      PetscInt kdof, koff;

      PetscCall(PetscSectionGetDof(mSec, k, &kdof));
      PetscCall(PetscSectionGetOffset(mSec, k, &koff));
      for (PetscInt j = koff; j < koff + kdof; ++j) {
        PetscInt r;

        PetscCall(PetscFindInt(mvalues[j], m, &dd->subdomains[off], &r));
        PetscCall(DMPlexTransformDDGetOwner_Private(mSec, moff[q], moff[q + 1], mranks, mvalues, mowned, mnew, mvalues[j], rank, r >= 0 ? newPoints[off + r] : -1, (PetscBool)(r >= 0 && owned[off + r]), &mowner[j]));
      }
    }
  }
  PetscCall(PetscSFBcastBegin(ssf, MPIU_SF_NODE, mowner, owner, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(ssf, MPIU_SF_NODE, mowner, owner, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&ssf));
  PetscCall(PetscSectionDestroy(&mSec));
  PetscCall(PetscFree4(mvalues, mowned, mnew, mowner));
  PetscCall(PetscFree2(mcounts, mranks));
  PetscCall(PetscFree(moff));
  // Replicas owned by another rank become leaves, sorted by their local number
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt m, off;

    PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
    PetscCall(PetscSectionGetOffset(dd->subSec, p, &off));
    for (PetscInt r = 0; r < m; ++r)
      if (owner[off + r].rank != rank) ++NlNew;
  }
  PetscCall(PetscMalloc3(NlNew, &leavesNew, NlNew, &remoteNew, NlNew, &perm));
  NlNew = 0;
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt m, off;

    PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
    PetscCall(PetscSectionGetOffset(dd->subSec, p, &off));
    for (PetscInt r = 0; r < m; ++r) {
      if (owner[off + r].rank == rank) continue;
      PetscCheck(owner[off + r].rank >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No owner for replica %" PetscInt_FMT " of point %" PetscInt_FMT, r, p);
      leavesNew[NlNew] = newPoints[off + r];
      remoteNew[NlNew] = owner[off + r];
      perm[NlNew]      = NlNew;
      ++NlNew;
    }
  }
  PetscCall(PetscSortIntWithPermutation(NlNew, leavesNew, perm));
  PetscCall(PetscMalloc1(NlNew, &ilocal));
  PetscCall(PetscMalloc1(NlNew, &iremote));
  for (PetscInt l = 0; l < NlNew; ++l) {
    ilocal[l]  = leavesNew[perm[l]];
    iremote[l] = remoteNew[perm[l]];
  }
  PetscCall(PetscSFSetGraph(sfNew, pEndNew - pStartNew, NlNew, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscFree3(leavesNew, remoteNew, perm));
  PetscCall(PetscFree4(newPoints, counts, ranks, owner));
  PetscCall(PetscFree(owned));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Mark each point of the transformed mesh with its subdomain, in the label named as the active label */
static PetscErrorCode DMPlexTransformCreateLabels_DD(DMPlexTransform tr, DM rdm)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;
  DM                  dm;
  DMLabel             active, label;
  const char         *name;
  PetscInt            pStart, pEnd;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCall(PetscObjectGetName((PetscObject)active, &name));
  PetscCall(DMCreateLabel(rdm, name));
  PetscCall(DMGetLabel(rdm, name, &label));
  PetscCall(DMLabelReset(label));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct;
    PetscInt       m, off;

    PetscCall(PetscSectionGetDof(dd->subSec, p, &m));
    if (!m) continue;
    PetscCall(PetscSectionGetOffset(dd->subSec, p, &off));
    PetscCall(DMPlexGetCellType(dm, p, &ct));
    for (PetscInt r = 0; r < m; ++r) {
      PetscInt pNew;

      PetscCall(DMPlexTransformGetTargetPoint(tr, ct, ct, p, r, &pNew));
      PetscCall(DMLabelSetValue(label, pNew, dd->subdomains[off + r]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetUpReferenceCell_DD(DMPlexTransform tr)
{
  DMLabel  active;
  DM       dm;
  PetscInt cStart, cEnd;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMCreateLabel(dm, "subdomain"));
  PetscCall(DMGetLabel(dm, "subdomain", &active));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) PetscCall(DMLabelSetValue(active, c, 0));
  PetscCall(DMPlexTransformSetActive(tr, active));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformMapLocalizedCoordinates_DD(DMPlexTransform tr, DMPolytopeType ct, DMPolytopeType rct, PetscInt r, const PetscScalar x[], PetscScalar xr[])
{
  DM       dm;
  PetscInt cdim;

  PetscFunctionBegin;
  PetscCheck(ct == rct, PetscObjectComm((PetscObject)tr), PETSC_ERR_PLIB, "Domain decomposition must preserve the cell type");
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(PetscArraycpy(xr, x, cdim * DMPolytopeTypeGetNumVertices(ct)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_DD(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                    = DMPlexTransformView_DD;
  tr->ops->setfromoptions          = DMPlexTransformSetFromOptions_DD;
  tr->ops->setup                   = DMPlexTransformSetUp_DD;
  tr->ops->setuprefcell            = DMPlexTransformSetUpReferenceCell_DD;
  tr->ops->destroy                 = DMPlexTransformDestroy_DD;
  tr->ops->setdimensions           = DMPlexTransformSetDimensions_Internal;
  tr->ops->celltransform           = DMPlexTransformCellTransform_DD;
  tr->ops->createsf                = DMPlexTransformCreateSF_DD;
  tr->ops->createlabels            = DMPlexTransformCreateLabels_DD;
  tr->ops->getsubcellorientation   = DMPlexTransformGetSubcellOrientationIdentity;
  tr->ops->mapcoordinates          = DMPlexTransformMapCoordinatesBarycenter_Internal;
  tr->ops->maplocalizedcoordinates = DMPlexTransformMapLocalizedCoordinates_DD;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  DMPLEXTRANSFORMDD - Transform producing the union of the possibly overlapping subdomains of a mesh

  Options Database Keys:
+ -dm_plex_transform_active name                 - The label whose strata mark the subdomain cells
- -dm_plex_transform_dd_ignore_halo (true|false) - Ignore labeled cells in the halo

  Level: intermediate

  Notes:
  Each stratum of the active label, set with `DMPlexTransformSetActive()`, marks the cells of one subdomain, and a cell
  may belong to several subdomains, for instance after growing the strata with `DMPlexLabelAddOverlap()`. Each point of the original mesh produces one replica for each subdomain whose
  cells contain it in their closure, and points outside every subdomain produce nothing. Replica r of a point belongs
  to its r-th smallest subdomain value, so `DMPlexTransformGetSourcePoint()` gives both the parent point and,
  with the label below, the subdomain of each new point.

  The transformed mesh has a label with the name of the active label, which marks every point, of every dimension,
  with its subdomain value.

  In parallel, the subdomain values are global. The replicas of point p in subdomain s are owned by the process owning
  p, if p is in the closure of one of its owned cells in s. Otherwise the lowest process with p in the closure of one of
  its owned cells in s owns them. If no process has an owned cell in s containing p, the lowest process holding the
  replica owns it. Thus every owned point is in the closure of an owned cell in the transformed mesh. In particular, if
  each subdomain only contains cells owned by one process, and no labeled cell is in the halo, then the point `PetscSF`
  of the transformed mesh is empty.

.seealso: [](plex_transform_table), `DMPlexTransform`, `DMPlexTransformType`, `DMPlexTransformSetActive()`, `DMPlexLabelAddOverlap()`,
          `DMPlexTransformDDSetIgnoreHalo()`, `DMPLEXTRANSFORMFILTER`, `DMPlexFilter()`
M*/
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_DD(DMPlexTransform tr)
{
  DMPlexTransform_DD *dd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&dd));
  for (PetscInt ct = 0; ct < DM_NUM_POLYTOPES; ++ct) dd->types[ct] = (DMPolytopeType)ct;
  tr->redFactor = 1.0;
  tr->data      = dd;
  PetscCall(DMPlexTransformInitialize_DD(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformDDGetIgnoreHalo - Get the flag to ignore labeled cells in the halo

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. ignoreHalo - The flag to ignore labeled cells which are leaves of the point `PetscSF`

  Level: intermediate

.seealso: [](plex_transform_table), `DMPlexTransform`, `DMPLEXTRANSFORMDD`, `DMPlexTransformDDSetIgnoreHalo()`
@*/
PetscErrorCode DMPlexTransformDDGetIgnoreHalo(DMPlexTransform tr, PetscBool *ignoreHalo)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(ignoreHalo, 2);
  *ignoreHalo = dd->ignoreHalo;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformDDSetIgnoreHalo - Set the flag to ignore labeled cells in the halo

  Not Collective

  Input Parameters:
+ tr         - The `DMPlexTransform`
- ignoreHalo - The flag to ignore labeled cells which are leaves of the point `PetscSF`, `PETSC_FALSE` by default

  Options Database Key:
. -dm_plex_transform_dd_ignore_halo (true|false) - Ignore labeled cells in the halo

  Level: intermediate

  Note:
  When the flag is `PETSC_FALSE`, a labeled cell in the halo produces a replica, which is shared with the other
  processes holding a replica of the same cell in the same subdomain, as described in `DMPLEXTRANSFORMDD`.

.seealso: [](plex_transform_table), `DMPlexTransform`, `DMPLEXTRANSFORMDD`, `DMPlexTransformDDGetIgnoreHalo()`, `DMPlexFilter()`
@*/
PetscErrorCode DMPlexTransformDDSetIgnoreHalo(DMPlexTransform tr, PetscBool ignoreHalo)
{
  DMPlexTransform_DD *dd = (DMPlexTransform_DD *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  dd->ignoreHalo = ignoreHalo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
