# --------------------------------------------------------------------

class PartitionerType(object):
    """The partitioner types."""
    PARMETIS        = S_(PETSCPARTITIONERPARMETIS)
    PTSCOTCH        = S_(PETSCPARTITIONERPTSCOTCH)
    CHACO           = S_(PETSCPARTITIONERCHACO)
    SIMPLE          = S_(PETSCPARTITIONERSIMPLE)
    SHELL           = S_(PETSCPARTITIONERSHELL)
    GATHER          = S_(PETSCPARTITIONERGATHER)
    MATPARTITIONING = S_(PETSCPARTITIONERMATPARTITIONING)
    MULTISTAGE      = S_(PETSCPARTITIONERMULTISTAGE)

# --------------------------------------------------------------------


cdef class Partitioner(Object):
    """A graph partitioner."""

    Type = PartitionerType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.part
        self.part = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the partitioner.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the graph.

        See Also
        --------
        petsc.PetscPartitionerView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(PetscPartitionerView(self.part, vwr))

    def destroy(self) -> Self:
        """Destroy the partitioner object.

        Collective.

        See Also
        --------
        petsc.PetscPartitionerDestroy

        """
        CHKERR(PetscPartitionerDestroy(&self.part))
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty partitioner object.

        Collective.

        The type can be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        setType, petsc.PetscPartitionerCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscPartitioner newpart = NULL
        CHKERR(PetscPartitionerCreate(ccomm, &newpart))
        CHKERR(PetscCLEAR(self.obj)); self.part = newpart
        return self

    def setType(self, part_type: Type | str) -> None:
        """Build a particular type of the partitioner.

        Collective.

        Parameters
        ----------
        part_type
            The kind of partitioner.

        See Also
        --------
        getType, petsc.PetscPartitionerSetType

        """
        cdef PetscPartitionerType cval = NULL
        part_type = str2bytes(part_type, &cval)
        CHKERR(PetscPartitionerSetType(self.part, cval))

    def getType(self) -> str:
        """Return the partitioner type.

        Not collective.

        See Also
        --------
        setType, petsc.PetscPartitionerGetType

        """
        cdef PetscPartitionerType cval = NULL
        CHKERR(PetscPartitionerGetType(self.part, &cval))
        return bytes2str(cval)

    def setFromOptions(self) -> None:
        """Set parameters in the partitioner from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscPartitionerSetFromOptions

        """
        CHKERR(PetscPartitionerSetFromOptions(self.part))

    def setUp(self) -> None:
        """Construct data structures for the partitioner.

        Collective.

        See Also
        --------
        petsc.PetscPartitionerSetUp

        """
        CHKERR(PetscPartitionerSetUp(self.part))

    def reset(self) -> None:
        """Reset data structures of the partitioner.

        Collective.

        See Also
        --------
        petsc.PetscPartitionerReset

        """
        CHKERR(PetscPartitionerReset(self.part))

    def setShellPartition(
        self,
        numProcs: int,
        sizes: Sequence[int] | None = None,
        points: Sequence[int] | None = None) -> None:
        """Set a custom partition for a mesh.

        Collective.

        Parameters
        ----------
        sizes
            The number of points in each partition.
        points
            A permutation of the points that groups those assigned to each
            partition in order (i.e., partition 0 first, partition 1 next,
            etc.).

        See Also
        --------
        petsc.PetscPartitionerShellSetPartition

        """
        cdef PetscInt cnumProcs = asInt(numProcs)
        cdef PetscInt *csizes = NULL
        cdef PetscInt *cpoints = NULL
        cdef PetscInt nsize = 0
        cdef PetscInt npoint = 0
        cdef PetscInt i = 0
        cdef PetscInt64 nrequired = 0
        if sizes is not None:
            sizes = iarray_i(sizes, &nsize, &csizes)
            if nsize != cnumProcs:
                raise ValueError("sizes array should have %d entries (has %d)" %
                                 (toInt(cnumProcs), toInt(nsize)))
            if points is None:
                raise ValueError("Must provide both sizes and points arrays")
        if points is not None:
            points = iarray_i(points, &npoint, &cpoints)
            for i from 0 <= i < nsize:
                if csizes[i] > 0: nrequired += csizes[i]
            if npoint < nrequired:
                raise ValueError(
                    "points array should have at least %d entries (has %d)" %
                    (nrequired, toInt(npoint)))
        CHKERR(PetscPartitionerShellSetPartition(self.part, cnumProcs,
                                                 csizes, cpoints))

    def partition(
        self,
        nparts: int,
        start: Sequence[int],
        adjacency: Sequence[int],
        Section vertexSection=None,
        Section edgeSection=None,
        Section targetSection=None) -> tuple[Section, IS]:
        """Partition a graph.

        Collective.

        Parameters
        ----------
        nparts
            The number of partitions.
        start
            The offsets of the adjacency of each local vertex, of length the
            number of local vertices plus one.
        adjacency
            The global numbers of the vertices adjacent to each local vertex.
        vertexSection
            The weight of each vertex, as its number of degrees of freedom.
        edgeSection
            The weight of each edge, as its number of degrees of freedom.
        targetSection
            The target weight of each partition.

        Returns
        -------
        partSection : Section
            The number of local vertices in each partition.
        partition : IS
            The local vertices, grouped by partition.

        See Also
        --------
        petsc.PetscPartitionerPartition

        """
        cdef PetscInt cnparts = asInt(nparts)
        cdef PetscInt nstart = 0, nadj = 0
        cdef PetscInt *cstart = NULL
        cdef PetscInt *cadj = NULL
        cdef PetscSection vsec = NULL, esec = NULL, tsec = NULL
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef Section partSection = Section()
        cdef IS partition = IS()
        start = iarray_i(start, &nstart, &cstart)
        adjacency = iarray_i(adjacency, &nadj, &cadj)
        if nstart < 1:
            raise ValueError("start array should have at least one entry")
        if nadj < cstart[nstart - 1]:
            raise ValueError("adjacency array should have at least %d entries (has %d)" %
                             (toInt(cstart[nstart - 1]), toInt(nadj)))
        if vertexSection is not None: vsec = vertexSection.sec
        if edgeSection is not None: esec = edgeSection.sec
        if targetSection is not None: tsec = targetSection.sec
        CHKERR(PetscObjectGetComm(<PetscObject>self.part, &comm))
        CHKERR(PetscSectionCreate(comm, &partSection.sec))
        CHKERR(PetscPartitionerPartition(self.part, cnparts, nstart - 1, cstart, cadj,
                                         vsec, esec, tsec, partSection.sec, &partition.iset))
        return partSection, partition

# --------------------------------------------------------------------

del PartitionerType

# --------------------------------------------------------------------
