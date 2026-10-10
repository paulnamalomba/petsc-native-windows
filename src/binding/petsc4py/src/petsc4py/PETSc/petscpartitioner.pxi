cdef extern from * nogil:

    ctypedef const char* PetscPartitionerType
    PetscPartitionerType PETSCPARTITIONERPARMETIS
    PetscPartitionerType PETSCPARTITIONERPTSCOTCH
    PetscPartitionerType PETSCPARTITIONERCHACO
    PetscPartitionerType PETSCPARTITIONERSIMPLE
    PetscPartitionerType PETSCPARTITIONERSHELL
    PetscPartitionerType PETSCPARTITIONERGATHER
    PetscPartitionerType PETSCPARTITIONERMATPARTITIONING
    PetscPartitionerType PETSCPARTITIONERMULTISTAGE

    PetscErrorCode PetscPartitionerCreate(MPI_Comm, PetscPartitioner*)
    PetscErrorCode PetscPartitionerDestroy(PetscPartitioner*)
    PetscErrorCode PetscPartitionerView(PetscPartitioner, PetscViewer)
    PetscErrorCode PetscPartitionerSetType(PetscPartitioner, PetscPartitionerType)
    PetscErrorCode PetscPartitionerGetType(PetscPartitioner, PetscPartitionerType*)
    PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner)
    PetscErrorCode PetscPartitionerSetUp(PetscPartitioner)
    PetscErrorCode PetscPartitionerReset(PetscPartitioner)

    PetscErrorCode PetscPartitionerPartition(PetscPartitioner, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscSection, PetscSection, PetscSection, PetscSection, PetscIS*)
    PetscErrorCode PetscPartitionerShellSetPartition(PetscPartitioner, PetscInt, PetscInt*, PetscInt*)
