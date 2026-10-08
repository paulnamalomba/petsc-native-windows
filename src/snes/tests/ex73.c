static char help[] = "Tests DMAdaptor pure refinement with no PetscDS fields.\n\n";

#include <petscsnes.h>
#include <petscdmadaptor.h>
#include <petscdmplex.h>
#include <petscdmshell.h>

static PetscErrorCode TransferNoOp(DMAdaptor adaptor, DM dm, Vec x, DM adm, Vec ax, PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ZeroResidual(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(f, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateInterpolation(DM dm, DM refined, Mat *interp, Vec *scale)
{
  PetscMPIInt size;
  MPI_Comm    comm = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatCreateAIJ(comm, 1, 1, size, size, 1, NULL, 0, NULL, interp));
  PetscCall(MatAssemblyBegin(*interp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*interp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(*interp, 1.0));
  if (scale) *scale = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RefineShell(DM dm, MPI_Comm comm, DM *refined)
{
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(DMShellCreate(comm, refined));
  PetscCall(VecCreateMPI(comm, 1, PETSC_DETERMINE, &x));
  PetscCall(DMShellSetGlobalVector(*refined, x));
  PetscCall(VecDestroy(&x));
  PetscCall(DMShellSetRefine(*refined, RefineShell));
  PetscCall(DMShellSetCreateInterpolation(*refined, CreateInterpolation));
  PetscCall(DMSNESSetFunction(*refined, ZeroResidual, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSNESSolve(void)
{
  DM       dm;
  SNES     snes;
  Vec      x;
  PetscInt refct;

  PetscFunctionBeginUser;
  PetscCall(RefineShell(NULL, PETSC_COMM_WORLD, &dm));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecSet(x, 0.0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetFromOptions(snes));
  // Keep a guard reference so the ownership assertion can report a failure.
  PetscCall(PetscObjectReference((PetscObject)x));
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscObjectGetReference((PetscObject)x, &refct));
  PetscCheck(refct == 2, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "SNESSolve() consumed the caller's vector reference: %" PetscInt_FMT, refct);
  PetscCall(PetscObjectDereference((PetscObject)x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDMAdaptor(void)
{
  DM           dm, adm = NULL;
  SNES         snes;
  Vec          x, ax = NULL;
  DMAdaptor    adaptor;
  PetscSection section;
  PetscInt     pStart, pEnd, sequence = 3;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-adapt_sequence", &sequence, NULL));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMCreateGlobalVector(dm, &x));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMAdaptorCreate(PETSC_COMM_WORLD, &adaptor));
  PetscCall(DMAdaptorSetSolver(adaptor, snes));
  PetscCall(DMAdaptorSetCriterion(adaptor, DM_ADAPTATION_REFINE));
  PetscCall(DMAdaptorSetTransferFunction(adaptor, TransferNoOp));
  PetscCall(DMAdaptorSetSequenceLength(adaptor, sequence));
  PetscCall(DMAdaptorSetUp(adaptor));
  /* DMAdaptorAdapt() consumes one reference to the solution and to the DM of the solver */
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(PetscObjectReference((PetscObject)x));
  PetscCall(DMAdaptorAdapt(adaptor, x, DM_ADAPTATION_INITIAL, &adm, &ax));

  PetscCheck(adm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMAdaptorAdapt() returned a NULL adapted DM");
  PetscCheck(ax, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMAdaptorAdapt() returned a NULL adapted vector");

  PetscCall(DMAdaptorDestroy(&adaptor));
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&ax));
  PetscCall(DMDestroy(&adm));
  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscBool testSNES = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_snes", &testSNES, NULL));
  if (testSNES) PetscCall(TestSNESSolve());
  else PetscCall(TestDMAdaptor());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: adaptor_refine
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -adapt_sequence 3

  test:
    suffix: snes_refine
    nsize: {{1 3}}
    output_file: output/empty.out
    args: -test_snes -adaptor_criterion refine -snes_adapt_sequence {{1 3}}

  test:
    suffix: snes_initial
    nsize: {{1 3}}
    output_file: output/empty.out
    args: -test_snes -adaptor_criterion refine -snes_adapt_initial {{1 3}}

  test:
    suffix: snes_initial_sequence
    nsize: {{1 3}}
    output_file: output/empty.out
    args: -test_snes -adaptor_criterion refine -snes_adapt_initial 1 -snes_adapt_sequence 1

TEST*/
