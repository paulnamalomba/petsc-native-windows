static char help[] = "Tests DMAdaptor pure refinement with no PetscDS fields.\n\n";

#include <petscsnes.h>
#include <petscdmadaptor.h>
#include <petscdmplex.h>

static PetscErrorCode TransferNoOp(DMAdaptor adaptor, DM dm, Vec x, DM adm, Vec ax, PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscCall(VecSet(f, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM           dm, adm = NULL;
  SNES         snes;
  Vec          x, ax = NULL;
  DMAdaptor    adaptor;
  PetscSection section;
  PetscInt     pStart, pEnd, sequence = 3, size, dim;
  PetscBool    criterionNone = PETSC_FALSE, useSnesAdapt = PETSC_FALSE, simplex;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-adapt_sequence", &sequence, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-criterion_none", &criterionNone, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_snes_adapt", &useSnesAdapt, NULL));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  if (useSnesAdapt && !criterionNone) {
    PetscFE fe;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexIsSimplex(dm, &simplex));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DEFAULT, &fe));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&fe));
  } else {
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
    PetscCall(PetscSectionSetChart(section, pStart, pEnd));
    PetscCall(PetscSectionSetUp(section));
    PetscCall(DMSetLocalSection(dm, section));
    PetscCall(PetscSectionDestroy(&section));
  }
  PetscCall(DMCreateGlobalVector(dm, &x));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  if (useSnesAdapt) {
    PetscCall(SNESSetFunction(snes, NULL, FormFunction, NULL));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSolve(snes, NULL, x));
    PetscCall(VecGetSize(x, &size));
    PetscCheck(criterionNone ? size == 0 : size > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Original solution vector has unexpected size %" PetscInt_FMT, size);
    PetscCall(DMGetDimension(dm, &dim));
  } else {
    PetscCall(DMAdaptorCreate(PETSC_COMM_WORLD, &adaptor));
    PetscCall(DMAdaptorSetSolver(adaptor, snes));
    if (!criterionNone) PetscCall(DMAdaptorSetCriterion(adaptor, DM_ADAPTATION_REFINE));
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
  }
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&ax));
  PetscCall(DMDestroy(&adm));
  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: adaptor_refine
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -adapt_sequence 3

  test:
    suffix: adaptor_none
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -adapt_sequence 3 -criterion_none

  test:
    suffix: snes_adapt_refine
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -dm_plex_hash_location -use_snes_adapt -snes_adapt_sequence 3 -adaptor_criterion refine

  test:
    suffix: snes_adapt_initial_refine
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -dm_plex_hash_location -use_snes_adapt -snes_adapt_initial 3 -adaptor_criterion refine

  test:
    suffix: snes_adapt_multigrid_refine
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -dm_plex_hash_location -use_snes_adapt -snes_adapt_multigrid 3 -adaptor_criterion refine

  test:
    suffix: snes_adapt_multigrid_none
    output_file: output/empty.out
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -use_snes_adapt -snes_adapt_multigrid 3 -adaptor_criterion none -criterion_none

TEST*/
