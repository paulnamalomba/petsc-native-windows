static char help[] = "Tests that SNESDestroy() clears the KSP operator callback that it set on every level of its DM.\n\n";

#include <petscsnes.h>
#include <petscdmda.h>

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckCleared(DM dm, const char name[])
{
  KSPComputeOperatorsFn *func;
  void                  *ctx;

  PetscFunctionBeginUser;
  PetscCall(DMKSPGetComputeOperators(dm, &func, &ctx));
  PetscCheck(!func && !ctx, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The %s DM still has the operator callback of the destroyed SNES", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM   dm, dmc;
  SNES snes;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetFunction(snes, NULL, FormFunction, NULL));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetUp(snes));

  /* The coarse DM shares the DMKSP of the fine DM until a write to the coarse DM gives it a copy, which keeps the operator callback */
  PetscCall(DMCoarsen(dm, PETSC_COMM_WORLD, &dmc));
  PetscCall(DMSetCoarseDM(dm, dmc));
  PetscCall(DMKSPSetComputeRHS(dmc, ComputeRHS, NULL));

  PetscCall(SNESDestroy(&snes));
  PetscCall(CheckCleared(dm, "fine"));
  PetscCall(CheckCleared(dmc, "coarse"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Operator callbacks cleared on every level\n"));

  PetscCall(DMDestroy(&dmc));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
