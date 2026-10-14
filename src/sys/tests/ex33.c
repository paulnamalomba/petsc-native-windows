static char help[] = "Tests bounded stack storage and unwinding.\n\n";

#include <petsc/private/petscimpl.h>

static PetscErrorCode CheckStack(int depth, int remaining, PetscBool check)
{
  PetscStack copy = {0};

  PetscFunctionBeginUser;
  PetscCheck(petscstack.currentsize == depth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected stack depth %d, got %d", depth, petscstack.currentsize);
  PetscCheck(petscstack.check == check, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack checking policy");
  PetscStackUpdateLine;
  if (depth <= PETSCSTACKSIZE) PetscCheck(petscstack.line[depth - 1] == __LINE__ - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack line number");
  PetscCall(PetscStackCopy(&petscstack, &copy));
  PetscCheck(copy.currentsize == depth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack copy depth");
  for (int i = 0; i < PetscMin(depth, PETSCSTACKSIZE); ++i) {
    PetscCheck(copy.function[i] == petscstack.function[i] && copy.file[i] == petscstack.file[i] && copy.line[i] == petscstack.line[i] && copy.petscroutine[i] == petscstack.petscroutine[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect copied frame %d", i);
  }
  if (remaining) PetscCall(CheckStack(depth + 1, remaining - 1, check));
  else {
    PetscStack *stack;
    FILE       *file;
    char        line[1024];
    char       *ptr;
    int         frames = 0;

    file = tmpfile();
    PetscCheck(file, PETSC_COMM_SELF, PETSC_ERR_SYS, "Cannot create temporary stack output file");
    PetscCall(PetscStackView(file));
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
      if (line[0] == '[') ++frames;
    }
    PetscCheck(frames == PetscMin(depth, PETSCSTACKSIZE), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect number of viewed frames: %d", frames);
    PetscCall(PetscMalloc1(8, &ptr));
    PetscCall(PetscMallocGetStack(ptr, &stack));
    PetscCheck(stack && stack->currentsize == depth + 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect allocation stack depth");
    PetscCall(PetscRealloc(16, &ptr));
    PetscCall(PetscMallocGetStack(ptr, &stack));
    PetscCheck(stack && stack->currentsize == depth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect reallocation stack depth");
    PetscCall(PetscMallocDump(file));
    PetscCall(PetscFree(ptr));
    PetscCallExternal(fclose, file);
  }
  PetscStackPushExternal("external");
  PetscCheck(petscstack.currentsize == depth + 1 && petscstack.hotdepth == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect external stack frame");
  PetscStackPop;
  PetscCheck(petscstack.currentsize == depth && !petscstack.hotdepth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack was not restored");
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscBool check    = PETSC_TRUE;
  const int depths[] = {2, PETSCSTACKSIZE - 1, PETSCSTACKSIZE, PETSCSTACKSIZE + 1, 2 * PETSCSTACKSIZE};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-checkstack", &check, NULL));
  {
    PetscStack stack = {0};

    PetscStackPop_Private(stack, PETSC_FUNCTION_NAME);
    PetscStackPop_Private(stack, PETSC_FUNCTION_NAME);
    PetscCheck(!stack.currentsize && !stack.hotdepth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Empty stack was not preserved");
    PetscStackPush_Private(stack, __FILE__, PETSC_FUNCTION_NAME, __LINE__, 1, PETSC_TRUE);
    PetscCheck(stack.currentsize == 1 && stack.hotdepth == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack could not be reused after an empty pop");
    PetscStackPop_Private(stack, PETSC_FUNCTION_NAME);
    PetscCheck(!stack.currentsize && !stack.hotdepth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack was not restored");
  }
  for (int i = 0; i < (int)PETSC_STATIC_ARRAY_LENGTH(depths); ++i) PetscCall(CheckStack(2, depths[i] - 2, check));
  PetscCheck(petscstack.currentsize == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Main stack was not restored");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: debug !threadsafety

  test:
    suffix: serial
    requires: !threadsafety
    args: -checkstack {{0 1}}
    output_file: output/empty.out

TEST*/
