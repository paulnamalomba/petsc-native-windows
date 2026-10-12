# PETSc

PETSc is a C library for parallel numerical computation using MPI. The codebase is primarily C, with Python bindings in `src/binding/petsc4py/`.

Shared rules are below; optional skills provide task procedures through the coding tool's skill system. Respect the user's skill settings, including for references between skills; do not load disabled skills through direct file reads. Read conditional convention references explicitly when relevant.

## Project Layout

- `src/<package>/` - source organized by package: `vec`, `mat`, `ksp`, `snes`, `ts`, `dm`, `tao`, `sys`
- `src/<package>/impls/` - concrete implementations of each package's abstract type
- `src/<package>/interface/` - public API for each package
- `src/<package>/tests/` and `src/<package>/tutorials/` - tests and examples
- `include/` - public headers such as `petsc*.h`
- `include/petsc/private/` - private headers such as `*impl.h`
- `src/binding/petsc4py/` - Python bindings and packaging logic
- `config/` - configure, build, and test harness generation
- `doc/` - user and developer documentation
- `.agents/pde-sim/` - opt-in multi-agent PDE simulation pipeline; in Claude Code run `/pde-sim <phenomenon>` to start it (nothing loads until invoked)

## Core Working Rules

- Keep edits minimal and local to the request. Read nearby code, preserve PETSc style, and reuse or extend existing logic. Do not add speculative abstractions or broad refactors unless requested.
- When several APIs make the same decision, for example which implementation or strategy to select, implement it once in a private helper that also reports what it chose, and call that helper from each API.
- Use PETSc accessors for object metadata instead of reconstructing internal layouts in callers or bindings. If metadata is missing, extend the owning API and document returned data's size, ownership, and lifetime. If only the documentation is unclear, clarify it.
- Fix only what the task requires. Report other improvements you notice, in scripts, makefiles, tools, or documentation, to the user separately with the file path and expected benefit; do not put them in the patch, review findings, or MR comments unless asked.
- For source changes, check test blocks, expected outputs, and documentation; include headers and examples for public interfaces. Run relevant checks and tests.
- Final test changes must exercise successful behavior: do not add expected-error cases, assertions of expected error codes, or `PetscPushErrorHandler()`/`PetscPopErrorHandler()` calls.
- The user selects `PETSC_ARCH`; reuse the value already supplied for the task. Ask if it is missing before configuring or building PETSc or petsc4py, running PETSc-dependent tests or executables, or importing petsc4py; pass it explicitly and never infer or change it unless asked. Architecture-independent checks need no architecture. Targets managing a dedicated default architecture, such as `make docs`, are exempt when using that default.
- A documentation audit or review does not authorize a documentation build; run it only when explicitly requested or approved for the task.
- All changes are expected to arrive through focused, reviewable GitLab merge requests.

## Conditional Conventions

Before working on any of the following, read its convention reference. Load only the references
relevant to the task, including during reviews and when no build or lint check will run.
Reread this file or a convention reference only if it changed or is no longer in context.

| Task | Reference |
| --- | --- |
| New features, API changes, or documentation edits/reviews | [Documentation](.agents/conventions/documentation.md) |
| C or C++ source, and `doc/changes/dev.md` entries | [PETSc Style and Usage Guide](doc/developers/style.md) |
| C API docstrings (`/*@ ... @*/`) | [C docstrings](.agents/conventions/c-docstrings.md) |
| Kokkos code | [Kokkos](.agents/conventions/kokkos.md) |
| petsc4py source or tests | [petsc4py](.agents/conventions/petsc4py.md) |

## Writing PETSc Contribution Materials

Apply these rules to PETSc contribution materials and their drafts: commit messages, MR
descriptions, review reports and comments, documentation, and code comments. They do not govern
unrelated conversations or prescribe the user's conversational style.

- Lead with the result and why it matters. Make the text understandable without the drafting conversation; include relevant verification, limitations, and reproduction details.
- Write clear, grammatical, concise prose for peer mathematicians and engineers. Explain unfamiliar terms, preserve PETSc terminology and exact API names, commands, and diagnostics, and remove repetition and boilerplate before presenting.
- Prefer self-explanatory code. Use comments and docstrings for non-obvious behavior, correctness constraints, rationale, or required documentation. Preserve accurate comments; omit process narration and edit history.

## Key References

External links are for human convenience only; do not assume linked Markdown files will be ingested automatically.

- Development docs: https://petsc.org/main/overview/
- Release docs: https://petsc.org/release/overview/
- GitLab project: https://gitlab.com/petsc/petsc
