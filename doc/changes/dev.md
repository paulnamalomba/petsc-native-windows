# Changes: Development

```{rubric} petsc4py:
```

- Add `DM.addRefineHook()` and `DM.interpolateHooks()` for callbacks on finer `DM`s, and
  document the optional callbacks to `DM.addRefineHook()` and `DM.addCoarsenHook()`. The
  dispatcher name avoids a collision with `DMPlex.interpolate()`.
