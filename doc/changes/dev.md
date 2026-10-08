# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

```{rubric} Configure/Build:
```

- Add `--download-cutest` and `--download-sifdecode` for the CUTEst optimization testing environment and SIF problem decoder

- Add Meson package builds, `--download-meson`, and `--download-package-meson-arguments` for additional Meson setup arguments

- Add `--download-ninja` and `--with-ninja-exec` for the Ninja backend used by Meson package builds

- Add `--with-meson-exec` to select an existing Meson executable

```{rubric} Sys:
```

- Remove `PetscDefaultCudaStream` and `PetscDefaultHipStream`; obtain the native stream from a concrete `PetscDeviceContext` with `PetscDeviceContextGetStreamHandle()` instead
- Replace implicit `NULL` contexts with `PetscDeviceContextDefault` in device-context and device-memory APIs to retain default-stream selection, or pass an explicit context. Add `PetscDeviceContextGetDefaultContext()` and petsc4py `DeviceContext.getDefault()` to capture the default context for the current device. Preserve destruction of null handles and the C-only device-interface fallback
- Add an alignment argument before the output pointer to `PetscDeviceMalloc()` and `PetscDeviceCalloc()`; pass `PETSC_DECIDE` to retain inferred alignment, or a positive power of two to request stronger alignment
- Add `PetscDeviceContextDelay()` to queue a timed host callback for testing CUDA and HIP stream ordering

```{rubric} Event Logging:
```

- Change the `CpuToGpu Count` and `GpuToCpu Count` columns of `-log_view` to report the maximum over the processes instead of the average, which rounded to 0 whenever the total number of copies was less than half the number of processes; the size columns remain averages

```{rubric} PetscViewer:
```

```{rubric} PetscDraw:
```

```{rubric} AO:
```

```{rubric} IS:
```

```{rubric} VecScatter / PetscSF:
```

- Prefer the native CUDA/HIP `PetscSF` backend when available so communication follows the current device context; retain Kokkos as a fallback and as an explicit choice with `-sf_backend kokkos`

```{rubric} PF:
```

```{rubric} Vec:
```

- Add a device implementation of `VecSqrtAbs()` for `VECKOKKOS`; previously it copied the vector to the host
- Add `VecGetArrayAndMemTypeAsync()`, `VecGetArrayReadAndMemTypeAsync()`, and `VecGetArrayWriteAndMemTypeAsync()` to acquire vector storage for read/write, read-only, and write-only access, respectively, without waiting for CUDA and HIP transfers on the current device context

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

- Add device implementations of `MatNorm()` with `NORM_1`, `NORM_FROBENIUS`, and `NORM_INFINITY` for `MATAIJKOKKOS`; previously all norms copied the matrix values to the host
- Add `-mat_spd` to set `MAT_SPD` from the options database in `MatSetFromOptions()`
- Change `MatSetOption()` to error when a symmetry-related option contradicts an already known property

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

```{rubric} KSP:
```

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

- Add support for CUTEst unconstrained problems in TAO tutorials and tests
- Map `TAOSNES` convergence and divergence reasons with the public `TaoConvergedReasonFromSNES()`, and add `TaoSNESGetSNES()` to access the underlying solver

```{rubric} TaoTerm:
```

```{rubric} PetscRegressor:
```

```{rubric} PetscDA:
```

```{rubric} DM:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

```{rubric} FE/FV:
```

```{rubric} DMNetwork:
```

```{rubric} DMStag:
```

```{rubric} DT:
```

```{rubric} Fortran:
```
