static char help[] = "Pushes DMSwarm particles with a user HIP kernel through the DMSwarmGetFieldAndMemType() accessors.\n\n";

/*
  The device accessor idiom for DMSWARM fields.

  DMSwarmGetFieldAndMemType() hands back the field array in the memory space native to the
  storage backend (-dm_swarm_backend host|hip) together with its PetscMemType. User code
  branches on the memtype: a host pointer is used in a plain loop, a device pointer is handed to
  a kernel. Nothing else changes between the two backends: the same swarm setup, the same
  restore call, the same DMSwarmMigrate(). Under the hip backend the pushed fields stay
  device-resident across push -> migrate -> push, so the time loop moves no field data between
  host and device (verify with -log_view: DMSwarmCopyToGPU/DMSwarmCopyFromGPU are independent of
  -nsteps).

  Each step is an explicit Euler push x += dt v with reflection at the box walls, so no particle
  ever leaves the domain (the example checks that after every migration); the velocity field is
  set once on the host from the initial coordinates. The final coordinate and velocity fields are
  reported with order-independent statistics, so the output is identical for both backends and
  any number of ranks.
*/

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscdevice_hip.h>

typedef struct {
  PetscInt  dim;
  PetscReal dt;
  PetscReal lower[3];
  PetscReal upper[3];
} PushCtx;

/*
  Push one particle. Shared by the host loop and the device kernel so that both backends perform
  exactly the same floating-point operations; contraction is disabled because the device compiler
  would otherwise fuse the multiply-add and the results would differ from the host in the last bit.
  Point location in DMPlex treats the upper walls as open, so a particle landing exactly on one is
  reflected as well.
*/
__host__ __device__ static void PushPoint(const PushCtx *ctx, PetscReal x[], PetscReal v[])
{
#pragma clang fp contract(off)
  for (PetscInt d = 0; d < ctx->dim; ++d) {
    x[d] += ctx->dt * v[d];
    if (x[d] < ctx->lower[d]) {
      x[d] = 2 * ctx->lower[d] - x[d];
      v[d] = -v[d];
    } else if (x[d] >= ctx->upper[d]) {
      x[d] = 2 * ctx->upper[d] - x[d];
      v[d] = -v[d];
    }
  }
}

// One thread per particle
__global__ static void PushKernel(PushCtx ctx, PetscInt Np, PetscReal x[], PetscReal v[])
{
  const PetscInt p = blockIdx.x * blockDim.x + threadIdx.x;

  if (p < Np) PushPoint(&ctx, &x[p * ctx.dim], &v[p * ctx.dim]);
}

/* The idiom: acquire both fields in native memory, dispatch on the memtype, restore */
static PetscErrorCode PushParticles(DM sw, const PushCtx *ctx)
{
  PetscReal   *x, *v;
  PetscMemType mtype, vmtype;
  PetscInt     Np;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetFieldAndMemType(sw, DMSwarmPICField_coor, NULL, NULL, &mtype, (void **)&x));
  PetscCall(DMSwarmGetFieldAndMemType(sw, "velocity", NULL, NULL, &vmtype, (void **)&v));
  PetscCheck(mtype == vmtype, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Fields of one swarm live in different memory spaces: %s and %s", PetscMemTypeToString(mtype), PetscMemTypeToString(vmtype));
  if (PetscMemTypeHost(mtype)) {
    for (PetscInt p = 0; p < Np; ++p) PushPoint(ctx, &x[p * ctx->dim], &v[p * ctx->dim]);
  } else {
    const unsigned int nthreads = 256;

    PetscCheck(PetscMemTypeHIP(mtype), PETSC_COMM_SELF, PETSC_ERR_SUP, "This example only has a HIP kernel, the swarm fields are in %s memory", PetscMemTypeToString(mtype));
    /* PetscDefaultHipStream is the stream on which the accessor refreshed the device copy, so the kernel is ordered after it; the synchronize makes the results visible to any later host access */
    if (Np > 0) hipLaunchKernelGGL(PushKernel, dim3(((unsigned int)Np + nthreads - 1) / nthreads), dim3(nthreads), 0, PetscDefaultHipStream, *ctx, Np, x, v);
    PetscCallHIP(hipGetLastError());
    PetscCallHIP(hipStreamSynchronize(PetscDefaultHipStream));
  }
  PetscCall(PetscInfo(sw, "Pushed %" PetscInt_FMT " points in %s memory\n", Np, PetscMemTypeToString(mtype)));
  PetscCall(DMSwarmRestoreFieldAndMemType(sw, "velocity", NULL, NULL, &vmtype, (void **)&v));
  PetscCall(DMSwarmRestoreFieldAndMemType(sw, DMSwarmPICField_coor, NULL, NULL, &mtype, (void **)&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Velocity from the initial position: v_d = 2 (x_{d+1} - c_{d+1}) with c the box center, nonzero almost everywhere so particles cross cell and rank boundaries */
static PetscErrorCode InitializeVelocity(DM sw, const PushCtx *ctx)
{
  PetscReal *x, *v;
  PetscInt   Np;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < ctx->dim; ++d) {
      const PetscInt e = (d + 1) % ctx->dim;

      v[p * ctx->dim + d] = 2 * (x[p * ctx->dim + e] - 0.5 * (ctx->lower[e] + ctx->upper[e]));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Order-independent summary of a field: per-component min, max and mean (the mean through an
  exactly summed fixed-point value) plus a checksum over the bit patterns of every entry, which
  changes if any entry differs in any bit
*/
static PetscErrorCode ReportField(DM sw, const char name[], PetscInt dim)
{
  const PetscReal  scale = 1099511627776.0; // 2^40, fixed-point resolution of the mean
  const PetscReal *a;
  PetscReal        lmin[3], lmax[3], gmin[3], gmax[3];
  PetscInt64       lsum[4] = {0, 0, 0, 0}, gsum[4];
  PetscInt         Np, N;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetSize(sw, &N));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, name, NULL, NULL, (void **)&a));
  for (PetscInt d = 0; d < dim; ++d) {
    lmin[d] = PETSC_MAX_REAL;
    lmax[d] = PETSC_MIN_REAL;
  }
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscReal val = a[p * dim + d];
      PetscInt64      bits;

      PetscCall(PetscMemcpy(&bits, &val, sizeof(bits)));
      lmin[d] = PetscMin(lmin[d], val);
      lmax[d] = PetscMax(lmax[d], val);
      lsum[d] += (PetscInt64)PetscRintReal(val * scale);
      lsum[3] += bits % 4294967291; // largest 32-bit prime, keeps the sum far from overflow
    }
  }
  PetscCall(DMSwarmRestoreField(sw, name, NULL, NULL, (void **)&a));
  PetscCallMPI(MPIU_Allreduce(lmin, gmin, dim, MPIU_REAL, MPIU_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(lmax, gmax, dim, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(lsum, gsum, 4, MPIU_INT64, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s:\n", name));
  for (PetscInt d = 0; d < dim; ++d) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  component %" PetscInt_FMT ": min %.8f max %.8f mean %.8f\n", d, (double)gmin[d], (double)gmax[d], (double)(gsum[d] / scale / N)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  checksum %" PetscInt64_FMT "\n", gsum[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       dm, sw;
  PushCtx  ctx;
  PetscInt nsteps = 10, N, N0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  ctx.dt = 0.03;
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Particle push options", "DMSWARM");
  PetscCall(PetscOptionsInt("-nsteps", "Number of push steps", NULL, nsteps, &nsteps, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step of the push", NULL, ctx.dt, &ctx.dt, NULL));
  PetscOptionsEnd();

  /* The cell DM is a box mesh configured entirely from the options (-dm_plex_dim, -dm_plex_simplex, -dm_plex_box_faces, ...) */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &ctx.dim));
  PetscCall(DMGetBoundingBox(dm, ctx.lower, ctx.upper));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &sw));
  PetscCall(DMSetType(sw, DMSWARM));
  PetscCall(DMSetDimension(sw, ctx.dim));
  PetscCall(DMSwarmSetType(sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "velocity", ctx.dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(sw));
  /* Options must be consumed before DMSwarmInsertPointsUsingCellDM() triggers DMSetUp(), so that
     -dm_swarm_backend can still be honored (it is a hard error after setup) */
  PetscCall(DMSetFromOptions(sw));
  PetscCall(DMSwarmInsertPointsUsingCellDM(sw, DMSWARMPIC_LAYOUT_GAUSS, 2));
  PetscCall(InitializeVelocity(sw, &ctx));
  PetscCall(DMViewFromOptions(sw, NULL, "-sw_view"));

  PetscCall(DMSwarmGetSize(sw, &N0));
  for (PetscInt step = 0; step < nsteps; ++step) {
    PetscCall(PushParticles(sw, &ctx));
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmGetSize(sw, &N));
    PetscCheck(N == N0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Step %" PetscInt_FMT ": %" PetscInt_FMT " of %" PetscInt_FMT " particles were lost in the migration", step, N0 - N, N0);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " particles after %" PetscInt_FMT " steps of dt %g\n", N, nsteps, (double)ctx.dt));
  PetscCall(ReportField(sw, DMSwarmPICField_coor, ctx.dim));
  PetscCall(ReportField(sw, "velocity", ctx.dim));

  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hip double !complex

  # The host backend and the hip backend (device push kernel + device migration) share one output
  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 8,8 -nsteps 20 -dt 0.03
    output_file: output/ex14_quad.out

    test:
      suffix: quad
    test:
      suffix: quad_backend_hip
      args: -dm_swarm_backend hip
    test:
      suffix: quad_2
      nsize: 2
      args: -petscpartitioner_type simple
    test:
      suffix: quad_2_backend_hip
      nsize: 2
      args: -petscpartitioner_type simple -dm_swarm_backend hip

  testset:
    args: -dm_plex_box_faces 6,6 -nsteps 20 -dt 0.03
    output_file: output/ex14_tri.out

    test:
      suffix: tri
    test:
      suffix: tri_backend_hip
      args: -dm_swarm_backend hip
    test:
      suffix: tri_2
      nsize: 2
      args: -petscpartitioner_type simple
    test:
      suffix: tri_2_backend_hip
      nsize: 2
      args: -petscpartitioner_type simple -dm_swarm_backend hip

TEST*/
