# PetscSF: NVSHMEM vs GPU-aware MPI on Janus (H100, single node)

Measured 2026-08-14 on `x2003c0s9b0n0` (4x H100, NVLink `NV6` all-pairs, node vetted clean),
`PETSC_ARCH=arch-janus-nvshmem`, `--with-debugging=0`. Vehicle: `src/snes/tutorials/ex19`
(lid-driven cavity, DMDA halo exchange), `-dm_vec_type cuda -dm_mat_type aijcusparse
-pc_type jacobi -lidvelocity 10 -grashof 100`, 1 MPI rank per GPU.

Baseline arm is `-use_nvshmem 0`, which is **GPU-aware MPI** (`use_gpu_aware_mpi` defaults to
`PETSC_TRUE`, `src/sys/objects/init.c:37`) over HPC-X Open MPI 4.1.7a1. Both arms use the same
binary, same `UCX_TLS`, best-of-N runs.

## Headline

**GPU-aware MPI is faster than NVSHMEM for this workload at these sizes.** NVSHMEM has a lower
marginal cost per exchange but pays a large fixed startup cost that this workload never amortizes.

| quantity | GPU-aware MPI | NVSHMEM | note |
| --- | --- | --- | --- |
| marginal cost per halo exchange | 25.85 us | **19.79 us** | NVSHMEM 23% cheaper |
| fixed one-time overhead | ~0 | **~0.385 s** | NVSHMEM init |
| break-even | | **~64,000 exchanges** | 0.385 s / 6.06 us |

This workload performs 2.5k-17k exchanges, i.e. well below break-even, so MPI wins end to end:

| run (np=4) | exchanges | scatter time MPI | scatter time NVSHMEM | SNESSolve MPI | SNESSolve NVSHMEM |
| --- | --- | --- | --- | --- | --- |
| `-da_refine 5`                 |  2,480 | 0.384 s | 0.754 s | 0.963 s | 1.351 s (+40%) |
| `-da_refine 6`                 |  5,852 | 0.466 s | 0.815 s | 1.829 s | 2.207 s (+21%) |
| `-da_refine 6 -ksp_rtol 1e-10` | 16,650 | 0.745 s | 1.029 s | 4.561 s | 4.871 s (+6.8%) |

The relative penalty shrinks monotonically as work grows (+40% -> +21% -> +6.8%), exactly as a
fixed cost plus a marginal advantage predicts. Extrapolating, NVSHMEM overtakes MPI somewhere
around 64k exchanges — beyond anything measured here, so treat that number as an extrapolation
from two clean points, not an observation.

## Two traps that produce badly wrong answers

### 1. CPU binding — worth 10-30x, in the wrong direction

NVSHMEM runs a **spin-polling proxy thread**. Open MPI's default binding on this node gives each
rank exactly **one** core:

    $ mpirun -n 4 bash -c 'echo $OMPI_COMM_WORLD_RANK: $(taskset -cp $$)'
    rank 0 cores: 0     rank 1 cores: 32     rank 2 cores: 1     rank 3 cores: 33

so the proxy thread contends with the main thread for a single core. NVSHMEM warns about it:

    NVSHMEM WARN Proxy thread shares a core with the main PE, performance may be impacted

The damage is severe, and it lands on **compute** events, not communication — which makes it easy
to misdiagnose as an NVSHMEM communication problem when it is CPU starvation:

| event (np=4, `-da_refine 5`) | MPI | NVSHMEM, 1 core/rank | NVSHMEM, `--bind-to none` |
| --- | --- | --- | --- |
| `VecMDot`        | 0.210 s | 6.395 s (30x) | 0.145 s |
| `VecNorm`        | 0.164 s | 4.928 s (30x) | — |
| `KSPGMRESOrthog` | 0.274 s | 6.523 s (24x) | — |
| `MatMult`        | 0.245 s | 3.424 s (14x) | — |
| `SNESSolve`      | 1.069 s | 11.556 s (11x) | 1.351 s |

**Always run NVSHMEM with more than one core per rank** (`--bind-to none`, or `--map-by
slot:pe=N`). All numbers in this note use `--bind-to none` for *both* arms.

### 2. `VecScatterEnd` alone is a misleading metric

The two implementations put their work in different halves of the begin/end pair. NVSHMEM issues
the put and synchronizes in `Begin`; MPI defers the wait to `End`. Reading only `VecScatterEnd`
suggests NVSHMEM is 7.7x faster; summing both shows it is 1.75x slower:

| np=4, `-da_refine 6` | VecScatterBegin | VecScatterEnd | **Begin+End** |
| --- | --- | --- | --- |
| MPI     | 0.076 s | 0.390 s | **0.466 s** |
| NVSHMEM | 0.765 s | 0.051 s | **0.815 s** |

Always sum `VecScatterBegin + VecScatterEnd` (or `SFBcastBegin + SFBcastEnd`).

## Where the fixed cost lands

It appears inside `SNESFunctionEval` (0.305 s -> 0.702 s at `-da_refine 6`), because
`PetscNvshmemInitializeCheck()` is lazy — NVSHMEM is initialized on the first SF operation that
passes `PetscSFLinkNvshmemCheck()`, which is the first residual evaluation. Consistency check
across the three runs gives a stable fixed cost:

    gap = fixed - 6.06us * calls
    refine6 loose : 0.3496 = fixed - 6.06us*5852   -> fixed = 0.385 s
    refine6 tight : 0.2842 = fixed - 6.06us*16650  -> fixed = 0.385 s

Both solve for the same value, which is why the fixed/marginal split is trustworthy.

## Caveats

- **Single node only.** NVSHMEM's inter-node transport is unavailable here (IBRC skipped: no
  `nvidia_peermem`), so this measures NVLink/IPC intra-node only. NVSHMEM's design advantages are
  strongest for fine-grained inter-node traffic, which is precisely what cannot be tested here.
  **Do not generalize this result to multi-node.** *(Caveat lifted 2026-09-01: section 28 has
  the multi-node measurements; the small-message verdict turned out to be the same.)*
- One application (`ex19`), one preconditioner, one message-size regime (AvgLen ~3.1e3 B).
- The marginal-cost figures come from two points at fixed message size; the break-even estimate
  extrapolates well beyond the measured range.
- `-use_nvshmem_get 1` (get-based protocol) was verified correct but not separately profiled.
- The put-based protocol was used throughout (PETSc's default).

## 17. The SC26 poster's ACGN splitting DAG, measured (`acgnbench.c`)

Question: the poster's decentralized operator splitting (four-node tomography, branch-specific
gradient aggregation) avoids global reductions -- does that layout benefit from NVSHMEM?
`nvshmem-tools/acgnbench.c` reproduces the exact communication DAG of the companion note's
worked example on 4 ranks / 4 H100s: box prox on rank 1, x1 broadcast, parallel shard
gradients, branch routing (g0,g1 -> slot 2; g0,g1,g2 -> slot 3 consumed *after* the late g3),
rowTV/colTV prox chain, state mixing. Compute emulated by VecAXPYs (G per gradient, P per
prox). Aggregation is switchable: `-allreduce` (global reduction) vs default branch routing.
Measured 2026-08-14 on `x2003c0s9b0n0`, 300 iters, best of 3, `--bind-to none`, NVSHMEM
engagement verified by the banner discriminator.

Per-iteration time (us), G=8 P=4:

| msg | branch MPI | branch NV | allred MPI | allred NV | compute ref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | **163** | 225 | 210 | 227 | 39 |
| 256 KB | **168** | 234 | 214 | 239 | 39 |
| 1 MB   | **219** | 320 | 264 | 292 | 44 |
| 4 MB   | **367** | 510 | 521 | 585 | 65 |

Heavier compute (G=32 P=16) does not change the picture; the NVSHMEM gap *widens*
(319 -> 460 us at 8 KB, 417 -> 651 us at 1 MB).

Two findings:

1. **The algorithmic claim is real and measurable today: branch-specific routing beats the
   global Allreduce by 22-30% under plain GPU-aware MPI.** This is the poster's splitting
   story, and it needs no NVSHMEM.
2. **The decentralized layout does NOT rescue NVSHMEM: 1.38-1.56x slower in every cell.**
   The reason is the section 16.3 discriminator: the ACGN schedule is a *dependency chain* --
   at the moment each exchange runs, the receiving rank is idle-waiting on it, so there is no
   same-rank independent compute to hide NVSHMEM's ~5 stream ops behind. The one overlap
   window the DAG has (g -> slot 3 sent early, consumed after g3) is exploited equally well
   by MPI's Isend/Irecv. "Decentralized" != "overlapped": the exchanges sit *on* the critical
   path, which is exactly the regime where MPI's lower per-exchange floor (~11.5 vs ~17 us)
   wins.

Routes that could still give NVSHMEM (or a stream-ordered transport) a win on this problem,
in increasing order of ambition:

- **Many shards per GPU** (m >> 4): each GPU computes several shard gradients and sends each
  while computing the next -- manufactures the section 16.2 overlap regime on the sender
  side. Expected gain is only the ~30-40% of *exposed* per-send cost, a few us per send.
- **NCCL for the edge set**: `ncclGroupStart/End` batches all sends/recvs of a DAG stage
  into one fused kernel -- attacking the same per-operation count that NVSHMEM's protocol
  pays 5 ops for, and it is stream-ordered (no host block) *and* CUDA-graph-capturable.
  Plausibly beats both current arms; needs a hand-rolled NCCL variant of acgnbench.
- **Frozen-schedule device-side execution**: the co-design already freezes coefficients and
  schedule before the solve; that is a license to compile the whole iteration into a CUDA
  graph (or persistent kernel) whose cross-GPU dependencies are NVSHMEM device-side
  puts/signal-waits -- no host involvement per iteration at all. This attacks the ~2 us
  per-stream-op floor itself (section 14.4 named graphs the only route past MPI on one
  node). MPI structurally cannot be captured. At 64 KB the ceiling is large: 163 us/iter
  (MPI) vs a ~39 us pipelined compute floor.

### Implications for the SC26 poster

- **Fill the 2x2 result matrix with the honest interaction, and make the NVSHMEM axis
  "GPU-orchestrated execution", not "PetscSF transport swap".** As designed, the NVSHMEM
  panels would show slowdowns if filled with the current PetscSF path. The honest story is
  *stronger* for the co-design thesis: a transport cannot fix a schedule -- splitting wins on
  its own (measured, 22-30%), NVSHMEM-as-transport-swap does not (measured, 1.4x slower),
  and only schedule+transport designed together (NCCL grouped stages, or the frozen-schedule
  device-side iteration) can beat MPI on this node. That interaction *is* the poster's
  claim.
- **Measurement protocol for the plots:** report per-iteration time or time-to-tolerance
  *excluding* the one-time ~0.385 s NVSHMEM init -- same doctrine as the companion note's
  "norm estimation is setup cost, not iteration-time work" (setup once, freeze, amortize).
  Always `--bind-to none` (section 10.1), always sum `Begin+End` (section 10.2), and verify
  NVSHMEM engagement with the `NVSHMEM_VERSION=1` banner discriminator (section 8).
- **Multi-node panels are out of reach on Janus** until `nvidia_peermem` is loaded (section
  13.2); worth an ALCF ticket, with IBGDA/IBDEVX as fallbacks. *(Unblocked 2026-09-01: section 28.)* Within-node scaling means
  more shards per GPU, not more GPUs.
- The TaoTerm API changes these schedules require (split-phase gradient/prox evaluation,
  gradient sinks for H/K routing, per-term streams, capture-safety, per-rank placement) are
  recorded in `TAOTERM-WISHLIST.md` in this worktree, each tied to the measurement that
  motivates it.

## 18. Two nodes, measured (2026-08-15): the ACGN DAG across the fabric, MPI arms

First 2-node allocation (`x2002c0s9b0n0` + `x2003c0s1b0n0`, 4x H100 each, both vetted
clean). Multi-node NVSHMEM remains impossible (all transports blocked on peermem/dmabuf;
build notes section 13.3) *-- as of this date; section 28 adds the NVSHMEM, NCCL-with-GDR and
MPI-with-GDR arms on the same placement --* so these are the GPU-aware-MPI arms only, host-staged over the
NIC (`UCX_TLS=sm,self,cuda_copy,cuda_ipc,rc`), ranks 0,1 on node 1 and ranks 2,3 on node 2
(`--map-by ppr:2:node`). Same binary and protocol as section 17; best of 3, 300 iters.

Same-day single-node control first: on today's node the section 17 numbers reproduce
within ~1% (161.8 vs 163, 209.1 vs 210, 216.3 vs 219, 366.8 vs 367, 523.6 vs 521 us) --
the measurement is stable across nodes and days.

Per-iteration time (us), G=8 P=4, MPI arms:

| msg | branch 1-node | branch 2-node | allred 1-node | allred 2-node | branch adv. 2-node |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | 162 | 206 | 209 | 259 | 21% |
| 512 KB | 189 | 297 | 234 | 332 | 10% |
| 1 MB   | 216 | 378 | 260 | 416 |  9% |
| 4 MB   | 367 | 886 | 524 | 942 |  6% |

Two observations:

1. **The splitting advantage survives the fabric but shrinks with size** (22-30% intra-node
   at all sizes -> 21% at 64 KB down to 6% at 4 MB inter-node). At small sizes the
   inter-node penalty is the ~19 us host-blocked NIC latency per crossing edge; branch
   routing crosses fewer edges than the allreduce and keeps its win. At large sizes both
   arms are bound by the same NIC bandwidth (~39 GB/s host-staged vs ~159 GB/s NVLink), and
   volume differences wash out in the shared bottleneck.
2. **The 2+2 placement puts 8 of the DAG's 12 edges across the fabric** (x1: 1->2, 1->3;
   g: 0->2, 1->2, 0->3, 1->3; x2: 2->1; x3: 3->1). A placement-aware
   mapping (the poster's u_iv variables) would hide more; this is the co-design lever the
   4xN instance makes real.

## 19. Route 1 executed: NCCL edge transport (`acgnbench-nccl.c`), single node

Measured 2026-08-15 on `x2002c0s9b0n0`, NCCL 2.28.3 (cuda12.9 build from `/soft`), same
DAG/compute/protocol as section 17, every DAG stage one fused `ncclGroupStart/End` of
`ncclSend/ncclRecv` on PETSc's compute stream (`PetscDeviceContextGetStreamHandle()`).
Correctness: checksums on every received lane, all PASS. Also new: `-monolithic` arm in
`acgnbench.c` -- the "dumb single-core" baseline, rank 0 does the whole iteration's compute
serially (4G+2P+12 AXPYs), zero communication.

Per-iteration (us), G=8 P=4, best of 3 (spread < 2%):

| msg | monolithic | branch MPI | allred MPI | branch NCCL | allred NCCL | compute ref |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | 118 | 162 | 209 | **107** | 109 | 39 |
| 512 KB | 128 | 189 | 234 | 149 | **144** | 42 |
| 1 MB   | 133 | 216 | 260 | 177 | **169** | 43 |
| 4 MB   | 195 | 367 | 524 | 363 | **348** | 65 |

Heavy compute (G=32 P=16): monolithic 369/423/450/642; branch NCCL 273/329/364/636;
allred NCCL 274/324/358/624; compute ref 122/131/134/203.

Three findings:

1. **NCCL beats MPI, but less than predicted eagerly** (107 vs the predicted 70-100 at
   64 KB; -34% vs branch MPI, converging to a tie at 4 MB where both are NVLink-bound).
2. **The threat test fired: under NCCL, branch and allreduce essentially tie.** Branch wins
   by ~2% at 64 KB; allreduce wins by 3-8% at every larger size, light or heavy compute,
   eager or graphed. The section 17 "branch beats allreduce by 22-30%" result is therefore
   a statement about *MPI's* cost model -- every exchange costs a host block, and the
   allreduce is the most exchange- and barrier-heavy pattern -- not about data volume.
   NCCL removes the host block entirely, and `ncclAllReduce` is one bandwidth-optimal ring
   driving all NVLink directions concurrently, while the branch DAG is four
   *dependency-serialized* fused groups; at np=4 the ring's extra volume (~1.5x buffer per
   GPU) costs less than the branch's stage serialization. Qualifiers before declaring
   branching dead: (a) this is np=4 on flat all-to-all NVLink -- at scale, on
   hierarchical/limited topologies, and inter-node under MPI (section 18: branch wins
   21% -> 6%), the volume and locality arguments return; (b) the allreduce arm is a
   performance *proxy*, not the algorithm -- ACGN's slots want different aggregations, and
   a global sum only emulates the cost of the naive design, so branch routing keeps its
   semantic and placement (co-design) value even where its raw transport advantage
   vanishes. Poster phrasing: splitting wins *given* the transport the ecosystem actually
   uses (MPI); co-designed transports flatten the routing difference and beat both.
3. **The monolithic baseline is only honest with the compute-weight axis attached.** At
   G=8 P=4 the serial no-comm version beats every parallel MPI arm at >= 512 KB (its 52
   launch-bound AXPYs cost less than the parallel arms' communication). At G=32 P=16 it
   loses ~1.4x to the parallel arms at small/medium sizes. The crossover -- "splitting pays
   once per-term compute exceeds the cost of routing" -- is itself the co-design statement.

## 20. Route 2a executed: CUDA-graph replay (`acgnbench-nccl-graph.c`) -- the headline

Same iteration body, but after an eager 30-iter warmup the whole body (VecAXPYs + NCCL
groups) is stream-captured once and the timed loop is `cudaGraphLaunch` replays
(`-graph_iters k` iterations per graph; k=1 and k=10 within ~4% of each other, so the
per-op launch tax, not the graph-launch count, was the enemy). Validation PASS everywhere.

What "eager" vs "graph replay" means -- same kernels, different host behavior:

    EAGER (acgnbench-nccl.c): the host re-issues every operation, every iteration

     HOST  |axpy|axpy|..|grp1|axpy|..|grp2|axpy|..|grp3|..|grp4|axpy|axpy|   ~18 enqueues/iter
           ~2us CPU cost each, every iteration; never blocks, never stops enqueueing
     GPU      |axpy|axpy|..|NCCL k1|axpy|..|NCCL k2| ...                     executes behind

    A "fused group" = one ncclGroupStart/End wrapping a DAG stage's sends/recvs; NCCL
    aggregates them into ONE kernel per rank (4 comm launches/iter instead of 12 edge ops).

    GRAPH REPLAY (acgnbench-nccl-graph.c): record the body once, then

     HOST  |launch|                            |launch|                      1 enqueue / k iters
     GPU   |axpy axpy .. NCCL1 axpy .. NCCL2 .. NCCL4 axpy axpy| (replay)    same kernels, no
            per-op gap: the GPU walks the pre-built dependency graph itself

The warmup runs eagerly first (connects all NCCL channels, settles PETSc lazy init), then
`cudaStreamBeginCapture` .. one iteration's calls .. `EndCapture` records the whole
stream-ordered sequence -- AXPY kernels *and* NCCL kernels -- into a `cudaGraph_t`,
instantiated once. Nothing about the kernels changes; what dies is the ~2 us-per-operation
CPU enqueue cost x ~18 ops. That is why 107 -> 32 us, and why the 39 us "compute floor"
also fell: that floor was itself mostly enqueue tax.

**Prerequisite discovered: PETSc's default device-context stream is the legacy NULL stream
(`PETSC_STREAM_DEFAULT` destroys the `CUPMStream` and hands out `NULL`), which
`cudaStreamBeginCapture()` rejects with `cudaErrorStreamCaptureUnsupported`.** Two
equivalent fixes (both verified): the command line `-root_device_context_stream_type
nonblocking` (the root context parses the `device_context_stream_type` key under the
`root_` prefix, `device.cxx:697`), or programmatically retype the global context to
`PETSC_STREAM_NONBLOCKING` right after `PetscInitialize()` and call
`PetscDeviceContextSetCurrentContext()` again so `PetscDefaultCudaStream` re-syncs (what
`acgnbench-nccl-graph.c` does, so it cannot be run wrong). This is the first concrete
capture-safety item for the TaoTerm wishlist: any future capture-based execution path in
PETSc needs a non-NULL stream type as a precondition.

Per-iteration (us), G=8 P=4, k=10, best of 2:

| msg | branch graph | allred graph | best eager NCCL | branch MPI | compute ref (eager) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | **32** | 34 | 107 | 162 | 39 |
| 512 KB | 68 | **63** | 144 | 189 | 42 |
| 1 MB   | 91 | **83** | 169 | 216 | 43 |
| 4 MB   | 227 | **211** | 348 | 367 | 65 |

- **5.0x vs branch-MPI at 64 KB, and below the eager compute floor** (32 < 39): the
  39 us "compute reference" was itself ~14 launch-bound AXPY launches; replay erases the
  launch tax for compute and comm alike, leaving ~20 us of AXPY execution + ~12 us of NCCL
  kernels.
- Heavy compute, 64 KB: **32 us** vs 273 eager branch-NCCL, 369 monolithic -- when
  everything is launch-bound, the graph win compounds: 8-11x against the eager arms.
- The route 2 prediction (~40-70 us at 64 KB) is beaten. The remaining gap to pure
  transfer time is AXPY execution serialization -- attacking that is route 2b (fused
  kernels), which is now about the last ~20 us, not the first 130.
- Allreduce-vs-branch under graphs: same story as eager NCCL (finding 2 above); branch
  wins only at 64 KB, by 2 us.

### Does graph replay require knowing the iteration count? (Reviewer question; no.)

The graph is the **fixed-point map, not the solve**: it records one application of T in
x_{k+1} = T(x_k), and the host decides at runtime how many times to replay it. What
capture requires is that T itself is data-independent -- fixed steps, fixed prox
parameters, fixed schedule, no line search or adaptive restart. That is exactly the
co-design doctrine ("obtain bounds once -> construct and schedule once -> freeze"): the
freeze is the *mathematical license* to compile the iteration, not just a performance
trick. The only data-dependent piece is the stopping test; the standard resolution is
chunked replay -- launch k iterations per graph, compute the residual inside the graph
(a fixed computation), read-and-decide on the host once per chunk. Cost: one sync per k
iterations plus overshoot of at most k-1 iterations, which is harmless for a
nonexpansive map (extra applications of T do not corrupt the iterate, they waste at most
k-1 iterations of work). Removable entirely with CUDA >= 12.4 conditional graph nodes
(device-evaluated while-loops), or in the device-NVSHMEM persistent kernel by computing
the norm in-kernel, allreducing it with the section 23 machinery, and breaking all PEs'
loops on the same signaled epoch -- the stopping test becomes one more edge of the
frozen protocol. `-graph_iters k` in `acgnbench-nccl-graph.c` is precisely this chunked
structure (k=1 and k=10 measured within ~4%).

## 21. Two-node NCCL: works, one mandatory knob, honest numbers are poor without GDR

*Correction (2026-09-01, build notes 13.4): `mlx5_bond_0` is the 25 GbE **management** bond,
not a bond of the data ports, so every 2-node NCCL number in this section and in section 22's
2-node column was taken over a 25 Gb/s link -- the ~2.7 GB/s ceiling below is that link's line
rate, which is why no knob moved it. The "dead routes on the bond slaves" were NCCL's RoCE v2
default stalling on the 400 Gb/s data port `mlx5_0`; RoCE v1 fixes it. Superseded by section
28.4 (177 us at 64 KB, 996 us at 4 MB, with GDR).*

`NCCL_IB_HCA=mlx5_bond_0` is **mandatory** on this fabric: by default NCCL enumerates the
raw bond-slave devices (`mlx5_0`, `mlx5_1`) alongside the bond and lands on dead routes --
**78,000+ us/iter of retransmit timeouts** instead of ~300. (Same family of fabric quirk as
the UCX `ud` failure in build notes section 13.3.) With the knob, 2+2 across nodes:

| msg | branch MPI | allred MPI | branch NCCL | allred NCCL |
| ---: | ---: | ---: | ---: | ---: |
| 64 KB  | **206** | 259 | 296 | 290 |
| 512 KB | **297** | 332 | 1542 | 1183 |
| 1 MB   | **378** | 416 | 3005 | 2268 |
| 4 MB   | **886** | 942 | 11709 | 8802 |

The right reading is three separate layers, not "NCCL is broken":

1. **A config trap (fixed):** default HCA enumeration includes the raw bond slaves and
   lands on dead routes -> the 78 ms pathology. `NCCL_IB_HCA=mlx5_bond_0` cures it
   completely. Same family of fabric quirk as the UCX `ud` failure (build notes 13.3).
2. **Small messages are fine:** 290-296 us vs MPI's 206 at 64 KB -- only 1.4x worse, with
   the launch-tax share removable by graph replay (280 us).
3. **Bulk bandwidth is the GDR-less fallback path:** without `nvidia_peermem` the NIC
   cannot DMA GPU memory, so NCCL's proxy stages through fixed host bounce buffers,
   reaching ~2.7 GB/s aggregate (32 MB of crossing edges / 11.7 ms at 4 MB), vs UCX's
   pipelined host staging at ~39 GB/s over the same wire.

Two side observations: graph replay works across nodes too (280 us vs 296 eager at 64 KB)
but only shaves the launch tax -- the NIC path dominates; and allreduce-NCCL *beats*
branch-NCCL inter-node at bandwidth sizes (8.8 vs 11.7 ms at 4 MB) because the ring
pipelines through the bond in both directions while the branch DAG serializes bulk edges.

### Can layer 3 be fixed without root? No -- measured, not assumed

The wire is provably capable (UCX does 39 GB/s host-staged), so the hypothesis "NCCL's
staging pipeline is just under-tuned" was worth testing. Every plausible user-space lever
was tried at the 4 MB size; the result is *completely flat*:

| attempt | per_iter |
| --- | ---: |
| baseline | 11705 us |
| `NCCL_IB_QPS_PER_CONNECTION=4` + `NCCL_IB_SPLIT_DATA_ON_QPS=1` | 11737 |
| `NCCL_BUFFSIZE=16M` | 11720 |
| `NCCL_MIN/MAX_NCHANNELS=16` | 11707 |
| `NCCL_P2P_NET_CHUNKSIZE=4M` | 11801 |
| all of the above combined | 11869 |
| HPC-X `nccl_rdma_sharp_plugin` (loads after extracting `libibumad.so.3` from `/soft/repos/hpe-doca-ofed-rhel9.4/` into `LD_LIBRARY_PATH`) | unchanged |

Knob-invariance at this level means the limiter is inside the proxy's bounce-buffer
protocol itself, not its parallelism parameters. The remaining routes all need root:
`modprobe nvidia_peermem` (the direct fix), or switching the nodes to the open-source
NVIDIA kernel modules for dmabuf (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0 on the
current stack). A newer NCCL cannot help -- 2.28.3 is already the newest installed, and no
NCCL version can register GPU memory with the NIC without one of those kernel-side pieces.

*(Resolved 2026-09-01: peermem loaded, and the real limiter was the 25 Gb/s management NIC --
section 28.4.)*

### Did the admin "miss" peermem? Almost certainly, and the ticket writes itself *(resolved 2026-09-01)*

The module **ships with the exact driver installed on these nodes** (`modinfo
nvidia_peermem` -> version 610.57.04, build notes 13.2); it is the standard GPUDirect RDMA
component for any IB/RoCE GPU cluster; it is simply not loaded (`lsmod` shows only
`nvidia`, `nvidia_uvm`) and not persisted in any `modules-load.d`. On H100 nodes with
bonded RoCE NICs that is an oversight or a deferral, not a design decision. One
`modprobe nvidia_peermem` (plus a one-line persist) unlocks three things at once:

    1. NVSHMEM multi-node -- ALL of its transports are gated on it (build notes 13.3)
    2. NCCL's inter-node fast path (this section's layer 3)
    3. GPUDirect RDMA for UCX/MPI itself -- today's 39 GB/s is host-staged; GDR removes
       the staging copies and the ~19 us host-blocked small-message floor drops too

**Verdict: inter-node, GPU-aware MPI remains the transport of record on this machine until
`nvidia_peermem` is loaded; every GPU-orchestrated transport (NVSHMEM entirely, NCCL's
data path) is gated on the same kernel module. The ALCF ticket is the single unlock.**

## 22. The poster's 2x2 matrix, with today's numbers (single node, 64 KB, G=8 P=4)

    monolithic serial (dumb baseline)      118 us
    term-per-node + Allreduce (MPI)        209 us   <- parallelism alone LOSES at light compute
    branch routing (MPI)                   162 us   <- splitting recovers it, still above serial
    branch routing (NCCL eager)            107 us   <- co-designed transport starts winning
    branch/allred + CUDA-graph replay       32 us   <- schedule+transport+execution co-design: 3.7x vs serial,
                                                       5x vs branch-MPI, at the compute floor

    At heavy compute (G=32 P=16): serial 369, branch-MPI ~320 (sec 17), graph 32 -- the
    ordering is strict and the co-design win is 10x.

## 23. Route 2b executed: device-side NVSHMEM skeleton (`acgnbench-nvdev.cu`)

Measured 2026-08-15, same node/protocol. Standalone MPI+NVSHMEM `.cu` (no PETSc, no
NCCL): each rank's whole ACGN iteration is ONE fused cooperative kernel (or one persistent
kernel for the entire run, `-persistent`). Cross-GPU dependencies are
`nvshmemx_signal_op()` / local signal-word polling inside the running kernel; data moves
by consumer-side pull through `nvshmem_ptr()` NVLink loads. Monotonic epoch signals
(`v = it+1`, `CMP_GE`), double-buffered slots (`it & 1`), reverse ack signals gate slot
reuse at depth 2. Race detection is built in: every published element is stamped
`role_base + v` and consumers validate every element, every iteration -- a missing fence
or stale slot is a counted error, not silence. **Zero mismatches across the whole matrix,
~100M validated element-reads per run.**

Per-iteration (us), G=8 P=4, best of 3, best block cap (32 at 64 KB, 64 above), final
(v4) protocol. The `-allreduce` arm is a **direct all-pull allreduce**: every PE
publishes its stamped contribution and pulls the other three lanes (one signal round, no
ring steps -- latency-optimal at np=4):

| msg | branch-dev (1 launch/iter) | branch-pk (1 launch/RUN) | allred-dev | allred-pk | graph NCCL | eager NCCL | branch MPI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | 48 | **41** | 44 | 42 | 32 | 107 | 162 |
| 512 KB | 92 | 88 | 86 | **76** | 63-68 | 144 | 189 |
| 1 MB   | 133 | 122 | 118 | **108** | 83-91 | 169 | 216 |
| 4 MB   | 387 | 382 | 326 | **316** | 211-227 | 348 | 363 |

The section 19 finding replicates on a third transport: allreduce ties branch at 64 KB
and *wins* above it (11-17%), despite moving ~2.4x the lane volume. The mechanism is
visible in the DAG: the all-pull g-phase lets rank 3 fetch its gradient lanes early and
concurrently with rank 2's pulls, taking the branch schedule's late-consume reads OFF the
critical path, while branch routing keeps them on it. Three transports now agree: the
splitting advantage is specific to host-blocking (MPI-style) cost models.

Heavy compute (G=32 P=16, 64 KB): 48 us -- compute inside the fused kernel is nearly free
(comm-only G=0 matches the full run), exactly the tier-b thesis.

### The intra-kernel synchronization protocol took four revisions; each step was measured

These are in-place revisions of the same file during one session; only v4 survives on
disk (v2's numbers are the untagged rows in `results-20260815-nvdev.txt`, v4's are tagged).

| protocol version | 64 KB persistent | lesson |
| --- | ---: | --- |
| v1: thread-0 waits + full `grid.sync()` after every phase (~10/iter), grid sized to d | 68-71 | grid barriers cost ~1-3 us each and scale with block count (264 blocks -> 283 us at 1 MB) |
| v2: all-thread polling for waits + grid capped at 32-64 blocks | 54 | wait-side barriers gone; publish/ack grid syncs remain |
| v3: barrier-free arrival counters, but ALL threads spin-poll signal and ack words | 74-75 | **worse than v2** -- 8192 threads hammering one L2 word starve the very L2 that serves incoming signals and peers' NVLink pulls |
| v4: arrival counters + one leader poller per block + `__syncthreads()` release | **41** | barrier-free only wins with contention-free polling: 32 pollers, not 8192 |

How each revision worked, and what its measurement forced next:

**v1 (conservative-correct).** Every dependency point was the textbook pattern: thread 0
of the grid performs `nvshmem_signal_wait_until()`, then `grid.sync()` broadcasts the
result; every publish was writes -> `__threadfence_system()` -> `grid.sync()` -> thread 0
signals. ~10 grid barriers per iteration, grid sized to the vector. It validated
immediately (68 us @ 64 KB) but two profile facts condemned it: comm-only (G=0) cost the
same as the full iteration -- the protocol, not compute or data, was the entire bill --
and 1 MB ran at 283 us because 264-block `grid.sync()`s are expensive. Persistent mode
was no faster than per-iteration launches, proving launch overhead was not the term that
mattered.

**v2 (wait-side barriers removed, grid capped).** Waits became all-thread polling of the
local signal word (each thread gates itself; the remote `signal_op` write lands in local
memory, so polling needs no NVSHMEM call), and the grid was capped at 32-64 blocks with
grid-stride loops covering any d. 54 us @ 64 KB, 112 @ 512 KB. Publish/ack still carried
~5 grid barriers. This was the version first reported as "route 2b done".

**v3 (publish/ack barriers removed -- the regression).** Grid syncs were replaced by
per-site-per-slot block-arrival counters (mechanism below), and ack-waits also became
all-thread polls: zero grid-wide synchronizations. It stayed correct (validation PASS)
but got SLOWER than v2 (74-75 us): with ~8192 threads spin-loading the same 8-byte
signal words, the consumer GPU's L2 is saturated by its own polling -- the same L2 that
must serve the incoming NVLink signal write and the peers' remote data pulls. The
barrier was gone but the detection path was self-jammed.

**v4 (contention-free polling).** Same barrier-free structure, but every poll (signal
waits and ack waits) is done by thread 0 of each block, releasing its block through
`__syncthreads()` -- 32 pollers instead of 8192, cross-block decoupling preserved.
41 us @ 64 KB, 88 @ 512 KB, 122 @ 1 MB.

v4's publish/ack protocol: each block fences (`__threadfence_system`) and bumps a local
per-site-per-slot atomic; the LAST block to arrive resets the counter and fires the
`nvshmemx_signal_op`s; all other blocks flow on immediately and self-gate at their next
signal wait. The reset-before-signal is provably safe: no block can re-enter a site+slot
(iteration it+2) before a consumer acks iteration it, and that ack requires the very
signal that follows the reset. There are ZERO grid-wide synchronizations per iteration --
blocks of one PE may run skewed across iterations, coupled only by the dataflow. The
epoch-stamped per-element validation ran through every revision, which is what made this
loop safe to iterate quickly: a fencing or reset mistake would have been a counted error,
not a silently wrong benchmark.

Findings:

1. **The execution model is proven.** A frozen schedule really can run as a device-side
   program: in persistent mode the host launches ONE kernel for 300 iterations -- zero
   host involvement per iteration, which no MPI-based path can structurally match. All
   dataflow, ordering, flow control, and validation live on the GPUs.
2. **It beats every eager arm** (4x vs branch-MPI, 2.6x vs eager NCCL at 64 KB) **and
   lands 1.28x behind graph-NCCL** (41 vs 32 us). The 10-30 us prediction was just
   missed: 41 us sits at its edge, and the remaining cost decomposes as the DAG's serial
   depth (~7 signal-hop segments on rank 3's critical path) x ~3-5 us per hop (NVLink
   signal write + poll detect + remote pull + arrival accounting) plus ~7 us compute --
   the realistic floor for THIS DAG on this fabric is high-20s us, i.e. the prediction's
   upper edge was about right and its lower edge was optimistic about hop latency.
3. **Remaining (unbuilt) headroom:** merge consume-validation into the following compute
   loop; per-warp specialization so the leader poller overlaps the consume reads;
   deferring off-critical-path acks (e.g. rank 3's x1 ack). Each is a ~2-4 us-class item;
   none changes the structural picture.
4. **What this arm uniquely demonstrates for the poster:** the co-design ladder's last
   rung is *feasible and correct* (device-resident iteration, validated), it is the only
   arm with literally zero per-iteration host cost, and it is the only route available
   when kernels must react to data mid-flight (graphs replay a fixed recording; NVSHMEM
   host APIs are not capture-safe -- device APIs go *inside* the kernel instead). As pure
   single-node throughput, graph-NCCL wins today.

Build is the notes-section-5 three-step recipe (`-rdc=true`, `-dlink` against
`libnvshmem_device.a`, host link); raw numbers in
`nvshmem-tools/results-20260815-nvdev.txt`.

## 24. The full measured configuration space, in one table

The DOF structure of everything measured on the ACGN DAG:

- **comm layer** (4): GPU-aware MPI; PetscSF-NVSHMEM (NVSHMEM host/stream API under
  PetscSF); NCCL fused groups; NVSHMEM device API inside fused kernels. (There is no
  "NCCL+NVSHMEM" arm -- the fourth layer replaces NCCL, it does not combine with it.)
- **algorithm** (2): global allreduce vs branch routing. Plus two baselines outside the
  axis: monolithic-serial (no comm) and compute-only floor.
- **execution model** (2 where it exists): eager (host re-enqueues every operation each
  iteration) vs replay/fused (graph replay for NCCL; per-iteration-launch vs persistent
  single-launch for the device arm). For MPI this axis is STRUCTURALLY EMPTY (blocking
  host calls cannot be captured); for PetscSF-NVSHMEM likewise (host API is not
  capture-safe, and its `<<<1,1>>>` spin kernels would deadlock a single fused kernel).
  Terminology: graph replay is ONE LAUNCH of a recorded multi-kernel sequence (kernels
  stay distinct); true single-kernel fusion is what the device arm does.
- **environment**: message size (64 KB-4 MB), compute weight (G8P4 / G32P16), placement
  (1 node; 2+2 across nodes for MPI and NCCL).

Master table, 64 KB, G=8 P=4, single node, us/iter (best-of-N; sources: sections 17-23):

| comm layer \ algo            | allreduce | branch | execution |
| --- | ---: | ---: | --- |
| GPU-aware MPI                | 209 | 162 | eager only (capture impossible) |
| PetscSF-NVSHMEM              | 227 | 225 | eager only (not capture-safe); 08-14 numbers |
| NCCL, eager                  | 109 | 107 | |
| NCCL, graph replay (k=10)    | 34  | **32** | |
| NVSHMEM device, 1 launch/iter| 44 | 48 | direct all-pull allreduce |
| NVSHMEM device, persistent   | 42 | **41** | |
| monolithic serial (baseline) | \-- 118 -- | | no comm at all |
| compute-only floor           | \-- 39 -- | | eager launches; the graph arms beat it |

Cell status is part of the result: the two "eager only" rows are *structural* absences
(and exactly the poster's point -- host-blocking transports cannot enter the replay/fused
regime). The device-side allreduce cells were filled after this section was first
written (direct all-pull; section 23); as the NCCL rows predicted, allreduce ties branch
at 64 KB -- and above 64 KB it wins 11-17%, because all-pull takes the late-consume reads
off the critical path. The matrix is now complete: every buildable cell is measured, and
every empty cell is structurally empty.

### How much of each arm is PETSc -- the deliverable ladder

The rows are NOT equally "PETSc results", and honesty about that is part of the story:

| rows | compute | transport | status |
| --- | --- | --- | --- |
| MPI, PetscSF-NVSHMEM | PETSc `VecAXPY` | **PetscSF** | in-tree today; no in-tree config beats 162 us |
| NCCL eager + graph | PETSc `VecAXPY` | hand-rolled NCCL, **bypasses PetscSF** | PETSc compute, foreign transport |
| NVSHMEM device | raw CUDA loops | NVSHMEM device API | PETSc-less showcase / upper bound |

The NCCL bypass is deliberate: SF's collectively-called per-exchange `Begin`/`End` cannot
express stage-fused groups or per-rank edge programs (the layering note in
`NVSHMEM-NEXT-STEPS.md`); an SF-NCCL backend MR could fuse within an exchange only, so it
would preserve part of the 107 us eager number and none of the cross-op structure. The
graph rows carry one genuinely PETSc-positive fact: **capture over real PETSc operations
works** -- the 32 us graph replays genuine `VecAXPY`s -- given the nonblocking-stream
precondition (section 20) and no host syncs in the captured region; making that a
supported pattern rather than folklore is wishlist item 4. The device arm's path from
showcase to deliverable is wishlist item 6. Deliverable ladder, in order of effort:
(1) this branch's build fixes upstream; (2) `sfnvshmem.cu` op fusion (section 13.1);
(3) SF-NCCL backend with partial fusion; (4) capture-safety support; (5) the device-API
execution mode.

Same table at other sizes/weights: section 19 (NCCL eager, monolithic), 20 (graph), 23
(device), 17 (MPI, PetscSF-NVSHMEM); 2-node variants in sections 18 and 21.

## 25. The sfnvshmem.cu op fusion, implemented and measured (2026-08-15, second job)

TODO item 7 executed: the section 14.4 fusion lever is now real code, behind
`-use_nvshmem_putsig` (default TRUE; 0 restores the legacy protocol for A/B).

**The change** (three edits, all local to the put protocol):

1. `PetscSFLinkPutDataBegin_NVSHMEM()`: for locally accessible PEs,
   `nvshmemx_putmem_nbi_on_stream()` becomes `nvshmemx_putmem_signal_nbi_on_stream()` --
   the receiver's arrival signal (`RecvSig` at its `sigdisp` offset) is delivered *with
   the data*, ordered after it by API contract.
2. The `nvshmemx_quiet_on_stream()` workaround (line ~590, "fence does not fence
   on_stream puts") is skipped -- its only job was ordering the puts before the
   End-time signals, which no longer exist for local PEs. One stream op removed.
3. `PutDataEnd<<<1,1>>>` signals only remotely-accessible dst ranks (an `nvshmem_ptr()`
   test in the kernel); its wait half is unchanged. The remote path's IB-ordering
   argument is untouched.

**Correctness:** `sf/tests/ex22` FetchAndOp at np=2,4 passes both protocols;
`snes/tutorials/ex19` (-da_refine 4, 3 Newton steps) is byte-identical to the
`-use_nvshmem 0` MPI baseline under both fused and legacy. (The first ex19 comparison
was vacuously "identical" -- the binary had failed to build for a missing `-lm` and all
outputs were empty. Check your file sizes before trusting a diff.)

**Measured, np=4, `--bind-to none`:**

| instrument | legacy | fused | delta |
| --- | ---: | ---: | --- |
| `sfbench` ring floor, 64 B-128 KB | 18.5-20.1 us | 17.4-19.0 us | ~-1.1-1.6 us (~7%), never worse |
| `acgnbench` branch DAG, 64 KB | 227-231 us/iter | **205-207** | **-9.5%** |
| `acgnbench` branch DAG, 1 MB | 320-325 us/iter | **292-296** | **-9%** |

The DAG gains ~4 us per exchange -- MORE than the ~1.4 us the ring microbenchmark shows,
and the difference is the second, subtler effect: the arrival signal now lands at
*Begin* time (with the data) instead of at the sender's *End* time, so a receiver
blocked on the critical path unblocks as soon as bytes arrive. A back-to-back
microbenchmark cannot see this latency term; a dependency chain feels it on every edge.

Follow-up cells (same protocol A/B on the other arms): allreduce-NVSHMEM 229 -> 219 at
64 KB, 292 -> 264 at 1 MB (-4.5%/-9.5%); the Reduce (LEAF2ROOT) direction runs correct
at parity within noise; the get protocol is unaffected by the new default. An in-tree
harness test now exists (`sf/tests/ex22` suffix `cuda_nvshmem`, both protocols x np
{1,4}, all pass through `make test`) -- PETSc's first NVSHMEM test.

Net position: branch-NVSHMEM's intra-node gap vs branch-MPI narrows from 1.39x
(225/162) to **1.27x** (206/162). Still not a win on one node -- as predicted, fusion
was never going to close the remaining ~3 ops x 2 us -- but it is free, strictly
nonnegative in every measurement, deletes a documented workaround, and its per-exchange
saving is fabric-independent (it survives to multi-node when peermem lands). This plus
the build fixes is the MR package; an in-tree NVSHMEM test should ride along (none
exist, build notes section 8).

## 26. Shards (m >> 1): the last algorithmic axis, measured and closed

TODO item 11 executed (`acgnbench.c -shards m`): each rank's gradient is split into m
slices; slice k's sends (its own pair of sharded SFs, explicit strided `ilocal`) are
posted immediately after slice k's compute, so slice k+1's compute overlaps them --
sender-side pipelining. Total compute (G AXPYs) and total bytes (d) are held constant;
m changes only the granularity (requires m | G and m | d). Sweep: d in {64 KB, 1 MB,
4 MB} x G in {8, 32} x m in {1,2,4,8} x {MPI, NVSHMEM-legacy, NVSHMEM-fused}, best of 2
(`results-20260815-shards.txt`).

**Result: m=1 is optimal in 23 of 24 cells.**

- **MPI degrades monotonically with m everywhere** (64 KB G=8: 165 -> 186 -> 237 -> 312).
  Every shard adds a host-blocked exchange; the per-exchange floor multiplies faster
  than any overlap can pay for.
- **The one predicted exception is real but small**: at 1 MB + heavy compute (G=32) the
  NVSHMEM arms' optimum shifts to m=4 -- fused 622 -> 569 (-8.6%), legacy 647 -> 596
  (-7.8%). This is exactly the section 16.2 regime: per-shard compute (~8 AXPYs) is
  finally large enough to hide a per-shard send, and the 256 KB shard is still
  latency-ish. At 4 MB the effect vanishes (bandwidth-bound; section 16.5's rule).
- **Even the best sharded NVSHMEM cell loses to unsharded MPI** (569 vs 410); sharding
  never reorders any transport ranking.
- The put+signal fusion (section 25) holds under sharding: fused < legacy in all 12
  NVSHMEM cells, by 5-9%.

One-line verdict for the poster: **sender-side sharding cannot rescue a host-API
transport on this DAG -- it recovers at most ~9% in one narrow window and changes no
conclusion.** With this, the section 17 route list is fully explored: every proposed
route has now been built and measured.

## 27. The device-side convergence test: the stopping decision as one more frozen edge

TODO item 14 executed (`acgnbench-nvdev.cu -converge`, persistent mode). The section 20
reviewer answer ("the graph/kernel freezes the map, not the iteration count") is now a
measured artifact: a synthetic residual vector decays by rho per iteration; every
`-check_every` iterations each PE grid-reduces its ||rv||^2, publishes the partial to a
symmetric slot, all-pulls the other three, and sums **in fixed rank order so every PE
computes the bit-identical global norm** -- hence the same stop decision with no extra
agreement round. An epoch-stamped decision word releases all blocks; all four resident
kernels break their loops on the same iteration. Flow control needs no ack signals: the
decision poll itself gates every block, so slot/counter reuse is provably serialized
behind the slowest block (the same ack-chain style argument as section 23).

Measured (64 KB, persistent, tol=24, rho=0.99 -> analytic stop at n*=202):

| configuration | stopped after | predicted | per-iter |
| --- | ---: | ---: | ---: |
| no convergence test (reference) | -- | -- | 43.1 us |
| `-converge -check_every 1` | 202 | 202 **MATCH** | 54.0 us |
| `-converge -check_every 10` | 210 | 210 **MATCH** (overshoot 8 <= k-1) | **42.2 us** |

Reading: checking *every* iteration costs ~11 us/iter (a grid reduction + a 4-double
all-pull + a decision gate); checking every 10th amortizes to noise while overshooting
at most k-1 iterations -- the exact chunked-replay tradeoff section 20 described for
graphs, here executed with zero host involvement: **one kernel launch runs the whole
solve, including deciding when it is over.** This is the capability that distinguishes
the device arm from graph replay (a graph cannot decide; a resident kernel can).

## Open issue found while building the benchmark

A hand-written PetscSF benchmark (`sfbench.c`: ring topology, raw `cudaMalloc` root/leaf buffers,
`PetscSFBcastWithMemTypeBegin` with `PETSC_MEMTYPE_CUDA`) **segfaults under GPU-aware MPI**:

    UCX ERROR cuMemGetAddressRange(0x7fabd7e00000) error: named symbol not found
    [1]PETSC ERROR: Caught signal number 11 SEGV

Established about it:
- Works with `-use_gpu_aware_mpi 0`, and works with `-use_nvshmem 1`.
- Not a UCX transport-selection issue: reproduced with `UCX_TLS` = `sm,self,cuda_copy,cuda_ipc`,
  `sm,self,cuda_ipc`, `...,gdr_copy`, and `all`.
- Not caused by leaf contiguity: reproduced with both contiguous and strided `ilocal`.
- Plain MPI+CUDA `MPI_Send`/`MPI_Recv` on raw `cudaMalloc` pointers works fine (no PETSc).
- PETSc's own `ex19`/`ex22` with GPU-aware MPI work fine.

So it is specific to this SF construction rather than to GPU-aware MPI generally. Unresolved;
the ex19-based measurements above avoid it entirely. Worth reducing further before reporting
upstream — note the CUDA driver here is 13.3 while the toolkit and HPC-X UCX are 12.x, which is a
plausible contributor.

## 28. Two nodes with GPUDirect: NVSHMEM across the fabric, measured (2026-09-01)

Peermem went live on Janus (build notes 13.4: DOCA-OFED 26.04, real `nvidia_peermem`,
and the fabric corrected -- `mlx5_0` is the 400 Gb/s RoCE data port, `mlx5_bond_0` is
the 25 GbE management bond). This section holds the first multi-node NVSHMEM numbers.
Nodes `x2000c0s5b0n0` + `x2001c0s9b0n0`, job 2507, pins as in build notes 13.4, raw logs
`nvshmem-tools/results-20260901-2node-ladder*.txt`. Placements: "8-rank ring" = 4 ranks
per node, rank r on GPU r%4 (ring edges 3->4 and 7->0 cross the fabric); "2+2" = ranks
0,1 on node 1 and 2,3 on node 2, exactly the section 18 placement (8 of the DAG's 12 edges
cross); "1 per node" = 2 ranks, GPU0 on each node.

### 28.1 Transport floors (NVSHMEM perftest, 2 PEs) -- what the wire and the proxy cost

| metric (8 B unless noted) | intra-node, NVLink | inter-node, IBRC + GDR | MPI inter-node, host-staged (UCX 1.17) |
| --- | ---: | ---: | ---: |
| device `put` latency, one-way | 2.0 us | 10.1 us | 19.4 us (ping-pong one-way) |
| device `get` latency | 2.5 us | 10.3 us | -- |
| `put_signal` ping-pong, round trip | 6.3 us | 25.5 us | 38.8 us (round trip) |
| host `put_on_stream` latency | 1.9 us | 12.7 us | -- |
| `put` bandwidth, 16 MB | 122 GB/s | 46.7 GB/s | 39.2 GB/s |
| `get` bandwidth, 4 MB | 115 GB/s | 37.2 GB/s | -- |

Three readings. (1) GDR halves the inter-node small-message floor relative to host-staged
MPI (10 vs 19 us) and adds ~20% bandwidth (47 vs 39 GB/s; 400 Gb/s line rate is 50). (2) A
remote put costs 5x an NVLink put and the proxy thread is that floor: the device-issued
put and the host on-stream put both land at 10-13 us, i.e. the GPU->proxy->NIC hop, not
the wire, dominates. (3) The `put_signal` round trip (25.5 us) is the number a
signal-driven DAG edge pays for a request/response across the fabric; intra-node it is 6.3.

Lane view of one small inter-node exchange under the mechanisms measured here (times are
the one-way floors above; IBGDA in its CPU-doorbell hybrid, section 28.6):

    MPI, host-staged (HPC-X UCX 1.17)                                ~19 us one-way
    GPU src  |D2H copy|
    CPU src  |        |stage, post RDMA|
    wire     |                         |====== 400G ======>|
    CPU dst  |                                             |recv|H2D copy|
    GPU dst  |                                                           |data|

    NVSHMEM IBRC (proxy thread), GDR                                 ~10 us one-way
    GPU src  |kernel: enqueue put|
    CPU proxy|                   |poll|post RDMA write (NIC reads GPU src directly)|
    wire     |                                           |====== 400G ======>|
    GPU dst  |                                                               |data|signal|

    NVSHMEM IBGDA, hybrid: GPU writes WQEs, CPU rings the doorbell      section 28.6
    GPU src  |kernel: write WQE in GPU memory|
    CPU      |                               |ring doorbell|
    wire     |                                             |====== 400G ======>|
    GPU dst  |                                                                 |data|signal|

    NVSHMEM IBGDA, true (GPU rings the doorbell) -- needs PeerMappingOverride; not available
    GPU src  |kernel: write WQE, ring doorbell|
    wire     |                                |====== 400G ======>|
    GPU dst  |                                                    |data|signal|

### 28.2 PetscSF across the fabric: `sfbench` ring, MPI vs NVSHMEM put vs NVSHMEM get

8-rank ring (4 per node; two edges cross), best of 2, us per `PetscSFBcast`:

| bytes | MPI | NVput | NVget | put/MPI | get/MPI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 22.8 | 25.3 | 37.1 | 1.11 | 1.63 |
| 1 K | 21.2 | 25.4 | 36.7 | 1.20 | 1.74 |
| 8 K | 21.9 | 25.4 | 36.9 | 1.16 | 1.69 |
| 16 K | 25.4 | 25.4 | 36.9 | 1.00 | 1.45 |
| 64 K | 27.8 | 25.5 | 37.4 | 0.92 | 1.35 |
| 256 K | 39.0 | 30.4 | 42.7 | 0.78 | 1.09 |
| 512 K | 54.7 | 34.7 | 49.9 | **0.63** | 0.91 |
| 1 M | 65.5 | 55.6 | 66.4 | 0.85 | 1.01 |
| 2 M | 89.9 | 98.4 | 113.1 | 1.09 | 1.26 |
| 4 M | 138.8 | 180.4 | 196.1 | 1.30 | 1.41 |
| 16 M | 426.6 | 673.8 | 703.3 | 1.58 | 1.65 |
| 32 M | 833.8 | 1335.4 | 1377.2 | 1.60 | 1.65 |

1 rank per node (every edge crosses; each rank both sends and receives):

| bytes | MPI | NVput | NVget | put/MPI | get/MPI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 21.8 | 29.6 | 40.3 | 1.36 | 1.85 |
| 8 K | 24.0 | 29.4 | 40.6 | 1.22 | 1.69 |
| 64 K | 31.4 | 29.3 | 41.8 | 0.93 | 1.33 |
| 512 K | 58.6 | 46.3 | 60.3 | 0.79 | 1.03 |
| 4 M | 250.0 | 189.9 | 207.5 | **0.76** | 0.83 |
| 16 M | 894.1 | 678.1 | 945.9 | 0.76 | 1.06 |
| 32 M | 1746.2 | 1657.6 | 1870.0 | 0.95 | 1.07 |

What the two tables say:

1. **Small messages: NVSHMEM loses despite the better transport.** MPI sits at 21-24 us,
   NVput at 25-29, NVget at 37-40. The transport floor is 10-13 us in NVSHMEM's favour
   (28.1), so the gap is the PetscSF-NVSHMEM protocol (sections 14-15: ~5 stream-ordered
   ops per exchange plus the completion wait) stacked on the proxy floor, while
   host-staged MPI's 19 us carries no such protocol. Get costs one more fabric round trip
   than put and shows it (+12 us).
2. **Crossover at 16 KB; NVput wins from 32 KB to 1 MB**, best 0.63x at 512 KB (34.7 vs
   54.7 us). This is the regime where staging costs per byte and GDR does not.
3. **Large messages depend on who does the staging.** In the 8-rank ring MPI wins again
   from 2 MB (1.3-1.6x); with 1 rank per node NVput wins (0.76x). MPI's 16 MB time doubles
   between the placements (427 -> 894 us) because a rank that both sends and receives
   pushes both directions through its own host-staging pipeline; NVput does not move
   (674 vs 678 us) because the NIC does the work. So the 8-rank column is MPI's best case
   (four ranks share the staging) and the 1-per-node column is its worst. NVput's 674 us
   at 16 MB is 25 GB/s per edge, about half the 47 GB/s perftest floor; whether that is the
   proxy chunking one large on-stream put or the put->signal->completion serialisation of
   one exchange at a time is not established (section 16's multi-exchange pipelining is
   the experiment that would tell).

### 28.3 The ACGN DAG across the fabric: MPI vs NVSHMEM arms, 2+2 placement

G=8, P=4, 300 iterations, us per iteration, two reps each; section 18's MPI numbers in
brackets:

| arm | 64 KB | 4 MB |
| --- | ---: | ---: |
| branch, MPI host-staged | 206, 208 [206] | 1015, 1041 [886] |
| allreduce, MPI host-staged | 263, 265 [259] | 991, 1026 [942] |
| branch, NVSHMEM put+signal (sfnvshmem fused) | 279, 281 | 956, 965 |
| allreduce, NVSHMEM | 282, 286 | 920, 924 |
| branch, NVSHMEM get | 280, 284 | 948 |

1. **At 64 KB the PetscSF-NVSHMEM arm loses to MPI by 35% across the fabric**, the same
   shape as intra-node (section 17: 1.4-1.6x). The wire favours NVSHMEM (28.1); the
   protocol cost is now paid on each of the 8 crossing edges and it is the protocol, not
   the transport, that the poster has to fix -- exactly the section 15 conclusion,
   re-confirmed with the fabric in the loop.
2. **At 4 MB NVSHMEM wins 6-10%**: 47 GB/s GDR against 39 GB/s staged, no host copies.
3. **Splitting vs allreduce**: under MPI, branch beats allreduce by 21% at 64 KB (as in
   section 18) and the two are within rep noise at 4 MB; under NVSHMEM, allreduce ties at
   64 KB and wins 4% at 4 MB, as it did under NCCL (section 19). Three transports and two
   placements now agree: splitting's advantage is a small-message, host-blocking-cost
   effect. (The MPI 4 MB arms are 10-15% slower than on 2026-08-15; the only deliberate
   change is the `UCX_NET_DEVICES=mlx5_0:1` pin -- in August UCX auto-selected, and the
   other node had both 400G ports up -- not investigated.)

### 28.4 NCCL across the fabric on the right NIC

Sections 21-22 measured 2-node NCCL on what turned out to be the 25 GbE management bond,
without GDR. Rerun on `mlx5_0` with `NCCL_IB_ROCE_VERSION_NUM=1 NCCL_IB_GID_INDEX=2` (build
notes 13.4) and GDRDMA channels; 2+2 placement, 300 iterations, two reps each, us per
iteration; the 2026-08-15 bond numbers in brackets:

| msg | G,P | branch-NCCL | allreduce-NCCL | branch adv. |
| ---: | --- | ---: | ---: | ---: |
| 64 KB | 8,4 | 176, 177 [296] | 188, 188 [291] | 6% |
| 512 KB | 8,4 | 272, 281 [1542] | 355, 358 [1183] | 24% |
| 1 MB | 8,4 | 372, 373 [3005] | 375, 382 [2268] | 2% |
| 4 MB | 8,4 | 995, 996 [11709] | 897, 903 [8802] | -10% |
| 64 KB | 32,16 | 341, 343 [431] | 354, 354 [450] | 3% |
| 512 KB | 32,16 | 452, 454 [1572] | 539, 561 [1257] | 19% |
| 1 MB | 32,16 | 554, 557 [3036] | 566, 571 [2284] | 2% |
| 4 MB | 32,16 | 1239, 1239 | 1170, 1170 | -6% |

1. **NCCL is now the fastest eager inter-node arm at 64 KB**: 177 us against 206
   (branch-MPI host-staged), 192 (branch-MPI with GDR, 28.5) and 280 (PetscSF-NVSHMEM).
   It fuses each DAG stage into one launch and never blocks the host; that advantage
   survives the fabric. Against its own single-node number (107 us, section 19) the
   fabric adds ~70 us at 64 KB for 8 crossing edges.
2. **The bond numbers were an artefact of the wrong NIC**: 11.7 ms -> 1.0 ms at 4 MB
   (11.8x), 296 -> 177 us at 64 KB. The section 21 statement that "inter-node NCCL stages
   at ~3 GB/s, so MPI stays the inter-node transport of record" is withdrawn: at 4 MB
   NCCL (996 us) matches MPI (1015-1041) and NVSHMEM (956-965), and at 64 KB it leads.
3. **Splitting vs allreduce under inter-node NCCL**: branch wins 6% at 64 KB and 19-24% at
   512 KB, ties at 1 MB, loses 6-10% at 4 MB. Different from the single-node NCCL result
   (allreduce matched or beat branch, section 19): with 8 of 12 edges on the wire, the
   volume argument does bite at mid sizes, and at 4 MB the ring allreduce's pipelining
   wins back. So the poster's phrasing stands: splitting is a transport-dependent
   advantage, largest where per-edge cost is host-side or where volume crosses a fabric.

### 28.5 GPU-aware MPI with GPUDirect (UCX 1.19 preload): the fair MPI baseline

HPC-X's UCX 1.17 cannot see this peermem (build notes 13.4), so every MPI number above is
host-staged. With the DOCA UCX 1.19 preloaded on both nodes (`JANUS_UCX119=1`), the same
`acgnbench` binary, 2+2 placement, two reps, us per iteration; host-staged (28.3) in
brackets:

| msg | branch-MPI, GDR | allreduce-MPI, GDR |
| ---: | ---: | ---: |
| 64 KB | 192, 194 [206, 208] | 229, 232 [263, 265] |
| 512 KB | 297, 297 | 333, 335 |
| 1 MB | 397, 399 | 420, 420 |
| 4 MB | 999, 1004 [1015, 1041] | 1015, 1044 [991, 1026] |

The preload is genuinely zero-copy: the 2-node GPU ping-pong under UCX 1.19 on both
nodes gives 10.8 us one-way at 8 B (19.4 host-staged), 38.8 GB/s at 2 MB (22.9) and
47.5 GB/s at 16 MB (39.2) -- i.e. MPI's transport floor is now the same 10-11 us as
NVSHMEM's proxy put (28.1). (One UCX artefact: 32 KB sits just above the rendezvous
threshold and costs 35.6 us one-way; `UCX_RNDV_THRESH` would move it; not tuned.)

GDR buys MPI 7% (branch) and 13% (allreduce) at 64 KB and nothing at 4 MB. The
host-staged path was already pipelining bulk transfers well (39 vs 47 GB/s); what it could
not hide was the per-message staging latency, which is what the small-message DAG pays.
Branch still beats allreduce under GDR-MPI by 16% at 64 KB, so the section 18 result is
not a host-staging artefact.

**At 4 MB every inter-node arm converges to ~1 ms** -- MPI host-staged 1015-1041,
MPI-GDR 999-1004, NCCL 995-996, PetscSF-NVSHMEM 956-965 -- because the 2+2 placement pushes
6 x 4 MB out of node 1 per iteration through one 400G port with DAG dependencies limiting
overlap: this is a placement/volume bound, not a transport one, and it is the regime the
4xN placement-aware instance (section 18, observation 2) is meant to attack.

### 28.6 IBGDA in CPU-doorbell hybrid mode: measured, and why the driver parameters matter

`NVSHMEM_IB_ENABLE_IBGDA=1` initializes here only in NVSHMEM's fallback ("NIC handler will
be CPU with host memory backend", build notes 13.4): the GPU writes the work-queue entries
in GPU memory but a CPU thread has to ring the NIC doorbell because the GPU cannot map the
doorbell page. Correctness is fine -- the PETSc suite is identical to the MPI path in all
three cases at np=8 (the harness first reported MISMATCH because IBGDA's warning lines
polluted the diff; the filter now drops them). Performance is not:

| metric, inter-node | IBRC proxy (28.1) | IBGDA hybrid |
| --- | ---: | ---: |
| device `put` latency, 8 B | 10.1 us | 16.1 us |
| `put_signal` ping-pong, round trip | 25.5 us | 36.6 us |
| `sfbench` NVput, 8-rank ring, 64 B - 64 KB | 25.3 - 25.5 us | 45.6 - 47.7 us |
| `sfbench` NVput, 8-rank ring, 16 MB | 674 us | 683 us |
| `sfbench` NVget, 8-rank ring, 64 B | 37.1 us | 68.0 us |

The hybrid adds a GPU->CPU doorbell hop on top of the proxy's own latency, so every
small-message number gets ~1.6-1.9x worse while bandwidth is unchanged. In this mode
IBGDA is strictly worse than IBRC and there is no reason to use it. What the true mode
would remove is the CPU from the issue path altogether (the fourth lane in 28.1); that
needs the `nvidia` module loaded with `PeerMappingOverride=1` / `EnableStreamMemOPs=1`
(ticket follow-up 5). The ACGN DAG under the hybrid is in the table below.

ACGN DAG, 2+2, G=8, P=4, two reps, us per iteration; IBRC-proxy numbers (28.3) in brackets:

| arm | 64 KB | 4 MB |
| --- | ---: | ---: |
| branch, NVSHMEM put+signal | 317, 322 [279, 281] | 949, 963 [956, 965] |
| allreduce, NVSHMEM | 320, 322 [282, 286] | 983, 987 [920, 924] |
| branch, NVSHMEM get | 361, 365 [280, 284] | 1168, 1175 [948] |

Same verdict: +15% at 64 KB, +30% for the get protocol (two doorbell hops per edge),
neutral for bandwidth-bound puts. Closed until the driver parameters change.

### 28.7 Graph-replayed NCCL across the fabric

The poster's single-node headline (section 20: one CUDA-graph launch per 10 iterations,
32 us/iter at 64 KB) rerun 2+2 on the 400G port, G=8, P=4, two reps, us per iteration:

| msg | graph-NCCL, 2+2 | eager NCCL (28.4) | 1-node graph-NCCL control |
| ---: | ---: | ---: | ---: |
| 64 KB | 122, 122 | 177 | 32.1 (August: 32) |
| 512 KB | 226, 227 | 272-281 | -- |
| 4 MB | 926, 927 | 995-996 | -- |

**122 us is the best inter-node number of the campaign**: 1.7x the host-staged MPI arm
(206), 1.6x the GDR-MPI arm (192), 2.3x PetscSF-NVSHMEM (280). It removes the per-op
launch tax the eager arm still pays (55 us of the 177), and the remaining 90 us over the
single-node figure is the fabric: 8 crossing edges at ~10-12 us each on the critical path.
That is the number a placement-aware 4xN instance would attack.

### 28.8 The inter-node matrix, and what it says for the poster

Every arm the campaign has, on the same DAG, 64 KB, G=8, P=4, us per iteration:

| arm | 1 node (sections 17-27) | 2 nodes, 2+2 (this section) | fabric cost |
| --- | ---: | ---: | ---: |
| branch-MPI, host-staged (UCX 1.17) | 162 | 206 | +44 |
| branch-MPI, GPUDirect (UCX 1.19) | -- | 192 | -- |
| allreduce-MPI, host-staged | 209 | 264 | +55 |
| branch-NCCL, eager | 107 | 177 | +70 |
| branch-NCCL, CUDA-graph replay | 32 | 122 | +90 |
| branch, PetscSF-NVSHMEM (fused put+signal) | 206 | 280 | +74 |
| branch, PetscSF-NVSHMEM, IBGDA hybrid | -- | 320 | -- |
| device-side NVSHMEM kernel (`acgnbench-nvdev`) | 41 | not yet (needs remote-edge variant) | -- |

Five statements the poster can now make with measured backing:

1. **The ordering of the arms survives the fabric.** Graph-NCCL < eager NCCL < MPI <
   PetscSF-NVSHMEM on one node and on two. Nothing reshuffles when GPUDirect is real.
2. **NVSHMEM's problem was never the transport.** Its raw inter-node put (10 us) equals
   GDR-MPI's (10.8) and halves host-staged MPI's (19.4); yet PetscSF-NVSHMEM is the slowest
   arm at small sizes on both node counts. The ~5 stream ops + completion wait per exchange
   (sections 14-15) is the whole story, and it is a PETSc-side protocol cost, which is the
   argument for the device-side route and for the TaoTerm items.
3. **GPUDirect is worth 7-13% to MPI at 64 KB and nothing at 4 MB**; the host-staged MPI
   numbers of section 18 were not far off a fair baseline.
4. **Splitting vs allreduce is transport- and placement-dependent, and the poster should
   say so**: branch wins 21% (MPI staged), 16% (MPI GDR), 6% (NCCL eager) at 64 KB, ties
   under PetscSF-NVSHMEM, and reverses at 4 MB under NCCL and NVSHMEM.
5. **The fabric cost is 44-90 us per iteration at 64 KB with 8 of 12 edges crossing**,
   largest for the arms that had the least slack. Placement-aware mapping (section 18,
   observation 2) is the next lever, and it is a scheduling result, not a transport one.

Left open, in order of poster value: the placement-aware 4xN instance; the device-side
kernel across nodes (replace the `nvshmem_ptr` pull with `nvshmem_getmem`/put edges; the
IBGDA hybrid can carry it today, the true IBGDA mode is what would make it fast); and a
GDR-capable HPC-X from ALCF so the MPI arms stop needing a preload.

### 28.9 Placement cuts of the 4-role DAG over two nodes (2026-09-01/02, same job)

The first placement-aware experiment needed no new code: the 4-rank DAG has role-pair edge
weights (1,2)=3, (1,3)=3, (2,3)=3 and only 1 for each pair involving role 0, so the cut that
isolates role 0 crosses 3 edges instead of 8. Five placements (rankfiles in
`nvshmem-tools/placements/`; hostfile order is ignored under PBS and a bare `slot=N` pins
each rank to one core, so rankfiles with `slot=0-63` are the only working mechanism), five
transports, G=8, P=4, best of 2, us per iteration:

64 KB:

| placement | cross | chain crossings | MPI staged | MPI GDR | NCCL eager | NCCL graph | PetscSF-NVSHMEM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {0,1} / {2,3} (section 28) | 8 | 3 | 206 | 193 | 176 | 121 | 279 |
| {1,2} / {0,3} | 8 | 4 | 224 | 208 | 205 | 142 | 278 |
| **{1,2,3} / {0}** | **3** | 2 | **192** | **164** | **172** | **102** | refused |
| {0,1,2} / {3} | 7 | 3 | 201 | 206 | 176 | 117 | refused |
| {0,2,3} / {1} | 7 | 3 | 209 | 196 | 177 | 114 | refused |

4 MB:

| placement | cross | out/in at the busier node | MPI staged | MPI GDR | NCCL eager | NCCL graph | PetscSF-NVSHMEM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {0,1} / {2,3} | 8 | 6 / 2 | 1002 | 1002 | 997 | 918 | 955 |
| {1,2} / {0,3} | 8 | 5 / 3 | 895 | 928 | 996 | 921 | 1075 |
| **{1,2,3} / {0}** | **3** | 2 / 1 | **563** | **576** | **796** | **670** | refused |
| {0,1,2} / {3} | 7 | 4 / 3 | 908 | 896 | 1018 | 933 | refused |
| {0,2,3} / {1} | 7 | 3 / 4 | 855 | 940 | 997 | 909 | refused |

("chain crossings" counts fabric hops on the serial dependency chain x1 -> g -> x2 -> x3 ->
w for the rank that closes the iteration; "out/in" counts messages leaving/entering the
node that sends most.)

1. **The placement-aware cut is worth 7-16% at 64 KB and 20-44% at 4 MB**, with no change
   to the algorithm or the transport. At 64 KB graph-NCCL goes 121 -> 102 us and GDR-MPI
   193 -> 164; at 4 MB every MPI arm nearly halves (1002 -> 563) because the busier node's
   outbound traffic through its one 400G port drops from 6 messages per iteration to 2.
   This is the section 18 observation 2 lever, measured: **placement recovers half to
   two thirds of the fabric cost** for the transports that could use it.
2. **Same crossing count, different time: the two 2+2 cuts differ by 9-17% at 64 KB.** The
   {1,2}/{0,3} cut puts one more fabric hop on the serial chain; the count of crossing
   edges is the wrong objective for latency-bound sizes. At 4 MB the ordering flips for
   MPI (895 vs 1002) because that cut balances the per-port volume (5/3 vs 6/2). Two
   sizes, two objectives, two optimal placements -- the joint problem the poster's step 1
   names, seen in one table.
3. **NVSHMEM cannot run an unbalanced placement at all.** Every 3+1 and 1+3 mapping aborts
   in `nvshmem_bootstrap` with `NVSHMEM requires the same number of PEs on all nodes`.
   That is a hard constraint of the library, not of PETSc: the placement space available
   to a PetscSF-NVSHMEM (or device-side NVSHMEM) design is only the balanced cuts, and the
   best cut above is outside it. Padding with idle PEs would satisfy NVSHMEM but trips
   PetscSF's eligibility check (an SF with an all-NULL rank falls back to MPI), so a
   4xN benchmark that wants the NVSHMEM arms must give every node the same number of
   role-shards -- a constraint the poster's step-2 family ("fix term-to-node placement")
   should state.
4. **NCCL benefits least at 4 MB** (997 -> 796, against 1002 -> 563 for MPI): with only
   role 0's two 4 MB sends crossing, NCCL's per-connection pipelining, not the port, sets
   its floor. Its graph version recovers more (918 -> 670).

What this settles for the 4xN design: the objective must be modelled as chain latency
(hops on the serial dependency path x per-hop cost) plus per-port volume (bytes out of the
busiest node / port bandwidth), not as a crossing-edge count; and balanced placements are
the only ones all transports can share.

### 28.10 Collective and point-to-point exposed-cost curves, and where the numbers went (2026-09-02)

`nvshmem-tools/collbench.c` (results in `results-20260902-collbench.txt`): each op issued and
synchronised before the next, so the number is the exposed cost a dependency chain pays.
Four ranks; "intra" = one host, "2+2" = {0,1}/{2,3}; p2p = rank 0 <-> rank 3 one-way. us:

| bytes | NCCL p2p intra | NCCL p2p 2+2 | NCCL allreduce intra | NCCL allreduce 2+2 | MPI allreduce intra | MPI allreduce 2+2 | MPI-GDR allreduce 2+2 | MPI p2p 2+2 | MPI-GDR p2p 2+2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 7.2 | 16.4 | 14.4 | 26.2 | 57.8 | 63.5 | 30.4 | 20.0 | 10.4 |
| 4 KB | 6.8 | 16.2 | 15.8 | 29.3 | 58.7 | 103.5 | 47.3 | 22.2 | 11.4 |
| 256 KB | 9.5 | 24.1 | 18.1 | 96.2 | 54.7 | 88.8 | 48.2 | 41.2 | 14.7 |
| 2 MB | 20.6 | 64.6 | 32.9 | 157.1 | 67.5 | 211.1 | 159.1 | 92.7 | 54.8 |
| 16 MB | 83.8 | 367.6 | 108.7 | 1032.7 | 129.0 | 1196.8 | 991.3 | 436.2 | 354.3 |

Two readings: NCCL's small-message collectives are 4x cheaper than HPC-X MPI's on one host
(14 vs 58 us) and 2.4x cheaper across hosts; a 16 MB allreduce across the two hosts costs
~1 ms under every library (one 400G port per host bounds it), so at 4 MB per vector the
DAG's ~1 ms per iteration (28.5) is that bound.

**Where all of section 28 now lives:** the measured catalog, a schedule simulator that
reproduces this section's per-iteration times to a factor 1.23 over 80 cells, and a
re-screen of the poster's 3072 ACGN skeletons under it, are in the research harness
`~/prox-latency/experiments/` (`harness/machine_janus.py`, `harness/schedule.py`,
`validate_schedule.py`, `rescreen_measured.py`; findings in that directory's `RESULTS.md`,
Result 7). Headline: the synthetic-constant winner is not top-ranked under measured costs
(the `[late,p2,late,p2] complete` family is, by 5-35%), branch routing still beats the
global allreduce by 1.07-2.8x, and once operators cost more than ~30 us the ranking is set
by iteration counts, not by transport.

### 28.11 Real operator costs on one H100, and what they do to the skeleton ranking (2026-09-02)

`nvshmem-tools/opbench.c` + `tvprox.cu` (`results-20260902-opbench.txt`), per shard, 45 of
180 angles, synthetic ray-like sparse projector (2n contiguous pixels per ray):

| image | vector | gradient Aᵀ(Ax−b) | rowTV prox | colTV prox | box | mixing |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128² | 128 KB | 26 us | 125 us | 136 us | 2.4 us | 4.8 us |
| 256² | 512 KB | 74 us | 245 us | 260 us | 2.4 us | 5.2 us |
| 512² | 2 MB | 257 us | 481 us | 507 us | 3.0 us | 6.4 us |
| 1024² | 8 MB | 1012 us | 952 us | 1014 us | 5.1 us | 10.8 us |

Against section 28's exchange costs (10-30 us per hop, ~1 ms for a 16 MB allreduce), the
ACGN iteration at 512 KB vectors is ~0.8 ms of which ~0.1 ms is communication. Fed into the
`~/prox-latency` schedule model (RESULTS.md Result 7): the poster's skeleton stays rank 1
of 114 finalists in every 64 KB cell and within 1-3% of the best elsewhere; the global
allreduce arm costs 1.3-2x; placement and transport, not the routing skeleton, are the
remaining design freedom. The TV kernel is a naive one-thread-per-row Condat (an upper
bound); a 10x faster prox moves the 4 MB two-host cells to gradient-routing skeletons.

`nvshmem-tools/acgnrun.c` is the generic executor for any skeleton (`-order`, `-grad`);
it matches `acgnbench` within 2% and is the instrument for closing the loop on the
model's ranking (`results-20260902-closeloop.txt`).
Loop closed 2026-09-02 (`results-20260902-closeloop.txt`, 80 cells, five skeletons, three
PetscSF transports, three placements, two sizes): model error x1.09, skeleton ranking
agreement 113/124 pairs; measured spread between skeletons 10-26% per cell.

Later the same night: `acgnrun-nccl.c` (NCCL / graph counterpart of the executor;
reproduces acgnbench-nccl 107 vs 110 us and the graph arm 31.9 vs 32.1 us) closed the loop
on 140 cells over five transports -- PetscSF transports modeled to ~10% with 113/124 ranking
pairs right, NCCL/graph ×1.23-1.36 with ~2/3 of pairs right (`results-20260902-closeloop.txt`).
Gang gradients (`gangbench.c`, `results-20260902-gangbench.txt`): a shard's gradient over
g GPUs of a node incl. the NVLink reduce: 256² 80 / 56 / 41 us, 512² 264 / 171 / 98 us,
1024² 1016 / 604 / 338 us for g = 1 / 2 / 4. Measured fastest DAG under NCCL and graph in
almost every cell: `box>colTV>rowTV [p3,split,late,split]` (30.7 us single-host graph vs
the poster winner's 31.8), a near-tie in time-to-tolerance at 261 vs 225 iterations.

### 28.12 The device-side kernel across the fabric (push variant), and five residuals answered (2026-09-02, 03:00-03:30)

`acgnbench-nvdev2.cu` = `acgnbench-nvdev.cu` plus `-push`: a producer no longer publishes
into its own buffer for consumers to pull through `nvshmem_ptr`; each thread block fills a
contiguous chunk and `nvshmemx_double_put_nbi_block`s it into a symmetric receive lane on
every consumer, the last-arriving block `nvshmem_fence`s and fires the signals, and
consumers read their local lane. The ack chain is unchanged (its meaning becomes "my
receive slot for it-2 is free"); the allreduce all-pull and the convergence test's
partial-norm pull become pushes the same way. Pull mode is kept as the control.
`results-20260902-nvdev-push.txt`, persistent kernel, G=8, P=4, best of 2, us/iteration:

| arm | 1 node, 64 KB | 2+2, 64 KB | 1 node, 4 MB | 2+2, 4 MB |
| --- | ---: | ---: | ---: | ---: |
| pull (NVLink only) | 45.6 | -- | 738 | -- |
| push, branch | **36.5** | **104** (85 seen once) | 740 | 1080 |
| push, allreduce (all-push) | 35.9 | 110 | 706 | 1068 |
| push, branch, device stopping test (k=10) | -- | 90-113, stops at exactly 210 = predicted | -- | -- |

Zero element mismatches in every run. Three things this says: (1) **the device-resident
iteration is now the fastest inter-node arm at 64 KB** (104 us vs graph-NCCL 122, eager
NCCL 177, GDR-MPI 192, PetscSF-NVSHMEM 280), over the IBRC proxy, with the stopping
decision taken on the GPUs across the fabric -- the capability no replayed graph has;
(2) on one node the push is 20% faster than the pull it replaces (block-contiguous puts
over NVLink beat strided volatile peer reads), so the single-node headline moves from 41
to 36.5 us; (3) at 4 MB the device arm sits at the same ~1 ms port bound as every other
arm. True IBGDA (ticket follow-up 5) is what would move the 2+2 number toward 70 us.

Residuals from 28.8/28.9 that the same night answered (`results-20260902-residuals.txt`):

- **NCCL fan-out is serial.** One GPU sending 1->2 remote peers in one group costs exactly
  two p2p times at every size (28.5 vs 16.7 us at 8 B; 729 vs 367 us at 16 MB). That is the
  4 MB under-prediction of 28.4/28.7; the harness model now serialises a GPU's NCCL sends.
- **PetscSF's put at 16 MB is proxy-throughput bound, not serialisation bound.** With K
  exchanges in flight (`sfbench2 -nexch`, 1 rank per node) NVSHMEM put gains 1.4x at 1 MB
  (68 -> 49.5 us per exchange) but nothing at 16 MB (677 -> 891-1033 us per exchange): one
  host-issued on-stream put streams at ~25 GB/s through the proxy and more of them do not
  add up; the device-issued 32-CTA put of 28.1 is what reaches 47 GB/s.
- **The 32 KB hole in GPUDirect MPI is the rendezvous threshold**: `UCX_RNDV_THRESH=16k`
  turns 35.6 us one-way into 10.0 with no change at any other size (now set by
  `JANUS_UCX119=1`).
- **NVSHMEM `get` bandwidth really collapses at 16 MB** (37 GB/s at 4 and 8 MB, 3.8 at 16,
  reproducible with 200 iterations; put stays at 46-47) -- a proxy-path pathology worth
  reporting; do not use the get protocol above 8 MB.
- **The MPI 4 MB drift is run-to-run variance, not the port pin**: pinned 912 us vs
  unpinned 1043 us today, 1002-1041 earlier tonight, 886 in August.

## 29. The SF-NCCL backend, built and measured: where NCCL beats GPU-aware MPI (2026-09-09)

Job 2779, nodes `x2003c0s9b0n0` + `x2003c0s5b0n0`, NCCL 2.28.3 (`--with-nccl-dir`), HPC-X Open MPI
4.1.7a1 / UCX 1.17 (host-staged across nodes; the `mpigdr` arm is UCX 1.19 with `JANUS_UCX119=1`),
NVSHMEM 3.4.5. Raw output: `nvshmem-tools/results-20260909-sfnccl.txt`. Everything below is the
`-use_nccl` transport of section 3.2 of `SESSION-SUMMARY-2026-09-04.md`, compiled today for the
first time; no tuning was done beyond what is listed.

### 29.1 Correctness and the engagement checks

`sf/tests/ex22`, `snes/tutorials/ex19` and `ksp/ksp/tutorials/ex45` are byte-identical to MPI at
np=4 (one node) and np=8 (two nodes); the harness test `ex22_cuda_nccl` passes at nsize 1 and 4.
Positive discriminators: `-info :sf` prints `PetscSF will use NCCL ...`, `-info` prints
`Created a NCCL (version 22803) communicator with N ranks ... in T seconds`. One rank: refused
(single-rank SF). Eight ranks on four GPUs: `Some ranks of MPI_Comm ... share a GPU ... PetscSF
will use MPI on it`, output still correct. `-log_view` counts NCCL sends/receives as MPI messages
(after switching the counters to `PetscAddLogDouble()`; raw `+=` misses the `_th` copies the
event accounting reads).

### 29.2 The ring: NCCL halves latency below 256 KB, loses above 2 MB

`sfbench` (us per `PetscSFBcast`, max over ranks, best of 3):

| bytes | 1 node MPI | 1 node NCCL | NVput | 2 nodes/8 MPI | NCCL | NCCL ch8 | GDR-MPI | 2+2 MPI | NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 B-8 KB | 13.1 | 6.4 | 17.3 | 19.6 | 14.8 | 15.0 | 20.5 | 21.3 | 16.2 |
| 32 KB | 13.5 | 8.1 | 17.8 | 25.9 | 16.7 | 18.2 | 16.7 | 25.8 | 18.9 |
| 256 KB | 15.3 | 13.4 | 19.8 | 39.0 | 25.9 | 25.7 | 42.7 | 39.0 | 29.0 |
| 1 MB | 21.1 | 23.1 | 26.6 | 66.1 | 57.5 | 45.7 | 70.9 | 72.1 | 67.3 |
| 4 MB | 45.2 | 58.0 | 53.2 | 138 | 171 | 156 | 164 | 171 | 213 |
| 32 MB | 265 | 305 | 338 | 828 | 1270 | 1220 | 1012 | 1438 | 1566 |

- One node: 2.0x at 64 B-32 KB (6.4 vs 13.1 us), parity at 512 KB, 15-28% slower from 1 MB up
  (NCCL's SM-driven p2p copies move less than UCX's `cuda_ipc` copy engines).
- Two nodes: 25-35% faster up to 1 MB, 20-55% slower from 2 MB. GDR-MPI (UCX 1.19) ties NCCL in
  its 16-64 KB rendezvous window and is worse elsewhere (eager host-staged below `UCX_RNDV_THRESH`).
- `NCCL_NCHANNELS_PER_NET_PEER=8 NCCL_MIN_NCHANNELS=8` helps only 512 KB-1 MB (38 -> 31, 58 -> 46 us).
- Against non-GPU-aware MPI (`-use_gpu_aware_mpi 0`, host staging) NCCL is 2x at 64 B and 17x at 4 MB.
- NVSHMEM put stays 1.2-1.3x slower than MPI on one node (section 21's protocol cost); NCCL wins
  because one group kernel replaces PetscSF's five stream ops.

### 29.3 The legacy null stream costs 35 us per exchange; use a nonblocking PETSc stream

`sfbench2 -n 4096` (32 KB) with `-naxpy k` VecAXPYs on a 1M work vector between Begin and End,
one node, us per iteration (`compute_alone` is the same loop without the exchange):

| PETSc stream | MPI k=8 | NCCL k=8 | MPI k=32 | NCCL k=32 | compute alone k=8 / 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| default (legacy null) | 49.7 | 55.7 | 178 | 218 | 42 / 169 |
| `-root_device_context_stream_type nonblocking` | 41.9 | 47.6 | 167 | 174 | 42 / 168 |

On the null stream NCCL adds ~45 us per iteration; on a nonblocking stream ~5 us (its host enqueue).
Independent of `NCCL_MAX_NCHANNELS` (1/2/4), `NCCL_PROTO` (Simple/LL) and `NCCL_P2P_LL_THRESHOLD`,
so it is not SM contention; it is the null stream's implicit synchronization with NCCL's internal
(blocking) streams. Rule: with `-use_nccl`, run with `-root_device_context_stream_type nonblocking`.
The same option is what CUDA-graph capture needs (29.5).

The `-w 16777216` arm (128 MB work vectors, GPU-bound AXPYs of 185 us each) shows the exchange
fully hidden under every transport (MPI +0.1 us, NCCL +1.5 us, NVput +11 us over compute alone at
32 KB, both placements): once there is enough GPU work to hide behind, the transport does not
matter, and the 1M default is host-launch-bound (5 us per `VecAXPY` on the host, ~2.5 us on the
GPU), which is why sections 21-23 saw exposed costs at all.

### 29.4 The ACGN DAG: NCCL wins the 64 KB skeleton by 37% on one node

`acgnbench` with the nonblocking stream, us per iteration (G=8, P=4):

| placement | msg | branch MPI | branch NCCL | branch NVput | allreduce MPI | allreduce NCCL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 node | 64 KB | 148 | **93** | 109 | 209 | 132 |
| 1 node | 4 MB | 318 | 377 | 301 | 506 | 448 |
| 2+2 | 64 KB | 190 | **165** | 183 | (see below) | |
| 2+2 | 4 MB | 927 | 1060 | 854 | | |

(On the null stream the same runs gave 160 / 212 / 183 at 64 KB: NCCL lost. The `-allreduce` arm
keeps its host `MPI_Allreduce`; only the three broadcasts use NCCL.) The 93 us sits between the
hand-rolled eager NCCL groups (107, which fuse a whole stage) and graph replay (32): the SF form
cannot fuse across SFs, but it removes the host block in every stage. At 4 MB the bulk-bandwidth
loss of 29.2 dominates and NVSHMEM put is best.

Independent exchanges in flight serialize under NCCL: `sfbench2 -nexch 4` costs 12.1 us per
exchange on one node (MPI 9.2) and 22.8 at 2+2 (MPI 11.9), because NCCL orders every operation on
one communicator, so four groups on four streams run back to back while MPI pipelines them. A
per-link communicator (`ncclCommSplit`) would fix it at the price of one init per link.

### 29.5 CUDA-graph capture works, and turns 45 us into 9

`nvshmem-tools/sfgraph.c` captures m Begin/End pairs (plus k AXPYs each) into one graph and
replays it. The MPI path aborts at `cudaStreamBeginCapture()` (host sync in Begin); NCCL captures.
us per exchange, 32 KB:

| placement | eager k=0 | graph m=1 | graph m=16 | eager k=8 | graph m=1 k=8 | graph m=16 k=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 node | 8.1 | 11.4 | 9.0 | 45 | 11.4 | 9.0 |
| 2+2 | 19.1 | 61 | 23.1 | 45 | 66 | 23.1 |

A single-exchange graph costs a graph launch (~3 us on one node, ~40 us across nodes, where the
NCCL network kernels' host proxy work moves inside the launch); sixteen per graph amortize it. The
k=8 columns are the point: eager, the host cannot launch 8 AXPYs faster than 42 us; replayed, the
same work plus the exchange runs in 9 us. This is the 5x of section 28.3 available from PetscSF
without hand-written NCCL code, for any loop body PETSc can enqueue (no host-side decisions).

### 29.6 Real solver: NCCL costs 2.5-3 s once, then is 20% better per iteration on a node and 2x worse across nodes

`ksp/ex45` (7-point Laplacian, GMRES/Jacobi, `aijcusparse`), `KSPSolve` seconds:

| grid, iterations | 1 node MPI | 1 node NCCL | 2 nodes MPI | 2 nodes NCCL |
| --- | ---: | ---: | ---: | ---: |
| 32^3, 80 | 0.344 | 3.239 | 0.351 | 2.828 |
| 64^3, 222 | 0.354 | 3.238 | 0.372 | 2.950 |
| 96^3, 381 | 0.408 | 3.290 | 0.424 | 2.982 |
| per iteration from the 32 -> 96 slope | 0.21 ms | 0.17 ms | 0.24 ms | 0.51 ms |

The 2.9 s is one-time: `ncclCommInitRank` takes 2.4-2.5 s (NCCL's own timing: `alloc 2.13 s`,
its cuMem-based buffers mapped to every peer; `NCCL_CUMEM_ENABLE=0` makes it 2.9 s), and NCCL
connects all 128 (peer, channel, direction) p2p links lazily inside the first exchange (nsys:
64 `cuMemExportToShareableHandle` at 5 ms each on the service thread, 162 `cuMemSetAccess` at 1 ms
on the main thread). `NCCL_MAX_NCHANNELS=1`, `NCCL_P2P_DISABLE=1`, `NCCL_LOCAL_REGISTER=0`,
`NCCL_LEGACY_CUDA_REGISTER=0` move it by at most 0.6 s. The backend now prints the init time in
`-info`. So: NCCL only pays off for solves longer than ~3 s, i.e. thousands of iterations or
repeated solves in one process, and on one node; across two nodes the per-iteration cost doubled
(0.51 vs 0.24 ms) even though the ring says 73 KB halos are 20% cheaper under NCCL -- the DMDA
pattern (3-5 peers, mixed intra/inter-node, plus host-scalar `MPI_Allreduce`s on the same NIC) is
not the ring, and this was not chased further.

### 29.7 Fabric trap: UCX and NCCL must use the same RoCE version on mlx5_0

With NCCL on RoCE v1 (the section 28 pin) and UCX on its RoCE v2 default, a host `MPI_Allreduce`
on 64 KB GPU buffers at 2+2 took 11.1 ms instead of 260 us (347 ms at 4 MB), regardless of
`NCCL_NET_GDR_LEVEL=0` or `UCX_MEMTYPE_CACHE=n`; `NCCL_IB_DISABLE=1` (NCCL over sockets) also
avoided it. `UCX_IB_GID_INDEX=2` (UCX on RoCE v1 too) restores 272 us and costs nothing for
MPI-only runs (266 us). It is now a default in `janus-env-nvshmem.sh`. The 2-node ex45 numbers
above were taken with it.

### 29.8 Answer to "when does NCCL help", in one place

NCCL (through PetscSF, no application change) beats GPU-aware MPI when the exchange is
latency-bound (< ~512 KB on a node, < ~1 MB across nodes: 1.3-2x), when the application runs on a
nonblocking PETSc stream, when it can capture its loop into a CUDA graph (5x on a host-bound loop),
when MPI is not GPU-aware (2-17x), and when the process lives long enough to amortize ~3 s of NCCL
setup. It loses for bulk messages above 2 MB (up to 1.5x), for several independent exchanges in
flight, and (unexplained) per iteration for the 2-node DMDA solve. NVSHMEM through PetscSF never
beats either on a node; its niche stays the device-initiated designs of sections 23-28.

### 29.9 Why NCCL is slower than GPU-aware MPI at 4 MB, taken apart (2026-09-09, later)

`nvshmem-tools/sfbw.c` exchanges the same device buffers through PetscSF, bare `MPI_Isend/Irecv`
or a bare `ncclGroup`, ring or pair, each message size in its own `PetscLogStage` (setup, the
NCCL communicator and first-touch connections sit in other stages, so `-log_view` reports the
exchange alone; the wall clock per stage agrees with it). One node, us per exchange (GB/s sent
per rank), 40 iterations:

| bytes | SF-MPI | bare MPI | SF-NCCL | bare NCCL | pair MPI | pair NCCL | uni MPI | uni NCCL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MB | 27.2 | 25.2 | 23.5 | 23.2 | 25.4 | 26.5 | 24.7 | 18.0 |
| 4 MB | 48.9 (86) | 46.9 (89) | 57.6 (73) | 56.9 (74) | 47.9 (88) | 64.9 (65) | 49.2 (85) | 44.5 (94) |
| 16 MB | 143 (117) | 141 (119) | 169 (99) | 165 (102) | 141 (119) | 218 (77) | 144 (116) | 148 (113) |
| 64 MB | 515 (130) | 518 (130) | 580 (116) | 579 (116) | 520 (129) | 816 (82) | 519 (129) | 567 (118) |

(ring = np 4, pair = np 2 exchanging with each other, uni = pair with one side sending only.)

1. **PetscSF is not the cause.** SF-MPI equals bare MPI and SF-NCCL equals bare NCCL within
   2 us at every size; the side-stream fork/join costs nothing measurable.
2. **The transports differ in kind.** nsys on the MPI arm shows one 4 MB peer-to-peer
   `cudaMemcpy` (copy kind PtoP, 39.9 us) per exchange: UCX `cuda_ipc` maps the peer's buffer
   and lets a copy engine DMA it, full rate from the first byte, both directions independent.
   The NCCL arm shows one `ncclDevKernel_SendRecv` of 4 blocks x 640 threads per 4 MB op (8
   blocks in the ring, where send and receive go to different peers): SMs doing loads and stores
   over NVLink through NCCL's staging buffers. Four SMs move ~94 GB/s one way (23 GB/s each),
   and in the bidirectional pair the same four blocks serve both directions, so each gets half
   (65 vs 45 us). A copy engine does not care about direction (pair MPI = uni MPI).
3. **The block count is NCCL's size heuristic, not a knob.** `NCCL_MIN/MAX_P2P_NCHANNELS=32`
   and `NCCL_MIN/MAX_NCHANNELS=32` leave the grid at 8 (16 only at 64 MB); 16 and 8 make it
   worse (65 and 109 us at 4 MB). `NCCL_BUFFSIZE`, `NCCL_NTHREADS`, `NCCL_PROTO=Simple`,
   `NCCL_P2P_READ_ENABLE`, `NCCL_P2P_DIRECT_DISABLE` do nothing. `NCCL_P2P_USE_CUDA_MEMCPY=1`
   (NCCL's own copy-engine path) is 2.4x worse (137 us): it stages through the proxy.
4. **What helps: a smaller NVLink chunk.** `NCCL_P2P_NVL_CHUNKSIZE` (default 512 KB) at 64 KB
   gives 49.3 us at 4 MB (bare MPI 46.9) and 555 us at 64 MB (MPI 518): the gap closes from
   21% to 5-7%. At 128 KB: 51.7 / 559. The smaller chunk pipelines the staging buffers better
   at the same block count; it is free at small sizes (1 MB: 22.1 vs 23.2).
5. **Across the fabric it is the same story with the roles reversed.** Cross-node pair,
   bidirectional, 4 MB: host-staged MPI 262 us (16 GB/s per direction), SF-NCCL 232 (18.1),
   GDR-MPI (UCX 1.19) 179 (23.5); unidirectional: NCCL 142 (29.6), MPI 143 (29.3), and at 64 MB
   NCCL 31 GB/s vs MPI 41. NCCL's default two channels per network peer are the limit:
   `NCCL_NCHANNELS_PER_NET_PEER=8` gives 190 us (22 GB/s) bidirectional, on par with GDR-MPI;
   `NCCL_NET_GDR_READ=1` changes nothing, `NCCL_NET_GDR_LEVEL=0` costs 25%, and a 2 MB net
   chunk stalls the 64 MB size (9 ms). Nobody reaches the 50 GB/s line rate per direction.
   The 8-rank ring of 29.2 lost because its cross-node links are unidirectional per rank while
   the same kernel also serves that rank's NVLink receive.

So "NCCL is slower at 4 MB" means: NCCL's point-to-point is an SM-driven, staged, chunked copy
whose parallelism NCCL picks from the message size, while GPU-aware MPI on NVLink is a DMA. Set
`NCCL_P2P_NVL_CHUNKSIZE=65536` (node) and `NCCL_NCHANNELS_PER_NET_PEER=8` (fabric) with
`-use_nccl` and the bulk loss is 5-7% on a node and gone across nodes; the wins of 29.2 are
unchanged. Both could become PETSc defaults set from `PetscNcclCommGet()` via `setenv` before
`ncclCommInitRank()`, if NVIDIA confirms they are safe generally.

### 29.10 What the two knobs are, and whether PETSc should set them (2026-09-09, later)

The gist of 29.2-29.9: below ~1 MB an exchange is fixed costs (host sync, handshake, launch)
and NCCL wins by having fewer of them; from 1-2 MB up it is bytes over copy rate, and NCCL's
rate is lower because it copies with a few SMs instead of a DMA engine. 4 MB is where that
crossover lands on this machine.

`NCCL_P2P_NVL_CHUNKSIZE` and `NCCL_NCHANNELS_PER_NET_PEER` are legitimate tuning knobs, but they
are process-wide environment variables that NCCL reads once at init from its hardware model
(NVLink present, NIC type, GPU generation); it measures nothing and never adapts them. The
defaults serve NCCL's main workload, collectives in training, where many channels are busy at
once and a large chunk amortizes the per-chunk handshake. Point-to-point with one or two peers
and four blocks, which is what PetscSF issues, is not that case.
- The NVLink chunk (512 KB default) is the size of one staged transfer. 64 KB lets the sender's
  block start chunk k+1 while the receiver drains chunk k, so the pipeline fills faster. It adds
  no SMs, so the 64 MB ceiling moves only from 116 to 121 GB/s, but at 4 MB it recovers most of
  the gap (57 -> 49 us).
- The network channels per peer (2 default) each carry their own queue pair and proxy path.
  Eight gives ~25% more cross-node bandwidth (19 -> 24 GB/s per direction), level with GDR-MPI.
Caveats: neither is per message, both apply to any collectives the application issues itself;
the sweeps had sharp edges (16 or 8 p2p channels worse, a 2 MB net chunk stalled 64 MB at 9 ms
on this RoCE fabric), so the values are Janus-tuned and other hardware needs its own sweep; and
the real limiter on NVLink, the block count per message, is not exposed at all
(`NCCL_MIN/MAX_P2P_NCHANNELS` do not change it), so that one is a question for NVIDIA.

For the backend: (1) document the two variables in the `-use_nccl` help and the notes, leave the
choice to the user (safe, what a reviewer accepts); or (2) `setenv` them in `PetscNcclCommGet()`
before `ncclCommInitRank()` when unset, which makes the bulk case a wash out of the box but
hard-codes a Janus sweep into PETSc and touches process-wide NCCL behavior; only behind an
explicit opt-in and after asking NVIDIA whether the values are generally safe. Both are now
defaults in `janus-env-nvshmem.sh`; 29.11 reruns the ring with them.

Poster summary: NCCL through PetscSF is the right transport for latency-bound exchanges and for
graph-captured loops; for bulk transfers it is at best a tuned tie with GPU-aware MPI, and the
tuning is manual.

### 29.11 The ring with the tuned defaults (2026-09-09, later)

Same `sfbench` ring as 29.2, best of 3, with `NCCL_P2P_NVL_CHUNKSIZE=65536`,
`NCCL_NCHANNELS_PER_NET_PEER=8` and `UCX_IB_GID_INDEX=2` from the env script. Untuned NCCL
values from 29.2 in parentheses; MPI moved by at most 2 us except where noted.

| bytes | 1 node MPI | 1 node NCCL | ratio | 8 ranks/2 nodes MPI | NCCL | ratio | 2+2 MPI | NCCL | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 B-8 KB | 11.8 | 6.4 (6.4) | 0.55 | 19.6 | 14.9 (14.8) | 0.76 | 21.4 | 16.2 (16.2) | 0.76 |
| 256 KB | 13.8 | 13.3 (13.4) | 0.97 | 39.8 | 25.5 (25.9) | 0.64 | 40.2 | 29.2 (29.0) | 0.73 |
| 512 KB | 16.1 | 16.6 (16.7) | 1.03 | 56.4 | 29.4 (38.0) | 0.52 | 57.6 | 46.7 (47.8) | 0.81 |
| 1 MB | 19.9 | 22.2 (23.1) | 1.12 | 68.3 | 48.9 (57.5) | 0.72 | 73.6 | 67.0 (67.3) | 0.91 |
| 4 MB | 44.1 | 50.3 (58.0) | 1.14 | 140 | 158 (171) | 1.12 | 178 | 214 (213) | 1.20 |
| 32 MB | 264 | 288 (305) | 1.09 | 827 | 1222 (1270) | 1.48 | unstable | 1570 (1566) | -- |

- One node: the bulk loss went from 21-28% to 9-14% (4 MB 58 -> 50 us); everything below
  256 KB is untouched. Not the 5% of the bare `sfbw` pair, so run-to-run spread of 2-4 us is
  part of the remaining gap.
- 8-rank ring over two nodes: the 8 net channels lift 512 KB-1 MB by 15-23% (38 -> 29, 58 ->
  49 us) and 4 MB by 8%; from 8 MB up the cross-node link still loses 1.3-1.5x, because each
  rank's kernel serves its NVLink receive and its network send together (29.9 point 5).
- 2+2: no change at 4-8 MB (each rank has one cross-node peer and the 4-block kernel is the
  limit); at 16-32 MB the *MPI* side is unstable, see below.

**MPI bulk at 2+2 is unstable and it is not the GID pin.** An A/B of MPI-only rings, 2 reps
each: at 2+2, 16 MB took 1232 / 4549 us with UCX on RoCE v1 (GID 2) and 711 / 692 with v2
(GID 3), 32 MB 6423 / 1789 vs 1414 / 5429 -- stalls of milliseconds on both settings, in
either rep, while 4-8 MB are steady within 2%. The 8-rank ring is steady on both GIDs and the
pin costs it 1-2% (4 MB 141.7 vs 138.6-140.4 us). So the UCX pin stays (it is what keeps host
MPI collectives alive next to NCCL, 29.7), and 2+2 host-staged MPI above 8 MB is not a usable
baseline on this fabric (the 28.x "MPI 4 MB drift" was the small end of the same thing).

## 30. The ladder rerun on a nonblocking PETSc stream (2026-09-14, job 2912)

Every eager arm of the sections 17-28 ladder (MPI, PetscSF-NVSHMEM, hand-rolled NCCL) was
launched without `-root_device_context_stream_type nonblocking`, i.e. on PETSc's default
legacy NULL stream. Section 29.3 later measured that stream costing NCCL ~45 us per
iteration of implicit synchronization, and 29.4 recorded a nonblocking-stream SF ladder
(148 / 109 / 93), but neither number ever propagated back to the ladder table (24, 28.8)
or to the interview prep. Rerun today on `x2002c0s1b0n0` + `x2003c0s9b0n0`, same binaries,
same launch lines as 2026-08-15 (1 node) and 2026-09-01 (2+2, RoCE v1 pin, `mlx5_0`),
64 KB, G=8, P=4, 300 iterations, best of 3. Raw log:
`nvshmem-tools/results-20260914-nonblocking-stream.txt`.

| arm | ladder value (24 / 28.8) | today, NULL stream | today, nonblocking stream |
| --- | ---: | ---: | ---: |
| PetscSF over GPU-aware MPI, 1 node | 162 | 165 | **149** |
| PetscSF over NVSHMEM host API (fused put+signal), 1 node | 206 | 180 | **106** |
| NCCL eager (`acgnbench-nccl`), 1 node | 107 | 106 | **59** |
| NCCL graph replay (k=10), 1 node | 32 | n/a (program forces nonblocking) | 32 |
| NVSHMEM device kernel, 1 node | 41 | n/a (no PETSc stream) | 41 (unchanged) |
| compute-only floor (`-skip_comm`), 1 node | 39 | 39 | 39 |
| NCCL eager, 2+2 | 177 | 173 | **118** |
| NCCL graph replay, 2+2 | 122 | n/a | 124 |
| NVSHMEM device kernel (push), 2+2 | 104 (28.12) | n/a | 104 (unchanged) |
| PetscSF over GPU-aware MPI, host-staged (UCX 1.17), 2+2 | 206 | 206 | **192** |
| PetscSF over GPU-aware MPI, GPUDirect (UCX 1.19 preload), 2+2 | 192 | 189 | **176** |
| PetscSF over NVSHMEM host API (fused put+signal, IBRC), 2+2 | 280 | 244 | **179** |

Four findings:

1. **The NULL stream, not the launch tax, was most of the eager NCCL number.** 106 -> 59 on
   one node and 173 -> 118 at 2+2: ~45-55 us per iteration, matching 29.3. The remaining
   59 -> 32 on one node is the per-op enqueue cost graph replay removes.
2. **PetscSF-NVSHMEM beats PetscSF-MPI by 1.4x on one node (106 vs 149) on a nonblocking
   stream.** The 24/28.8 ordering "PetscSF-NVSHMEM is the slowest arm" was a NULL-stream
   artefact: the legacy NULL stream serializes against every other blocking stream in the
   process, and the NVSHMEM path's dedicated high-priority comm stream (section 14) paid
   for that on every one of its ~7 ops per exchange. MPI barely moves (165 -> 149) because
   it blocks on the host regardless of stream. Section 28.8 statement 2 ("NVSHMEM's problem
   was never the transport ... yet PetscSF-NVSHMEM is the slowest arm") must be restated:
   the protocol cost is real (106 is still 1.8x eager NCCL's 59) but the arm is second on
   the in-tree ladder, not last.
3. **Across the fabric, graph replay gives nothing over a properly streamed eager loop**
   (118 eager vs 124 graph). See "why" below. The device kernel (104) is the only arm that
   still improves at 2+2, and it does so for a different reason than launch cost.
4. **The compute-only floor is stream-invariant (39 on both)**, which confirms it is the
   host's `VecAXPY` issue rate (17 launches on the busiest rank x ~2.3 us), not GPU time.

**Why graph replay stops helping across nodes.** Graph replay removes exactly one thing:
the gap between consecutive host enqueues. That gap only matters when the host is the
bottleneck, i.e. when the GPU finishes each kernel before the host has issued the next.
On one node that is the case: daxpy on 8192 doubles is ~1-2 us, an NVLink hop ~3-5 us,
so the GPU starves behind a ~2 us-per-op host. Across the fabric the critical path is 8
crossing edges at ~10-12 us each (28.7): GPU -> NCCL proxy thread -> NIC -> wire -> remote
(28.1). The eager host never blocks, so its ~40 us of enqueueing per iteration overlaps
with the ~100 us the GPU spends waiting on the network; the launch tax is hidden, and
there is nothing left for the graph to remove. Two costs the graph cannot touch: the NCCL
proxy thread still posts and polls every RDMA work request whether the kernel came from a
graph or not (29.5: a graph launch costs ~40 us across nodes for that reason, ~4 us/iter
amortized at k=10, which is the 118 -> 124), and NCCL fan-out to two remote peers is
serial (28.12). The device kernel's 104 comes from fusing signal with data and skipping
NCCL's per-stage kernel and proxy hand-off, not from launch count.

**Why graph replay (32) is below the "compute-only floor" (39).** The floor is
host-bound: 17 `VecAXPY` calls at ~2.3 us of host issue cost each = 39 us, during which
the GPU runs ~1.5 us kernels and idles ~0.8 us between them. Stream type does not change
it (39 on both), because nothing is waiting on anything; the host simply cannot issue
faster. In the graph the same 17 daxpy kernels execute back to back on the GPU (~20-25 us)
and the four NCCL NVLink kernels add ~8-10 us, total 32. A graph-replayed compute-only
run would land near 20-25 us; it was not measured (`acgnbench-nccl-graph` has no
`-skip_comm`). "Compute floor, eager" is therefore a floor for host-driven loops only,
and it is itself the strongest evidence for the ladder's thesis.

**What the slide/prep should say.** Use the nonblocking numbers with "nonblocking PETSc
stream" in the caption: 149 / 106 / 59 / 41 / 32 on one node (4.7x MPI -> graph, not 5x);
118 / 124 / 104 at 2+2. The story is unchanged on one node (each rung is a change of
orchestrator) and sharper across nodes (graph replay stops paying; device-initiated
execution is what still does). The 2+2 SF arms were rerun the same night (rows above): PetscSF-NVSHMEM
280 -> 179 on the nonblocking stream, level with GDR-MPI (176) and 7% ahead of
host-staged MPI (192); so across the fabric the three in-tree arms tie within noise once
the stream is right, and the 28.8 statement "slowest arm on both node counts" is retracted
on both. Not rerun: the put+signal fusion delta (25, NULL stream) and the IBGDA hybrid.
