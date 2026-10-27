# Replay-free CUDA checkpointing: implementation brief

This brief is for the agent implementing replay-free CUDA checkpoint-restart in the split-process codebase. A proof of concept and the supporting measurements live in `/home/yao/crac-old`. The design report is at https://claude.ai/artifact/Jz11vR3KmgDH5bmD4zXchj (it is private to its owner).

## Goal

Checkpoint and restart CUDA applications without NVIDIA's checkpoint API (`cuCheckpointProcess*`) and without log-and-replay. At restart, rebuild GPU state from what was **alive** at the checkpoint, not from the history of calls:

- **Device memory:** reserve the same virtual address range with CUDA VMM and map new physical memory at the saved offsets.
- **Other objects:** keep one record per live object (creation parameters plus current settings) and recreate it from that record.

## Decisions already made (do not revisit)

- **Split process.** The upper half holds the application and all shim state, and DMTCP saves it. The lower half holds libcudart, libcuda, the CUDA libraries and driver state. It is discarded at checkpoint and created fresh at restart.
- **Checkpoints happen only at quiesced points.** The GPU is drained first, and no thread is inside a CUDA wrapper.
- **No call logs on the restart path.** The one exception is the fixed set of fatbin registrations made at load time (see Kernels).

## What is already proven

| Item | Where | Result |
|---|---|---|
| VMM-backed `cudaMalloc`/`cudaFree`, save and restore in a fresh process | `vmm-ckpt.cpp`, `vmm-ckpt.h`, `test/vmm_poc.cu`; run `make check-vmm` | Same addresses after restart; pointers stored inside device memory stay valid; 20/20 restarts passed; holes freed before the checkpoint are reused after restart |
| Driver behaviors the design depends on | `test/vmm_assumptions.cu`; run `make check-assumptions` | All PASS on RTX 3070 Laptop, driver 595.99 (open kernel module), CUDA 13.1 |

The PoC does not use DMTCP. A fresh process stands in for the new lower half, and the test passes its host state explicitly. In the PoC:

- `stash_device_memory()` is the checkpoint step.
- `restore_device_memory()` is the restart step.
- The file I/O around them exists only because DMTCP is absent.

**Run `make check-assumptions` on every target GPU/driver before relying on the design there.** Its exit status is the number of failures.

## Invariants

1. **Every device address the app can see comes from the VMM reservation.** The reservation sits at a fixed base (PoC default `0x200000000000`, 64 GiB).
   - Before reserving at restart, `munmap` the range on the host side. DMTCP restores the old `PROT_NONE` placeholder there, and the driver then **silently** returns a different address with no error (measured).
   - Always compare the address you got with the one you asked for, and abort the restart on a mismatch.
2. **All shim state lives in upper-half memory.** Real lower-half handles stored in records are stale after restart and get overwritten during the rebuild.
3. **Records describe only live objects.** Destroying an object deletes its record. Never append to a history.
4. **Virtual handles are pointers to records.** They survive restart for free.
   - Pass through special values untranslated: stream `0`, `cudaStreamLegacy`, `cudaStreamPerThread`.
5. **Rebuild in dependency order:** device settings → memory → arrays → modules and kernels → streams and events → texture and surface objects → library handles → graphs → graph execs → NCCL communicators.
6. **Texture and surface objects are rebuilt before anything else in the new context can create one.** Their values must come back exactly (see below).

## Interception

- **Coverage:** wrap every API that takes or returns a handle or device memory, across cudart, the driver API and each supported library. Generate the wrappers from the headers; don't hand-write them.
  - Include the `_ptsz`/`_ptds` variants that apps built with `--default-stream per-thread` call.
- **Entry-point lookups:** `cudaGetDriverEntryPoint`, `cudaGetDriverEntryPointByVersion`, `cuGetProcAddress` and `cuGetProcAddress_v2` are called from the upper half. They must return wrapped function pointers, not real ones.
- **cudart and libraries stay in the lower half.**
  - cudart talks to the driver through undocumented export tables (`cuGetExportTable`), so it can't be virtualized from below.
  - A library saved in the upper half would keep stale CUfunction and stream handles inside its own state.
  - Allocations libraries make internally in the lower half are therefore never saved. They are recreated along with the library.
- **Static cudart is an open problem.** Apps built with nvcc's default (static cudart) carry cudart inside the upper-half binary, so the dynamic-symbol interposition the PoC uses cannot reach it. Until the user decides how to handle this, require `-cudart=shared` and detect the static case at startup: check whether `cudaMalloc` resolves inside the executable.
- **Checkpoint gate.** Wrappers enter a gate that the checkpoint can close.
  - Implement the gate as an atomic flag plus an in-flight counter in upper-half memory. **Do not use a pthread rwlock**: the gate's state is checkpointed, and a different thread opens it after restart.
  - Closing the gate means: wait until no calls are in flight, then drain.
  - Add a drain timeout. A kernel spinning on a host flag that a gated thread was supposed to write would otherwise deadlock the drain.

## Components

### 1. Device memory allocator

Port `vmm-ckpt.cpp`. Extend it as follows:

- **Entry points:** `cudaMalloc`, `cudaMallocPitch`, `cudaMalloc3D`, `cuMemAlloc_v2`, `cuMemAllocPitch_v2`, `cudaFree`, `cuMemFree_v2`.
  - Pick the pitch yourself, as a multiple of `cudaDevAttrTexturePitchAlignment`.
  - Unknown pointers passed to free go to the real function.
- **Sub-allocation:** the minimum granularity is 2 MiB. Pack small allocations into 2 MiB chunks, with records per chunk and a sub-allocator inside each. Keep `cudaMalloc`'s 256-byte alignment.
- **Multiple GPUs:** use one reservation. Each chunk's `CUmemAllocationProp.location.id` is the device that is current at allocation time.
  - `cudaDeviceEnablePeerAccess` has no effect on VMM memory. Record peer pairs and call `cuMemSetAccess` for every peer on existing and future chunks.
- **Stream-ordered allocation:** `cudaMallocAsync`, `cudaFreeAsync`, `cudaMallocFromPoolAsync`, and the `cudaMemPool*` calls. Implement them as your own allocator on the same range.
  - Returning already-mapped memory immediately is valid.
  - `cudaFreeAsync` records an event on the stream; reclaim the memory once the event completes.
  - Pools become records holding their attributes.
- **Allocation flags:** set `allocFlags.gpuDirectRDMACapable` when the target uses GPUDirect RDMA. Set `requestedHandleTypes` (POSIX fd) if cross-process sharing is needed.
- **Checkpoint:** after the drain, copy each live allocation to a host shadow in the upper half. Copy only the requested bytes.
- **Restart:** `munmap` the range, reserve it, verify the address. Then run `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` for each chunk, copy the shadow back, and free it.
- **Resume without restart:** free the shadows. Nothing else.
- **Known differences from `cudaMalloc` memory:**
  - Legacy IPC (`cudaIpcGetMemHandle`) does not work.
  - `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`/`RANGE_SIZE` report the chunk, not the allocation.
  - Check that the target's CUDA-aware MPI handles VMM memory.

### 2. Host memory

- **Pinned memory** (`cudaMallocHost`, `cudaHostAlloc`, `cuMemHostAlloc`): use fixed-address anonymous `mmap` in the upper half, then call `cudaHostRegister` with flags mapped from the request.
  - The device pointer equals the host pointer (measured).
  - The write-combined flag cannot be reproduced; document that.
  - App calls to `cudaHostRegister`: record `(ptr, size, flags)`.
  - At restart, register every live range again.
- **Managed memory** (`cudaMallocManaged`, `cuMemAllocManaged`): requires HMM or ATS. Check `cudaDevAttrPageableMemoryAccess == 1`.
  - Use fixed-address anonymous `mmap` in the upper half. DMTCP saves it as ordinary memory; nothing needs restoring.
  - Before checkpoint, prefetch it to the CPU with `cudaMemPrefetchAsync`. DMTCP reading GPU-resident pages migrates them back anyway, just slowly.
  - Translate `cudaMemAdvise` and prefetch calls.
  - `cudaPointerGetAttributes` reports this memory as unregistered host memory, not managed. Check what that does to the target's MPI.
  - Without HMM/ATS, refuse `cudaMallocManaged` with a clear error. The only fallback is zero-copy pinned memory, which is correct but slow; offer it only behind a flag.
- `__managed__` variables are not supported (see Unsupported).

### 3. Virtual-handle framework

```cpp
struct record_t {
  uint32_t kind;          // stream, event, module, texture, ...
  uint32_t generation;    // lower-half instance this real handle belongs to
  void *real;             // handle in the current lower half
  // kind-specific: creation parameters + current settings
};
```

- **Handle values:** the virtual handle is `(T)record_ptr`. Look up the real handle in O(1) and check `kind` in debug builds.
- **Generations:** a global generation counter is bumped at each restart.
  - Records are rebuilt eagerly during restart, in dependency order.
  - Per-thread state, such as the current device kept in upper-half TLS, is re-applied lazily on each thread's first call after restart, by comparing the thread's generation with the global one.
- **Settings:** setters update the record (last value wins). Getters can go to the real object.
- **Thread safety:** use one lock per record table, or lock-free reads with a writer lock. Wrappers run on app threads.

### 4. Runtime objects

| Object | Record | Rebuild |
|---|---|---|
| Device and context | Device flags; limits from `cudaDeviceGetLimit` (stack, printf FIFO, malloc heap, ...); cache config; peer pairs; per-thread current device (TLS) | Set flags and limits, enable peers, call `cuMemSetAccess` on peers |
| Stream | Flags, priority, attributes | `cudaStreamCreateWithPriority`, then set the attributes. Refuse to checkpoint while any stream is capturing (`cudaStreamIsCapturing`). |
| Event | Flags, whether it has been recorded; at checkpoint, `cudaEventElapsedTime` against a reference event | Create, then record it once on a fresh stream if it had been recorded. The wrapper for `cudaEventElapsedTime` between two pre-checkpoint events returns the difference of the saved offsets. |
| Kernels (runtime API) | The `__cudaRegisterFatBinary`/`__cudaRegisterFatBinaryEnd`/`__cudaRegisterFunction`/`__cudaRegisterVar` calls, with their arguments. This fixed set is made at load and `dlopen` time. Fatbin data and host stubs live in the upper half, so the pointers stay valid. Remove entries on `__cudaUnregisterFatBinary`. | Make the same registrations in the new lower half. Kernels are identified by host-stub address, so `cudaLaunchKernel` needs no translation of the function. |
| Modules (driver API) | Copy of the image bytes at `cuModuleLoadData`/`cuLibraryLoadData` time (the app may free its buffer); path for `cuModuleLoad`; JIT options; function and kernel names; `cuFuncSetAttribute`/`cudaFuncSetAttribute` values. For `cuLink*`, keep the cubin from `cuLinkComplete` and load that. NVRTC output arrives through `cuModuleLoadData` anyway. | Load, look up functions by name, apply attributes |
| `__device__`/`__constant__` contents | Symbols from `__cudaRegisterVar` | At checkpoint, `cudaMemcpyFromSymbol` into a shadow. At restart, `cudaMemcpyToSymbol`. The address moves (measured). Wrap `cudaGetSymbolAddress` to mark the symbol as escaped; abort the restart if an escaped symbol's new address differs. |
| CUDA arrays, mipmapped arrays | `cudaArrayGetInfo` (desc, extent, flags); contents copied to a linear host shadow with `cudaMemcpy3D` | Create, copy back |
| Texture and surface objects | Old value; resource, texture and view descriptors; arrays referenced in them translated at rebuild | See below |
| Graphs | Serialize the graph into the record when `cudaGraphInstantiate*` is called: nodes, edges, node types, per-type params. For kernel node arguments, get sizes from `cuFuncGetParamInfo` (CUfunction via `cudaGetFuncBySymbol`) and copy the argument bytes. Host-node function and `userData` pointers are upper-half pointers, so keep them as they are. Execs often outlive their graph (capture → instantiate → `cudaGraphDestroy`), so the serialized copy belongs to the exec. | Rebuild the graph with translated handles, instantiate with the saved flags (`cudaGraphExecGetFlags`), then apply updates |
| Graph execs | Exec updates have no getters (`cuGraphExec*Get` only has `GetId` and `GetFlags`). Apply `cudaGraphExecUpdate`, `cudaGraphExec*NodeSetParams` and `cudaGraphNodeSetEnabled` to the serialized copy. | Covered by the graph rebuild |

**Texture and surface objects.** Their values are passed to kernels and stored in device memory, so wrappers cannot translate them and the new value must equal the old one. Measured behavior:

- Values start at 1 in a fresh context.
- Values are sequential as long as nothing is destroyed.
- Freed values come back after a delay that varies between runs, and textures and surfaces appear to share one table.

Rebuild procedure:

1. Sort live objects by old value.
2. For each one, create throwaway fillers until the next create returns the wanted value. **Keep every filler alive**; destroying one mid-rebuild lets the driver reuse its value unpredictably.
3. Verify each value; abort the restart on a mismatch.
4. Destroy all fillers at the end.

`make check-assumptions` verifies the policy on a new driver. Do not trust it without that check.

### 5. Libraries (lower half; handles rebuilt from records)

| Library | Record | Notes |
|---|---|---|
| cuBLAS | Stream, pointer mode, math mode, atomics mode, workspace pointer and size, SM count target | cublasLt: matmul and layout descriptors with their attributes; `cublasLtMatmulAlgo_t` is a plain value, so store it |
| cuFFT | Plan type and parameters (`cufftPlanMany`/`cufftMakePlan*` arguments), stream, work area, auto-allocation flag | `cufftHandle` is an int; virtualize it as an index into the record table |
| cuDNN | Handle stream; last-set values of each legacy descriptor; for the backend API, every attribute set before `cudnnBackendFinalize`, including engine configs | Re-finalizing with the same engine config gives the same plan; do not re-run heuristics |
| cuRAND (host API) | Type, seed, ordering, quasi-random dimensions, stream, and the offset | There is no offset getter (`curand.h` only has `curandSetGeneratorOffset`). Count the values generated in the wrappers. Some types reject offsets (e.g. MTGP32); refuse to checkpoint those, or document it. Verify per type against an uninterrupted run. |
| NCCL | nranks, rank, config; for split communicators, the parent, color and key; registered buffers | Rebuild collectively: rank 0 calls `ncclGetUniqueId` and sends it through MPI or the DMTCP coordinator, then every rank calls `ncclCommInitRankConfig`. Children are rebuilt with `ncclCommSplit` from the rebuilt parent. Then run `ncclCommRegister` again. `ncclMemAlloc` must come from the VMM range with the properties NCCL needs. Refuse to checkpoint inside `ncclGroupStart`/`ncclGroupEnd`. |

## Checkpoint procedure

1. Close the gate: wait for in-flight wrapper calls to finish and refuse to start if any stream is capturing or an NCCL group is open. Do this where the codebase already does lower-half work during checkpoint, while the lower-half driver can still make progress.
2. Call `cudaDeviceSynchronize` on every device.
3. Stash device memory, symbol contents and array contents into upper-half shadows. Prefetch managed memory to the CPU. Save event offsets.
4. Let DMTCP write the image. The lower half is excluded.
5. On resume: free the shadows, open the gate.

## Restart procedure

1. DMTCP restores the upper half.
2. Create a fresh lower half.
3. Bump the generation.
4. Restore memory: `munmap` the range → reserve → verify → map chunks → copy shadows back. Register pinned ranges again.
5. Make the fatbin registrations again, then restore symbol contents.
6. Rebuild records in the order from invariant 5, with textures before any library.
7. Free the shadows and open the gate. Threads re-apply their TLS state on their first call.

## Unsupported: detect these and refuse clearly

- Device-side `malloc` pointers that live across a checkpoint. The heap is owned by the driver.
- `__managed__` variables.
- Global variable addresses that escaped and moved (detected through `cudaGetSymbolAddress`).
- Graph memory nodes (`cudaGraphAddMemAllocNode`). Optional workaround: rewrite them into VMM allocations at capture time, accepting a different lifetime.
- Handles used inside kernels: graphs launched from the device, dynamic-parallelism streams, and conditional-node handles.
- Managed memory without HMM/ATS, unless the zero-copy flag is set.
- Legacy CUDA IPC (`cudaIpcGetMemHandle`) on VMM memory.

## Measured pitfalls

- Plain `cudaMalloc` and `__device__` addresses change with every process. With ASLR off they repeat, but a restarted lower half has a different layout, so don't rely on that.
- The reservation hint is silently ignored if anything occupies the host side of the range.
- Freed texture values are reused after a varying delay. A naive rebuild loop that creates and destroys can spin forever; one probe did.
- The driver creates a host `PROT_NONE` mapping covering each reservation. It shows up in `/proc/self/maps` and in the DMTCP image.

## Tests to write (each must pass after 2+ chained restarts and after 20 restarts of one image)

1. The PoC scenario (`test/vmm_poc.cu`) under the real split process and DMTCP.
2. Small allocations (sub-2 MiB packing), pitch/3D, async allocator with frees still pending at checkpoint.
3. Pinned memory: kernel writes through the host pointer before and after restart.
4. Managed memory under HMM: data touched by both GPU and CPU before the checkpoint is intact after restart.
5. Streams with priorities and events: `cudaStreamWaitEvent` across restart; `cudaEventElapsedTime` between events recorded before the checkpoint.
6. Driver-API module loaded from a buffer the app has already freed; `cuLink*` output.
7. Texture objects stored in a device-memory array and read by a kernel after restart, with destroys in the history.
8. A graph captured from a stream, its graph destroyed, the exec updated, then launched after restart.
9. cuBLAS GEMM loop, cuFFT plan reuse, and a cuRAND sequence identical to an uninterrupted run.
10. NCCL all-reduce on 2+ ranks, including a split communicator.
11. Refusal paths: checkpoint during stream capture, escaped symbol that moved, `cudaMallocManaged` without HMM.

## Open questions for the user

- How to support apps built with static cudart (require `-cudart=shared`, or redirect at link time).
- Whether to try pinning global variables in the VMM range by rewriting modules. `CU_JIT_GLOBAL_SYMBOL_ADDRESSES` places *unresolved* globals at chosen addresses (dynamic linker only); turning definitions into externs is untested.
- HMM performance against `cudaMallocManaged`, and how CUDA-aware MPI treats HMM memory.
- Whether the texture value policy holds on the target GPUs and drivers; `make check-assumptions` answers this per machine.
