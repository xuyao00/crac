# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CRAC is a DMTCP plugin that checkpoints and restarts CUDA programs using the CUDA driver checkpoint API (`cuCheckpointProcess*`, CUDA 12.4+). The entire plugin lives in `cuda-ckpt.cpp`. `test/` holds small CUDA programs (and MPI+CUDA variants) that serve as checkpoint targets.

## Build and run

`DMTCP_ROOT` must point to a DMTCP **source/build tree**, not just an install prefix. The plugin includes internal headers from `$(DMTCP_ROOT)/src` and `$(DMTCP_ROOT)/jalib` (`procselfmaps.h`, `trampolines.h`, `plugin/ipc/event/eventwrappers.h`). The Makefile takes the CUDA location from `nvcc` on `PATH`, then `CUDA_PATH`, then `/usr/local/cuda`.

```sh
make DMTCP_ROOT=/path/to/dmtcp            # builds libdmtcp_cuda.so and test/counter
make check DMTCP_ROOT=/path/to/dmtcp      # launch test/counter, checkpoint, quit, restart, quit
make -C test counter                      # build only the single-GPU test
make -C test mpi                          # MPI tests (not built by default)
make clean
```

- The built library is `libdmtcp_cuda.so`. The README's `libdmtcp_crac.so` predates a rename and is out of date.
- `test/Makefile` hardcodes Cray MPICH paths (`/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3`) for the MPI targets. Change them on other systems.
- Test binaries are built with `nvcc -cudart=shared`.
- There is no linter and no unit-test framework. `make check` is the only test: it needs a real NVIDIA GPU and a driver that supports the checkpoint API. It passes if the counter keeps counting from where it stopped after restart, so read its output.

Manual checkpoint/restart cycle:

```sh
$DMTCP_ROOT/bin/dmtcp_launch --with-plugin $PWD/libdmtcp_cuda.so ./test/counter &
$DMTCP_ROOT/bin/dmtcp_command --checkpoint
$DMTCP_ROOT/bin/dmtcp_command --quit
$DMTCP_ROOT/bin/dmtcp_restart ckpt_*.dmtcp
```

## Architecture: the DMTCP event lifecycle

Everything is driven by `cuda_event_hook()`. The order of work across DMTCP events is the core design, and it is not obvious from any single function:

1. **`DMTCP_EVENT_INIT`**: installs a trampoline on `eventfd` with `dmtcp_setup_trampoline`. The trampoline is uninstalled at PRECHECKPOINT and reinstalled in RUNNING before the GPU is restored.
2. **`DMTCP_EVENT_PRESUSPEND`** → `checkpoint_gpu()`: this must happen *before* DMTCP suspends user threads, because the CUDA checkpoint API needs them running to talk to the GPU. Steps: `cuCheckpointProcessLock` → record every `/dev/nvidia*` mapping from `/proc/self/maps` → `cuCheckpointProcessCheckpoint` → `mmap(PROT_NONE, MAP_FIXED)` placeholders over those device-file ranges. Older drivers release the ranges on checkpoint, and the placeholders stop DMTCP or anything else from reusing those addresses before restore.
3. **`DMTCP_EVENT_PRECHECKPOINT`**: `inspect_pipes()` scans `/proc/self/fd` and records each pipe fd's number, inode, and read/write end into the global `pipe_list`. fds 0–2 are skipped on purpose.
4. **`DMTCP_EVENT_RESTART`** (restart only): `recreate_pipes()` makes fresh pipe pairs and `dup2`s them onto the original fd numbers. Ends that shared an inode stay paired, but pipe contents are not preserved. It calls libc's real `pipe()` through `dlopen`/`dlsym` so that DMTCP's `pipe` wrapper is bypassed. It first fills the target fd slots with a `/dev/zero` fd so the new pipes land elsewhere.
5. **`DMTCP_EVENT_RESUME`**: does nothing on purpose. After checkpoint the CUDA process is in the CHECKPOINTED state, and both resume and restart have to go through restore and unlock, which need user threads running. Because of that, restore happens in RUNNING.
6. **`DMTCP_EVENT_RUNNING`** (initial launch, resume, and restart): if the global `cuda_state == CU_PROCESS_STATE_CHECKPOINTED`, it reinstalls the `eventfd` trampoline and calls `restore_gpu()`: `munmap` the placeholders → `cuCheckpointProcessRestore` → `cuCheckpointProcessUnlock` → assert the state is RUNNING. This section is wrapped in `DMTCP_PLUGIN_DISABLE_CKPT()`/`ENABLE_CKPT()`, because DMTCP allows a new checkpoint to start during RUNNING.

The global `cuda_state` is what tells the RUNNING hook whether this is a first launch or a post-checkpoint path. Keep it accurate if you change any checkpoint or restore step.

## VMM proof of concept (no checkpoint API, no DMTCP)

`vmm-ckpt.cpp` builds `libvmm_ckpt.so` and is separate from the DMTCP plugin. It is an `LD_PRELOAD` shim that replaces `cudaMalloc`/`cudaFree` with CUDA VMM calls (`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`). All allocations go inside one VA reservation at a fixed hint, `0x200000000000` by default; override it with `VMM_CKPT_VA_BASE` and `VMM_CKPT_VA_SIZE`. To restart, the shim reserves the same range again in a fresh process and maps new physical memory at each saved address, so allocation history does not have to be replayed.

```sh
make check-vmm            # launch -> ckpt -> restart -> ckpt -> restart, using test/vmm_poc
make check-assumptions    # PASS/FAIL for each driver behavior the design relies on
```

`docs/replay-free-implementation.md` is the full design brief: per-object records, rebuild order, and tests.

- Interposition only works when the target is built with `-cudart=shared`.
- `vmm_ckpt_save`/`vmm_ckpt_restore` write and read an image file. The test passes its host state struct to them in place of the process memory DMTCP would save. `stash_device_memory()`/`restore_device_memory()` mark where the PRECHECKPOINT and RESTART hooks would go in a real plugin.
- The driver treats the address only as a hint. If anything is mapped on the host side of the range, it silently returns a different address, so restore `munmap`s the range first and then checks the result.
- `__device__`/`__constant__` globals are not covered: the driver places them at module load, and they land at a different address in each new process.

## Code notes

- Error handling uses DMTCP's `JASSERT(cond)(vars...)`. A few places use plain `assert()` around calls with side effects (for example `assert(dup2(...) != -1)`). The build always uses `-O0 -g3` and never defines `NDEBUG`. Keep it that way, or move those calls out of `assert()`.
- `pipe2_trampoline_info` is declared but not used yet.
