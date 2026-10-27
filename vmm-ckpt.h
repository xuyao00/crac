// Proof of concept: VMM-backed cudaMalloc/cudaFree that can be restored at the
// same device addresses in a fresh process without log-and-replay and without
// the CUDA checkpoint API. Load with LD_PRELOAD=libvmm_ckpt.so; the target
// must be built with -cudart=shared so cudaMalloc/cudaFree can be interposed.
#ifndef VMM_CKPT_H
#define VMM_CKPT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Copy every live allocation to host memory and write it to path, together
// with app_state (a stand-in for the host memory DMTCP would save).
int vmm_ckpt_save(const char *path, const void *app_state,
                  size_t app_state_size);

// In a fresh process, before any cudaMalloc: re-reserve the same VA range,
// remap and refill every allocation saved by vmm_ckpt_save(), and copy the
// saved app_state back into the caller's buffer.
int vmm_ckpt_restore(const char *path, void *app_state, size_t app_state_size);

#ifdef __cplusplus
}
#endif

#endif
