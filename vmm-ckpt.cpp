#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <map>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vmm-ckpt.h"

// All device allocations live inside one VA reservation at a fixed address.
// 32 TiB is far from where Linux places the heap, mmap area, and stack, and
// from where the driver puts its own reservations (0x3xxxxxxxx and 0x7fxx...).
#define DEFAULT_VA_BASE 0x200000000000ULL
#define DEFAULT_VA_SIZE (64ULL << 30)
#define IMAGE_MAGIC 0x564d4d434b505431ULL  // "VMMCKPT1"

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult ret_ = (call);                                                  \
    if (ret_ != CUDA_SUCCESS) {                                              \
      const char *msg_ = NULL;                                               \
      cuGetErrorString(ret_, &msg_);                                         \
      fprintf(stderr, "vmm-ckpt: %s failed: %s (%s:%d)\n", #call, msg_,      \
              __FILE__, __LINE__);                                           \
      abort();                                                               \
    }                                                                        \
  } while (0)

typedef struct {
  size_t size;                          // mapped size, multiple of granularity
  size_t requested;                     // size passed to cudaMalloc
  CUmemGenericAllocationHandle handle;  // stale after restart until remapped
  void *shadow;                         // host copy while checkpointed
} alloc_t;

typedef struct {
  uint64_t magic;
  uint64_t va_base;
  uint64_t va_size;
  uint64_t va_top;
  uint64_t num_allocs;
  uint64_t num_free_ranges;
  uint64_t app_state_size;
} image_header_t;

// Allocator metadata. It is plain host memory, so under DMTCP it would be
// checkpointed with the rest of the process; only the handles need refreshing.
static std::mutex vmm_lock;
static int initialized = 0;
static CUdevice device;
static size_t granularity;
static CUdeviceptr va_base;
static size_t va_size;
static size_t va_top;                               // bump offset into range
static std::map<CUdeviceptr, alloc_t> live_allocs;  // keyed by start VA
static std::map<CUdeviceptr, size_t> free_ranges;   // holes below va_top
static cudaError_t (*real_cudaFree)(void *) = NULL;

static void resolve_real_cudaFree() {
  if (real_cudaFree == NULL) {
    *(void **)(&real_cudaFree) = dlsym(RTLD_NEXT, "cudaFree");
    if (real_cudaFree == NULL) {
      fprintf(stderr, "vmm-ckpt: cannot find cudaFree; is the target built "
                      "with -cudart=shared?\n");
      abort();
    }
  }
}

static CUmemAllocationProp device_alloc_prop() {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}

static size_t env_or(const char *name, size_t dflt) {
  const char *val = getenv(name);
  return val ? strtoull(val, NULL, 0) : dflt;
}

// Bring up the runtime, then reserve [hint, hint + size). The driver treats
// the address only as a hint, so callers that need it must check va_base.
static void vmm_init(CUdeviceptr hint, size_t size) {
  resolve_real_cudaFree();
  real_cudaFree(0);  // Forces runtime and primary context initialization.
  CU_CHECK(cuCtxGetDevice(&device));

  int supported = 0;
  CU_CHECK(cuDeviceGetAttribute(
      &supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      device));
  if (!supported) {
    fprintf(stderr, "vmm-ckpt: device %d does not support VMM\n", device);
    abort();
  }
  CUmemAllocationProp prop = device_alloc_prop();
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CU_CHECK(cuMemAddressReserve(&va_base, size, 0, hint, 0));
  va_size = size;
  initialized = 1;
  fprintf(stderr, "vmm-ckpt: reserved VA [0x%llx, 0x%llx), granularity %zu\n",
          va_base, va_base + va_size, granularity);
}

static CUresult map_physical(CUdeviceptr va, size_t size,
                             CUmemGenericAllocationHandle *handle) {
  CUmemAllocationProp prop = device_alloc_prop();
  CUresult ret = cuMemCreate(handle, size, &prop, 0);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }
  CU_CHECK(cuMemMap(va, size, 0, *handle, 0));
  CUmemAccessDesc access = {};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(va, size, &access, 1));
  return CUDA_SUCCESS;
}

// First fit over holes left by cudaFree, otherwise bump.
static CUdeviceptr take_va(size_t size) {
  for (auto it = free_ranges.begin(); it != free_ranges.end(); ++it) {
    if (it->second >= size) {
      CUdeviceptr va = it->first;
      size_t rest = it->second - size;
      free_ranges.erase(it);
      if (rest) {
        free_ranges[va + size] = rest;
      }
      return va;
    }
  }
  if (va_top + size > va_size) {
    return 0;
  }
  CUdeviceptr va = va_base + va_top;
  va_top += size;
  return va;
}

static void give_back_va(CUdeviceptr va, size_t size) {
  auto next = free_ranges.lower_bound(va);
  if (next != free_ranges.end() && va + size == next->first) {
    size += next->second;
    free_ranges.erase(next);
  }
  auto prev = free_ranges.lower_bound(va);
  if (prev != free_ranges.begin()) {
    --prev;
    if (prev->first + prev->second == va) {
      prev->second += size;
      return;
    }
  }
  free_ranges[va] = size;
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  std::lock_guard<std::mutex> guard(vmm_lock);
  if (!initialized) {
    vmm_init(env_or("VMM_CKPT_VA_BASE", DEFAULT_VA_BASE),
             env_or("VMM_CKPT_VA_SIZE", DEFAULT_VA_SIZE));
  }
  if (devPtr == NULL) {
    return cudaErrorInvalidValue;
  }
  if (size == 0) {
    *devPtr = NULL;
    return cudaSuccess;
  }
  size_t rounded = (size + granularity - 1) / granularity * granularity;
  CUdeviceptr va = take_va(rounded);
  if (va == 0) {
    return cudaErrorMemoryAllocation;
  }
  alloc_t alloc = {rounded, size, 0, NULL};
  if (map_physical(va, rounded, &alloc.handle) != CUDA_SUCCESS) {
    give_back_va(va, rounded);
    return cudaErrorMemoryAllocation;
  }
  live_allocs[va] = alloc;
  *devPtr = (void *)va;
  return cudaSuccess;
}

extern "C" cudaError_t cudaFree(void *devPtr) {
  std::unique_lock<std::mutex> guard(vmm_lock);
  auto it = live_allocs.find((CUdeviceptr)devPtr);
  if (it == live_allocs.end()) {
    // NULL, or memory we did not allocate.
    guard.unlock();
    resolve_real_cudaFree();
    return real_cudaFree(devPtr);
  }
  // cudaFree() implicitly synchronizes; keep that so unmapping is safe.
  cudaDeviceSynchronize();
  CU_CHECK(cuMemUnmap(it->first, it->second.size));
  CU_CHECK(cuMemRelease(it->second.handle));
  give_back_va(it->first, it->second.size);
  live_allocs.erase(it);
  return cudaSuccess;
}

// DMTCP_EVENT_PRECHECKPOINT equivalent: stash device contents in host memory.
static void stash_device_memory() {
  cudaDeviceSynchronize();
  for (auto &entry : live_allocs) {
    alloc_t &alloc = entry.second;
    alloc.shadow = malloc(alloc.requested);
    cudaError_t ret = cudaMemcpy(alloc.shadow, (void *)entry.first,
                                 alloc.requested, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
      fprintf(stderr, "vmm-ckpt: D2H copy failed: %s\n",
              cudaGetErrorString(ret));
      abort();
    }
  }
}

// DMTCP_EVENT_RESTART equivalent (in a fresh lower half): reserve the same VA
// range, then map new physical memory at every saved address and refill it.
// No allocation history is replayed; only the final layout matters.
static int restore_device_memory(CUdeviceptr saved_base, size_t saved_size) {
  // A host mapping over the range (e.g. the old PROT_NONE reservation that
  // DMTCP restores) makes the driver silently ignore the hint.
  munmap((void *)saved_base, saved_size);
  vmm_init(saved_base, saved_size);
  if (va_base != saved_base) {
    fprintf(stderr, "vmm-ckpt: driver placed reservation at 0x%llx instead "
                    "of 0x%llx; cannot restore\n", va_base, saved_base);
    return -1;
  }
  for (auto &entry : live_allocs) {
    alloc_t &alloc = entry.second;
    CU_CHECK(map_physical(entry.first, alloc.size, &alloc.handle));
    cudaError_t ret = cudaMemcpy((void *)entry.first, alloc.shadow,
                                 alloc.requested, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
      fprintf(stderr, "vmm-ckpt: H2D copy failed: %s\n",
              cudaGetErrorString(ret));
      abort();
    }
    free(alloc.shadow);
    alloc.shadow = NULL;
  }
  return 0;
}

extern "C" int vmm_ckpt_save(const char *path, const void *app_state,
                             size_t app_state_size) {
  std::lock_guard<std::mutex> guard(vmm_lock);
  if (!initialized) {
    fprintf(stderr, "vmm-ckpt: nothing allocated yet\n");
    return -1;
  }
  stash_device_memory();

  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    perror("vmm-ckpt: fopen");
    return -1;
  }
  image_header_t hdr = {IMAGE_MAGIC, va_base, va_size, va_top,
                        live_allocs.size(), free_ranges.size(), app_state_size};
  fwrite(&hdr, sizeof hdr, 1, f);
  fwrite(app_state, app_state_size, 1, f);
  for (auto &range : free_ranges) {
    uint64_t rec[2] = {range.first, range.second};
    fwrite(rec, sizeof rec, 1, f);
  }
  for (auto &entry : live_allocs) {
    uint64_t rec[3] = {entry.first, entry.second.size, entry.second.requested};
    fwrite(rec, sizeof rec, 1, f);
    fwrite(entry.second.shadow, entry.second.requested, 1, f);
    free(entry.second.shadow);
    entry.second.shadow = NULL;
  }
  int ret = ferror(f) ? -1 : 0;
  fclose(f);
  return ret;
}

extern "C" int vmm_ckpt_restore(const char *path, void *app_state,
                                size_t app_state_size) {
  std::lock_guard<std::mutex> guard(vmm_lock);
  if (initialized) {
    fprintf(stderr, "vmm-ckpt: restore must run before any cudaMalloc\n");
    return -1;
  }
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    perror("vmm-ckpt: fopen");
    return -1;
  }
  image_header_t hdr;
  if (fread(&hdr, sizeof hdr, 1, f) != 1 || hdr.magic != IMAGE_MAGIC ||
      hdr.app_state_size != app_state_size) {
    fprintf(stderr, "vmm-ckpt: %s is not a matching image\n", path);
    fclose(f);
    return -1;
  }
  // Rebuild the allocator metadata. Under DMTCP this step is free: the maps
  // are restored along with the rest of the process memory.
  int ok = fread(app_state, app_state_size, 1, f) == 1;
  for (uint64_t i = 0; ok && i < hdr.num_free_ranges; i++) {
    uint64_t rec[2];
    ok = fread(rec, sizeof rec, 1, f) == 1;
    free_ranges[rec[0]] = rec[1];
  }
  for (uint64_t i = 0; ok && i < hdr.num_allocs; i++) {
    uint64_t rec[3];
    ok = fread(rec, sizeof rec, 1, f) == 1;
    alloc_t alloc = {rec[1], rec[2], 0, malloc(rec[2])};
    ok = ok && fread(alloc.shadow, alloc.requested, 1, f) == 1;
    live_allocs[rec[0]] = alloc;
  }
  fclose(f);
  if (!ok) {
    fprintf(stderr, "vmm-ckpt: %s is truncated\n", path);
    return -1;
  }
  va_top = hdr.va_top;
  return restore_device_memory(hdr.va_base, hdr.va_size);
}
