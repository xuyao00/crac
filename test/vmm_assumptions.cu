// Checks the driver behaviors that replay-free restore depends on. Run it on
// every new GPU/driver combination before relying on the design there.
// Exit status is the number of FAIL lines; INFO lines are for comparison
// across runs (e.g. run twice and see whether an address moves).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define VA_HINT 0x200000000000ULL
#define VA_SIZE (64ULL << 30)
#define HMM_ADDR 0x300000000000ULL
#define PINNED_ADDR 0x310000000000ULL
#define N (1 << 20)

static int failures = 0;

static void report(const char *level, const char *name, const char *detail) {
  printf("%-4s  %-36s %s\n", level, name, detail);
  if (!strcmp(level, "FAIL")) {
    failures++;
  }
}

static void check(bool ok, const char *name, const char *detail) {
  report(ok ? "PASS" : "FAIL", name, detail);
}

__device__ int g_module_global;

__global__ void increment(int *p, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    p[i] += 1;
  }
}

static cudaResourceDesc tex_res;
static cudaTextureDesc tex_desc;

static cudaTextureObject_t make_texture() {
  cudaTextureObject_t t = 0;
  cudaCreateTextureObject(&t, &tex_res, &tex_desc, NULL);
  return t;
}

int main() {
  char msg[256];
  cudaFree(0);

  // 1. VMM is available.
  int vmm = 0;
  cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0);
  check(vmm == 1, "VMM supported", vmm ? "yes" : "no");

  // 2. A fixed VA hint is honored when the range is free.
  CUdeviceptr va = 0;
  CUresult ret = cuMemAddressReserve(&va, VA_SIZE, 0, VA_HINT, 0);
  snprintf(msg, sizeof msg, "asked 0x%llx, got 0x%llx (err %d)", VA_HINT, va, ret);
  check(ret == CUDA_SUCCESS && va == VA_HINT, "fixed VA hint honored", msg);
  if (ret == CUDA_SUCCESS) {
    cuMemAddressFree(va, VA_SIZE);
  }

  // 3. What happens when the host side of the range is occupied, as it is
  //    after DMTCP restores the old PROT_NONE placeholder. Restore must
  //    munmap first; this shows why.
  void *block = mmap((void *)VA_HINT, 1 << 21, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  ret = cuMemAddressReserve(&va, VA_SIZE, 0, VA_HINT, 0);
  snprintf(msg, sizeof msg, "err %d, got 0x%llx (%s)", ret, va,
           ret != CUDA_SUCCESS ? "error returned"
           : va == VA_HINT     ? "hint honored anyway"
                               : "hint silently ignored");
  report("INFO", "hint with host range occupied", msg);
  if (ret == CUDA_SUCCESS) {
    cuMemAddressFree(va, VA_SIZE);
  }
  munmap(block, 1 << 21);
  ret = cuMemAddressReserve(&va, VA_SIZE, 0, VA_HINT, 0);
  snprintf(msg, sizeof msg, "got 0x%llx", va);
  check(ret == CUDA_SUCCESS && va == VA_HINT, "hint honored after munmap", msg);
  if (ret == CUDA_SUCCESS) {
    cuMemAddressFree(va, VA_SIZE);
  }

  // 4. Registered host memory at a fixed address: device pointer == host
  //    pointer, so pinned memory can live in the upper half.
  int *pinned = (int *)mmap((void *)PINNED_ADDR, N * sizeof(int),
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                            -1, 0);
  memset(pinned, 0, N * sizeof(int));
  cudaError_t err = cudaHostRegister(pinned, N * sizeof(int), cudaHostRegisterMapped);
  void *dev_alias = NULL;
  cudaHostGetDevicePointer(&dev_alias, pinned, 0);
  increment<<<1, 256>>>(pinned, N);
  cudaDeviceSynchronize();
  snprintf(msg, sizeof msg, "register %s, device ptr %p, kernel wrote %d",
           cudaGetErrorString(err), dev_alias, pinned[7]);
  check(err == cudaSuccess && dev_alias == pinned && pinned[7] == 1,
        "registered memory devptr == hostptr", msg);

  // 5. Pageable memory access (HMM or ATS): managed memory can be plain
  //    fixed-address mmap in the upper half.
  int pageable = 0, ats = 0;
  cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0);
  cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
  if (pageable) {
    int *hmm = (int *)mmap((void *)HMM_ADDR, N * sizeof(int),
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                           -1, 0);
    for (int i = 0; i < N; i++) {
      hmm[i] = i;
    }
    increment<<<1, 256>>>(hmm, N);
    err = cudaDeviceSynchronize();
#if CUDART_VERSION >= 13000
    cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
    cudaError_t pf = cudaMemPrefetchAsync(hmm, N * sizeof(int), loc, 0, 0);
#else
    cudaError_t pf = cudaMemPrefetchAsync(hmm, N * sizeof(int), 0, 0);
#endif
    cudaDeviceSynchronize();
    cudaGetLastError();
    snprintf(msg, sizeof msg, "%s, kernel %s, hmm[7]=%d, prefetch %s",
             ats ? "ATS" : "HMM", cudaGetErrorString(err), hmm[7],
             cudaGetErrorString(pf));
    check(err == cudaSuccess && hmm[7] == 8, "GPU access to plain mmap", msg);
  } else {
    check(false, "GPU access to plain mmap",
          "no HMM/ATS: managed memory has no replay-free plan here");
  }

  // 6. Texture/surface object values. Rebuild relies on values starting at 1
  //    in a fresh context and being sequential when nothing is destroyed.
  int *lin;
  cudaMalloc(&lin, N * sizeof(int));
  tex_res.resType = cudaResourceTypeLinear;
  tex_res.res.linear.devPtr = lin;
  tex_res.res.linear.desc = cudaCreateChannelDesc<int>();
  tex_res.res.linear.sizeInBytes = N * sizeof(int);
  tex_desc.readMode = cudaReadModeElementType;

  cudaTextureObject_t first = make_texture();
  cudaTextureObject_t second = make_texture();
  cudaTextureObject_t third = make_texture();
  snprintf(msg, sizeof msg, "%llu %llu %llu", first, second, third);
  check(first == 1 && second == 2 && third == 3,
        "texture values 1,2,3 in fresh ctx", msg);

  // Reproduce a sparse set by keeping fillers alive until each value is hit.
  unsigned long long want[] = {third + 4, third + 5, third + 9, third + 25};
  std::vector<cudaTextureObject_t> fillers;
  bool all_hit = true;
  int len = snprintf(msg, sizeof msg, "wanted/got:");
  for (unsigned long long w : want) {
    cudaTextureObject_t t = make_texture();
    for (int guard = 0; t < w && guard < 1000; guard++) {
      fillers.push_back(t);
      t = make_texture();
    }
    all_hit = all_hit && t == w;
    len += snprintf(msg + len, sizeof msg - len, " %llu/%llu", w, t);
  }
  check(all_hit, "sparse texture set via fillers", msg);
  for (cudaTextureObject_t t : fillers) {
    cudaDestroyTextureObject(t);
  }
  cudaTextureObject_t a = make_texture();
  cudaDestroyTextureObject(a);
  cudaTextureObject_t b = make_texture();
  snprintf(msg, sizeof msg, "create %llu, destroy, create -> %llu", a, b);
  report("INFO", "freed texture value reuse", msg);

  // 7. Addresses the design does not control. Run twice to compare.
  void *sym = NULL;
  cudaGetSymbolAddress(&sym, g_module_global);
  snprintf(msg, sizeof msg, "%p", sym);
  report("INFO", "__device__ global address", msg);
  snprintf(msg, sizeof msg, "%p", (void *)lin);
  report("INFO", "plain cudaMalloc address", msg);

  printf("%d failure(s)\n", failures);
  return failures;
}
