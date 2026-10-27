// Driver for the VMM checkpoint proof of concept (../vmm-ckpt.cpp).
//
// Builds a linked list in device memory whose nodes hold device pointers to
// each other and to payload buffers, with freed buffers in between so the
// final layout depends on allocation history. Checkpointing writes an image
// and exits; --restart continues from it in a brand-new process (new driver
// instance, new context, different ASLR layout).
//
//   LD_PRELOAD=../libvmm_ckpt.so ./vmm_poc --ckpt-at 3 --image a.img
//   LD_PRELOAD=../libvmm_ckpt.so ./vmm_poc --restart a.img

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../vmm-ckpt.h"

// Resolved from the LD_PRELOADed library; NULL if it is not loaded.
#pragma weak vmm_ckpt_save
#pragma weak vmm_ckpt_restore

#define NUM_NODES 6
#define THREADS 256

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err_));   \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

struct Node {
  Node *next;    // device pointer stored in device memory
  int *payload;  // device pointer stored in device memory
  int len;
  int id;
};

// All host-side state the program needs to keep running. Under DMTCP this is
// just process memory; here it rides along in the checkpoint image.
struct AppState {
  int iter;
  Node *head;
  int *mismatches;
  void *junk[NUM_NODES];
  void *holes[NUM_NODES];  // addresses of junk buffers freed during setup
  void *symbol_addr;  // where the driver placed g_module_global
};

__device__ int g_module_global;

__global__ void init_payload(int *payload, int len, int id) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < len;
       j += gridDim.x * blockDim.x) {
    payload[j] = j % 977 + id;
  }
}

__global__ void step(Node *head) {
  for (Node *n = head; n != NULL; n = n->next) {
    for (int j = threadIdx.x; j < n->len; j += blockDim.x) {
      n->payload[j] += n->id + 1;
    }
  }
}

__global__ void verify(Node *head, int iter, int *mismatches) {
  for (Node *n = head; n != NULL; n = n->next) {
    for (int j = threadIdx.x; j < n->len; j += blockDim.x) {
      if (n->payload[j] != j % 977 + n->id + iter * (n->id + 1)) {
        atomicAdd(mismatches, 1);
      }
    }
  }
}

static void setup(AppState *s) {
  memset(s, 0, sizeof *s);
  CHECK(cudaMalloc((void **)&s->mismatches, sizeof(int)));

  Node host_nodes[NUM_NODES];
  Node *dev_nodes[NUM_NODES];
  for (int k = 0; k < NUM_NODES; k++) {
    CHECK(cudaMalloc(&s->junk[k], (size_t)(k + 1) << 20));
    CHECK(cudaMalloc((void **)&dev_nodes[k], sizeof(Node)));
    int len = (k + 1) * 300000;
    int *payload;
    CHECK(cudaMalloc((void **)&payload, len * sizeof(int)));
    init_payload<<<64, THREADS>>>(payload, len, k);
    host_nodes[k] = {NULL, payload, len, k};
  }
  for (int k = 0; k < NUM_NODES; k++) {
    host_nodes[k].next = k + 1 < NUM_NODES ? dev_nodes[k + 1] : NULL;
    CHECK(cudaMemcpy(dev_nodes[k], &host_nodes[k], sizeof(Node),
                     cudaMemcpyHostToDevice));
  }
  // Punch holes. Replaying this history is what log-and-replay has to do.
  for (int k = 1; k < NUM_NODES; k += 2) {
    CHECK(cudaFree(s->junk[k]));
    s->holes[k] = s->junk[k];
    s->junk[k] = NULL;
  }
  s->head = dev_nodes[0];
  CHECK(cudaGetSymbolAddress(&s->symbol_addr, g_module_global));
  printf("launch: head=%p mismatches=%p __device__ global=%p\n", s->head,
         s->mismatches, s->symbol_addr);
}

static void after_restart(AppState *s) {
  printf("restart: head=%p mismatches=%p (pointers taken from the image)\n",
         s->head, s->mismatches);

  // Module globals are placed by the driver at module load, not by cudaMalloc,
  // so VMM does not pin them. Report where they ended up.
  void *symbol_addr;
  CHECK(cudaGetSymbolAddress(&symbol_addr, g_module_global));
  printf("restart: __device__ global %p -> %p (%s)\n", s->symbol_addr,
         symbol_addr, symbol_addr == s->symbol_addr ? "same" : "MOVED");
  s->symbol_addr = symbol_addr;

  // The allocator keeps working without replay: a new allocation lands in a
  // hole that cudaFree() left before the checkpoint.
  void *extra;
  CHECK(cudaMalloc(&extra, 3 << 20));
  int hole = -1;
  for (int k = 0; k < NUM_NODES; k++) {
    if (s->holes[k] != NULL && s->holes[k] == extra) {
      hole = k;
    }
  }
  if (hole >= 0) {
    printf("restart: new 3 MiB allocation at %p reuses the hole junk[%d] "
           "left before the checkpoint\n", extra, hole);
    s->junk[hole] = extra;
    s->holes[hole] = NULL;
  } else {
    printf("restart: new 3 MiB allocation at %p\n", extra);
  }
}

int main(int argc, char **argv) {
  int iters = 9;
  int ckpt_at = -1;
  const char *image = "vmm-ckpt.img";
  const char *restart = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
      iters = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--ckpt-at") && i + 1 < argc) {
      ckpt_at = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--image") && i + 1 < argc) {
      image = argv[++i];
    } else if (!strcmp(argv[i], "--restart") && i + 1 < argc) {
      restart = argv[++i];
    } else {
      fprintf(stderr, "usage: %s [--iters N] [--ckpt-at N] [--image FILE] "
                      "[--restart FILE]\n", argv[0]);
      return 2;
    }
  }
  if (vmm_ckpt_save == NULL || vmm_ckpt_restore == NULL) {
    fprintf(stderr, "run with LD_PRELOAD=/path/to/libvmm_ckpt.so\n");
    return 2;
  }

  AppState s;
  if (restart) {
    if (vmm_ckpt_restore(restart, &s, sizeof s) != 0) {
      return 1;
    }
    after_restart(&s);
  } else {
    setup(&s);
  }

  while (s.iter < iters) {
    step<<<1, THREADS>>>(s.head);
    s.iter++;
    int bad = 0;
    CHECK(cudaMemset(s.mismatches, 0, sizeof(int)));
    verify<<<1, THREADS>>>(s.head, s.iter, s.mismatches);
    CHECK(cudaMemcpy(&bad, s.mismatches, sizeof bad, cudaMemcpyDeviceToHost));
    printf("iter %d: %s\n", s.iter, bad ? "MISMATCH" : "ok");
    if (bad) {
      return 1;
    }
    if (s.iter == ckpt_at) {
      if (vmm_ckpt_save(image, &s, sizeof s) != 0) {
        return 1;
      }
      printf("checkpointed at iter %d to %s; exiting\n", s.iter, image);
      return 0;
    }
  }
  printf("done: %d iterations verified\n", s.iter);
  return 0;
}
