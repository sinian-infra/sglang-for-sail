// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cstdint>

namespace sglang::sail_ipc {

namespace {

// Lock-free per-tile IPC AllGather for small/medium decode batches on Sail/PPU.
// Each CTA independently pushes its tile to every peer's symmetric double-buffer
// slot, publishes a per-tile release flag, waits for peers, and copies out.
// Layout per rank workspace:
//   [0, 65536):        per-tile peer ready flags: tile * (World * Stride) + peer * Stride
//   [65536, 131072):   per-tile local epoch counter: tile * 128
//   [131072, ...):     2 double-buffered payload slots of (World * shard_bytes) each
template <int World, int Tiles, int Threads>
__global__ __launch_bounds__(Threads) void sail_ipc_allgather_kernel(
    const uint4* __restrict__ input,
    uint4* __restrict__ output,
    const uint64_t* __restrict__ peers,
    int rank,
    int shard_u4,
    int* __restrict__ error) {
  static_assert(World == 4 || World == 8, "Supported world sizes are 4 and 8");
  static_assert(
      Tiles == 1 || Tiles == 4 || Tiles == 8 || Tiles == 16 || Tiles == 32 || Tiles == 64,
      "Supported tile counts are 1, 4, 8, 16, 32, 64");
  static_assert(Threads % World == 0, "Threads must be divisible by World");

  const int V = shard_u4 / Tiles;
  __shared__ unsigned epoch;
  __shared__ int failed;
  const int tile = blockIdx.x;
  char* own = reinterpret_cast<char*>(peers[rank]);

  if (threadIdx.x == 0) {
    epoch = ++*reinterpret_cast<unsigned*>(own + 65536 + tile * 128);
    failed = 0;
  }
  __syncthreads();

  constexpr int Group = Threads / World;
  const int r = threadIdx.x / Group;
  const int lane = threadIdx.x % Group;
  const size_t slot_offset = 131072 + size_t(epoch % 2) * World * size_t(shard_u4) * 16;

  uint4* remote =
      reinterpret_cast<uint4*>(reinterpret_cast<char*>(peers[r]) + slot_offset) +
      size_t(rank) * shard_u4 + size_t(tile) * V;
  for (int j = lane; j < V; j += Group) {
    remote[j] = input[size_t(tile) * V + j];
  }
  __syncthreads();

  if (threadIdx.x < World) {
    const int peer = threadIdx.x;
    auto* flag = reinterpret_cast<unsigned*>(
        reinterpret_cast<char*>(peers[peer]) + tile * (World * 128) + rank * 128);
    asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(flag), "r"(epoch) : "memory");

    auto* local_flag =
        reinterpret_cast<unsigned*>(own + tile * (World * 128) + peer * 128);
    unsigned spins = 0;
    while (true) {
      unsigned observed;
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                   : "=r"(observed)
                   : "l"(local_flag)
                   : "memory");
      if (observed >= epoch) {
        break;
      }
      if (++spins == 20000000) {
        atomicExch(&failed, 100 + tile * World + peer);
        break;
      }
    }
  }
  __syncthreads();

  if (!failed) {
    const uint4* local_in =
        reinterpret_cast<const uint4*>(own + slot_offset) + size_t(r) * shard_u4 + size_t(tile) * V;
    for (int j = lane; j < V; j += Group) {
      output[size_t(r) * shard_u4 + size_t(tile) * V + j] = local_in[j];
    }
  }
  if (threadIdx.x == 0 && failed) {
    atomicExch(error, failed);
  }
}

// Lock-free per-tile IPC ReduceScatter for small/medium decode batches on Sail/PPU.
// Uses deterministic ring-order BF16 __hadd2 accumulation matching PCCL ring order.
template <int World, int Tiles, int Threads, int Stride>
__global__ __launch_bounds__(Threads) void sail_ipc_reduce_scatter_kernel(
    const uint4* __restrict__ input,
    uint4* __restrict__ output,
    const uint64_t* __restrict__ peers,
    int rank,
    int shard_u4,
    int* __restrict__ error) {
  static_assert(World == 4 || World == 8, "Supported world sizes are 4 and 8");
  static_assert(
      Tiles == 1 || Tiles == 4 || Tiles == 8 || Tiles == 16 || Tiles == 32 || Tiles == 64,
      "Supported tile counts are 1, 4, 8, 16, 32, 64");
  static_assert(Threads % World == 0, "Threads must be divisible by World");

  const int V = shard_u4 / Tiles;
  __shared__ unsigned epoch;
  __shared__ int failed;
  const int tile = blockIdx.x;
  char* own = reinterpret_cast<char*>(peers[rank]);

  if (threadIdx.x == 0) {
    epoch = ++*reinterpret_cast<unsigned*>(own + 65536 + tile * 128);
    failed = 0;
  }
  __syncthreads();

  constexpr int Group = Threads / World;
  const int r = threadIdx.x / Group;
  const int lane = threadIdx.x % Group;
  const size_t slot_offset = 131072 + size_t(epoch % 2) * World * size_t(shard_u4) * 16;

  uint4* remote =
      reinterpret_cast<uint4*>(reinterpret_cast<char*>(peers[r]) + slot_offset) +
      size_t(rank) * shard_u4 + size_t(tile) * V;
  for (int j = lane; j < V; j += Group) {
    remote[j] = input[size_t(r) * shard_u4 + size_t(tile) * V + j];
  }
  __syncthreads();

  if (threadIdx.x < World) {
    const int peer = threadIdx.x;
    auto* flag = reinterpret_cast<unsigned*>(
        reinterpret_cast<char*>(peers[peer]) + tile * (World * Stride) + rank * Stride);
    asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(flag), "r"(epoch) : "memory");

    auto* local_flag =
        reinterpret_cast<unsigned*>(own + tile * (World * Stride) + peer * Stride);
    unsigned spins = 0;
    while (true) {
      unsigned observed;
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                   : "=r"(observed)
                   : "l"(local_flag)
                   : "memory");
      if (observed >= epoch) {
        break;
      }
      if (++spins == 20000000) {
        atomicExch(&failed, 100 + tile * World + peer);
        break;
      }
    }
  }
  __syncthreads();

  if (!failed) {
    const auto* base = reinterpret_cast<const __nv_bfloat162*>(own + slot_offset);
    const int pairs_shard = shard_u4 * 4;
    const int pairs_tile = pairs_shard / Tiles;
    const int first = (rank + 1) % World;
    for (int j = threadIdx.x; j < pairs_tile; j += Threads) {
      __nv_bfloat162 sum = base[size_t(first) * pairs_shard + size_t(tile) * pairs_tile + j];
#pragma unroll
      for (int k = 1; k < World; ++k) {
        const int sender = (rank + 1 + k) % World;
        sum = __hadd2(sum, base[size_t(sender) * pairs_shard + size_t(tile) * pairs_tile + j]);
      }
      reinterpret_cast<__nv_bfloat162*>(output)[size_t(tile) * pairs_tile + j] = sum;
    }
  }
  if (threadIdx.x == 0 && failed) {
    atomicExch(error, failed);
  }
}

}  // namespace

template <int World, int Tiles, int Threads>
struct SailIpcAllGather {
  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView peers,
      int64_t rank,
      tvm::ffi::TensorView error) {
    using namespace host;
    RuntimeCheck(input.ndim() == 2, "SailIpcAllGather expects a 2D BF16 tensor");
    const int64_t rows = input.size(0);
    const int64_t width = input.size(1);
    RuntimeCheck(rows > 0 && rows <= 16, "SailIpcAllGather supports 1..16 local rows");
    RuntimeCheck(width > 0 && width % 8 == 0, "SailIpcAllGather width must be a multiple of 8");
    const int64_t shard_u4 = (rows * width) / 8;
    RuntimeCheck(shard_u4 % Tiles == 0, "SailIpcAllGather shard_u4 must be divisible by Tiles");
    RuntimeCheck(output.numel() == World * input.numel(), "Output size mismatch for AllGather");

    LaunchKernel(dim3(Tiles), Threads, input.device())(
        sail_ipc_allgather_kernel<World, Tiles, Threads>,
        static_cast<const uint4*>(input.data_ptr()),
        static_cast<uint4*>(output.data_ptr()),
        static_cast<const uint64_t*>(peers.data_ptr()),
        int(rank),
        int(shard_u4),
        static_cast<int*>(error.data_ptr()));
  }
};

template <int World, int Tiles, int Threads, int Stride>
struct SailIpcReduceScatter {
  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView peers,
      int64_t rank,
      tvm::ffi::TensorView error) {
    using namespace host;
    RuntimeCheck(input.ndim() == 2, "SailIpcReduceScatter expects a 2D BF16 tensor");
    const int64_t rows = input.size(0);
    const int64_t width = input.size(1);
    RuntimeCheck(rows > 0 && rows % World == 0, "SailIpcReduceScatter rows must be divisible by World");
    const int64_t local_rows = rows / World;
    RuntimeCheck(local_rows <= 16, "SailIpcReduceScatter supports at most 16 local rows per rank");
    RuntimeCheck(width > 0 && width % 8 == 0, "SailIpcReduceScatter width must be a multiple of 8");
    const int64_t shard_u4 = (local_rows * width) / 8;
    RuntimeCheck(shard_u4 % Tiles == 0, "SailIpcReduceScatter shard_u4 must be divisible by Tiles");
    RuntimeCheck(output.numel() * World == input.numel(), "Output size mismatch for ReduceScatter");

    LaunchKernel(dim3(Tiles), Threads, input.device())(
        sail_ipc_reduce_scatter_kernel<World, Tiles, Threads, Stride>,
        static_cast<const uint4*>(input.data_ptr()),
        static_cast<uint4*>(output.data_ptr()),
        static_cast<const uint64_t*>(peers.data_ptr()),
        int(rank),
        int(shard_u4),
        static_cast<int*>(error.data_ptr()));
  }
};

}  // namespace sglang::sail_ipc
