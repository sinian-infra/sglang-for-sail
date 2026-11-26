// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cstdint>

namespace sglang::sail_glm_router {

namespace {

constexpr uint32_t kQuantWarpSize = 32;

template <typename T, int kVecSize>
__device__ __forceinline__ void per_token_fp8_quant(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    uint32_t hidden_dim) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t num_vecs = hidden_dim / kVecSize;
  float max_value = 0.0f;
  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    input_vec_t input_vec;
    input_vec.load(input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
  }

  __shared__ float reduction_smem[kQuantWarpSize];
  __shared__ float scale_smem;
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    scale_smem = math::max(reduction_smem[0], 1e-10f) / math::FP8_E4M3_MAX;
    output_s[0] = scale_smem;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale_smem;

  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    input_vec_t input_vec;
    output_vec_t output_vec;
    input_vec.load(input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float value = static_cast<float>(input_vec[j]) * scale_inv;
      output_vec[j] = static_cast<fp8_e4m3_t>(
          math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    output_vec.store(output_q, i);
  }
}

__device__ __forceinline__ float full_div_f32(float a, float b) {
  float r;
  asm volatile("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float fast_sigmoid_f32(float s) {
  float r;
  float x = __fmul_rn(-s, __int_as_float(0x3fb8aa3b));
  asm volatile("ex2.approx.f32 %0,%1;" : "=f"(r) : "f"(x));
  return full_div_f32(1.0f, __fadd_rn(1.0f, r));
}

// Fused small-batch decode MoE router GEMV + sigmoid + correction bias +
// shared-expert per-token FP8 quantization.
// Grid: (1 + NumExperts / 2, Batch), Block: 512 threads.
//   blockIdx.x == 0: per-token FP8 quant of hidden_states [token, 4096].
//   blockIdx.x >= 1: 2 teams of 256 threads each compute 1 expert logit row,
//                    reduce across 8 warps, apply sigmoid + bias.
__global__ __launch_bounds__(512) void router_gemv_sigmoid_fp8_quant_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    sglang::fp8_e4m3_t* __restrict__ q,
    float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ scores,
    float* __restrict__ active,
    float* __restrict__ ranked) {
  const int token = blockIdx.y;
  x += size_t(token) * 4096;
  q += size_t(token) * 4096;
  scale += token;
  scores += size_t(token) * 288;
  active += size_t(token) * 288;
  ranked += size_t(token) * 288;

  if (blockIdx.x == 0) {
    per_token_fp8_quant<__nv_bfloat16, 8>(x, q, scale, 4096);
    return;
  }

  const int team = threadIdx.x / 256;
  const int tid = threadIdx.x % 256;
  const int row = (blockIdx.x - 1) * 2 + team;
  const int lane = tid % 32;
  const int split = tid / 32;

  float acc = 0.0f;
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int base = split * 256 + lane * 8 + it * 2048;
    const uint4 xp = *reinterpret_cast<const uint4*>(x + base);
    const uint4 wp = *reinterpret_cast<const uint4*>(w + size_t(row) * 4096 + base);
    const auto* a = reinterpret_cast<const __nv_bfloat16*>(&xp);
    const auto* b = reinterpret_cast<const __nv_bfloat16*>(&wp);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc = __fmaf_rn(__bfloat162float(a[j]), __bfloat162float(b[j]), acc);
    }
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, delta));
  }

  __shared__ float parts_store[16];
  float* parts = parts_store + team * 8;
  if (lane == 0) {
    parts[split] = acc;
  }
  __syncthreads();
  if (tid == 0) {
    const float v = __fadd_rn(
        __fadd_rn(__fadd_rn(parts[0], parts[4]), __fadd_rn(parts[2], parts[6])),
        __fadd_rn(__fadd_rn(parts[1], parts[5]), __fadd_rn(parts[3], parts[7])));
    const float a_val = fast_sigmoid_f32(v);
    const float b_val = __fadd_rn(a_val, bias[row]);
    scores[row] = v;
    active[row] = a_val;
    ranked[row] = (b_val == b_val) ? b_val : -1e30f;
  }
}

// Warp-parallel Top-8 selection + renorm + routed_scaling_factor + optional
// logical->physical expert placement mapping.
template <int Threads, int Batch>
__global__ __launch_bounds__(Threads) void router_topk8_renorm_kernel(
    const float* __restrict__ active_input,
    const float* __restrict__ ranked_input,
    const int* __restrict__ valid_count,
    float* __restrict__ weights,
    int64_t* __restrict__ ids,
    int* __restrict__ completion,
    const int* __restrict__ physical_table,
    float routed_scaling_factor) {
  const int token = blockIdx.y;
  active_input += size_t(token) * 288;
  ranked_input += size_t(token) * 288;
  weights += size_t(token) * 8;
  ids += size_t(token) * 8;
  completion += token;

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;

  __shared__ float active[288];
  __shared__ __align__(4096) float ranked[1024];
  __shared__ int is_last;

  for (int n = tid; n < 288; n += Threads) {
    active[n] = active_input[n];
    ranked[((n & 3) * 256) + (n >> 2)] = ranked_input[n];
  }
  __syncthreads();

#pragma unroll
  for (int round = 0; round < (Batch + Threads / 32 - 1) / (Threads / 32); ++round) {
    const int target = blockIdx.x * Batch + warp + round * (Threads / 32);
    if (target < (blockIdx.x + 1) * Batch) {
      float v = 0.0f;
      if (lane == 0) {
        v = ranked[((target & 3) * 256) + (target >> 2)];
      }
      v = __shfl_sync(0xffffffffu, v, 0);
      int rank = 0;
#pragma unroll
      for (int j = 0; j < 9; ++j) {
        const int n = lane + 32 * j;
        const float other = ranked[((n & 3) * 256) + (n >> 2)];
        rank += (other > v) || (other == v && n < target);
      }
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        rank += __shfl_down_sync(0xffffffffu, rank, d);
      }
      if (lane == 0 && rank < 8) {
        weights[rank] = active[target];
        ids[rank] = (*valid_count > token)
                        ? (physical_table ? int64_t(physical_table[target]) : int64_t(target))
                        : -1;
      }
    }
  }

  __threadfence();
  __syncthreads();
  if (tid == 0) {
    is_last = (atomicAdd(completion, 1) == (288 / Batch - 1));
  }
  __syncthreads();
  if (is_last) {
    __threadfence();
    if (tid < 8) {
      volatile float* w = weights;
      const float a = __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[1]), w[2]), w[3]);
      const float b = __fadd_rn(__fadd_rn(__fadd_rn(w[4], w[5]), w[6]), w[7]);
      float s = __fadd_rn(a, b);
      s = s > 0.0f ? s : 1.0f;
      active[tid] = full_div_f32(w[tid], s);
    }
    __syncthreads();
    if (tid < 8) {
      weights[tid] = __fmul_rn(active[tid], routed_scaling_factor);
    }
    __syncthreads();
    if (tid == 0) {
      atomicExch(completion, 0);
    }
  }
}

}  // namespace

struct SailGlmRouterQuant {
  static void producer(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView w,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView active,
      tvm::ffi::TensorView ranked) {
    using namespace sglang::host;
    const int rows = int(x.numel() / 4096);
    RuntimeCheck(
        rows >= 1 && rows <= 16 && x.numel() == rows * 4096 &&
            w.numel() == 288 * 4096 && q.numel() == rows * 4096 &&
            scale.numel() == rows && bias.numel() == 288 &&
            scores.numel() == rows * 288 && active.numel() == rows * 288 &&
            ranked.numel() == rows * 288,
        "SailGlmRouterQuant::producer expects H=4096, E=288, 1<=B<=16");
    LaunchKernel(dim3(145, rows), 512, x.device())(
        router_gemv_sigmoid_fp8_quant_kernel,
        static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<const __nv_bfloat16*>(w.data_ptr()),
        static_cast<sglang::fp8_e4m3_t*>(q.data_ptr()),
        static_cast<float*>(scale.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<float*>(scores.data_ptr()),
        static_cast<float*>(active.data_ptr()),
        static_cast<float*>(ranked.data_ptr()));
  }

  static void rank(
      tvm::ffi::TensorView active,
      tvm::ffi::TensorView ranked,
      tvm::ffi::TensorView valid_count,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView completion,
      tvm::ffi::TensorView physical_table,
      double routed_scaling_factor) {
    using namespace sglang::host;
    const int rows = int(active.numel() / 288);
    RuntimeCheck(
        rows >= 1 && rows <= 16 && active.numel() == rows * 288 &&
            ranked.numel() == rows * 288 && weights.numel() == rows * 8 &&
            ids.numel() == rows * 8 && valid_count.numel() == 1 &&
            completion.numel() == rows,
        "SailGlmRouterQuant::rank expects E=288, topk=8, 1<=B<=16");
    const int* phys_ptr =
        physical_table.numel() == 288
            ? static_cast<const int*>(physical_table.data_ptr())
            : nullptr;
    LaunchKernel(dim3(36, rows), 320, active.device())(
        router_topk8_renorm_kernel<320, 8>,
        static_cast<const float*>(active.data_ptr()),
        static_cast<const float*>(ranked.data_ptr()),
        static_cast<const int*>(valid_count.data_ptr()),
        static_cast<float*>(weights.data_ptr()),
        static_cast<int64_t*>(ids.data_ptr()),
        static_cast<int*>(completion.data_ptr()),
        phys_ptr,
        static_cast<float>(routed_scaling_factor));
  }
};

}  // namespace sglang::sail_glm_router
