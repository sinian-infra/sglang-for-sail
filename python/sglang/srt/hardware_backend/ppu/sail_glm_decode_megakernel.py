"""Sail/PPU GLM-5.3 small-batch decode MegaKernels.

Provides two fused, zero-external-dependency decode kernels for Sail/PPU:
1. ``sail_dual_rmsnorm_fp8_quant``: Single-launch Triton kernel replacing
   separate ``q_a_layernorm``, ``kv_a_layernorm``, and ``q_b_proj`` per-token
   FP8 quantization across all 45 attention layers, while retaining BF16 Q for
   the DSA indexer.
2. ``SailGlmRouterQuantRunner``: Fused CUDA MegaKernel (compiled via the
   content-addressed cross-rank JIT cache in ``sail_jit_source_cache``) that
   fuses MoE router GEMV, sigmoid + correction bias, shared-expert FP8 input
   quantization, and warp-parallel Top-8 expert ranking + renormalization.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.hardware_backend.ppu.sail_jit_source_cache import load_generated_jit


@triton.jit
def _dual_rmsnorm_fp8_quant_kernel(
    q_in_ptr,
    k_in_ptr,
    wq_ptr,
    wk_ptr,
    oq_ptr,
    ok_ptr,
    fq_ptr,
    sq_ptr,
    stride_q_in,
    stride_k_in,
    stride_oq,
    stride_ok,
    stride_fq,
    NUM_TOKENS,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    eps_q: tl.constexpr,
    eps_k: tl.constexpr,
    fp8_max: tl.constexpr,
    fp8_min: tl.constexpr,
):
    """Fuse Q RMSNorm + Q per-token FP8 quant + K RMSNorm into a single launch.

    Grid: (2 * NUM_TOKENS,).
      pid < NUM_TOKENS:  Q RMSNorm -> BF16 (for DSA indexer) + FP8 (for q_b_proj).
      pid >= NUM_TOKENS: K RMSNorm -> BF16.
    """
    pid = tl.program_id(0).to(tl.int64)
    if pid < NUM_TOKENS:
        q_row = pid
        q_offs = tl.arange(0, Q_BLOCK)
        q_mask = q_offs < Q_DIM
        qx = tl.load(q_in_ptr + q_row * stride_q_in + q_offs, mask=q_mask, other=0.0).to(
            tl.float32
        )
        qw = tl.load(wq_ptr + q_offs, mask=q_mask, other=0.0).to(tl.float32)
        q_var = tl.sum(qx * qx, axis=0) / Q_DIM
        q_rstd = tl.math.rsqrt(q_var + eps_q)
        # Preserve BF16 rounding before FP8 quantization so q_b_proj matches unfused.
        y_bf16 = (qx * q_rstd * qw).to(oq_ptr.dtype.element_ty)
        tl.store(oq_ptr + q_row * stride_oq + q_offs, y_bf16, mask=q_mask)

        y_f32 = tl.where(q_mask, y_bf16.to(tl.float32), 0.0)
        absmax = tl.max(tl.abs(y_f32), axis=0)
        scale = tl.maximum(absmax, 1e-10) / fp8_max
        scale_inv = 1.0 / scale
        q_fp8 = tl.clamp(y_f32 * scale_inv, fp8_min, fp8_max).to(
            fq_ptr.dtype.element_ty
        )
        tl.store(fq_ptr + q_row * stride_fq + q_offs, q_fp8, mask=q_mask)
        tl.store(sq_ptr + q_row, scale)
    else:
        k_row = pid - NUM_TOKENS
        k_offs = tl.arange(0, K_BLOCK)
        k_mask = k_offs < K_DIM
        kx = tl.load(k_in_ptr + k_row * stride_k_in + k_offs, mask=k_mask, other=0.0).to(
            tl.float32
        )
        kw = tl.load(wk_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_var = tl.sum(kx * kx, axis=0) / K_DIM
        k_rstd = tl.math.rsqrt(k_var + eps_k)
        ky = (kx * k_rstd * kw).to(ok_ptr.dtype.element_ty)
        tl.store(ok_ptr + k_row * stride_ok + k_offs, ky, mask=k_mask)


def sail_dual_rmsnorm_fp8_quant(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    eps_q: float,
    eps_k: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch fused Q/K RMSNorm + per-token FP8 quantization for Q.

    Accepts strided views directly from ``qkv_a_proj`` output (``stride(-1) == 1``)
    with zero intermediate copy.

    Returns:
        ``(q_bf16, k_bf16, q_fp8, q_scale)`` where ``q_bf16`` feeds the DSA
        indexer and ``(q_fp8, q_scale)`` feeds ``q_b_proj`` directly.
    """
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        fp8_max,
        fp8_min,
    )

    assert q.ndim == 2 and k_nope.ndim == 2
    assert q.shape[0] == k_nope.shape[0]
    assert q.stride(-1) == 1 and k_nope.stride(-1) == 1
    num_tokens, q_dim = q.shape
    k_dim = k_nope.shape[1]

    oq = torch.empty((num_tokens, q_dim), dtype=q.dtype, device=q.device)
    ok = torch.empty((num_tokens, k_dim), dtype=k_nope.dtype, device=k_nope.device)
    fq = torch.empty((num_tokens, q_dim), dtype=fp8_dtype, device=q.device)
    sq = torch.empty((num_tokens, 1), dtype=torch.float32, device=q.device)
    if num_tokens == 0:
        return oq, ok, fq, sq

    q_block = triton.next_power_of_2(q_dim)
    k_block = triton.next_power_of_2(k_dim)
    num_warps = 8 if max(q_block, k_block) >= 1024 else 4

    _dual_rmsnorm_fp8_quant_kernel[(2 * num_tokens,)](
        q,
        k_nope,
        wq,
        wk,
        oq,
        ok,
        fq,
        sq,
        stride_q_in=q.stride(0),
        stride_k_in=k_nope.stride(0),
        stride_oq=oq.stride(0),
        stride_ok=ok.stride(0),
        stride_fq=fq.stride(0),
        NUM_TOKENS=num_tokens,
        Q_DIM=q_dim,
        K_DIM=k_dim,
        Q_BLOCK=q_block,
        K_BLOCK=k_block,
        eps_q=eps_q,
        eps_k=eps_k,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        num_warps=num_warps,
    )
    return oq, ok, fq, sq


@lru_cache(maxsize=2)
def load_sail_glm_router_quant_module():
    """Compile and cache ``sail_glm_router_quant.cuh`` via atomic cross-rank source cache."""
    import sglang.kernels.jit as jit_pkg
    from sglang.kernels.jit.utils import load_jit

    src_path = (
        Path(jit_pkg.__file__).resolve().parent
        / "csrc"
        / "moe"
        / "sail_glm_router_quant.cuh"
    )
    source = src_path.read_text(encoding="utf-8")
    return load_generated_jit(
        load_jit,
        "sail_glm_router_quant.cuh",
        source,
        "sail_glm_router_quant",
        cuda_wrappers=[
            ("producer", "sglang::sail_glm_router::SailGlmRouterQuant::producer"),
            ("rank", "sglang::sail_glm_router::SailGlmRouterQuant::rank"),
        ],
    )


class SailGlmRouterQuantRunner:
    """Graph-safe fused Router GEMV + Sigmoid/Bias + TopK8 + Shared-Expert FP8 Quant."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        max_rows: int = 16,
        routed_scaling_factor: float = 2.5,
        physical_table: Optional[torch.Tensor] = None,
    ):
        assert weight.is_cuda and weight.is_contiguous()
        assert weight.shape == (288, 4096) and weight.dtype == torch.bfloat16
        assert bias.device == weight.device and bias.is_contiguous()
        assert bias.shape == (288,) and bias.dtype == torch.float32
        assert 1 <= max_rows <= 16

        self.weight = weight
        self.bias = bias
        self.max_rows = max_rows
        self.routed_scaling_factor = float(routed_scaling_factor)
        device = weight.device
        self.physical_table = (
            physical_table.to(device=device, dtype=torch.int32).contiguous()
            if physical_table is not None
            else torch.empty((0,), device=device, dtype=torch.int32)
        )
        self.q = torch.empty((max_rows, 4096), device=device, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty((max_rows, 1), device=device, dtype=torch.float32)
        self.scores = torch.empty((max_rows, 288), device=device, dtype=torch.float32)
        self.active = torch.empty((max_rows, 288), device=device, dtype=torch.float32)
        self.ranked = torch.empty((max_rows, 288), device=device, dtype=torch.float32)
        self.weights = torch.empty((max_rows, 8), device=device, dtype=torch.float32)
        self.ids = torch.empty((max_rows, 8), device=device, dtype=torch.int64)
        self.completion = torch.zeros((max_rows,), device=device, dtype=torch.int32)
        self._valid_count_buf = torch.empty((1,), device=device, dtype=torch.int32)
        self._mod = load_sail_glm_router_quant_module()

    def run(
        self,
        x: torch.Tensor,
        valid_count: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run fused router + FP8 quant for ``x`` of shape ``(B, 4096)`` (1 <= B <= max_rows).

        Returns:
            ``(topk_weights, topk_ids, x_fp8, x_scale)``
        """
        rows = x.shape[0]
        assert 1 <= rows <= self.max_rows and x.shape[1] == 4096
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        if valid_count is None:
            self._valid_count_buf.fill_(rows)
            valid_count = self._valid_count_buf

        q = self.q[:rows]
        scale = self.scale[:rows]
        scores = self.scores[:rows]
        active = self.active[:rows]
        ranked = self.ranked[:rows]
        weights = self.weights[:rows]
        ids = self.ids[:rows]
        completion = self.completion[:rows]

        self._mod.producer(x, self.weight, q, scale, self.bias, scores, active, ranked)
        self._mod.rank(
            active,
            ranked,
            valid_count,
            weights,
            ids,
            completion,
            self.physical_table,
            self.routed_scaling_factor,
        )
        return weights, ids, q, scale
