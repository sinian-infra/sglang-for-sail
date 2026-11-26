import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    q_ptr=None,
    s_ptr=None,
    QUANT: tl.constexpr = False,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

    result = attn * tl.sigmoid(g)
    if QUANT:  # the block is one row: per-token FP8 as _per_token_group_quant_8bit
        y = tl.where(mask, result.to(output_ptr.dtype.element_ty).to(tl.float32), 0.0)
        y_s = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
        y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + attn_off, y_q, mask=mask)
        tl.store(s_ptr + pid_row, y_s)
    else:
        tl.store(output_ptr + attn_off, result, mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
    quant: bool = False,
) -> torch.Tensor:
    """
    Fused sigmoid-mul for attention output gating.

    Equivalent to: attn_output * sigmoid(gate)

    The production Qwen3.5 path passes a 3D strided gate. A single hidden-block
    Triton kernel handles both that path and flat contiguous inputs.

    When inplace=True, writes result back to attn_output and returns it.

    Supports strided gate: if gate is 3D (num_tokens, num_heads, head_dim)
    and attn_output is 2D (num_tokens, hidden_dim), the kernel reads gate
    via explicit strides without requiring a contiguous copy.
    """
    if gate.ndim == 3 and attn_output.ndim == 2:
        # Strided gate path: gate is 3D (num_tokens, num_heads, head_dim)
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        # Flat path: both tensors have the same shape
        assert attn_output.shape == gate.shape, (
            "attn_output and gate must have the same shape"
        )
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if guard_or_true(num_tokens < 1024) else 2048
    kwargs = {}
    if quant:  # one block per row; returns per-token FP8 (q, scale [T, 1])
        block_h = triton.next_power_of_2(hidden_dim)
        q = torch.empty_like(attn_output, dtype=torch.float8_e4m3fn)
        s = torch.empty((num_tokens, 1), dtype=torch.float32, device=q.device)
        kwargs = {"q_ptr": q, "s_ptr": s, "QUANT": True}
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
        **kwargs,
    )
    return (q, s) if quant else out


@triton.jit
def _sigmoid_gate_mul_broadcast_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
    out = x * g
    tl.store(
        out_ptr + row * hidden_dim + offs,
        out.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


def sigmoid_gate_mul_broadcast(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` where gate is (N, 1) and x is (N, D)."""
    bs, hidden_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(
        min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
    )
    _sigmoid_gate_mul_broadcast_kernel[(bs,)](
        out,
        gate,
        x,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _fused_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    gate_weight_ptr,  # [hidden_dim]
    shared_output_ptr,  # [num_tokens, hidden_dim]
    output_ptr,  # [num_tokens, hidden_dim], optionally also the addend
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_ADD: tl.constexpr = True,
    USE_PDL: tl.constexpr = False,
    GATE_ONLY: tl.constexpr = False,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if not GATE_ONLY:
        s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if DO_ADD:
            f = tl.load(output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
                tl.float32
            )

        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    if GATE_ONLY:
        # Preserve FP32 for the later shared-add FMA, without an intermediate
        # BF16 gated shared output. Keep the standalone gate's PDL ordering.
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(output_ptr + pid, gate_val)
    else:
        result = gate_val * s
        if DO_ADD:
            result += f
        tl.store(output_ptr + row_offset + offsets, result, mask=mask)


def _launch_fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    output: torch.Tensor,
    *,
    do_add: bool,
) -> None:
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert gate_weight.is_contiguous(), "gate_weight must be contiguous"
    assert shared_output.is_contiguous(), "shared_output must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"

    num_tokens, hidden_dim = hidden_states.shape
    assert gate_weight.shape == (hidden_dim,)
    assert shared_output.shape == (num_tokens, hidden_dim)
    assert output.shape == (num_tokens, hidden_dim)

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    if guard_or_false(num_tokens >= 1024) or is_batch_invariant_mode_enabled():
        config["num_warps"] = min(config["num_warps"], 8)

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        shared_output,
        output,
        hidden_dim=hidden_dim,
        DO_ADD=do_add,
        USE_PDL=use_pdl,
        GATE_ONLY=False,
        **config,
        **pdl_kwargs,
    )


def fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    """Materialize the gated shared-expert contribution without an add/copy."""
    output = torch.empty_like(shared_output)
    _launch_fused_gate_sigmoid_mul(
        hidden_states,
        gate_weight,
        shared_output,
        output,
        do_add=False,
    )
    return output


def fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Add the gated shared-expert contribution to routed-expert output."""
    _launch_fused_gate_sigmoid_mul(
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        do_add=True,
    )


def _get_fp8_quant_params():
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        fp8_max,
        fp8_min,
    )

    return fp8_dtype, fp8_max, fp8_min


@triton.jit
def _gdn_gated_rmsnorm_fp8_quant_2d_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    stride_x,
    stride_z,
    stride_out,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    fp8_min: tl.constexpr,
    HEAD_COUNT_BLOCK: tl.constexpr,
    HEAD_DIM_BLOCK: tl.constexpr,
):
    """Single-pass fused GDN per-head SiLU-gated RMSNorm + per-token FP8 quant.

    Keeps the entire [NUM_HEADS, HEAD_DIM] token slice in registers and preserves
    exact BF16/FP16 rounding before per-token absmax and E4M3 quantization,
    avoiding any intermediate global-memory round-trip.
    """
    pid = tl.program_id(0).to(tl.int64)
    h_offs = tl.arange(0, HEAD_COUNT_BLOCK)[:, None]
    d_offs = tl.arange(0, HEAD_DIM_BLOCK)[None, :]
    mask = (h_offs < NUM_HEADS) & (d_offs < HEAD_DIM)
    elem_offs = h_offs * HEAD_DIM + d_offs

    x = tl.load(x_ptr + pid * stride_x + elem_offs, mask=mask, other=0.0).to(
        tl.float32
    )
    z = tl.load(z_ptr + pid * stride_z + elem_offs, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(weight_ptr + d_offs, mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1)[:, None] / HEAD_DIM
    rrms = tl.math.rsqrt(var + eps)
    # Match unfused RMSNormGated output rounding prior to FP8 quantization.
    y = (x * rrms * w * z * tl.sigmoid(z)).to(x_ptr.dtype.element_ty).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    overall_max = tl.max(tl.abs(y))
    scale = tl.maximum(overall_max, 1e-10) / fp8_max
    scale_inv = 1.0 / scale

    q = tl.clamp(y * scale_inv, fp8_min, fp8_max).to(out_fp8_ptr.dtype.element_ty)
    tl.store(out_fp8_ptr + pid * stride_out + elem_offs, q, mask=mask)
    tl.store(out_scale_ptr + pid, scale)


@triton.jit
def _gdn_gated_rmsnorm_fp8_quant_multipass_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    x_norm_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    stride_x,
    stride_z,
    stride_x_norm,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    fp8_min: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    """Multi-pass fallback when [HEAD_COUNT_BLOCK, HEAD_DIM_BLOCK] exceeds register budget."""
    pid = tl.program_id(0).to(tl.int64)
    x_base = pid * stride_x
    z_base = pid * stride_z
    norm_base = pid * stride_x_norm

    for h in range(NUM_HEADS):
        col_offs = tl.arange(0, HEAD_BLOCK)
        head_mask = col_offs < HEAD_DIM
        elem_offs = h * HEAD_DIM + col_offs

        x = tl.load(x_ptr + x_base + elem_offs, mask=head_mask, other=0.0).to(
            tl.float32
        )
        z = tl.load(z_ptr + z_base + elem_offs, mask=head_mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(weight_ptr + col_offs, mask=head_mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / HEAD_DIM
        rrms = tl.math.rsqrt(var + eps)
        y = x * rrms * w * z * tl.sigmoid(z)

        tl.store(
            x_norm_ptr + norm_base + elem_offs,
            y.to(x_norm_ptr.dtype.element_ty),
            mask=head_mask,
        )

    g_max = tl.zeros([QUANT_BLOCK], dtype=tl.float32)
    for start in range(0, VALUE_DIM, QUANT_BLOCK):
        offs = start + tl.arange(0, QUANT_BLOCK)
        mask = offs < VALUE_DIM
        vals = tl.load(x_norm_ptr + norm_base + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        g_max = tl.maximum(g_max, tl.where(mask, tl.abs(vals), 0.0))

    overall_max = tl.max(g_max, axis=0)
    scale = tl.maximum(overall_max, 1e-10) / fp8_max
    scale_inv = 1.0 / scale

    for start in range(0, VALUE_DIM, QUANT_BLOCK):
        offs = start + tl.arange(0, QUANT_BLOCK)
        mask = offs < VALUE_DIM
        vals = tl.load(x_norm_ptr + norm_base + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        q = tl.clamp(vals * scale_inv, fp8_min, fp8_max).to(
            out_fp8_ptr.dtype.element_ty
        )
        tl.store(out_fp8_ptr + norm_base + offs, q, mask=mask)

    tl.store(out_scale_ptr + pid, scale)


def gdn_gated_rmsnorm_fp8_quant(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
):
    """Fused GDN per-head SiLU-gated RMSNorm + per-token dynamic FP8 quantization.

    Args:
        x: Contiguous 2D tensor of shape ``(T * num_heads, head_dim)``.
        z: Contiguous 2D gate tensor of shape ``(T * num_heads, head_dim)``.
        weight: 1D RMSNorm weight of shape ``(head_dim,)``.
        num_heads: Local number of value heads per token.
        head_dim: Head dimension.
        eps: RMSNorm epsilon.

    Returns:
        ``(out_fp8, out_scale)`` where ``out_fp8`` has shape
        ``(T, num_heads * head_dim)`` in FP8 and ``out_scale`` has shape
        ``(T, 1)`` in FP32.
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert z.is_contiguous(), "z must be contiguous"
    assert x.shape == z.shape, f"x shape {x.shape} != z shape {z.shape}"
    assert x.ndim == 2 and x.shape[1] == head_dim
    assert x.shape[0] % num_heads == 0

    T = x.shape[0] // num_heads
    value_dim = num_heads * head_dim
    fp8_dtype, fp8_max, fp8_min = _get_fp8_quant_params()

    out_fp8 = torch.empty((T, value_dim), dtype=fp8_dtype, device=x.device)
    out_scale = torch.empty((T, 1), dtype=torch.float32, device=x.device)
    if T == 0:
        return (out_fp8, out_scale)

    head_count_block = triton.next_power_of_2(num_heads)
    head_dim_block = triton.next_power_of_2(head_dim)
    max_warps = 16 if _is_hip else 16

    if head_count_block * head_dim_block <= 8192:
        num_warps = max(
            min(
                triton.next_power_of_2(
                    triton.cdiv(head_count_block * head_dim_block, 256)
                ),
                max_warps,
            ),
            4,
        )
        _gdn_gated_rmsnorm_fp8_quant_2d_kernel[(T,)](
            x,
            z,
            weight,
            out_fp8,
            out_scale,
            stride_x=value_dim,
            stride_z=value_dim,
            stride_out=out_fp8.stride(0),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            eps=eps,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            HEAD_COUNT_BLOCK=head_count_block,
            HEAD_DIM_BLOCK=head_dim_block,
            num_warps=num_warps,
        )
    else:
        x_norm = torch.empty((T, value_dim), dtype=x.dtype, device=x.device)
        quant_block = min(triton.next_power_of_2(value_dim), 4096)
        num_warps = max(
            min(triton.next_power_of_2(triton.cdiv(quant_block, 256)), max_warps), 4
        )
        _gdn_gated_rmsnorm_fp8_quant_multipass_kernel[(T,)](
            x,
            z,
            weight,
            x_norm,
            out_fp8,
            out_scale,
            stride_x=value_dim,
            stride_z=value_dim,
            stride_x_norm=x_norm.stride(0),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            eps=eps,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            HEAD_BLOCK=head_dim_block,
            QUANT_BLOCK=quant_block,
            num_warps=num_warps,
        )
    return (out_fp8, out_scale)
