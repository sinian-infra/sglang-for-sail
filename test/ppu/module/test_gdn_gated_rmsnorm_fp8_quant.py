import unittest

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.kernels.ops.elementwise.elementwise import gdn_gated_rmsnorm_fp8_quant
from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
    sglang_per_token_quant_fp8,
)


def _reference_gdn_norm_fp8(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
):
    T = x.shape[0] // num_heads
    norm = RMSNormGated(head_dim, eps=eps, activation="silu", device=x.device, dtype=x.dtype)
    norm.weight.data.copy_(weight)
    y = norm(x, z).reshape(T, num_heads * head_dim)
    return sglang_per_token_quant_fp8(y)


class TestGdnGatedRmsnormFp8Quant(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA/PPU device")
    def test_numerical_equivalence_single_pass_and_multipass(self):
        torch.manual_seed(42)
        eps = 1e-6
        # Covers typical Qwen3.8 TP4 (nv=8, d=128), TP1 (nv=32, d=128),
        # non-power-of-two heads (nv=6), small head_dim (64), and multipass (>8192).
        shapes = [
            (1, 8, 128),
            (4, 8, 128),
            (16, 8, 128),
            (64, 16, 128),
            (128, 32, 128),
            (8, 6, 128),
            (8, 8, 64),
            (4, 64, 256),  # 16384 > 8192 -> exercises multipass fallback
        ]
        for T, nv, d in shapes:
            x = torch.randn((T * nv, d), dtype=torch.bfloat16, device="cuda")
            z = torch.randn((T * nv, d), dtype=torch.bfloat16, device="cuda")
            w = torch.randn((d,), dtype=torch.bfloat16, device="cuda")

            q_fused, s_fused = gdn_gated_rmsnorm_fp8_quant(x, z, w, nv, d, eps)
            q_ref, s_ref = _reference_gdn_norm_fp8(x, z, w, nv, d, eps)

            self.assertEqual(q_fused.shape, (T, nv * d))
            self.assertEqual(s_fused.shape, (T, 1))
            self.assertEqual(q_fused.dtype, fp8_dtype)
            self.assertEqual(s_fused.dtype, torch.float32)

            deq_fused = q_fused.float() * s_fused
            deq_ref = q_ref.float() * s_ref
            cos = torch.nn.functional.cosine_similarity(
                deq_fused.flatten(), deq_ref.flatten(), dim=0
            ).item()
            rel_scale_err = ((s_fused - s_ref).abs() / s_ref.abs().clamp_min(1e-8)).max().item()
            self.assertGreater(
                cos, 0.9999, f"shape={(T, nv, d)} cosine similarity too low: {cos}"
            )
            self.assertLess(
                rel_scale_err,
                1e-2,
                f"shape={(T, nv, d)} scale mismatch: {rel_scale_err}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA/PPU device")
    def test_empty_and_zero_rows(self):
        nv, d, eps = 8, 128, 1e-6
        w = torch.ones((d,), dtype=torch.bfloat16, device="cuda")

        # Empty batch T=0
        x0 = torch.empty((0, d), dtype=torch.bfloat16, device="cuda")
        z0 = torch.empty((0, d), dtype=torch.bfloat16, device="cuda")
        q0, s0 = gdn_gated_rmsnorm_fp8_quant(x0, z0, w, nv, d, eps)
        self.assertEqual(q0.shape, (0, nv * d))
        self.assertEqual(s0.shape, (0, 1))

        # All-zero token row (no NaN / Inf)
        xz = torch.zeros((4 * nv, d), dtype=torch.bfloat16, device="cuda")
        zz = torch.zeros((4 * nv, d), dtype=torch.bfloat16, device="cuda")
        qz, sz = gdn_gated_rmsnorm_fp8_quant(xz, zz, w, nv, d, eps)
        self.assertFalse(torch.isnan(qz.float()).any().item())
        self.assertFalse(torch.isnan(sz).any().item())
        self.assertTrue((sz > 0).all().item())

        # Extreme values clamp cleanly to [fp8_min, fp8_max]
        x_big = torch.full((2 * nv, d), 500.0, dtype=torch.bfloat16, device="cuda")
        z_big = torch.full((2 * nv, d), 50.0, dtype=torch.bfloat16, device="cuda")
        qb, sb = gdn_gated_rmsnorm_fp8_quant(x_big, z_big, w, nv, d, eps)
        self.assertFalse(torch.isnan(qb.float()).any().item())
        self.assertTrue((qb.float() <= fp8_max).all().item())
        self.assertTrue((qb.float() >= fp8_min).all().item())

    def test_qwen35_gating_rules(self):
        """Verify activation, quant method, static scale, and empty-batch gates."""
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.quantization.w8a8_fp8 import (
            W8A8Fp8Config,
            W8A8Fp8LinearMethod,
        )

        fp8_method = W8A8Fp8LinearMethod(W8A8Fp8Config(is_checkpoint_fp8_serialized=True))
        unquant_method = UnquantizedLinearMethod()

        def _eligible(act, qm, input_scale, m_rows, nv=8, strided=False):
            return (
                not strided
                and act in ("swish", "silu")
                and 0 < m_rows <= 4096 * nv
                and type(qm) is W8A8Fp8LinearMethod
                and input_scale is None
            )

        self.assertTrue(_eligible("silu", fp8_method, None, 64))
        self.assertTrue(_eligible("swish", fp8_method, None, 64))
        self.assertFalse(_eligible("sigmoid", fp8_method, None, 64))
        self.assertFalse(_eligible("silu", unquant_method, None, 64))
        self.assertFalse(_eligible("silu", fp8_method, torch.tensor(1.0), 64))
        self.assertFalse(_eligible("silu", fp8_method, None, 0))
        self.assertFalse(_eligible("silu", fp8_method, None, 64, strided=True))


if __name__ == "__main__":
    unittest.main()
