import unittest

import torch

from sglang.srt.utils import is_sail


@unittest.skipUnless(
    torch.cuda.is_available() and is_sail(), "Requires Sail/PPU device"
)
class TestDeepGemmFp4FusedEpilogue(unittest.TestCase):
    def _make_mxfp4_pair(self, shape, device="cuda"):
        """Generate valid packed MXFP4 tensor and raw uint8 E8M0 scales."""
        import deep_gemm

        x = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5
        flat = x.reshape(-1, shape[-1])
        q_2d, s_2d = deep_gemm.per_token_cast_to_fp4(
            flat, use_ue8m0=True, gran_k=32, use_packed_ue8m0=False
        )
        if s_2d.dtype != torch.uint8:
            s_2d = (s_2d.to(torch.float32).view(torch.int32) >> 23).to(torch.uint8)
        q = q_2d.view(torch.uint8).reshape(*shape[:-1], shape[-1] // 2).contiguous()
        s = s_2d.reshape(*shape[:-1], shape[-1] // 32).contiguous()
        return q, s

    def test_masked_fused_epilogue_matches_unfused_two_gemm_moe(self):
        import deep_gemm
        from sglang.kernels.ops.elementwise.silu_mul_quant import (
            silu_and_mul_masked_post_quant_mxfp4,
        )
        from sglang.srt.layers import deep_gemm_wrapper

        torch.manual_seed(0)
        E, M_max, K, N = 16, 64, 2048, 3072  # inter = 1536 (GLM-5.3 Flash)
        a_q, a_s_raw = self._make_mxfp4_pair((E, M_max, K))
        a_s = deep_gemm.preprocess_mxfp4_scales(a_s_raw)
        w13_q, w13_s_raw = self._make_mxfp4_pair((E, N, K))
        w2_q, w2_s_raw = self._make_mxfp4_pair((E, K, N // 2))
        w2_s = deep_gemm.preprocess_mxfp4_scales(w2_s_raw)
        masked_m = torch.randint(1, 33, (E,), dtype=torch.int32, device="cuda")
        expected_m = 16

        # Unfused reference path
        w13_s_unfused = deep_gemm.preprocess_mxfp4_scales(w13_s_raw)
        gateup = torch.empty((E, M_max, N), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_masked(
            (a_q, a_s), (w13_q, w13_s_unfused), None, gateup, masked_m, expected_m
        )
        mid_q_ref, mid_s_ref = silu_and_mul_masked_post_quant_mxfp4(
            gateup, masked_m, expected_m=expected_m
        )
        out_ref = torch.empty((E, M_max, K), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_masked(
            (mid_q_ref, mid_s_ref.permute(0, 2, 1)),
            (w2_q, w2_s),
            None,
            out_ref,
            masked_m,
            expected_m,
        )

        # Fused epilogue path
        w13_q_fused, w13_s_fused = (
            deep_gemm.preprocess_mxfp4_weight_for_act_and_quant_fusing(
                w13_q.contiguous(), w13_s_raw.contiguous()
            )
        )
        mid_q_fus = torch.empty((E, M_max, N // 4), dtype=torch.uint8, device="cuda")
        mid_s_fus = torch.empty((E, M_max, N // 128), dtype=torch.uint16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_masked(
            (a_q, a_s),
            (w13_q_fused, w13_s_fused),
            None,
            mid_q_fus,
            masked_m,
            expected_m,
            out_scale=mid_s_fus,
        )
        out_fus = torch.empty((E, M_max, K), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_masked(
            (mid_q_fus, mid_s_fus),
            (w2_q, w2_s),
            None,
            out_fus,
            masked_m,
            expected_m,
        )

        # Compare only active tokens [0:masked_m[e]] per expert
        ref_slices = [out_ref[e, : int(masked_m[e].item())] for e in range(E)]
        fus_slices = [out_fus[e, : int(masked_m[e].item())] for e in range(E)]
        ref_cat = torch.cat(ref_slices, dim=0).float()
        fus_cat = torch.cat(fus_slices, dim=0).float()
        cos = torch.nn.functional.cosine_similarity(
            ref_cat.flatten(), fus_cat.flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.998, f"Masked fused epilogue cosine too low: {cos}")

    def test_nopad_fused_epilogue_matches_unfused_two_gemm_moe(self):
        import deep_gemm
        from sglang.kernels.ops.elementwise.silu_mul_quant import (
            silu_and_mul_post_quant_mxfp4,
        )
        from sglang.srt.layers import deep_gemm_wrapper

        torch.manual_seed(1)
        E, K, N = 8, 2048, 3072
        counts = [32, 16, 48, 0, 24, 16, 40, 24]
        all_tokens = sum(counts)
        m_rows = torch.tensor(counts, dtype=torch.int32, device="cuda")
        m_indices = torch.cat(
            [torch.full((c,), e, dtype=torch.int32, device="cuda") for e, c in enumerate(counts) if c > 0]
        )

        a_q, a_s_raw = self._make_mxfp4_pair((all_tokens, K))
        a_s = deep_gemm.preprocess_mxfp4_scales(a_s_raw)
        w13_q, w13_s_raw = self._make_mxfp4_pair((E, N, K))
        w2_q, w2_s_raw = self._make_mxfp4_pair((E, K, N // 2))
        w2_s = deep_gemm.preprocess_mxfp4_scales(w2_s_raw)

        # Unfused reference
        w13_s_unfused = deep_gemm.preprocess_mxfp4_scales(w13_s_raw)
        gateup = torch.empty((all_tokens, N), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
            (a_q, a_s), (w13_q, w13_s_unfused), None, gateup, m_indices, m_rows
        )
        mid_q_ref, mid_s_ref = silu_and_mul_post_quant_mxfp4(gateup)
        out_ref = torch.empty((all_tokens, K), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
            (mid_q_ref, mid_s_ref), (w2_q, w2_s), None, out_ref, m_indices, m_rows
        )

        # Fused epilogue
        w13_q_fused, w13_s_fused = (
            deep_gemm.preprocess_mxfp4_weight_for_act_and_quant_fusing(
                w13_q.contiguous(), w13_s_raw.contiguous()
            )
        )
        mid_q_fus = torch.empty((all_tokens, N // 4), dtype=torch.uint8, device="cuda")
        mid_s_fus = torch.empty((all_tokens, N // 128), dtype=torch.uint16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
            (a_q, a_s),
            (w13_q_fused, w13_s_fused),
            None,
            mid_q_fus,
            m_indices,
            m_rows,
            out_scale=mid_s_fus,
        )
        out_fus = torch.empty((all_tokens, K), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
            (mid_q_fus, mid_s_fus), (w2_q, w2_s), None, out_fus, m_indices, m_rows
        )

        cos = torch.nn.functional.cosine_similarity(
            out_ref.float().flatten(), out_fus.float().flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.998, f"NoPad fused epilogue cosine too low: {cos}")


if __name__ == "__main__":
    unittest.main()
