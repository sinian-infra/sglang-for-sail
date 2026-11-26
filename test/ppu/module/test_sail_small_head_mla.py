import unittest

import torch

from sglang.srt.utils import is_sail


@unittest.skipUnless(
    torch.cuda.is_available() and is_sail(), "Requires Sail/PPU device"
)
class TestSailSmallHeadSparseMla(unittest.TestCase):
    def test_small_head_sparse_mla_matches_padded_h64_and_value_bmm(self):
        from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
            flash_mla_sparse_fwd,
        )

        torch.manual_seed(7)
        s_kv, topk, d_qk, d_v = 4096, 2048, 576, 512
        kv = torch.randn((s_kv, 1, d_qk), dtype=torch.bfloat16, device="cuda") * 0.25
        scale = d_qk**-0.5

        for num_heads in (8, 16, 32):
            for batch in (1, 4, 8, 16, 32):
                q = (
                    torch.randn(
                        (batch, num_heads, d_qk), dtype=torch.bfloat16, device="cuda"
                    )
                    * 0.25
                )
                indices = torch.randint(
                    0, s_kv, (batch, 1, topk), dtype=torch.int32, device="cuda"
                )
                lengths = torch.randint(
                    64, topk + 1, (batch,), dtype=torch.int32, device="cuda"
                )
                w_vc = (
                    torch.randn(
                        (num_heads, d_v, 128), dtype=torch.bfloat16, device="cuda"
                    )
                    * 0.1
                )

                # Reference padded-64 path
                q_pad = q.new_zeros((batch, 64, d_qk))
                q_pad[:, :num_heads, :] = q
                o_pad, _, _ = flash_mla_sparse_fwd(
                    q=q_pad,
                    kv=kv,
                    indices=indices,
                    sm_scale=scale,
                    d_v=d_v,
                    topk_length=lengths,
                )
                o_ref = o_pad[:, :num_heads, :]

                # Native unpadded small-head path
                o_small, _, _ = flash_mla_sparse_fwd(
                    q=q,
                    kv=kv,
                    indices=indices,
                    sm_scale=scale,
                    d_v=d_v,
                    topk_length=lengths,
                )

                self.assertEqual(o_small.shape, (batch, num_heads, d_v))
                self.assertTrue(o_small.is_contiguous())
                max_diff = (o_ref.float() - o_small.float()).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(
                    o_ref.float().flatten(), o_small.float().flatten(), dim=0
                ).item()
                self.assertGreater(
                    cos,
                    0.9999,
                    f"H={num_heads} B={batch} attention cosine too low: {cos}",
                )
                self.assertLess(
                    max_diff,
                    0.02,
                    f"H={num_heads} B={batch} attention max_diff too high: {max_diff}",
                )

                # Downstream absorbed value BMM contract check: [H, B, 512] @ [H, 512, 128]
                bmm_ref = torch.bmm(o_ref.transpose(0, 1), w_vc)
                bmm_small = torch.bmm(o_small.transpose(0, 1), w_vc)
                bmm_cos = torch.nn.functional.cosine_similarity(
                    bmm_ref.float().flatten(), bmm_small.float().flatten(), dim=0
                ).item()
                self.assertGreater(
                    bmm_cos,
                    0.9999,
                    f"H={num_heads} B={batch} value BMM cosine too low: {bmm_cos}",
                )

    def test_small_head_with_attn_sink(self):
        from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
            flash_mla_sparse_fwd,
        )

        torch.manual_seed(11)
        batch, num_heads, s_kv, topk, d_qk, d_v = 8, 8, 2048, 1024, 576, 512
        q = torch.randn((batch, num_heads, d_qk), dtype=torch.bfloat16, device="cuda") * 0.25
        kv = torch.randn((s_kv, 1, d_qk), dtype=torch.bfloat16, device="cuda") * 0.25
        indices = torch.randint(0, s_kv, (batch, 1, topk), dtype=torch.int32, device="cuda")
        sink = torch.randn((num_heads,), dtype=torch.float32, device="cuda")
        scale = d_qk**-0.5

        q_pad = q.new_zeros((batch, 64, d_qk))
        q_pad[:, :num_heads, :] = q
        sink_pad = sink.new_zeros((64,))
        sink_pad[:num_heads] = sink

        o_ref = flash_mla_sparse_fwd(
            q=q_pad, kv=kv, indices=indices, sm_scale=scale, d_v=d_v, attn_sink=sink_pad
        )[0][:, :num_heads, :]
        o_small = flash_mla_sparse_fwd(
            q=q, kv=kv, indices=indices, sm_scale=scale, d_v=d_v, attn_sink=sink
        )[0]
        cos = torch.nn.functional.cosine_similarity(
            o_ref.float().flatten(), o_small.float().flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.9999, f"Sink H=8 cosine too low: {cos}")


if __name__ == "__main__":
    unittest.main()
