import errno
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.ppu.sail_jit_source_cache import (
    is_cache_directory_error,
    load_generated_jit,
    publish_source,
)
from sglang.srt.utils import is_sail


class TestSailJitSourceCache(unittest.TestCase):
    def test_content_addressed_atomic_publish_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "// test kernel\nint foo() { return 1; }\n"
            p1 = publish_source("kernel.cuh", src, cache_dir=tmp)
            p2 = publish_source("kernel.cuh", src, cache_dir=tmp)
            self.assertEqual(p1, p2)
            self.assertEqual(p1.read_text(encoding="utf-8"), src)

            p3 = publish_source("kernel.cuh", src + "// v2\n", cache_dir=tmp)
            self.assertNotEqual(p1, p3)

            with self.assertRaises(ValueError):
                publish_source("../escape.cuh", src, cache_dir=tmp)

    def test_readonly_cache_fallback_and_compiler_error_propagation(self):
        calls = []

        def fake_load_jit(*args, cuda_files, **kwargs):
            calls.append((Path(os.environ.get("SGLANG_JIT_CACHE_DIR", "")), cuda_files))
            if len(calls) == 1:
                raise OSError(errno.EROFS, "Read-only file system", cuda_files[0])
            return "compiled_ok"

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SGLANG_JIT_CACHE_DIR")
            os.environ["SGLANG_JIT_CACHE_DIR"] = tmp
            try:
                res = load_generated_jit(
                    fake_load_jit, "demo.cuh", "// demo\n", "demo_mod"
                )
            finally:
                if old is None:
                    os.environ.pop("SGLANG_JIT_CACHE_DIR", None)
                else:
                    os.environ["SGLANG_JIT_CACHE_DIR"] = old
        self.assertEqual(res, "compiled_ok")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0][0], calls[1][0])

        # Compiler error outside cache_dir must NOT be swallowed as cache error
        err = OSError(errno.ENOENT, "nvcc not found", "/usr/bin/nvcc")
        self.assertFalse(is_cache_directory_error(err, Path("/tmp/jit_cache")))


@unittest.skipUnless(
    torch.cuda.is_available() and is_sail(), "Requires Sail/PPU device"
)
class TestSailGlmDecodeMegaKernels(unittest.TestCase):
    def test_dual_rmsnorm_fp8_quant_matches_unfused(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8,
        )
        from sglang.srt.hardware_backend.ppu.sail_glm_decode_megakernel import (
            sail_dual_rmsnorm_fp8_quant,
        )
        from sglang.srt.layers.layernorm import RMSNorm

        torch.manual_seed(17)
        for q_dim, k_dim in ((1536, 512), (2048, 512)):
            norm_q = RMSNorm(q_dim, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
            norm_k = RMSNorm(k_dim, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
            norm_q.weight.data.copy_(
                torch.randn(q_dim, device="cuda", dtype=torch.bfloat16)
            )
            norm_k.weight.data.copy_(
                torch.randn(k_dim, device="cuda", dtype=torch.bfloat16)
            )
            for B in (1, 4, 8, 16, 32, 64):
                # Strided split views matching qkv_a_proj output [B, q_dim + k_dim + 64]
                latent = torch.randn(
                    (B, q_dim + k_dim + 64), device="cuda", dtype=torch.bfloat16
                )
                q_in, kv_in = latent.split([q_dim, k_dim + 64], dim=-1)
                k_in = kv_in[..., :k_dim]

                # Reference unfused 3-kernel sequence
                q_ref = norm_q(q_in)
                k_ref = norm_k(k_in)
                fq_ref, sq_ref = sglang_per_token_quant_fp8(q_ref)

                # Fused single-launch kernel
                q_fus, k_fus, fq_fus, sq_fus = sail_dual_rmsnorm_fp8_quant(
                    q_in, k_in, norm_q.weight, norm_k.weight, 1e-5, 1e-5
                )

                cos_q = F.cosine_similarity(
                    q_ref.float().flatten(), q_fus.float().flatten(), dim=0
                ).item()
                cos_k = F.cosine_similarity(
                    k_ref.float().flatten(), k_fus.float().flatten(), dim=0
                ).item()
                deq_ref = fq_ref.float() * sq_ref
                deq_fus = fq_fus.float() * sq_fus
                cos_fp8 = F.cosine_similarity(
                    deq_ref.flatten(), deq_fus.flatten(), dim=0
                ).item()
                self.assertGreater(cos_q, 0.9999, f"Q BF16 mismatch B={B} cos={cos_q}")
                self.assertGreater(cos_k, 0.9999, f"K BF16 mismatch B={B} cos={cos_k}")
                self.assertGreater(
                    cos_fp8, 0.9999, f"Q FP8 mismatch B={B} cos={cos_fp8}"
                )

    def test_router_quant_megakernel_matches_unfused(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8,
        )
        from sglang.srt.hardware_backend.ppu.sail_glm_decode_megakernel import (
            SailGlmRouterQuantRunner,
        )

        torch.manual_seed(23)
        E, H, topk = 288, 4096, 8
        scale_factor = 2.5
        w = torch.randn((E, H), device="cuda", dtype=torch.bfloat16) * 0.05
        bias = torch.randn((E,), device="cuda", dtype=torch.float32) * 0.05
        runner = SailGlmRouterQuantRunner(
            w, bias, max_rows=16, routed_scaling_factor=scale_factor
        )

        for B in (1, 2, 4, 8, 16):
            x = torch.randn((B, H), device="cuda", dtype=torch.bfloat16) * 0.5
            # Reference unfused router + sigmoid + bias + topk + renorm + FP8 quant
            logits_ref = F.linear(x.float(), w.float())
            active_ref = torch.sigmoid(logits_ref)
            ranked_ref = active_ref + bias
            _, ids_ref = torch.topk(ranked_ref, k=topk, dim=-1, largest=True, sorted=True)
            w_sel = torch.gather(active_ref, 1, ids_ref)
            w_ref = (w_sel / w_sel.sum(dim=-1, keepdim=True).clamp_min(1e-12)) * scale_factor
            fq_ref, sq_ref = sglang_per_token_quant_fp8(x)

            # Fused megakernel
            w_fus, ids_fus, fq_fus, sq_fus = runner.run(x)

            # Check Top-8 expert set & weight agreement
            self.assertTrue(
                torch.equal(ids_ref, ids_fus),
                f"B={B} topk ids mismatch:\nref={ids_ref}\nfus={ids_fus}",
            )
            max_w_err = (w_ref - w_fus).abs().max().item()
            self.assertLess(max_w_err, 1e-3, f"B={B} router weights err={max_w_err}")

            deq_ref = fq_ref.float() * sq_ref
            deq_fus = fq_fus.float() * sq_fus
            cos_fp8 = F.cosine_similarity(
                deq_ref.flatten(), deq_fus.flatten(), dim=0
            ).item()
            self.assertGreater(cos_fp8, 0.9999, f"B={B} shared FP8 cos={cos_fp8}")


if __name__ == "__main__":
    unittest.main()
