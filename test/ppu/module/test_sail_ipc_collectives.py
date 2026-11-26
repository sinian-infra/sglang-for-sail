import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.utils import is_sail


def _worker_verify_and_bench(rank: int, world_size: int, port: int, do_bench: bool):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    from sglang.kernels.ops.communication.sail_ipc_collectives import SailIpcWorkspace
    from sglang.srt.hardware_backend.ppu.sail_ipc_collectives import (
        SailFusedGemmReduceScatter,
    )

    ws_ag = SailIpcWorkspace(cpu_group=cpu_group, max_hidden_size=6144)
    ws_rs = SailIpcWorkspace(cpu_group=cpu_group, max_hidden_size=6144)

    try:
        # 1. Verify byte-exact AllGather across GLM-5.3 Flash (4096) and Full (6144)
        for hidden_size in (4096, 6144):
            for local_rows in (1, 2, 4, 8, 16):
                torch.manual_seed(1000 + rank * 37 + local_rows)
                x = torch.randn(
                    (local_rows, hidden_size), dtype=torch.bfloat16, device="cuda"
                )
                ref = torch.empty(
                    (world_size * local_rows, hidden_size),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                out = torch.empty_like(ref)
                dist.all_gather_into_tensor(ref, x)
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)

                ws_ag.all_gather(out, x)
                ws_ag.check()
                assert torch.equal(
                    ref.view(torch.uint8), out.view(torch.uint8)
                ), f"AllGather mismatch rank={rank} H={hidden_size} rows={local_rows}"

        # 2. Verify ReduceScatter accuracy vs PCCL
        for hidden_size in (4096, 6144):
            for local_rows in (1, 2, 4, 8, 16):
                torch.manual_seed(2000 + rank * 41 + local_rows)
                x = (
                    torch.randn(
                        (world_size * local_rows, hidden_size),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    * 0.25
                )
                ref = torch.empty(
                    (local_rows, hidden_size), dtype=torch.bfloat16, device="cuda"
                )
                out = torch.empty_like(ref)
                dist.reduce_scatter_tensor(ref, x)
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)

                ws_rs.reduce_scatter(out, x)
                ws_rs.check()
                cos = torch.nn.functional.cosine_similarity(
                    ref.float().flatten(), out.float().flatten(), dim=0
                ).item()
                max_diff = (ref.float() - out.float()).abs().max().item()
                assert (
                    cos > 0.9999 and max_diff < 0.05
                ), f"ReduceScatter mismatch rank={rank} H={hidden_size} rows={local_rows} cos={cos} max={max_diff}"

        # 3. Verify BF16 GEMM + IPC ReduceScatter
        fused = SailFusedGemmReduceScatter(ws_rs)
        for hidden_size, k_dim in ((4096, 1024), (6144, 1024)):
            for local_rows in (1, 2, 4):
                m = world_size * local_rows
                torch.manual_seed(3000 + rank * 19 + local_rows)
                a = torch.randn((m, k_dim), dtype=torch.bfloat16, device="cuda") * 0.1
                w = (
                    torch.randn((hidden_size, k_dim), dtype=torch.bfloat16, device="cuda")
                    * 0.1
                )
                full_ref = torch.matmul(a, w.t())
                rs_ref = torch.empty(
                    (local_rows, hidden_size), dtype=torch.bfloat16, device="cuda"
                )
                dist.reduce_scatter_tensor(rs_ref, full_ref)
                torch.cuda.synchronize()

                rs_fus = fused.bf16_gemm_reduce_scatter(a, w)
                ws_rs.check()
                cos = torch.nn.functional.cosine_similarity(
                    rs_ref.float().flatten(), rs_fus.float().flatten(), dim=0
                ).item()
                assert cos > 0.9999, f"GEMM+RS mismatch rank={rank} cos={cos}"

        # 4. Optional paired A/B benchmark (PCCL vs Sail IPC)
        if do_bench:
            if rank == 0:
                print(
                    f"\n=== PR3 PPU TP{world_size} A/B Benchmark (PCCL vs Sail Lock-Free IPC) ===",
                    flush=True,
                )
                print(
                    f"{'Op & Shape':<30} {'PCCL(us)':>10} {'IPC(us)':>10} {'Speedup':>9} {'90x_save(ms)':>13}",
                    flush=True,
                )
            for hidden_size in (4096, 6144):
                for local_rows in (1, 2, 4, 8):
                    # AllGather bench
                    x_ag = torch.randn(
                        (local_rows, hidden_size), dtype=torch.bfloat16, device="cuda"
                    )
                    o_ag = torch.empty(
                        (world_size * local_rows, hidden_size),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    for _ in range(10):
                        dist.all_gather_into_tensor(o_ag, x_ag)
                        ws_ag.all_gather(o_ag, x_ag)
                    torch.cuda.synchronize()
                    dist.barrier(group=cpu_group)
                    iters = 200
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        dist.all_gather_into_tensor(o_ag, x_ag)
                    torch.cuda.synchronize()
                    t_pccl = (time.perf_counter() - t0) * 1e6 / iters

                    dist.barrier(group=cpu_group)
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        ws_ag.all_gather(o_ag, x_ag)
                    torch.cuda.synchronize()
                    t_ipc = (time.perf_counter() - t0) * 1e6 / iters
                    if rank == 0:
                        save = (t_pccl - t_ipc) * 90 / 1000.0
                        print(
                            f"AllGather    H={hidden_size} B={world_size*local_rows:<3}   {t_pccl:10.2f} {t_ipc:10.2f} {t_pccl/t_ipc:8.2f}x {save:12.3f}ms",
                            flush=True,
                        )

                    # ReduceScatter bench
                    x_rs = torch.randn(
                        (world_size * local_rows, hidden_size),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    o_rs = torch.empty(
                        (local_rows, hidden_size), dtype=torch.bfloat16, device="cuda"
                    )
                    for _ in range(10):
                        dist.reduce_scatter_tensor(o_rs, x_rs)
                        ws_rs.reduce_scatter(o_rs, x_rs)
                    torch.cuda.synchronize()
                    dist.barrier(group=cpu_group)
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        dist.reduce_scatter_tensor(o_rs, x_rs)
                    torch.cuda.synchronize()
                    t_pccl = (time.perf_counter() - t0) * 1e6 / iters

                    dist.barrier(group=cpu_group)
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        ws_rs.reduce_scatter(o_rs, x_rs)
                    torch.cuda.synchronize()
                    t_ipc = (time.perf_counter() - t0) * 1e6 / iters
                    if rank == 0:
                        save = (t_pccl - t_ipc) * 90 / 1000.0
                        print(
                            f"ReduceScatter H={hidden_size} B={world_size*local_rows:<3}  {t_pccl:10.2f} {t_ipc:10.2f} {t_pccl/t_ipc:8.2f}x {save:12.3f}ms",
                            flush=True,
                        )
    finally:
        ws_ag.close()
        ws_rs.close()
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.is_available() and is_sail() and torch.cuda.device_count() >= 4,
    "Requires >= 4 Sail/PPU devices",
)
class TestSailIpcCollectives(unittest.TestCase):
    def test_tp4_ipc_allgather_reducescatter_and_gemm_rs(self):
        do_bench = os.environ.get("RUN_PR3_BENCH", "0") == "1"
        mp.spawn(
            _worker_verify_and_bench,
            args=(4, 29543, do_bench),
            nprocs=4,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
