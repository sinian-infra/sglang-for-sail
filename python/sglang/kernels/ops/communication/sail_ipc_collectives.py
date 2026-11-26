"""JIT loaders and symmetric IPC workspace for Sail/PPU low-latency collectives."""

import ctypes
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import load_jit

SUPPORTED_WORLD_SIZES = (4, 8)
MAX_LOCAL_ROWS = 16
FLAG_AND_EPOCH_PREFIX_BYTES = 131072  # 64 KiB flags + 64 KiB epoch counters


@lru_cache(maxsize=None)
def load_sail_ipc_allgather(world_size: int, tiles: int = 32, threads: int = 128):
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size} (expected 4 or 8)")
    if tiles not in (16, 32, 64):
        raise ValueError(f"Unsupported tiles={tiles}")
    if threads not in (128, 256, 512):
        raise ValueError(f"Unsupported threads={threads}")
    return load_jit(
        "sail_ipc_allgather",
        str(world_size),
        str(tiles),
        str(threads),
        cuda_files=["distributed/sail_ipc_collectives.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::sail_ipc::SailIpcAllGather<{world_size}, {tiles}, {threads}>::run",
            )
        ],
    )


@lru_cache(maxsize=None)
def load_sail_ipc_reduce_scatter(world_size: int, tiles: int = 32, threads: int = 128):
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size} (expected 4 or 8)")
    if tiles not in (16, 32, 64):
        raise ValueError(f"Unsupported tiles={tiles}")
    if threads not in (128, 256, 512):
        raise ValueError(f"Unsupported threads={threads}")
    return load_jit(
        "sail_ipc_reduce_scatter",
        str(world_size),
        str(tiles),
        str(threads),
        cuda_files=["distributed/sail_ipc_collectives.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::sail_ipc::SailIpcReduceScatter<{world_size}, {tiles}, {threads}, 128>::run",
            )
        ],
    )


def select_launch_config(local_rows: int, hidden_size: int) -> Tuple[int, int]:
    """Select (tiles, threads) ensuring shard_u4 = (local_rows * hidden_size) // 8 is divisible by tiles."""
    shard_u4 = (local_rows * hidden_size) // 8
    if local_rows >= 8 and shard_u4 % 64 == 0:
        tiles = 64
    elif shard_u4 % 32 == 0:
        tiles = 32
    else:
        tiles = 16
    threads = 256 if local_rows >= 4 else 128
    return tiles, threads


class SailIpcWorkspace:
    """Symmetric double-buffered IPC workspace for one (group, device, stream)."""

    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        max_hidden_size: int = 8192,
        max_local_rows: int = MAX_LOCAL_ROWS,
        device: Optional[torch.device] = None,
    ):
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
        )

        self.group = cpu_group
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"SailIpcWorkspace requires world_size in {SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )
        self.max_hidden_size = max_hidden_size
        self.max_local_rows = max_local_rows
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        self.lib = CudaRTLibrary()

        # Double-buffered payload: 2 * world_size * max_local_rows * max_hidden_size * sizeof(bf16)
        payload_bytes = 2 * self.world_size * max_local_rows * max_hidden_size * 2
        self.workspace_bytes = FLAG_AND_EPOCH_PREFIX_BYTES + payload_bytes

        with torch.cuda.device(self.device):
            self.pointer = self.lib.cudaMalloc(self.workspace_bytes)
            self.lib.cudaMemset(self.pointer, 0, self.workspace_bytes)
            handle = self.lib.cudaIpcGetMemHandle(self.pointer)
            handles = [None] * self.world_size
            dist.all_gather_object(handles, handle, group=cpu_group)

            self.raw_ptrs = [
                (
                    self.pointer.value
                    if i == self.rank
                    else self.lib.cudaIpcOpenMemHandle(h).value
                )
                for i, h in enumerate(handles)
            ]
            self.peers = torch.tensor(self.raw_ptrs, dtype=torch.int64, device=self.device)
            self.error = torch.zeros(1, dtype=torch.int32, device=self.device)
        dist.barrier(group=cpu_group)
        self.closed = False

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        local_rows, hidden_size = input_.shape
        tiles, threads = select_launch_config(local_rows, hidden_size)
        mod = load_sail_ipc_allgather(self.world_size, tiles, threads)
        mod.run(input_, output, self.peers, self.rank, self.error)

    def reduce_scatter(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        total_rows, hidden_size = input_.shape
        local_rows = total_rows // self.world_size
        tiles, threads = select_launch_config(local_rows, hidden_size)
        mod = load_sail_ipc_reduce_scatter(self.world_size, tiles, threads)
        mod.run(input_, output, self.peers, self.rank, self.error)

    def check(self) -> None:
        torch.cuda.synchronize(self.device)
        err = int(self.error.item())
        if err != 0:
            raise RuntimeError(
                f"Sail IPC collective synchronization timeout on rank {self.rank}: code={err}"
            )

    def close(self) -> None:
        if self.closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            close_fn = self.lib.lib.cudaIpcCloseMemHandle
            close_fn.restype = ctypes.c_int
            close_fn.argtypes = [ctypes.c_void_p]
            for i, ptr in enumerate(self.raw_ptrs):
                if i != self.rank:
                    self.lib.CUDART_CHECK(close_fn(ctypes.c_void_p(ptr)))
            dist.barrier(group=self.group)
            self.lib.cudaFree(self.pointer)
        self.closed = True

    def __del__(self):
        # Never call collective barriers inside GC; explicit close() handles teardown.
        pass
