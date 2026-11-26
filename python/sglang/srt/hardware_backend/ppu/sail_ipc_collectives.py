"""Sail/PPU IPC Collective Manager & Fused GEMM + IPC ReduceScatter operator."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.sail_ipc_collectives import (
    MAX_LOCAL_ROWS,
    SUPPORTED_WORLD_SIZES,
    SailIpcWorkspace,
    load_sail_ipc_allgather,
    load_sail_ipc_reduce_scatter,
    select_launch_config,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_sail

logger = logging.getLogger(__name__)
_is_sail = is_sail()


def _is_eligible_tensor(
    t: torch.Tensor,
    max_hidden_size: int = 8192,
) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.ndim == 2
        and t.dtype == torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and 0 < t.shape[1] <= max_hidden_size
        and t.shape[1] % 128 == 0
    )


class SailIpcCollectiveManager:
    """Manages per-(kind, group, device, stream) Sail IPC workspaces.

    Safe with CUDA graph capture: never allocates or exchanges IPC handles while
    a stream is capturing. Pre-warmed workspaces replay directly inside graphs;
    uninitialized streams fall back transparently to PCCL.
    """

    def __init__(self, max_hidden_size: int = 8192):
        self.max_hidden_size = max_hidden_size
        self._workspaces: Dict[Tuple[str, int, int, int], SailIpcWorkspace] = {}
        self._disabled_groups: set[int] = set()

    def _get_workspace(
        self, kind: str, group, device: torch.device
    ) -> Optional[SailIpcWorkspace]:
        group_id = id(group)
        if group_id in self._disabled_groups:
            return None
        cpu_group = getattr(group, "cpu_group", None)
        if cpu_group is None:
            return None
        world_size = getattr(group, "world_size", 0)
        if world_size not in SUPPORTED_WORLD_SIZES:
            return None
        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        key = (kind, group_id, device.index, stream_id)
        ws = self._workspaces.get(key)
        if ws is not None and not ws.closed:
            return ws
        # Never open IPC handles during CUDA graph capture.
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            # Preload default tile kernels before allocating IPC memory.
            if kind == "all_gather":
                load_sail_ipc_allgather(world_size, 32, 128)
            else:
                load_sail_ipc_reduce_scatter(world_size, 32, 128)
            ws = SailIpcWorkspace(
                cpu_group=cpu_group,
                max_hidden_size=self.max_hidden_size,
                device=device,
            )
            self._workspaces[key] = ws
            return ws
        except Exception as exc:
            logger.warning(
                "Sail IPC collective (%s) unavailable for group world=%d, falling back to PCCL: %s",
                kind,
                world_size,
                exc,
            )
            self._disabled_groups.add(group_id)
            return None

    def try_all_gather(
        self, group, output: torch.Tensor, input_: torch.Tensor
    ) -> bool:
        if not (_is_sail and envs.SGLANG_OPT_SAIL_IPC_COLLECTIVES.get()):
            return False
        world_size = getattr(group, "world_size", 0)
        if world_size not in SUPPORTED_WORLD_SIZES:
            return False
        if not (
            _is_eligible_tensor(input_, self.max_hidden_size)
            and _is_eligible_tensor(output, self.max_hidden_size)
            and input_.device == output.device
            and 1 <= input_.shape[0] <= MAX_LOCAL_ROWS
            and output.shape == (world_size * input_.shape[0], input_.shape[1])
        ):
            return False
        ws = self._get_workspace("all_gather", group, input_.device)
        if ws is None:
            return False
        tiles, threads = select_launch_config(input_.shape[0], input_.shape[1])
        if torch.cuda.is_current_stream_capturing():
            # Ensure the exact (world, tiles, threads) module is already compiled.
            load_sail_ipc_allgather(world_size, tiles, threads)
        ws.all_gather(output, input_)
        return True

    def try_reduce_scatter(
        self, group, output: torch.Tensor, input_: torch.Tensor
    ) -> bool:
        if not (_is_sail and envs.SGLANG_OPT_SAIL_IPC_COLLECTIVES.get()):
            return False
        world_size = getattr(group, "world_size", 0)
        if world_size not in SUPPORTED_WORLD_SIZES:
            return False
        if not (
            _is_eligible_tensor(input_, self.max_hidden_size)
            and _is_eligible_tensor(output, self.max_hidden_size)
            and input_.device == output.device
            and input_.shape[0] % world_size == 0
            and 1 <= input_.shape[0] // world_size <= MAX_LOCAL_ROWS
            and output.shape == (input_.shape[0] // world_size, input_.shape[1])
        ):
            return False
        ws = self._get_workspace("reduce_scatter", group, input_.device)
        if ws is None:
            return False
        local_rows = input_.shape[0] // world_size
        tiles, threads = select_launch_config(local_rows, input_.shape[1])
        if torch.cuda.is_current_stream_capturing():
            load_sail_ipc_reduce_scatter(world_size, tiles, threads)
        ws.reduce_scatter(output, input_)
        return True

    def close_all(self) -> None:
        for ws in list(self._workspaces.values()):
            ws.close()
        self._workspaces.clear()


class SailFusedGemmReduceScatter:
    """Graph-safe BF16/FP8 GEMM + Sail IPC ReduceScatter operator.

    Pairs local RowParallel linear projection directly with lock-free Sail IPC
    ReduceScatter on the same stream using a pre-allocated full staging buffer
    (zero host sync, zero PCCL launch overhead for small/medium decode batches).
    """

    def __init__(self, workspace: SailIpcWorkspace):
        self.workspace = workspace
        self._full_bufs: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _get_full_buffer(self, m: int, n: int, device: torch.device) -> torch.Tensor:
        key = (m, n, device.index)
        buf = self._full_bufs.get(key)
        if buf is None:
            buf = torch.empty((m, n), dtype=torch.bfloat16, device=device)
            self._full_bufs[key] = buf
        return buf

    def bf16_gemm_reduce_scatter(
        self,
        a: torch.Tensor,
        b_nk: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ReduceScatter(a @ b_nk.T) where a=[M, K], b_nk=[N, K], out=[M/W, N]."""
        m = a.shape[0]
        n = b_nk.shape[0]
        world = self.workspace.world_size
        assert m % world == 0 and 1 <= m // world <= MAX_LOCAL_ROWS
        full = self._get_full_buffer(m, n, a.device)
        torch.matmul(a, b_nk.t(), out=full)
        if out is None:
            out = torch.empty((m // world, n), dtype=torch.bfloat16, device=a.device)
        self.workspace.reduce_scatter(out, full)
        return out

    def fp8_gemm_reduce_scatter(
        self,
        a_fp8: torch.Tensor,
        a_scale: torch.Tensor,
        b_fp8_kn: torch.Tensor,
        b_scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ReduceScatter(FP8_GEMM(a, b)) where a=[M, K], b=[K, N], out=[M/W, N]."""
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        m = a_fp8.shape[0]
        n = b_fp8_kn.shape[1]
        world = self.workspace.world_size
        assert m % world == 0 and 1 <= m // world <= MAX_LOCAL_ROWS
        full = apply_fp8_linear(
            a_fp8,
            b_fp8_kn,
            b_scale,
            input_scale=a_scale,
            pre_quant_output_dtype=torch.bfloat16,
        )
        if out is None:
            out = torch.empty((m // world, n), dtype=torch.bfloat16, device=a_fp8.device)
        self.workspace.reduce_scatter(out, full.contiguous())
        return out


_GLOBAL_MANAGER = SailIpcCollectiveManager()


def get_sail_ipc_collective_manager() -> SailIpcCollectiveManager:
    return _GLOBAL_MANAGER
