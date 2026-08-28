"""Experimental graph-captured Indexer-as-a-Service data plane.

This module intentionally keeps control-plane coordination out of captured
graphs. Model workers retain their TP process group and use a separate
StatelessProcessGroup + PyNcclCommunicator with one service rank.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

import deep_gemm
from sgl_kernel import fast_topk_v2
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _store_fp4_index_k_cache_kernel,
    quantize_fp4_indexer_tensor,
)
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.utils import StatelessProcessGroup


IAAS_WORLD_SIZE = 5
IAAS_CLIENTS = 4
IAAS_PAGE_SIZE = 64
IAAS_HEADS = 64
IAAS_HEAD_DIM = 128
IAAS_TOPK = 2048
IAAS_LAYERS = 61


_client: Optional["IAASClient"] = None


class IAASClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        rank: int,
        device: torch.device,
    ) -> None:
        if rank <= 0 or rank >= IAAS_WORLD_SIZE:
            raise ValueError(f"IAAS client rank must be 1..4, got {rank}")
        self.rank = rank
        self.device = device
        self.group = StatelessProcessGroup.create(
            host, port, rank, IAAS_WORLD_SIZE
        )
        self.group.store.set_timeout(timedelta(hours=1))
        self.comm = PyNcclCommunicator(self.group, device)
        if not self.comm.available:
            raise RuntimeError("IAAS PyNcclCommunicator is unavailable")
        self.responses_by_batch: Dict[int, torch.Tensor] = {}

    def control_barrier(self) -> None:
        self.group.barrier()

    def query(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        batch = query.shape[0]
        if query.shape != (batch, IAAS_HEADS, IAAS_HEAD_DIM):
            raise ValueError(f"unexpected IAAS query shape {query.shape}")
        if key.shape != (batch, IAAS_HEAD_DIM):
            raise ValueError(f"unexpected IAAS key shape {key.shape}")
        if weights.shape != (batch, IAAS_HEADS):
            raise ValueError(f"unexpected IAAS weights shape {weights.shape}")

        q_data, q_scale = quantize_fp4_indexer_tensor(query.contiguous())
        k_data, k_scale = quantize_fp4_indexer_tensor(key.contiguous())
        q_data = q_data.view(batch, 1, IAAS_HEADS, IAAS_HEAD_DIM // 2)
        q_scale = q_scale.view(batch, 1, IAAS_HEADS)
        k_data = k_data.view(batch, IAAS_HEAD_DIM // 2)
        k_scale = k_scale.view(batch)
        weights = weights.to(torch.float32).contiguous()
        response = torch.empty(
            (batch, IAAS_TOPK), dtype=torch.int32, device=query.device
        )
        self.responses_by_batch[batch] = response

        with self.comm.change_state(enable=True):
            self.comm.send(q_data, dst=0)
            self.comm.send(q_scale, dst=0)
            self.comm.send(weights, dst=0)
            self.comm.send(k_data, dst=0)
            self.comm.send(k_scale, dst=0)
            self.comm.recv(response, src=0)
        return response

    def response_is_valid(self, batch: int, context_len: int) -> bool:
        response = self.responses_by_batch.get(batch)
        if response is None:
            return False
        return bool(
            torch.all((response >= 0) & (response < context_len)).item()
        )

    def shutdown(self) -> None:
        self.group.send_obj("done", dst=0)
        if self.group.recv_obj(src=0) != "ack":
            raise RuntimeError("IAAS service returned invalid shutdown ack")
        self.group.send_obj("exit_ready", dst=0)


def init_iaas_client(
    *, host: str, port: int, rank: int, device: torch.device
) -> IAASClient:
    global _client
    if _client is not None:
        raise RuntimeError("IAAS client is already initialized")
    _client = IAASClient(host=host, port=port, rank=rank, device=device)
    return _client


def get_iaas_client() -> IAASClient:
    if _client is None:
        raise RuntimeError("IAAS client has not been initialized")
    return _client


def get_iaas_client_if_initialized() -> Optional[IAASClient]:
    return _client


@dataclass
class _ServiceBuffers:
    batch_per_client: int
    q_data: torch.Tensor
    q_scale: torch.Tensor
    weights: torch.Tensor
    k_data: torch.Tensor
    k_scale: torch.Tensor
    responses: torch.Tensor
    seqlens_2d: torch.Tensor
    lengths: torch.Tensor
    block_table: torch.Tensor
    loc: torch.Tensor
    schedule: object


class IAASService:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        device: torch.device,
        max_batch_per_client: int,
        context_len: int,
        capture_batches: Sequence[int],
    ) -> None:
        self.device = device
        self.max_batch_per_client = max_batch_per_client
        self.context_len = context_len
        self.capture_batches = tuple(sorted(set(capture_batches)))
        if not self.capture_batches or self.capture_batches[-1] > max_batch_per_client:
            raise ValueError("invalid IAAS capture batches")
        self.group = StatelessProcessGroup.create(
            host, port, 0, IAAS_WORLD_SIZE
        )
        self.group.store.set_timeout(timedelta(hours=1))
        self.comm = PyNcclCommunicator(self.group, device)
        if not self.comm.available:
            raise RuntimeError("IAAS service PyNcclCommunicator is unavailable")

        pages_per_req = (context_len + IAAS_PAGE_SIZE - 1) // IAAS_PAGE_SIZE
        self.pages_per_req = pages_per_req
        self.total_rows = IAAS_CLIENTS * max_batch_per_client
        self.total_pages = self.total_rows * pages_per_req
        self.caches = [
            torch.zeros(
                (self.total_pages, IAAS_PAGE_SIZE * 68),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(IAAS_LAYERS)
        ]
        for cache in self.caches:
            cache[:, IAAS_PAGE_SIZE * 64 :].fill_(127)
        self.full_block_table = torch.arange(
            self.total_pages, dtype=torch.int32, device=device
        ).view(self.total_rows, pages_per_req)
        self.buffers: Dict[int, _ServiceBuffers] = {
            batch: self._make_buffers(batch) for batch in self.capture_batches
        }
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}

    def _make_buffers(self, batch: int) -> _ServiceBuffers:
        rows = IAAS_CLIENTS * batch
        slots = torch.cat(
            [
                torch.arange(
                    client * self.max_batch_per_client,
                    client * self.max_batch_per_client + batch,
                    dtype=torch.long,
                    device=self.device,
                )
                for client in range(IAAS_CLIENTS)
            ]
        )
        block_table = self.full_block_table.index_select(0, slots).contiguous()
        seqlens_2d = torch.full(
            (rows, 1), self.context_len, dtype=torch.int32, device=self.device
        )
        lengths = seqlens_2d[:, 0]
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_2d, IAAS_PAGE_SIZE, deep_gemm.get_num_sms()
        )
        loc = block_table[:, -1] * IAAS_PAGE_SIZE + (
            self.context_len - 1
        ) % IAAS_PAGE_SIZE
        return _ServiceBuffers(
            batch_per_client=batch,
            q_data=torch.empty(
                (rows, 1, IAAS_HEADS, IAAS_HEAD_DIM // 2),
                dtype=torch.int8,
                device=self.device,
            ),
            q_scale=torch.empty(
                (rows, 1, IAAS_HEADS), dtype=torch.int32, device=self.device
            ),
            weights=torch.empty(
                (rows, IAAS_HEADS), dtype=torch.float32, device=self.device
            ),
            k_data=torch.empty(
                (rows, IAAS_HEAD_DIM // 2), dtype=torch.int8, device=self.device
            ),
            k_scale=torch.empty((rows,), dtype=torch.int32, device=self.device),
            responses=torch.empty(
                (rows, IAAS_TOPK), dtype=torch.int32, device=self.device
            ),
            seqlens_2d=seqlens_2d,
            lengths=lengths,
            block_table=block_table,
            loc=loc.contiguous(),
            schedule=schedule,
        )

    def _recv_from_clients(self, tensor: torch.Tensor, batch: int) -> None:
        self.comm.group_start()
        for peer in range(1, IAAS_WORLD_SIZE):
            self.comm.recv(tensor[(peer - 1) * batch : peer * batch], src=peer)
        self.comm.group_end()

    def _send_to_clients(self, tensor: torch.Tensor, batch: int) -> None:
        self.comm.group_start()
        for peer in range(1, IAAS_WORLD_SIZE):
            self.comm.send(tensor[(peer - 1) * batch : peer * batch], dst=peer)
        self.comm.group_end()

    def _once(self, buffers: _ServiceBuffers) -> None:
        batch = buffers.batch_per_client
        rows = IAAS_CLIENTS * batch
        with self.comm.change_state(enable=True):
            for layer in range(IAAS_LAYERS):
                self._recv_from_clients(buffers.q_data, batch)
                self._recv_from_clients(buffers.q_scale, batch)
                self._recv_from_clients(buffers.weights, batch)
                self._recv_from_clients(buffers.k_data, batch)
                self._recv_from_clients(buffers.k_scale, batch)
                cache = self.caches[layer]
                _store_fp4_index_k_cache_kernel[(rows,)](
                    buffers.k_data.view(torch.uint8),
                    buffers.k_scale,
                    cache,
                    buffers.loc,
                    IAAS_PAGE_SIZE,
                    cache.stride(0),
                    BLOCK=64,
                )
                logits = deep_gemm.fp8_fp4_paged_mqa_logits(
                    (buffers.q_data, buffers.q_scale),
                    cache.view(self.total_pages, IAAS_PAGE_SIZE, 1, 68),
                    buffers.weights,
                    buffers.seqlens_2d,
                    buffers.block_table,
                    buffers.schedule,
                    self.context_len,
                    False,
                )
                buffers.responses.copy_(
                    fast_topk_v2(logits, buffers.lengths, IAAS_TOPK)
                )
                self._send_to_clients(buffers.responses, batch)

    def capture(self) -> None:
        for batch in reversed(self.capture_batches):
            buffers = self.buffers[batch]
            for _ in range(2):
                self.group.barrier()
                self._once(buffers)
                torch.cuda.synchronize(self.device)
            self.group.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._once(buffers)
            torch.cuda.synchronize(self.device)
            self.graphs[batch] = graph

    def _set_sequence_length(self, buffers: _ServiceBuffers, seq_len: int) -> None:
        page = (seq_len - 1) // IAAS_PAGE_SIZE
        offset = (seq_len - 1) % IAAS_PAGE_SIZE
        buffers.seqlens_2d.fill_(seq_len)
        buffers.loc.copy_(
            buffers.block_table[:, page] * IAAS_PAGE_SIZE + offset
        )

    def serve_decode_points(
        self,
        *,
        active_batches: Sequence[int],
        context_len: int,
        warmup_steps: int,
        decode_steps: int,
    ) -> Dict[int, float]:
        total_steps = warmup_steps + decode_steps
        start_len = context_len - total_steps
        elapsed: Dict[int, float] = {}
        for batch in active_batches:
            buffers = self.buffers[batch]
            graph = self.graphs[batch]
            self.group.barrier()
            steady_begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for step in range(total_steps):
                if step == warmup_steps:
                    steady_begin.record()
                self._set_sequence_length(buffers, start_len + step + 1)
                graph.replay()
            end.record()
            end.synchronize()
            elapsed[batch] = (
                steady_begin.elapsed_time(end) / decode_steps
            )
            self.group.barrier()
        return elapsed

    def result(self, elapsed_ms: Dict[int, float]) -> dict:
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "status": "ok",
            "capture_batches": list(self.capture_batches),
            "max_batch_per_client": self.max_batch_per_client,
            "context_len": self.context_len,
            "layers": IAAS_LAYERS,
            "fp4_cache_gib": (
                IAAS_LAYERS * self.total_rows * self.context_len * 68 / 2**30
            ),
            "torch_allocated_gib": torch.cuda.memory_allocated(self.device) / 2**30,
            "torch_reserved_gib": torch.cuda.memory_reserved(self.device) / 2**30,
            "device_used_gib": (total_bytes - free_bytes) / 2**30,
            "device_total_gib": total_bytes / 2**30,
            "service_step_elapsed_ms": {
                str(batch): value for batch, value in elapsed_ms.items()
            },
        }

    def shutdown(self) -> None:
        for peer in range(1, IAAS_WORLD_SIZE):
            if self.group.recv_obj(src=peer) != "done":
                raise RuntimeError(f"IAAS client {peer} invalid shutdown ready")
        for peer in range(1, IAAS_WORLD_SIZE):
            self.group.send_obj("ack", dst=peer)
        for peer in range(1, IAAS_WORLD_SIZE):
            if self.group.recv_obj(src=peer) != "exit_ready":
                raise RuntimeError(f"IAAS client {peer} invalid exit confirmation")


def run_iaas_service(
    *,
    host: str,
    port: int,
    gpu_id: int,
    max_batch_per_client: int,
    capture_batches: Sequence[int],
    context_len: int,
    warmup_steps: int,
    decode_steps: int,
    result_filename: str,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    service = IAASService(
        host=host,
        port=port,
        device=device,
        max_batch_per_client=max_batch_per_client,
        context_len=context_len,
        capture_batches=capture_batches,
    )
    service.capture()
    elapsed = service.serve_decode_points(
        active_batches=capture_batches,
        context_len=context_len,
        warmup_steps=warmup_steps,
        decode_steps=decode_steps,
    )
    result = service.result(elapsed)
    path = Path(result_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    service.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
