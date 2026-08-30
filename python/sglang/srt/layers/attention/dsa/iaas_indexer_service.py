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
import triton
import triton.language as tl

import deep_gemm
from sgl_kernel import fast_topk_v2
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _store_fp4_index_k_cache_kernel,
    quantize_fp4_indexer_tensor,
    quantize_fp4_indexer_tensor_out,
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
# DeepSeek-V3.2 executes the first ``first_k_dense_replace`` layers outside the
# sparse-layer TBO scheduler.  Their attention requests therefore remain full
# batch, while layers at and after this boundary issue two half-batch requests.
IAAS_TBO_FIRST_LAYER = 3
IAAS_Q_DATA_BYTES_PER_ROW = IAAS_HEADS * IAAS_HEAD_DIM // 2
IAAS_Q_SCALE_BYTES_PER_ROW = IAAS_HEADS * 4
IAAS_WEIGHTS_BYTES_PER_ROW = IAAS_HEADS * 4
IAAS_K_DATA_BYTES_PER_ROW = IAAS_HEAD_DIM // 2
IAAS_K_SCALE_BYTES_PER_ROW = 4
IAAS_REQUEST_BYTES_PER_ROW = (
    IAAS_Q_DATA_BYTES_PER_ROW
    + IAAS_Q_SCALE_BYTES_PER_ROW
    + IAAS_WEIGHTS_BYTES_PER_ROW
    + IAAS_K_DATA_BYTES_PER_ROW
    + IAAS_K_SCALE_BYTES_PER_ROW
)


_client: Optional["IAASClient"] = None


@triton.jit
def _unpack_fused_requests_kernel(
    mailbox,
    aggregate,
    batch: tl.constexpr,
    per_client_bytes: tl.constexpr,
    total_bytes: tl.constexpr,
    clients: tl.constexpr,
    q_bytes_per_row: tl.constexpr,
    q_scale_bytes_per_row: tl.constexpr,
    weights_bytes_per_row: tl.constexpr,
    k_bytes_per_row: tl.constexpr,
    k_scale_bytes_per_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Transpose client-major fused mailboxes into field-major tensors."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total_bytes
    client = offsets // per_client_bytes
    local = offsets - client * per_client_bytes

    q_bytes = batch * q_bytes_per_row
    q_scale_bytes = batch * q_scale_bytes_per_row
    weights_bytes = batch * weights_bytes_per_row
    k_bytes = batch * k_bytes_per_row
    k_scale_bytes = batch * k_scale_bytes_per_row

    q_end = q_bytes
    q_scale_end = q_end + q_scale_bytes
    weights_end = q_scale_end + weights_bytes
    k_end = weights_end + k_bytes

    dst = client * q_bytes + local
    dst = tl.where(
        (local >= q_end) & (local < q_scale_end),
        clients * q_bytes + client * q_scale_bytes + local - q_end,
        dst,
    )
    dst = tl.where(
        (local >= q_scale_end) & (local < weights_end),
        clients * (q_bytes + q_scale_bytes)
        + client * weights_bytes
        + local
        - q_scale_end,
        dst,
    )
    dst = tl.where(
        (local >= weights_end) & (local < k_end),
        clients * (q_bytes + q_scale_bytes + weights_bytes)
        + client * k_bytes
        + local
        - weights_end,
        dst,
    )
    dst = tl.where(
        local >= k_end,
        clients * (q_bytes + q_scale_bytes + weights_bytes + k_bytes)
        + client * k_scale_bytes
        + local
        - k_end,
        dst,
    )
    value = tl.load(mailbox + offsets, mask=valid)
    tl.store(aggregate + dst, value, mask=valid)


class IAASClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        rank: int,
        device: torch.device,
        fused_request_mailbox: bool = False,
    ) -> None:
        if rank <= 0 or rank >= IAAS_WORLD_SIZE:
            raise ValueError(f"IAAS client rank must be 1..4, got {rank}")
        self.rank = rank
        self.device = device
        self.fused_request_mailbox = fused_request_mailbox
        self.group = StatelessProcessGroup.create(
            host, port, rank, IAAS_WORLD_SIZE
        )
        self.group.store.set_timeout(timedelta(hours=1))
        self.comm = PyNcclCommunicator(self.group, device)
        if not self.comm.available:
            raise RuntimeError("IAAS PyNcclCommunicator is unavailable")
        self.responses_by_batch: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _encode_fused_request(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize directly into one client-major graph-captured mailbox."""
        batch = query.shape[0]
        request = torch.empty(
            batch * IAAS_REQUEST_BYTES_PER_ROW,
            dtype=torch.uint8,
            device=query.device,
        )
        q_bytes = batch * IAAS_Q_DATA_BYTES_PER_ROW
        q_scale_bytes = batch * IAAS_Q_SCALE_BYTES_PER_ROW
        weights_bytes = batch * IAAS_WEIGHTS_BYTES_PER_ROW
        k_bytes = batch * IAAS_K_DATA_BYTES_PER_ROW
        q_end = q_bytes
        q_scale_end = q_end + q_scale_bytes
        weights_end = q_scale_end + weights_bytes
        k_end = weights_end + k_bytes
        q_data = request[:q_end].view(torch.int8).view(
            batch, 1, IAAS_HEADS, IAAS_HEAD_DIM // 2
        )
        q_scale = request[q_end:q_scale_end].view(torch.int32).view(
            batch, 1, IAAS_HEADS
        )
        weights_out = request[q_scale_end:weights_end].view(torch.float32).view(
            batch, IAAS_HEADS
        )
        k_data = request[weights_end:k_end].view(torch.int8).view(
            batch, IAAS_HEAD_DIM // 2
        )
        k_scale = request[k_end:].view(torch.int32).view(batch)
        quantize_fp4_indexer_tensor_out(
            query, q_data.view(-1, IAAS_HEAD_DIM // 2), q_scale.view(-1)
        )
        quantize_fp4_indexer_tensor_out(key, k_data, k_scale)
        weights_out.copy_(weights)
        return request

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

        weights = weights.to(torch.float32).contiguous()
        if self.fused_request_mailbox:
            request = self._encode_fused_request(query, key, weights)
        else:
            q_data, q_scale = quantize_fp4_indexer_tensor(query.contiguous())
            k_data, k_scale = quantize_fp4_indexer_tensor(key.contiguous())
            q_data = q_data.view(batch, 1, IAAS_HEADS, IAAS_HEAD_DIM // 2)
            q_scale = q_scale.view(batch, 1, IAAS_HEADS)
            k_data = k_data.view(batch, IAAS_HEAD_DIM // 2)
            k_scale = k_scale.view(batch)
        response = torch.empty(
            (batch, IAAS_TOPK), dtype=torch.int32, device=query.device
        )
        self.responses_by_batch[batch] = response

        with self.comm.change_state(enable=True):
            if self.fused_request_mailbox:
                self.comm.send(request, dst=0)
            else:
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
    *,
    host: str,
    port: int,
    rank: int,
    device: torch.device,
    fused_request_mailbox: bool = False,
) -> IAASClient:
    global _client
    if _client is not None:
        raise RuntimeError("IAAS client is already initialized")
    _client = IAASClient(
        host=host,
        port=port,
        rank=rank,
        device=device,
        fused_request_mailbox=fused_request_mailbox,
    )
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
    first_client: int
    num_clients: int
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
    request_mailbox: Optional[torch.Tensor] = None
    aggregate_request: Optional[torch.Tensor] = None


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
        fused_request_mailbox: bool = False,
        two_batch_overlap: bool = False,
        client_wave_size: int = IAAS_CLIENTS,
        tbo_first_layer: int = IAAS_TBO_FIRST_LAYER,
    ) -> None:
        self.device = device
        self.max_batch_per_client = max_batch_per_client
        self.context_len = context_len
        self.capture_batches = tuple(sorted(set(capture_batches)))
        self.fused_request_mailbox = fused_request_mailbox
        self.two_batch_overlap = two_batch_overlap
        self.client_wave_size = client_wave_size
        self.tbo_first_layer = tbo_first_layer
        if not self.capture_batches or self.capture_batches[-1] > max_batch_per_client:
            raise ValueError("invalid IAAS capture batches")
        if self.two_batch_overlap and any(batch % 2 for batch in self.capture_batches):
            raise ValueError("IAAS TBO currently requires even capture batches")
        if self.client_wave_size not in (1, 2, IAAS_CLIENTS):
            raise ValueError("IAAS client wave size must be 1, 2, or 4")
        if not 0 <= self.tbo_first_layer <= IAAS_LAYERS:
            raise ValueError("invalid IAAS TBO first-layer boundary")
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
        self.buffers: Dict[int, tuple[tuple[_ServiceBuffers, ...], ...]] = {
            batch: self._make_point_buffers(batch) for batch in self.capture_batches
        }
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}

    def _make_point_buffers(
        self, batch: int
    ) -> tuple[tuple[_ServiceBuffers, ...], ...]:
        full = self._make_phase_buffers(batch, request_offset=0)
        if not self.two_batch_overlap:
            return (full,)
        child_batch = batch // 2
        return (
            full,
            self._make_phase_buffers(child_batch, request_offset=0),
            self._make_phase_buffers(child_batch, request_offset=child_batch),
        )

    def _make_phase_buffers(
        self, batch: int, *, request_offset: int
    ) -> tuple[_ServiceBuffers, ...]:
        return tuple(
            self._make_buffers(
                batch,
                request_offset=request_offset,
                first_client=first_client,
                num_clients=min(self.client_wave_size, IAAS_CLIENTS - first_client),
            )
            for first_client in range(0, IAAS_CLIENTS, self.client_wave_size)
        )

    def _make_buffers(
        self,
        batch: int,
        *,
        request_offset: int,
        first_client: int,
        num_clients: int,
    ) -> _ServiceBuffers:
        rows = num_clients * batch
        slots = torch.cat(
            [
                torch.arange(
                    client * self.max_batch_per_client + request_offset,
                    client * self.max_batch_per_client + request_offset + batch,
                    dtype=torch.long,
                    device=self.device,
                )
                for client in range(first_client, first_client + num_clients)
            ]
        )
        block_table = self.full_block_table.index_select(0, slots).contiguous()
        seqlens_2d = torch.full(
            (rows, 1), self.context_len, dtype=torch.int32, device=self.device
        )
        lengths = seqlens_2d[:, 0]
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_2d,
            IAAS_PAGE_SIZE,
            deep_gemm.get_num_sms(),
        )
        loc = block_table[:, -1] * IAAS_PAGE_SIZE + (
            self.context_len - 1
        ) % IAAS_PAGE_SIZE
        request_mailbox = None
        aggregate_request = None
        if self.fused_request_mailbox:
            request_mailbox = torch.empty(
                (num_clients, batch * IAAS_REQUEST_BYTES_PER_ROW),
                dtype=torch.uint8,
                device=self.device,
            )
            aggregate_request = torch.empty(
                rows * IAAS_REQUEST_BYTES_PER_ROW,
                dtype=torch.uint8,
                device=self.device,
            )
            q_bytes = rows * IAAS_Q_DATA_BYTES_PER_ROW
            q_scale_bytes = rows * IAAS_Q_SCALE_BYTES_PER_ROW
            weights_bytes = rows * IAAS_WEIGHTS_BYTES_PER_ROW
            k_bytes = rows * IAAS_K_DATA_BYTES_PER_ROW
            q_data = aggregate_request[:q_bytes].view(torch.int8).view(
                rows, 1, IAAS_HEADS, IAAS_HEAD_DIM // 2
            )
            q_scale = aggregate_request[
                q_bytes : q_bytes + q_scale_bytes
            ].view(torch.int32).view(rows, 1, IAAS_HEADS)
            weights = aggregate_request[
                q_bytes + q_scale_bytes : q_bytes + q_scale_bytes + weights_bytes
            ].view(torch.float32).view(rows, IAAS_HEADS)
            k_data = aggregate_request[
                q_bytes
                + q_scale_bytes
                + weights_bytes : q_bytes
                + q_scale_bytes
                + weights_bytes
                + k_bytes
            ].view(torch.int8).view(rows, IAAS_HEAD_DIM // 2)
            k_scale = aggregate_request[-rows * IAAS_K_SCALE_BYTES_PER_ROW :].view(
                torch.int32
            )
        else:
            q_data = torch.empty(
                (rows, 1, IAAS_HEADS, IAAS_HEAD_DIM // 2),
                dtype=torch.int8,
                device=self.device,
            )
            q_scale = torch.empty(
                (rows, 1, IAAS_HEADS), dtype=torch.int32, device=self.device
            )
            weights = torch.empty(
                (rows, IAAS_HEADS), dtype=torch.float32, device=self.device
            )
            k_data = torch.empty(
                (rows, IAAS_HEAD_DIM // 2), dtype=torch.int8, device=self.device
            )
            k_scale = torch.empty((rows,), dtype=torch.int32, device=self.device)

        return _ServiceBuffers(
            batch_per_client=batch,
            first_client=first_client,
            num_clients=num_clients,
            q_data=q_data,
            q_scale=q_scale,
            weights=weights,
            k_data=k_data,
            k_scale=k_scale,
            responses=torch.empty(
                (rows, IAAS_TOPK), dtype=torch.int32, device=self.device
            ),
            seqlens_2d=seqlens_2d,
            lengths=lengths,
            block_table=block_table,
            loc=loc.contiguous(),
            schedule=schedule,
            request_mailbox=request_mailbox,
            aggregate_request=aggregate_request,
        )

    def _recv_fused_request(self, buffers: _ServiceBuffers) -> None:
        batch = buffers.batch_per_client
        assert buffers.request_mailbox is not None
        assert buffers.aggregate_request is not None
        self.comm.group_start()
        for local_client in range(buffers.num_clients):
            peer = buffers.first_client + local_client + 1
            self.comm.recv(buffers.request_mailbox[local_client], src=peer)
        self.comm.group_end()
        total_bytes = buffers.num_clients * batch * IAAS_REQUEST_BYTES_PER_ROW
        _unpack_fused_requests_kernel[(triton.cdiv(total_bytes, 256),)](
            buffers.request_mailbox,
            buffers.aggregate_request,
            batch=batch,
            per_client_bytes=batch * IAAS_REQUEST_BYTES_PER_ROW,
            total_bytes=total_bytes,
            clients=buffers.num_clients,
            q_bytes_per_row=IAAS_Q_DATA_BYTES_PER_ROW,
            q_scale_bytes_per_row=IAAS_Q_SCALE_BYTES_PER_ROW,
            weights_bytes_per_row=IAAS_WEIGHTS_BYTES_PER_ROW,
            k_bytes_per_row=IAAS_K_DATA_BYTES_PER_ROW,
            k_scale_bytes_per_row=IAAS_K_SCALE_BYTES_PER_ROW,
            BLOCK=256,
        )

    def _recv_from_clients(
        self, tensor: torch.Tensor, batch: int, first_client: int, num_clients: int
    ) -> None:
        self.comm.group_start()
        for local_client in range(num_clients):
            peer = first_client + local_client + 1
            self.comm.recv(
                tensor[local_client * batch : (local_client + 1) * batch], src=peer
            )
        self.comm.group_end()

    def _send_to_clients(
        self, tensor: torch.Tensor, batch: int, first_client: int, num_clients: int
    ) -> None:
        self.comm.group_start()
        for local_client in range(num_clients):
            peer = first_client + local_client + 1
            self.comm.send(
                tensor[local_client * batch : (local_client + 1) * batch], dst=peer
            )
        self.comm.group_end()

    def _once_layer(self, buffers: _ServiceBuffers, layer: int) -> None:
        batch = buffers.batch_per_client
        rows = buffers.num_clients * batch
        with self.comm.change_state(enable=True):
            if self.fused_request_mailbox:
                self._recv_fused_request(buffers)
            else:
                self._recv_from_clients(
                    buffers.q_data,
                    batch,
                    buffers.first_client,
                    buffers.num_clients,
                )
                self._recv_from_clients(
                    buffers.q_scale,
                    batch,
                    buffers.first_client,
                    buffers.num_clients,
                )
                self._recv_from_clients(
                    buffers.weights,
                    batch,
                    buffers.first_client,
                    buffers.num_clients,
                )
                self._recv_from_clients(
                    buffers.k_data,
                    batch,
                    buffers.first_client,
                    buffers.num_clients,
                )
                self._recv_from_clients(
                    buffers.k_scale,
                    batch,
                    buffers.first_client,
                    buffers.num_clients,
                )
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
            self._send_to_clients(
                buffers.responses,
                batch,
                buffers.first_client,
                buffers.num_clients,
            )

    def _once(
        self, point_buffers: tuple[tuple[_ServiceBuffers, ...], ...]
    ) -> None:
        full_phase = point_buffers[0]
        child_phases = point_buffers[1:]
        for layer in range(IAAS_LAYERS):
            if not child_phases or layer < self.tbo_first_layer:
                phases = (full_phase,)
            else:
                # The model's TBO operation strategy always reaches child A's
                # attention before child B's for a layer.  Serving in that order
                # lets A's MoE work overlap service compute for B without adding
                # an extra control message to the captured graph.
                phases = child_phases
            for phase in phases:
                # A smaller wave returns the earliest DP response before later
                # DP requests are serviced.  This can overlap those ranks' local
                # sparse-attention tail with service work; the downstream TP/EP
                # collective still provides the correctness rendezvous.
                for wave in phase:
                    self._once_layer(wave, layer)

    def capture(self) -> None:
        for batch in reversed(self.capture_batches):
            point_buffers = self.buffers[batch]
            for _ in range(2):
                self.group.barrier()
                self._once(point_buffers)
                torch.cuda.synchronize(self.device)
            self.group.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._once(point_buffers)
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
            point_buffers = self.buffers[batch]
            graph = self.graphs[batch]
            self.group.barrier()
            steady_begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for step in range(total_steps):
                if step == warmup_steps:
                    steady_begin.record()
                for phase in point_buffers:
                    for buffers in phase:
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
            "fused_request_mailbox": self.fused_request_mailbox,
            "two_batch_overlap": self.two_batch_overlap,
            "client_wave_size": self.client_wave_size,
            "tbo_first_layer": self.tbo_first_layer,
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
    fused_request_mailbox: bool = False,
    two_batch_overlap: bool = False,
    client_wave_size: int = IAAS_CLIENTS,
    decode_point_batches: Optional[Sequence[int]] = None,
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
        fused_request_mailbox=fused_request_mailbox,
        two_batch_overlap=two_batch_overlap,
        client_wave_size=client_wave_size,
    )
    service.capture()
    elapsed = service.serve_decode_points(
        active_batches=(
            decode_point_batches
            if decode_point_batches is not None
            else capture_batches
        ),
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
