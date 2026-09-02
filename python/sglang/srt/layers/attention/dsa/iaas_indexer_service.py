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


IAAS_PAGE_SIZE = 64
IAAS_DEFAULT_CLIENTS = 4
IAAS_DEFAULT_HEADS = 64
IAAS_DEFAULT_HEAD_DIM = 128
IAAS_DEFAULT_TOPK = 2048
IAAS_DEFAULT_LAYERS = 61
# DeepSeek-V3.2 executes the first ``first_k_dense_replace`` layers outside the
# sparse-layer TBO scheduler.  Their attention requests therefore remain full
# batch, while layers at and after this boundary issue two half-batch requests.
IAAS_TBO_FIRST_LAYER = 3


@dataclass(frozen=True)
class IAASSpec:
    clients: int = IAAS_DEFAULT_CLIENTS
    heads: int = IAAS_DEFAULT_HEADS
    head_dim: int = IAAS_DEFAULT_HEAD_DIM
    topk: int = IAAS_DEFAULT_TOPK
    layers: int = IAAS_DEFAULT_LAYERS
    cache_dtype: str = "fp4"

    def __post_init__(self) -> None:
        if self.clients <= 0:
            raise ValueError("IAAS clients must be positive")
        if self.heads <= 0 or self.head_dim != 128 or self.topk <= 0:
            raise ValueError(f"invalid IAAS model shape: {self}")
        if self.layers <= 0:
            raise ValueError("IAAS layers must be positive")
        if self.cache_dtype not in ("fp4", "fp8"):
            raise ValueError("IAAS cache dtype must be fp4 or fp8")

    @property
    def world_size(self) -> int:
        return self.clients + 1

    @property
    def q_data_bytes_per_row(self) -> int:
        divisor = 2 if self.cache_dtype == "fp4" else 1
        return self.heads * self.head_dim // divisor

    @property
    def q_scale_bytes_per_row(self) -> int:
        return self.heads * 4

    @property
    def weights_bytes_per_row(self) -> int:
        return self.heads * 4

    @property
    def k_data_bytes_per_row(self) -> int:
        return self.head_dim // (2 if self.cache_dtype == "fp4" else 1)

    @property
    def k_scale_bytes_per_row(self) -> int:
        return 4

    @property
    def request_bytes_per_row(self) -> int:
        return (
            self.q_data_bytes_per_row
            + self.q_scale_bytes_per_row
            + self.weights_bytes_per_row
            + self.k_data_bytes_per_row
            + self.k_scale_bytes_per_row
        )

    @property
    def cache_bytes_per_token(self) -> int:
        return 68 if self.cache_dtype == "fp4" else 132


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
        spec: IAASSpec,
        fused_request_mailbox: bool = False,
    ) -> None:
        if rank <= 0 or rank >= spec.world_size:
            raise ValueError(
                f"IAAS client rank must be 1..{spec.clients}, got {rank}"
            )
        self.rank = rank
        self.device = device
        self.spec = spec
        self.fused_request_mailbox = fused_request_mailbox
        self.group = StatelessProcessGroup.create(
            host, port, rank, spec.world_size
        )
        self.group.store.set_timeout(timedelta(hours=1))
        self.comm = PyNcclCommunicator(self.group, device)
        if not self.comm.available:
            raise RuntimeError("IAAS PyNcclCommunicator is unavailable")
        self.responses_by_batch: Dict[int, torch.Tensor] = {}

    def _encode_fused_request(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize directly into one client-major graph-captured mailbox."""
        batch = query.shape[0]
        request = torch.empty(
            batch * self.spec.request_bytes_per_row,
            dtype=torch.uint8,
            device=query.device,
        )
        q_bytes = batch * self.spec.q_data_bytes_per_row
        q_scale_bytes = batch * self.spec.q_scale_bytes_per_row
        weights_bytes = batch * self.spec.weights_bytes_per_row
        k_bytes = batch * self.spec.k_data_bytes_per_row
        q_end = q_bytes
        q_scale_end = q_end + q_scale_bytes
        weights_end = q_scale_end + weights_bytes
        k_end = weights_end + k_bytes
        data_dtype = (
            torch.int8
            if self.spec.cache_dtype == "fp4"
            else torch.float8_e4m3fn
        )
        q_data = request[:q_end].view(data_dtype).view(
            batch, 1, self.spec.heads, self.spec.q_data_bytes_per_row // self.spec.heads
        )
        scale_dtype = (
            torch.int32 if self.spec.cache_dtype == "fp4" else torch.float32
        )
        q_scale = request[q_end:q_scale_end].view(scale_dtype).view(
            batch, 1, self.spec.heads
        )
        weights_out = request[q_scale_end:weights_end].view(torch.float32).view(
            batch, self.spec.heads
        )
        k_data = request[weights_end:k_end].view(data_dtype).view(
            batch, self.spec.k_data_bytes_per_row
        )
        k_scale = request[k_end:].view(scale_dtype).view(batch)
        if self.spec.cache_dtype == "fp4":
            quantize_fp4_indexer_tensor_out(
                query,
                q_data.view(-1, self.spec.head_dim // 2),
                q_scale.view(-1),
            )
            quantize_fp4_indexer_tensor_out(key, k_data, k_scale)
        else:
            from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

            q_quant, q_quant_scale = act_quant(query.contiguous())
            k_quant, k_quant_scale = act_quant(key.contiguous())
            q_data.copy_(q_quant.view_as(q_data))
            q_scale.copy_(q_quant_scale.view_as(q_scale))
            k_data.copy_(k_quant.view_as(k_data))
            k_scale.copy_(k_quant_scale.view_as(k_scale))
        weights_out.copy_(
            weights
            if self.spec.cache_dtype == "fp4"
            else weights * q_scale.view(batch, self.spec.heads)
        )
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
        if query.shape != (batch, self.spec.heads, self.spec.head_dim):
            raise ValueError(f"unexpected IAAS query shape {query.shape}")
        if key.shape != (batch, self.spec.head_dim):
            raise ValueError(f"unexpected IAAS key shape {key.shape}")
        if weights.shape != (batch, self.spec.heads):
            raise ValueError(f"unexpected IAAS weights shape {weights.shape}")

        weights = weights.to(torch.float32).contiguous()
        if self.fused_request_mailbox:
            request = self._encode_fused_request(query, key, weights)
        else:
            if self.spec.cache_dtype == "fp4":
                q_data, q_scale = quantize_fp4_indexer_tensor(query.contiguous())
                k_data, k_scale = quantize_fp4_indexer_tensor(key.contiguous())
                q_data = q_data.view(
                    batch, 1, self.spec.heads, self.spec.head_dim // 2
                )
                q_scale = q_scale.view(batch, 1, self.spec.heads)
                k_data = k_data.view(batch, self.spec.head_dim // 2)
                k_scale = k_scale.view(batch)
            else:
                from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

                q_data, q_scale = act_quant(query.contiguous())
                k_data, k_scale = act_quant(key.contiguous())
                q_data = q_data.view(
                    batch, 1, self.spec.heads, self.spec.head_dim
                )
                q_scale = q_scale.view(batch, 1, self.spec.heads)
                k_data = k_data.view(batch, self.spec.head_dim)
                k_scale = k_scale.view(batch)
                weights = weights * q_scale.view(batch, self.spec.heads)
        response = torch.empty(
            (batch, self.spec.topk), dtype=torch.int32, device=query.device
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
    clients: int = IAAS_DEFAULT_CLIENTS,
    heads: int = IAAS_DEFAULT_HEADS,
    head_dim: int = IAAS_DEFAULT_HEAD_DIM,
    topk: int = IAAS_DEFAULT_TOPK,
    layers: int = IAAS_DEFAULT_LAYERS,
    cache_dtype: str = "fp4",
    fused_request_mailbox: bool = False,
) -> IAASClient:
    global _client
    if _client is not None:
        raise RuntimeError("IAAS client is already initialized")
    spec = IAASSpec(
        clients=clients,
        heads=heads,
        head_dim=head_dim,
        topk=topk,
        layers=layers,
        cache_dtype=cache_dtype,
    )
    _client = IAASClient(
        host=host,
        port=port,
        rank=rank,
        device=device,
        spec=spec,
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
    k_row: Optional[torch.Tensor]
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
        spec: IAASSpec,
        fused_request_mailbox: bool = False,
        two_batch_overlap: bool = False,
        client_wave_size: Optional[int] = None,
        tbo_first_layer: int = IAAS_TBO_FIRST_LAYER,
    ) -> None:
        self.device = device
        self.spec = spec
        self.max_batch_per_client = max_batch_per_client
        self.context_len = context_len
        self.capture_batches = tuple(sorted(set(capture_batches)))
        self.fused_request_mailbox = fused_request_mailbox
        self.two_batch_overlap = two_batch_overlap
        self.client_wave_size = client_wave_size or spec.clients
        self.tbo_first_layer = tbo_first_layer
        if not self.capture_batches or self.capture_batches[-1] > max_batch_per_client:
            raise ValueError("invalid IAAS capture batches")
        if self.two_batch_overlap and any(batch % 2 for batch in self.capture_batches):
            raise ValueError("IAAS TBO currently requires even capture batches")
        if not 1 <= self.client_wave_size <= spec.clients:
            raise ValueError(
                f"IAAS client wave size must be in 1..{spec.clients}"
            )
        if spec.clients % self.client_wave_size:
            raise ValueError("IAAS client wave size must divide client count")
        if not 0 <= self.tbo_first_layer <= spec.layers:
            raise ValueError("invalid IAAS TBO first-layer boundary")
        self.group = StatelessProcessGroup.create(
            host, port, 0, spec.world_size
        )
        self.group.store.set_timeout(timedelta(hours=1))
        self.comm = PyNcclCommunicator(self.group, device)
        if not self.comm.available:
            raise RuntimeError("IAAS service PyNcclCommunicator is unavailable")

        pages_per_req = (context_len + IAAS_PAGE_SIZE - 1) // IAAS_PAGE_SIZE
        self.pages_per_req = pages_per_req
        self.total_rows = spec.clients * max_batch_per_client
        self.total_pages = self.total_rows * pages_per_req
        self.caches = [
            torch.zeros(
                (
                    self.total_pages,
                    IAAS_PAGE_SIZE * spec.cache_bytes_per_token,
                ),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(spec.layers)
        ]
        if spec.cache_dtype == "fp4":
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
                num_clients=min(
                    self.client_wave_size, self.spec.clients - first_client
                ),
            )
            for first_client in range(
                0, self.spec.clients, self.client_wave_size
            )
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
        loc = loc.to(torch.int64)
        request_mailbox = None
        aggregate_request = None
        data_dtype = (
            torch.int8
            if self.spec.cache_dtype == "fp4"
            else torch.float8_e4m3fn
        )
        scale_dtype = (
            torch.int32 if self.spec.cache_dtype == "fp4" else torch.float32
        )
        if self.fused_request_mailbox:
            request_mailbox = torch.empty(
                (num_clients, batch * self.spec.request_bytes_per_row),
                dtype=torch.uint8,
                device=self.device,
            )
            aggregate_request = torch.empty(
                rows * self.spec.request_bytes_per_row,
                dtype=torch.uint8,
                device=self.device,
            )
            q_bytes = rows * self.spec.q_data_bytes_per_row
            q_scale_bytes = rows * self.spec.q_scale_bytes_per_row
            weights_bytes = rows * self.spec.weights_bytes_per_row
            k_bytes = rows * self.spec.k_data_bytes_per_row
            q_data = aggregate_request[:q_bytes].view(data_dtype).view(
                rows,
                1,
                self.spec.heads,
                self.spec.q_data_bytes_per_row // self.spec.heads,
            )
            q_scale = aggregate_request[
                q_bytes : q_bytes + q_scale_bytes
            ].view(scale_dtype).view(rows, 1, self.spec.heads)
            weights = aggregate_request[
                q_bytes + q_scale_bytes : q_bytes + q_scale_bytes + weights_bytes
            ].view(torch.float32).view(rows, self.spec.heads)
            k_data = aggregate_request[
                q_bytes
                + q_scale_bytes
                + weights_bytes : q_bytes
                + q_scale_bytes
                + weights_bytes
                + k_bytes
            ].view(data_dtype).view(rows, self.spec.k_data_bytes_per_row)
            k_scale = aggregate_request[
                -rows * self.spec.k_scale_bytes_per_row :
            ].view(
                scale_dtype
            )
        else:
            q_data = torch.empty(
                (
                    rows,
                    1,
                    self.spec.heads,
                    self.spec.q_data_bytes_per_row // self.spec.heads,
                ),
                dtype=data_dtype,
                device=self.device,
            )
            q_scale = torch.empty(
                (rows, 1, self.spec.heads),
                dtype=scale_dtype,
                device=self.device,
            )
            weights = torch.empty(
                (rows, self.spec.heads),
                dtype=torch.float32,
                device=self.device,
            )
            k_data = torch.empty(
                (rows, self.spec.k_data_bytes_per_row),
                dtype=data_dtype,
                device=self.device,
            )
            k_scale = torch.empty(
                (rows,), dtype=scale_dtype, device=self.device
            )

        k_row = (
            torch.empty(
                (rows, self.spec.cache_bytes_per_token),
                dtype=torch.uint8,
                device=self.device,
            )
            if self.spec.cache_dtype == "fp8"
            else None
        )

        return _ServiceBuffers(
            batch_per_client=batch,
            first_client=first_client,
            num_clients=num_clients,
            q_data=q_data,
            q_scale=q_scale,
            weights=weights,
            k_data=k_data,
            k_scale=k_scale,
            k_row=k_row,
            responses=torch.empty(
                (rows, self.spec.topk), dtype=torch.int32, device=self.device
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
        total_bytes = (
            buffers.num_clients * batch * self.spec.request_bytes_per_row
        )
        _unpack_fused_requests_kernel[(triton.cdiv(total_bytes, 256),)](
            buffers.request_mailbox,
            buffers.aggregate_request,
            batch=batch,
            per_client_bytes=batch * self.spec.request_bytes_per_row,
            total_bytes=total_bytes,
            clients=buffers.num_clients,
            q_bytes_per_row=self.spec.q_data_bytes_per_row,
            q_scale_bytes_per_row=self.spec.q_scale_bytes_per_row,
            weights_bytes_per_row=self.spec.weights_bytes_per_row,
            k_bytes_per_row=self.spec.k_data_bytes_per_row,
            k_scale_bytes_per_row=self.spec.k_scale_bytes_per_row,
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
            if self.spec.cache_dtype == "fp4":
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
            else:
                assert buffers.k_row is not None
                buffers.k_row[:, : self.spec.head_dim].view(
                    torch.float8_e4m3fn
                ).copy_(buffers.k_data)
                buffers.k_row[:, self.spec.head_dim :].view(
                    torch.float32
                ).copy_(buffers.k_scale.view(rows, 1))
                cache.view(-1, self.spec.cache_bytes_per_token).index_copy_(
                    0, buffers.loc, buffers.k_row
                )
                logits = deep_gemm.fp8_paged_mqa_logits(
                    buffers.q_data,
                    cache.view(
                        self.total_pages,
                        IAAS_PAGE_SIZE,
                        1,
                        self.spec.cache_bytes_per_token,
                    ),
                    buffers.weights,
                    buffers.seqlens_2d,
                    buffers.block_table,
                    buffers.schedule,
                    self.context_len,
                    False,
                )
            buffers.responses.copy_(
                fast_topk_v2(logits, buffers.lengths, self.spec.topk)
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
        for layer in range(self.spec.layers):
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
            "clients": self.spec.clients,
            "heads": self.spec.heads,
            "head_dim": self.spec.head_dim,
            "topk": self.spec.topk,
            "layers": self.spec.layers,
            "cache_dtype": self.spec.cache_dtype,
            "fused_request_mailbox": self.fused_request_mailbox,
            "two_batch_overlap": self.two_batch_overlap,
            "client_wave_size": self.client_wave_size,
            "tbo_first_layer": self.tbo_first_layer,
            "index_cache_gib": (
                self.spec.layers
                * self.total_rows
                * self.context_len
                * self.spec.cache_bytes_per_token
                / 2**30
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
        for peer in range(1, self.spec.world_size):
            if self.group.recv_obj(src=peer) != "done":
                raise RuntimeError(f"IAAS client {peer} invalid shutdown ready")
        for peer in range(1, self.spec.world_size):
            self.group.send_obj("ack", dst=peer)
        for peer in range(1, self.spec.world_size):
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
    clients: int = IAAS_DEFAULT_CLIENTS,
    heads: int = IAAS_DEFAULT_HEADS,
    head_dim: int = IAAS_DEFAULT_HEAD_DIM,
    topk: int = IAAS_DEFAULT_TOPK,
    layers: int = IAAS_DEFAULT_LAYERS,
    cache_dtype: str = "fp4",
    fused_request_mailbox: bool = False,
    two_batch_overlap: bool = False,
    client_wave_size: Optional[int] = None,
    decode_point_batches: Optional[Sequence[int]] = None,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    spec = IAASSpec(
        clients=clients,
        heads=heads,
        head_dim=head_dim,
        topk=topk,
        layers=layers,
        cache_dtype=cache_dtype,
    )
    service = IAASService(
        host=host,
        port=port,
        device=device,
        max_batch_per_client=max_batch_per_client,
        context_len=context_len,
        capture_batches=capture_batches,
        spec=spec,
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
