"""Decode-only HiSparse/indexer capacity-wall benchmark.

The harness loads real model weights and runs the production ModelRunner decode
path, but constructs an already-prefilled dummy KV state.  Historical MLA KV is
backed by a small shared pinned-host pool; each retrieval still executes the real
HiSparse plan/copy kernel into per-request device hot buffers.  Normal mode keeps
request-private index-cache pages, while Oracle-No-Index may share logical pages
because no index cache exists.

This module is research-only.  Synthetic indexer output has no accuracy meaning.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import time
from array import array
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist

from sglang.benchmark.one_batch import (
    TreeCacheNamespace,
    _maybe_prepare_mlp_sync_batch,
    load_model,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import get_moe_runner_backend, initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id


MODES = (
    "normal",
    "no_indexer_compute",
    "oracle_no_index",
    "oracle_recent",
    "iaas_fp4",
)


@dataclasses.dataclass
class DummyDecodeArgs:
    indexer_mode: str = "normal"
    # Allocation/capacity batch per attention-DP rank.  --batch-sizes may run
    # smaller active prefixes without reloading the model.
    batch_size: int = 8
    batch_sizes: str = ""
    context_len: int = 65536
    device_buffer_size: int = 4096
    warmup_steps: int = 4
    decode_steps: int = 20
    dummy_host_size_gb: float = 0.2
    result_filename: str = "decode_dummy.jsonl"
    force_synthetic_retrieval: bool = True
    synthetic_overlap_fraction: float = -1.0
    iaas_host: str = "127.0.0.1"
    iaas_port: int = 29649
    iaas_service_gpu_id: int = 4
    iaas_service_device: str = ""
    iaas_service_result_filename: str = "iaas_service.json"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--indexer-mode", choices=MODES, default="normal")
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument(
            "--batch-sizes",
            default="",
            help="Comma-separated active batches; each must be <= --batch-size.",
        )
        parser.add_argument("--context-len", type=int, default=65536)
        parser.add_argument("--device-buffer-size", type=int, default=4096)
        parser.add_argument("--warmup-steps", type=int, default=4)
        parser.add_argument("--decode-steps", type=int, default=20)
        parser.add_argument("--dummy-host-size-gb", type=float, default=0.2)
        parser.add_argument("--result-filename", default="decode_dummy.jsonl")
        parser.add_argument(
            "--synthetic-overlap-fraction",
            type=float,
            default=-1.0,
            help=(
                "If in [0,1], preserve this fraction of a deterministic top-k "
                "between adjacent steps after running the configured indexer."
            ),
        )
        parser.add_argument("--iaas-host", default="127.0.0.1")
        parser.add_argument("--iaas-port", type=int, default=29649)
        parser.add_argument("--iaas-service-gpu-id", type=int, default=4)
        parser.add_argument(
            "--iaas-service-device",
            default="",
            help="Optional CUDA_VISIBLE_DEVICES value for the service child.",
        )
        parser.add_argument(
            "--iaas-service-result-filename", default="iaas_service.json"
        )
        parser.add_argument(
            "--no-force-synthetic-retrieval",
            action="store_false",
            dest="force_synthetic_retrieval",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "DummyDecodeArgs":
        return cls(
            **{
                field.name: getattr(args, field.name)
                for field in dataclasses.fields(cls)
            }
        )

    def active_batch_sizes(self) -> List[int]:
        values = (
            [int(value) for value in self.batch_sizes.split(",") if value.strip()]
            if self.batch_sizes
            else [self.batch_size]
        )
        values = sorted(set(values))
        if not values or values[0] <= 0:
            raise ValueError("all active batch sizes must be positive")
        if values[-1] > self.batch_size:
            raise ValueError(
                f"active batch {values[-1]} exceeds allocation batch {self.batch_size}"
            )
        return values


def _ceil_to(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _configure_cli_for_point(
    args: argparse.Namespace, bench_args: DummyDecodeArgs
) -> Dict[str, Any]:
    page_size = args.page_size or 64
    assert page_size == 64, "DeepSeek-V3.2 DSA benchmark requires page_size=64"
    assert bench_args.device_buffer_size >= 2048

    padded_hot = bench_args.device_buffer_size + page_size
    device_tokens = _ceil_to(bench_args.batch_size * padded_hot, page_size)
    aligned_context = _ceil_to(bench_args.context_len, page_size)
    private_index_pages = bench_args.indexer_mode not in (
        "oracle_no_index",
        "oracle_recent",
        "iaas_fp4",
    )
    logical_tokens = aligned_context * (
        bench_args.batch_size if private_index_pages else 1
    )
    ratio = max(1, math.ceil(logical_tokens / device_tokens))

    # ModelRunner sizes the physical MLA pool from max_total_tokens. HiSparse
    # multiplies it by ratio for logical/index capacity.  Oracle shares the
    # dummy historical page table, so it only needs one context worth of logical
    # pages; the per-request MLA hot buffers remain private.
    args.max_total_tokens = device_tokens
    # ServerArgs.max_running_requests is node-global; the KV configurator
    # divides it by attention-DP size when creating each rank-local req pool.
    args.max_running_requests = bench_args.batch_size * args.dp_size
    args.context_length = (
        bench_args.context_len + bench_args.warmup_steps + bench_args.decode_steps + page_size
    )
    args.enable_hisparse = True
    args.hisparse_config = json.dumps(
        {
            "top_k": 2048,
            "device_buffer_size": bench_args.device_buffer_size,
            "host_to_device_ratio": ratio,
        }
    )
    args.disable_radix_cache = True
    args.cuda_graph_bs_decode = bench_args.active_batch_sizes()

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    os.environ["SGLANG_DSA_BENCHMARK_MODE"] = bench_args.indexer_mode
    os.environ["SGLANG_DSA_BENCHMARK_FORCE_SYNTHETIC_OUTPUT"] = (
        "1" if bench_args.force_synthetic_retrieval else "0"
    )
    if not (-1.0 <= bench_args.synthetic_overlap_fraction <= 1.0):
        raise ValueError("--synthetic-overlap-fraction must be -1 or in [0, 1]")
    os.environ["SGLANG_DSA_BENCHMARK_OVERLAP_FRACTION"] = str(
        bench_args.synthetic_overlap_fraction
    )
    os.environ["SGLANG_DSA_DUMMY_KV_HOST_SIZE_GB"] = str(
        bench_args.dummy_host_size_gb
    )

    return {
        "page_size": page_size,
        "padded_hot_tokens_per_request": padded_hot,
        "device_pool_tokens": device_tokens,
        "logical_pool_tokens": device_tokens * ratio,
        "requested_logical_tokens": logical_tokens,
        "host_to_device_ratio": ratio,
        "node_max_running_requests": args.max_running_requests,
        "allocation_batch_size_per_dp_rank": bench_args.batch_size,
        "active_batch_sizes_per_dp_rank": bench_args.active_batch_sizes(),
    }


def _make_reqs(batch_size: int) -> List[Req]:
    sampling = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=f"dummy-{i}",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=sampling,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(0, 1)
        reqs.append(req)
    return reqs


def _build_dummy_batch(
    model_runner,
    reqs: List[Req],
    tree_cache: TreeCacheNamespace,
) -> ScheduleBatch:
    req_pool = model_runner.req_to_token_pool
    allocator = model_runner.token_to_kv_pool_allocator
    coordinator = model_runner.hisparse_coordinator
    slots = [req.req_pool_idx for req in reqs]
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=model_runner.device)

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.hisparse_coordinator = coordinator
    batch.forward_mode = ForwardMode.DECODE
    batch.req_pool_indices = slot_tensor
    batch.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
    batch.input_ids = torch.ones(
        len(reqs), dtype=torch.int64, device=model_runner.device
    )
    batch.multimodal_inputs = [None] * len(reqs)
    return batch


def _prepare_dummy_state(model_runner, bench_args: DummyDecodeArgs):
    req_pool = model_runner.req_to_token_pool
    allocator = model_runner.token_to_kv_pool_allocator
    coordinator = model_runner.hisparse_coordinator
    assert coordinator is not None

    reqs = _make_reqs(bench_args.batch_size)
    slots = req_pool.alloc(reqs)
    if slots is None:
        raise RuntimeError("request pool allocation failed")

    page_size = allocator.page_size
    aligned_context = _ceil_to(bench_args.context_len, page_size)
    logical_allocator = allocator.logical_attn_allocator
    private_pages = bench_args.indexer_mode not in (
        "oracle_no_index",
        "oracle_recent",
        "iaas_fp4",
    )
    alloc_tokens = aligned_context * (bench_args.batch_size if private_pages else 1)
    logical = logical_allocator.alloc(alloc_tokens)
    if logical is None:
        raise RuntimeError(
            f"logical pool allocation failed: need={alloc_tokens}, "
            f"available={logical_allocator.available_size()}"
        )

    if private_pages:
        logical_rows = logical.view(bench_args.batch_size, aligned_context)
    else:
        logical_rows = logical.view(1, aligned_context).expand(
            bench_args.batch_size, -1
        )

    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=model_runner.device)
    req_pool.req_to_token[slot_tensor, : bench_args.context_len] = logical_rows[
        :, : bench_args.context_len
    ].to(torch.int32)

    # Shared dummy host backing. Every logical position has a valid host-cache
    # location, but positions/requests alias a small pinned pool. The production
    # swap-in kernel still copies the selected real bytes for every miss.
    host_size = coordinator.mem_pool_host.size
    host_map = torch.arange(
        bench_args.context_len, dtype=torch.int64, device=model_runner.device
    ).remainder_(host_size)
    coordinator.req_to_host_pool[slot_tensor, : bench_args.context_len] = host_map
    coordinator.req_to_host_pool_allocated_len[slots] = aligned_context
    coordinator.mem_pool_host.kv_buffer.zero_()

    for req in reqs:
        req.kv = ReqKvInfo(
            kv_allocated_len=bench_args.context_len,
            swa_evicted_seqlen=0,
        )
        req.kv_committed_len = bench_args.context_len
        coordinator.alloc_device_buffer(req)
        coordinator.req_device_buffer_tokens[
            :, req.req_pool_idx, : bench_args.device_buffer_size
        ] = -1
        coordinator._skip_first_backup[req.req_pool_idx] = True

    tree_cache = TreeCacheNamespace(
        page_size=page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=allocator,
    )
    return reqs, logical_rows[:, : bench_args.context_len], tree_cache


def _reset_active_dummy_state(
    model_runner,
    reqs: List[Req],
) -> None:
    """Reset hot state so each active batch starts from the same all-cold KV."""
    coordinator = model_runner.hisparse_coordinator
    slots = [req.req_pool_idx for req in reqs]
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=model_runner.device)
    coordinator.wait_for_pending_backup()
    coordinator.req_device_buffer_tokens[:, slot_tensor, :] = -1
    coordinator.lru_slots[:, slot_tensor, :] = coordinator._lru_init
    coordinator.mem_pool_host.kv_buffer.zero_()
    for req in reqs:
        coordinator._skip_first_backup[req.req_pool_idx] = True


def _set_step_state(batch, logical_rows, model_runner, seq_len: int) -> None:
    batch.seq_lens = torch.full(
        (len(batch.reqs),), seq_len, dtype=torch.int64, device=model_runner.device
    )
    batch.seq_lens_cpu = torch.full((len(batch.reqs),), seq_len, dtype=torch.int64)
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.seq_lens_sum = seq_len * len(batch.reqs)
    batch.out_cache_loc = logical_rows[:, seq_len - 1].to(torch.int64)
    batch.input_ids.fill_(1)
    model_runner.hisparse_coordinator.map_last_loc_to_buffer(
        batch.seq_lens,
        batch.out_cache_loc,
        batch.req_pool_indices,
        batch.seq_lens_cpu,
        batch.req_pool_indices_cpu,
    )


def _tensor_bytes(tensors) -> int:
    seen = set()
    total = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        key = (tensor.device.type, tensor.device.index, tensor.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += tensor.nbytes
    return total


def _memory_breakdown(
    model_runner,
    bench_args: DummyDecodeArgs,
    actual_batch_size: int,
) -> Dict[str, int]:
    pool = model_runner.token_to_kv_pool
    allocator = model_runner.token_to_kv_pool_allocator
    coord = model_runner.hisparse_coordinator
    model = model_runner.model

    weights = _tensor_bytes(list(model.parameters()) + list(model.buffers()))
    mla_device = _tensor_bytes(pool.kv_buffer)
    index_cache = _tensor_bytes(pool.index_k_with_scale_buffer)
    req_to_token = req_pool_bytes = model_runner.req_to_token_pool.req_to_token.nbytes
    allocator_mapping = _tensor_bytes(
        [allocator.full_to_hisparse_device_index_mapping]
    )
    coordinator_request_maps = _tensor_bytes(
        [
            coord.req_to_device_buffer,
            coord.req_to_host_pool,
            coord.req_device_buffer_size,
            coord.req_to_host_pool_allocated_len,
        ]
    )
    coordinator_layer_hot_metadata = _tensor_bytes(
        [
            coord.req_device_buffer_tokens,
            coord.req_device_buffer_token_locs,
            coord.lru_slots,
        ]
    )
    coordinator_workspaces = _tensor_bytes(
        [
            coord.top_k_device_locs_buffer,
            coord.raw_indices_buffer,
            coord.num_real_reqs,
        ]
    )
    torch_allocated = torch.cuda.memory_allocated(model_runner.device)
    torch_reserved = torch.cuda.memory_reserved(model_runner.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(model_runner.device)
    known = (
        weights
        + mla_device
        + index_cache
        + req_to_token
        + allocator_mapping
        + coordinator_request_maps
        + coordinator_layer_hot_metadata
        + coordinator_workspaces
    )
    return {
        "weights_and_model_buffers": weights,
        "mla_device_hot_pool": mla_device,
        "indexer_k_scale_cache": index_cache,
        "req_to_token_map": req_pool_bytes,
        "logical_to_device_map": allocator_mapping,
        "coordinator_request_maps": coordinator_request_maps,
        "coordinator_layer_hot_metadata": coordinator_layer_hot_metadata,
        "coordinator_workspaces": coordinator_workspaces,
        "other_torch_allocated": max(0, torch_allocated - known),
        "torch_allocated": torch_allocated,
        "torch_reserved": torch_reserved,
        "device_total_minus_free": total_bytes - free_bytes,
        "device_total": total_bytes,
        "dummy_host_kv_bytes": coord.mem_pool_host.kv_buffer.nbytes,
        "index_cache_tensor_count": len(pool.index_k_with_scale_buffer),
        "actual_batch_size": actual_batch_size,
        "allocation_batch_size": bench_args.batch_size,
        "active_logical_tokens": (
            bench_args.context_len
            if bench_args.indexer_mode
            in ("oracle_no_index", "oracle_recent", "iaas_fp4")
            else actual_batch_size * bench_args.context_len
        ),
        "context_len": bench_args.context_len,
    }


def _run_forward(batch, model_runner) -> bool:
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    forward_batch = ForwardBatch.init_new(
        batch,
        model_runner,
        return_hidden_states_before_norm=False,
    )
    output = model_runner.forward(forward_batch)
    return bool(output.can_run_graph)


def _worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    bench_args: DummyDecodeArgs,
    point_config: Dict[str, Any],
    gpu_id: int,
    tp_rank: int,
) -> None:
    # _set_envs_and_config() selects the multiprocessing "spawn" method.  A
    # spawned worker does not inherit the parent's process-local resolved config,
    # so publish the exact point config before constructing any model layers.
    publish(server_args, role="scheduler")
    # Production Scheduler.init_moe_gemm_config() initializes a second set of
    # backend flags consumed directly by quantized layers.  The generic
    # one_batch loader omits this scheduler lifecycle step.
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    initialize_bf16_gemm_config(server_args)
    iaas_client = None
    if bench_args.indexer_mode == "iaas_fp4":
        from sglang.srt.layers.attention.dsa.iaas_indexer_service import (
            init_iaas_client,
        )

        torch.cuda.set_device(gpu_id)
        iaas_client = init_iaas_client(
            host=bench_args.iaas_host,
            port=bench_args.iaas_port,
            rank=tp_rank + 1,
            device=torch.device("cuda", gpu_id),
        )
    resolved_moe_backend = get_exec().moe.moe_runner_backend
    flag_moe_backend = get_moe_runner_backend().value
    if not (
        resolved_moe_backend == server_args.moe_runner_backend == flag_moe_backend
    ):
        raise RuntimeError(
            "worker resolved MoE backend diverged from ServerArgs: "
            f"resolved={resolved_moe_backend}, flags={flag_moe_backend}, "
            f"server={server_args.moe_runner_backend}"
        )
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    rank_print(
        f"resolved_moe_runner_backend={resolved_moe_backend}, "
        f"flags_moe_runner_backend={flag_moe_backend}"
    )
    wrapped_runner, _ = load_model(server_args, port_args, gpu_id, tp_rank)
    model_runner = wrapped_runner.torch_runner
    reqs, logical_rows, tree_cache = _prepare_dummy_state(model_runner, bench_args)

    total_steps = bench_args.warmup_steps + bench_args.decode_steps
    start_len = bench_args.context_len - total_steps
    if start_len <= 2048:
        raise ValueError("context_len must exceed top-k plus all benchmark steps")

    for actual_batch_size in bench_args.active_batch_sizes():
        active_reqs = reqs[:actual_batch_size]
        active_logical_rows = logical_rows[:actual_batch_size]
        batch = _build_dummy_batch(model_runner, active_reqs, tree_cache)
        _reset_active_dummy_state(model_runner, active_reqs)
        if dist.is_initialized():
            dist.barrier()
        if iaas_client is not None:
            iaas_client.control_barrier()

        latencies_ms: List[float] = []
        cuda_graph_replay_steps: List[bool] = []
        for step in range(total_steps):
            seq_len = start_len + step + 1
            _set_step_state(batch, active_logical_rows, model_runner, seq_len)
            torch.cuda.synchronize(model_runner.device)
            tic = time.perf_counter()
            replayed_cuda_graph = _run_forward(batch, model_runner)
            torch.cuda.synchronize(model_runner.device)
            elapsed_ms = (time.perf_counter() - tic) * 1000

            # Node TPOT is the slowest synchronized TP rank for this step.
            elapsed = torch.tensor(
                elapsed_ms, dtype=torch.float64, device=model_runner.device
            )
            if dist.is_initialized():
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if step >= bench_args.warmup_steps:
                latencies_ms.append(float(elapsed.item()))
                graph_flag = torch.tensor(
                    int(replayed_cuda_graph),
                    dtype=torch.int32,
                    device=model_runner.device,
                )
                if dist.is_initialized():
                    # Only report graph replay when every rank used it.
                    dist.all_reduce(graph_flag, op=dist.ReduceOp.MIN)
                cuda_graph_replay_steps.append(bool(graph_flag.item()))

        # Let the service leave the timed replay phase before model ranks do
        # rank-local memory accounting and object collectives.
        if iaas_client is not None:
            iaas_client.control_barrier()

        iaas_response_valid = (
            iaas_client.response_is_valid(
                actual_batch_size, bench_args.context_len
            )
            if iaas_client is not None
            else None
        )
        if iaas_client is not None and dist.is_initialized():
            valid_flag = torch.tensor(
                int(iaas_response_valid),
                dtype=torch.int32,
                device=model_runner.device,
            )
            dist.all_reduce(valid_flag, op=dist.ReduceOp.MIN)
            iaas_response_valid = bool(valid_flag.item())

        breakdown = _memory_breakdown(
            model_runner, bench_args, actual_batch_size
        )
        all_breakdowns: List[Dict[str, Any]] = [None] * server_args.tp_size
        if dist.is_initialized():
            dist.all_gather_object(all_breakdowns, breakdown)
        else:
            all_breakdowns[0] = breakdown

        if tp_rank == 0:
            mean_ms = float(np.mean(latencies_ms))
            p99_ms = float(np.percentile(latencies_ms, 99))
            result = {
                "status": "ok",
                "mode": bench_args.indexer_mode,
                "batch_size_per_dp_rank": actual_batch_size,
                "allocation_batch_size_per_dp_rank": bench_args.batch_size,
                "node_actual_batch_size": actual_batch_size * server_args.dp_size,
                "context_len": bench_args.context_len,
                "warmup_steps": bench_args.warmup_steps,
                "decode_steps": bench_args.decode_steps,
                "mean_tpot_ms": mean_ms,
                "p99_tpot_ms": p99_ms,
                "steady_state_node_tokens_per_s": (
                    actual_batch_size * server_args.dp_size * 1000.0 / mean_ms
                ),
                "step_tpot_ms": latencies_ms,
                "cuda_graph_replay_all_ranks": all(cuda_graph_replay_steps),
                "cuda_graph_replay_steps": cuda_graph_replay_steps,
                "decode_cuda_graph_runner": (
                    type(model_runner.decode_cuda_graph_runner).__name__
                    if model_runner.decode_cuda_graph_runner is not None
                    else None
                ),
                "decode_cuda_graph_captured_batch_sizes": list(
                    getattr(
                        model_runner.decode_cuda_graph_runner, "capture_bs", []
                    )
                    or []
                ),
                "model_runner_max_total_num_tokens": (
                    model_runner.max_total_num_tokens
                ),
                "max_hot_buffer_batch_by_physical_pool": (
                    model_runner.max_total_num_tokens
                    // point_config["padded_hot_tokens_per_request"]
                ),
                "point_config": point_config,
                "hbm_breakdown_per_rank": all_breakdowns,
                "synthetic_retrieval": bench_args.force_synthetic_retrieval,
                "synthetic_overlap_fraction": (
                    bench_args.synthetic_overlap_fraction
                    if bench_args.synthetic_overlap_fraction >= 0
                    else None
                ),
                "iaas_response_valid_all_ranks": iaas_response_valid,
                "real_hisparse_io": True,
                "real_sparse_attention": True,
                "real_moe": True,
            }
            path = Path(bench_args.result_filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fout:
                fout.write(json.dumps(result) + "\n")
            rank_print(json.dumps(result, indent=2))

    if iaas_client is not None:
        iaas_client.shutdown()


def main(
    server_args: ServerArgs,
    bench_args: DummyDecodeArgs,
    point_config: Dict[str, Any],
) -> None:
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    workers = []
    service_proc = None
    device_offset = 0
    if bench_args.indexer_mode == "iaas_fp4":
        from sglang.srt.layers.attention.dsa.iaas_indexer_service import (
            run_iaas_service,
        )

        if server_args.tp_size != 4 or server_args.dp_size != 4:
            raise ValueError("iaas_fp4 currently requires TP=4 and DP=4")
        if bench_args.force_synthetic_retrieval:
            raise ValueError(
                "iaas_fp4 must use --no-force-synthetic-retrieval so the "
                "service top-k is not mislabeled as synthetic"
            )
        original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if bench_args.iaas_service_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = bench_args.iaas_service_device
            service_device_context = maybe_reindex_device_id(0)
        else:
            service_device_context = maybe_reindex_device_id(
                bench_args.iaas_service_gpu_id
            )
        with service_device_context as service_gpu_id:
            service_proc = multiprocessing.Process(
                target=run_iaas_service,
                kwargs={
                    "host": bench_args.iaas_host,
                    "port": bench_args.iaas_port,
                    "gpu_id": service_gpu_id,
                    "max_batch_per_client": bench_args.batch_size,
                    "capture_batches": bench_args.active_batch_sizes(),
                    "context_len": bench_args.context_len,
                    "warmup_steps": bench_args.warmup_steps,
                    "decode_steps": bench_args.decode_steps,
                    "result_filename": bench_args.iaas_service_result_filename,
                },
            )
            service_proc.start()
        if original_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
        workers.append(service_proc)

    for tp_rank in range(server_args.tp_size):
        with maybe_reindex_device_id(tp_rank + device_offset) as gpu_id:
            proc = multiprocessing.Process(
                target=_worker,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    point_config,
                    gpu_id,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

    # A failed client leaves the service blocked in NCCL/store coordination.
    # Poll all children together so one failure terminates its siblings rather
    # than hanging the benchmark parent indefinitely.
    failed = False
    alive = set(workers)
    while alive:
        for proc in list(alive):
            proc.join(timeout=0.2)
            if proc.exitcode is None:
                continue
            alive.remove(proc)
            if proc.exitcode != 0:
                failed = True
        if failed:
            for proc in alive:
                proc.terminate()
            for proc in alive:
                proc.join(timeout=10)
            break
    if failed:
        raise RuntimeError(
            "one or more dummy-decode ranks failed; inspect the per-point log"
        )


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    DummyDecodeArgs.add_cli_args(parser)
    args = parser.parse_args()
    bench_args = DummyDecodeArgs.from_cli_args(args)
    point_config = _configure_cli_for_point(args, bench_args)
    server_args = ServerArgs.from_cli_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main(server_args, bench_args, point_config)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    cli_main()
