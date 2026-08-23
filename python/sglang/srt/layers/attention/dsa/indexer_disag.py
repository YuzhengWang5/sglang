"""Indexer-as-a-service disaggregation client (experimental).

This module is the registration point for the indexer-disag RPC client.
The indexer worker process (which owns the full index-key pool and runs
SCORE+SELECT on a dedicated GPU) registers a client here at startup;
`DSAIndexer._disag_topk_remote` then uses it to ship q_fp8 and recv
topk_result over NVLink.

Gated by SGLANG_INDEX_DISAG=1 (see dsa_indexer.py forward_cuda). With no
client registered, `get_client()` returns None and the disag path is a
no-op (the engine falls back to the co-located _get_topk_paged).

See runs/20260823_033353_r3_indexer_service/ for the real multi-process PoC
(batched comm = 1 round-trip/step) that validates this mechanism on real
kernels, and runs/20260822_021415_r2_disag_2gpu_design/ for the full
server-side design (KV pool placement, CUDA-graph split-op).
"""
from __future__ import annotations

from typing import Optional

# Pluggable client singleton. The indexer worker registers it via
# register_client(); model ranks fetch it via get_client().
_CLIENT: Optional["IndexerDisagClient"] = None  # type: ignore  # noqa: F821


class IndexerDisagClient:
    """Minimal interface the indexer worker's RPC client must implement.

    topk_remote() ships q_fp8 (+ weights + page table + seqlens) to the
    indexer GPU and returns the (B*next_n, index_topk) int tensor of selected
    token indices -- the same shape _get_topk_paged returns.
    """

    def topk_remote(
        self,
        layer_id: int,
        q_fp8,
        weights,
        seqlens,
        block_table,
        max_seq_len: int,
        index_topk: int,
    ):
        raise NotImplementedError


def register_client(client: IndexerDisagClient) -> None:
    """Called by the indexer worker process at startup."""
    global _CLIENT
    _CLIENT = client


def get_client() -> Optional[IndexerDisagClient]:
    """Called by DSAIndexer._get_indexer_disag_client (lazy, off hot path)."""
    return _CLIENT
