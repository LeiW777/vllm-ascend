from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    PoolKey,
    ReqMeta,
)


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@dcp:{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@layer_id:{self.layer_id}"
            f"@{self.chunk_hash}"
        )


def get_partial_block_index(
    token_count: int,
    block_size: int,
    hash_count: int,
    enabled: bool,
) -> int | None:
    """Index of the trailing partial block to transfer, if any.

    Returns None when disabled, when the request carries no tokens, or when
    the token count aligns exactly with completed hash blocks.
    """
    if not enabled or token_count <= 0:
        return None
    full_blocks, remainder = divmod(token_count, block_size)
    if remainder:
        return full_blocks
    if full_blocks > hash_count:
        return full_blocks - 1
    return None


def masked_block_runs(
    mask: Sequence[bool] | None,
    start_block: int,
    end_block: int,
) -> list[tuple[int, int]]:
    """Split [start_block, end_block) into maximal runs of mask-allowed blocks.

    A None mask (or blocks at or beyond the mask length) disables filtering,
    so callers never transfer fewer blocks than the mask actually covers.
    """
    if end_block <= start_block:
        return []
    if mask is None:
        return [(start_block, end_block)]
    mask_len = len(mask)
    if start_block >= mask_len:
        return [(start_block, end_block)]
    scan_end = min(end_block, mask_len)
    # True/False map to b"\x01"/b"\x00"; bytes.find scans at C speed and
    # never raises (unlike list.index), so both dense and holey masks skip
    # the per-block Python loop.
    bits = bytes(mask[start_block:scan_end])
    runs: list[tuple[int, int]] = []
    pos = 0
    while True:
        hole = bits.find(b"\x00", pos)
        if hole == -1:
            runs.append((start_block + pos, end_block))
            return runs
        if hole > pos:
            runs.append((start_block + pos, start_block + hole))
        pos = bits.find(b"\x01", hole + 1)
        if pos == -1:
            # Remaining mask blocks are all disallowed, but blocks at or
            # beyond mask_len stay allowed.
            if end_block > mask_len:
                runs.append((mask_len, end_block))
            return runs


@dataclass
class LayerBlockRange:
    request: ReqMeta
    start_block: int
    end_block: int
    partial_block_index: int | None = None


@dataclass
class LayerTransferMeta:
    """One layerwise transfer description covering both data planes.

    Built in two stages with one type: ``LayerBatchBuilder.build_shared``
    fills the layer-invariant block rows once per task (attached to every
    layer's ``LayerTransferTask.shared_block_data``), and
    ``LayerBatchBuilder.build_addrs`` projects those rows onto one layer's
    addresses. Exactly one plane payload is populated per projection: flat
    GVA batch arrays for the memcache plane, or per-key range rows for the
    block-key plane.
    """

    # --- request scope (layer-invariant) ---
    req_ids: list[str]
    is_last_chunks: list[bool | None] = field(default_factory=list)

    # --- block rows shared by every layer of the task ---
    block_ids_arr: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    block_gvas_arr: np.ndarray | None = None
    block_keys: list[str] | None = None
    save_keys: list[str] = field(default_factory=list)
    load_keys: list[str] = field(default_factory=list)

    # --- per-layer projection (filled by build_addrs) ---
    layer_id: int = -1
    # GVA plane: flat device/pool address descriptors.
    addr_array: np.ndarray | None = None
    size_array: np.ndarray | None = None
    gvas_array: np.ndarray | None = None
    # Block-key plane: one buffer/size/offset row per block key.
    all_buffers: list[list[int]] | None = None
    all_sizes: list[list[int]] | None = None
    all_offsets: list[list[int]] | None = None

    @property
    def uses_key_ranges(self) -> bool:
        """Whether build_addrs projected this meta onto per-key range rows."""
        return self.all_buffers is not None


@dataclass
class LayerTransferTask:
    layer_id: int
    block_ranges: list[LayerBlockRange]
    shared_block_data: LayerTransferMeta | None = None
    group_id: int = 0
    layer_idx_in_group: int = 0
    # Publish newly allocated GVA keys after this task completes the final
    # actual layer copy for the batch.
    write_finish_keys: list[str] = field(default_factory=list)
    # Block-key backends use one remote object per block/rank with per-layer ranges.
    use_key_major_ranges: bool = False
    # Group-local completion differs from the physical model layer boundary.
    final_group_layer: bool = False


@dataclass
class LayerLoadTask:
    wait_for_save_layer: int | None
    transfer_tasks: list[LayerTransferTask]
    layer_id: int
    attention_start_gate: AttentionComputeStartGate | None = None
