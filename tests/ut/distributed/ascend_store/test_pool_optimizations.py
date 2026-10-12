#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the KV pool Python-side hot-path optimizations.

Covers the store-path skip-chunk-range iteration (including shard candidate
invariance), the vectorized batch address builder, the reachable-mask result
cache, the single-rank lookup key expansion fast path, and the scheduler-side
per-request pool lookup result cache.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
# isort: on


@pytest.fixture(autouse=True)
def _patch_pool_scheduler_importlib():
    """KVPoolScheduler resolves its backend dynamically via
    ``importlib.import_module``; point it at a MagicMock so the scheduler's
    ``store_scheduler`` is a mock (mirrors test_pool_scheduler.py)."""
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib:
        mock_importlib.import_module.return_value = MagicMock()
        yield


_BLOCK_SIZE = 16


def _full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _make_config(kv_role="kv_producer", extra_config=None, block_size=16):
    config = MagicMock()
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = block_size
    config.cache_config.hash_block_size = block_size
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    return config


class RecordingFakeStore:
    def __init__(self, exists_result=None):
        self.requires_exists_before_put = True
        self.exists_result = exists_result or []
        self.exists_calls = []
        self.put_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        self.exists_calls.append(list(keys))
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))


class TestStoreSkipSpanMath(unittest.TestCase):
    """The chunk-id span passed as skip_chunk_range must reject exactly the
    chunks the store path's should_skip predicate rejects (unclamped chunk
    end, as chunk_filter evaluates it)."""

    def test_span_matches_should_skip_predicate(self):
        for skip_start in range(0, 130, 7):
            for skip_end in range(0, 260, 5):
                if skip_end <= skip_start:
                    continue
                with self.subTest(skip_start=skip_start, skip_end=skip_end):
                    span_lo = -(-skip_start // _BLOCK_SIZE)
                    span_hi = skip_end // _BLOCK_SIZE
                    for chunk in range(16):
                        start = chunk * _BLOCK_SIZE
                        end = start + _BLOCK_SIZE
                        skipped_by_predicate = start >= skip_start and end <= skip_end
                        skipped_by_span = span_lo <= chunk < span_hi
                        self.assertEqual(skipped_by_predicate, skipped_by_span)


class TestSkipChunkRangeIteration(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("m", 0, 0, 0)
        self.db = ChunkedTokenDatabase([self.meta], block_size=[_BLOCK_SIZE], partitions=None)
        self.hashes = [f"h{idx}" for idx in range(12)]
        self.block_ids = list(range(100, 112))

    @staticmethod
    def _filter_with_skip(skip_start: int, skip_end: int, allowed_mask=None):
        def chunk_filter(start: int) -> bool:
            block_idx = start // _BLOCK_SIZE
            if allowed_mask is not None and not (block_idx < len(allowed_mask) and allowed_mask[block_idx]):
                return False
            chunk_start = block_idx * _BLOCK_SIZE
            return not (skip_end > skip_start and chunk_start >= skip_start and chunk_start + _BLOCK_SIZE <= skip_end)

        return chunk_filter

    @staticmethod
    def _filter_mask_only(allowed_mask=None):
        def chunk_filter(start: int) -> bool:
            if allowed_mask is None:
                return True
            block_idx = start // _BLOCK_SIZE
            return block_idx < len(allowed_mask) and allowed_mask[block_idx]

        return chunk_filter

    def _span(self, skip_start: int, skip_end: int):
        if skip_end <= skip_start:
            return None
        return (-(-skip_start // _BLOCK_SIZE), skip_end // _BLOCK_SIZE)

    def test_skip_range_split_matches_combined_filter(self):
        cases = [
            (0, 0),
            (0, 64),
            (16, 64),
            (23, 61),
            (0, 192),
            (5, 7),
            (100, 190),
            (48, 1000),
            (16, 16),
        ]
        masks = [None, [True] * 12, [True, False] * 6]
        for skip_start, skip_end in cases:
            for allowed_mask in masks:
                with self.subTest(skip_start=skip_start, skip_end=skip_end, mask=bool(allowed_mask)):
                    baseline = list(
                        self.db.process_token_key_strings_with_block_ids(
                            192,
                            self.hashes,
                            self.block_ids,
                            chunk_filter=self._filter_with_skip(skip_start, skip_end, allowed_mask),
                        )
                    )
                    optimized = list(
                        self.db.process_token_key_strings_with_block_ids(
                            192,
                            self.hashes,
                            self.block_ids,
                            chunk_filter=self._filter_mask_only(allowed_mask),
                            skip_chunk_range=self._span(skip_start, skip_end),
                        )
                    )
                    self.assertEqual(baseline, optimized)

    def test_skip_range_preserves_shard_candidates(self):
        for skip_start, skip_end in ((16, 96), (23, 61), (0, 32)):
            for shard_size in (2, 3):
                for shard_rank in range(shard_size):
                    with self.subTest(skip=(skip_start, skip_end), shard=shard_rank, size=shard_size):
                        baseline = list(
                            self.db.process_token_key_strings_with_block_ids(
                                192,
                                self.hashes,
                                self.block_ids,
                                chunk_filter=self._filter_with_skip(skip_start, skip_end),
                                shard_rank=shard_rank,
                                shard_size=shard_size,
                            )
                        )
                        optimized = list(
                            self.db.process_token_key_strings_with_block_ids(
                                192,
                                self.hashes,
                                self.block_ids,
                                chunk_filter=self._filter_mask_only(),
                                skip_chunk_range=self._span(skip_start, skip_end),
                                shard_rank=shard_rank,
                                shard_size=shard_size,
                            )
                        )
                        self.assertEqual(baseline, optimized)

    def test_skip_range_with_mask_num(self):
        baseline = list(
            self.db.process_token_key_strings(
                192,
                self.hashes,
                mask_num=16,
                chunk_filter=self._filter_with_skip(32, 96),
            )
        )
        optimized = list(
            self.db.process_token_key_strings(
                192,
                self.hashes,
                mask_num=16,
                chunk_filter=self._filter_mask_only(),
                skip_chunk_range=self._span(32, 96),
            )
        )
        self.assertEqual(baseline, optimized)


class TestPrepareValuesBatch(unittest.TestCase):
    def _make_db(self, base_addrs, block_len, stride):
        db = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], block_size=[_BLOCK_SIZE], partitions=None)
        db.set_group_buffers(
            {0: base_addrs},
            {0: block_len},
            {0: stride} if stride is not None else None,
            group_num_layers={0: 1},
        )
        return db

    def test_batch_matches_per_chunk_prepare_value(self):
        layouts = [
            ([1000, 2000, 3000], [32], None),
            ([1000, 2000, 3000, 4000], [16, 32], [4, 8]),
            ([1000], [16], [1]),
            ([1000, 2000], [16, 16], [2, 2]),
        ]
        for base_addrs, block_len, stride in layouts:
            with self.subTest(base_addrs=base_addrs, block_len=block_len, stride=stride):
                db = self._make_db(base_addrs, block_len, stride)
                starts = [0, 16, 32]
                ends = [16, 32, 48]
                key_block_ids = [3, 0, 17]
                addrs, sizes = db.prepare_values_batch(starts, ends, key_block_ids)
                self.assertEqual(len(addrs), len(starts))
                self.assertEqual(len(sizes), len(starts))
                for index in range(len(starts)):
                    expected_addrs, expected_sizes, _ = db.prepare_value(
                        starts[index],
                        ends[index],
                        key_block_ids,
                        kv_cache_group_id=0,
                        block_id=key_block_ids[index],
                    )
                    self.assertEqual(addrs[index], expected_addrs)
                    self.assertEqual(sizes[index], expected_sizes)

    def test_partial_chunk_tokens_and_empty_input(self):
        db = self._make_db([1000, 2000], [32], None)
        addrs, sizes = db.prepare_values_batch([0, 8], [8, 16], [5, 6])
        # Both chunks span 8 tokens: 32 bytes / 16 tokens * 8 tokens = 16 per entry.
        self.assertEqual(sizes[0], [16, 16])
        self.assertEqual(sizes[1], [16, 16])
        expected_addrs, expected_sizes, _ = db.prepare_value(8, 16, [5, 6], kv_cache_group_id=0, block_id=6)
        self.assertEqual(addrs[1], expected_addrs)
        self.assertEqual(sizes[1], expected_sizes)

        self.assertEqual(db.prepare_values_batch([], [], []), ([], []))


class TestReachableMaskCache(unittest.TestCase):
    def _make_coordinator(self):
        return AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _full_spec(128))],
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )

    def test_store_mask_computed_once_per_shape(self):
        coordinator = self._make_coordinator()
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[True, True, True, True],
        ) as mask_fn:
            first = coordinator.store_mask(512)
            second = coordinator.store_mask(512)
        self.assertEqual(first, second)
        self.assertEqual(mask_fn.call_count, 1)

    def test_distinct_shapes_and_prompt_lengths_cached_separately(self):
        coordinator = self._make_coordinator()
        results = [[True, True], [False, False], [True, False, True, False]]
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            side_effect=results,
        ) as mask_fn:
            first = coordinator.store_mask(256, num_prompt_tokens=256)
            second = coordinator.store_mask(256, num_prompt_tokens=512)
            again = coordinator.store_mask(256, num_prompt_tokens=256)
        self.assertEqual(first, ([True, True],))
        self.assertEqual(second, ([False, False],))
        self.assertEqual(again, first)
        self.assertEqual(mask_fn.call_count, 2)

    def test_lookup_mask_reuses_cached_masks(self):
        coordinator = self._make_coordinator()
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[True, False, True, True],
        ) as mask_fn:
            first = coordinator.lookup_mask(512)
            second = coordinator.lookup_mask(512)
        self.assertEqual(first, second)
        self.assertEqual(mask_fn.call_count, 1)

    def test_cache_bound_still_correct_after_eviction(self):
        coordinator = self._make_coordinator()
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            side_effect=lambda manager_cls, **kwargs: [True] * kwargs["end_block"],
        ):
            for num_chunks in range(1, 140):
                aligned_len = num_chunks * 128
                self.assertEqual(coordinator.store_mask(aligned_len), ([True] * num_chunks,))
            self.assertGreater(len(coordinator._reachable_mask_cache), 0)
            # The earliest entry has been evicted; recomputing must stay correct.
            self.assertEqual(coordinator.store_mask(128), ([True],))
            self.assertEqual(coordinator.store_mask(128), ([True],))
        self.assertLessEqual(len(coordinator._reachable_mask_cache), 128)


def _load_pool_worker_class():
    try:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        return KVPoolWorker
    except ImportError:
        return None


class TestIdentityRankExpansion(unittest.TestCase):
    """The single-variant fast path of _expand_lookup_keys_by_rank."""

    KEY = "m@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@hash0"

    def _make_worker(self, *, head_or_tp_rank=0, pp_size=1, dcp_size=1, tp_variants=1):
        pool_worker_cls = _load_pool_worker_class()
        if pool_worker_cls is None:
            self.skipTest("pool_worker import requires a full vllm installation (run in CI)")
        worker = pool_worker_cls.__new__(pool_worker_cls)
        worker._identity_rank_expansion_cache = {}
        worker.token_database = SimpleNamespace(
            metadata=[KeyMetadata("m", head_or_tp_rank=head_or_tp_rank, dcp_rank=0, pp_rank=0)]
        )
        worker.pp_size = pp_size
        worker.dcp_size = dcp_size
        worker.get_group_tp_size = lambda group_id: tp_variants
        return worker

    def test_single_variant_returns_keys_unchanged(self):
        worker = self._make_worker()
        keys = [self.KEY]
        self.assertIs(worker._expand_lookup_keys_by_rank(keys, 0), keys)

    def test_nonzero_local_rank_still_expands(self):
        worker = self._make_worker(head_or_tp_rank=2)
        expanded = worker._expand_lookup_keys_by_rank([self.KEY.replace("head_or_tp_rank:0", "head_or_tp_rank:2")], 0)
        self.assertEqual(expanded, [self.KEY])

    def test_multiple_variants_expand(self):
        worker = self._make_worker(tp_variants=2)
        expanded = worker._expand_lookup_keys_by_rank([self.KEY], 0)
        self.assertEqual(
            expanded,
            [
                self.KEY,
                self.KEY.replace("head_or_tp_rank:0", "head_or_tp_rank:1"),
            ],
        )

    def test_pp_stages_expand(self):
        worker = self._make_worker(pp_size=2)
        expanded = worker._expand_lookup_keys_by_rank([self.KEY], 0)
        self.assertEqual(
            expanded,
            [
                self.KEY,
                self.KEY.replace("pp_rank:0", "pp_rank:1"),
            ],
        )


class TestHitLookupCache(unittest.TestCase):
    def _make_request(self, request_id="r1"):
        return MagicMock(
            prompt_token_ids=list(range(64)),
            num_tokens=64,
            request_id=request_id,
            block_hashes=[b"h"] * 4,
        )

    def _make_scheduler_output(self, finished=(), preempted=()):
        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = set(finished)
        scheduler_output.preempted_req_ids = set(preempted)
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock(req_ids=[], new_block_ids=[])
        return scheduler_output

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_repeated_lookup_reuses_cached_result(self, mock_client_cls):
        scheduler = KVPoolScheduler(_make_config(), use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48
        request = self._make_request()

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (48, False))
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (48, False))

        self.assertEqual(mock_client_cls.return_value.lookup.call_count, 1)
        self.assertIn("r1", scheduler._hit_lookup_cache)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_changed_computed_tokens_triggers_relookup(self, mock_client_cls):
        scheduler = KVPoolScheduler(_make_config(), use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48
        request = self._make_request()

        scheduler.get_num_new_matched_tokens(request, 16)
        scheduler.get_num_new_matched_tokens(request, 16)
        scheduler.get_num_new_matched_tokens(request, 32)

        self.assertEqual(mock_client_cls.return_value.lookup.call_count, 2)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_finished_request_invalidates_cache(self, mock_client_cls):
        scheduler = KVPoolScheduler(_make_config(), use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48
        request = self._make_request()

        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.build_connector_meta(self._make_scheduler_output(finished={"r1"}))
        self.assertNotIn("r1", scheduler._hit_lookup_cache)

        scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(mock_client_cls.return_value.lookup.call_count, 2)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_preempted_request_invalidates_cache(self, mock_client_cls):
        scheduler = KVPoolScheduler(_make_config(), use_layerwise=False)
        mock_client_cls.return_value.lookup.return_value = 48
        request = self._make_request()

        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.build_connector_meta(self._make_scheduler_output(preempted={"r1"}))
        self.assertNotIn("r1", scheduler._hit_lookup_cache)


class TestStoreThreadSkipAndBatch(unittest.TestCase):
    def _make_thread(self, store, block_size=_BLOCK_SIZE):
        db = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], block_size=[block_size], partitions=None)
        db.set_group_buffers(
            {0: [1000, 2000]},
            {0: [32]},
            {0: [4, 8]},
            group_num_layers={0: 1},
        )
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=block_size,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        thread.pcp_rank = 0
        thread.pcp_size = 1
        return thread, db

    def test_put_addrs_match_per_chunk_prepare_value(self):
        store = RecordingFakeStore([0, 0, 0, 0])
        thread, db = self._make_thread(store)
        block_ids = [10, 11, 12, 13]
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=block_ids,
            block_hashes=[f"h{idx}" for idx in range(4)],
            current_event=None,
        )
        thread.request_queue.put(req)

        thread._handle_request(req)

        keys, addrs, sizes = store.put_calls[0]
        self.assertEqual(len(keys), 4)
        for index in range(4):
            expected_addrs, expected_sizes, _ = db.prepare_value(
                index * _BLOCK_SIZE,
                (index + 1) * _BLOCK_SIZE,
                block_ids,
                kv_cache_group_id=0,
                block_id=block_ids[index],
            )
            self.assertEqual(addrs[index], expected_addrs)
            self.assertEqual(sizes[index], expected_sizes)

    def test_pooled_prefix_skipped_from_exists_and_put(self):
        store = RecordingFakeStore([0, 0])
        thread, db = self._make_thread(store)
        thread.add_stored_request("r1")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[10, 11, 12, 13],
            block_hashes=[f"h{idx}" for idx in range(4)],
            current_event=None,
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True),
        )
        thread.request_queue.put(req)

        thread._handle_request(req)

        # Chunks 0-1 (tokens 0-32) are already pooled: exists and put only see
        # the trailing two chunks.
        self.assertEqual(len(store.exists_calls[0]), 2)
        keys, _, _ = store.put_calls[0]
        self.assertEqual([key.rsplit("@", 1)[1] for key in keys], ["h2", "h3"])


if __name__ == "__main__":
    unittest.main()
