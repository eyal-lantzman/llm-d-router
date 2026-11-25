# SPDX-License-Identifier: Apache-2.0
"""Write a synthetic sha256_cbor checkpoint vector from the pinned vLLM producer."""

import sys
from pathlib import Path

import msgspec
from tests.v1.core.hybrid_checkpoint.test_recovery_unit_events import (
    COST,
    UNIT_EVENTS,
    EventRecorder,
)
from tests.v1.core.hybrid_checkpoint.utils import EOS, make_hybrid_scheduler, run
from vllm import SamplingParams
from vllm.distributed.kv_events import KVEventBatch, RecoveryUnitStored
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

init_none_hash(sha256_cbor)
scheduler = make_hybrid_scheduler(
    num_blocks=94,
    ring=0,
    kv_events_config=UNIT_EVENTS,
    mamba_frontier_checkpoint=True,
    prefix_cache_eviction="gds",
    prefill_cost_coefficients=COST,
    prefix_cache_leaf_last=True,
    prefix_caching_hash_algo="sha256_cbor",
)
recorder = EventRecorder(scheduler)
scheduler.kv_event_publisher = recorder
params = SamplingParams(max_tokens=5, ignore_eos=True)
params.update_from_generation_config({}, EOS)
request = Request(
    request_id="parent",
    prompt_token_ids=[7 + i % 17 for i in range(100)],
    sampling_params=params,
    pooling_params=None,
    cache_salt="tenant",
    block_hasher=get_request_block_hasher(scheduler.hash_block_size, sha256_cbor),
)
scheduler.add_request(request)
run(scheduler, request)
recorder.take_pending()
stores = [event for event in recorder.events if isinstance(event, RecoveryUnitStored)]
assert [(event.num_tokens, event.frontier) for event in stores] == [
    (64, False),
    (104, True),
]
batch = KVEventBatch(ts=0.0, events=stores, data_parallel_rank=0)
vector = {"tokens": list(request.all_token_ids), "batch": msgspec.msgpack.encode(batch)}
Path(sys.argv[1]).write_bytes(msgspec.msgpack.encode(vector))
