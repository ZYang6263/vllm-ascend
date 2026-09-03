# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import asyncio
from collections import Counter
from types import SimpleNamespace

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from vllm_ascend.ascend_config import ComputeAwareRoutingConfig
from vllm_ascend.core.compute_aware_routing import (
    PREFILL_PROGRESS_OUTPUT_ID,
    PREFILL_PROGRESS_PAYLOAD_KEY,
)
from vllm_ascend.patch.platform.patch_compute_aware_routing import _append_prefill_progress


def _make_client(*, shadow_mode: bool = False) -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.engine_inflight = Counter()
    client.core_engines = [b"engine-0", b"engine-1"]
    client.lb_engines = [[0, 0, 0.0], [0, 0, 0.0]]
    client.eng_start_index = 0
    client._ascend_compute_aware_routing_config = ComputeAwareRoutingConfig(
        enabled=True,
        shadow_mode=shadow_mode,
    )
    client._ascend_max_num_batched_tokens = 4096
    client._ascend_request_prefill_tokens = {}
    client._ascend_request_prefill_engine = {}
    client._ascend_engine_prefill_tokens = {}
    client._ascend_compute_routing_decisions = 0
    client._ascend_compute_routing_fallbacks = 0
    client._ascend_compute_routing_shadow_disagreements = 0
    return client


def _make_request(request_id: str, prompt_tokens: int):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_tokens)),
        prompt_embeds=None,
        mm_features=None,
        sampling_params=SimpleNamespace(max_tokens=32),
        pooling_params=None,
        data_parallel_rank=None,
    )


def test_active_routing_spreads_long_prompt_burst():
    client = _make_client()

    first_engine = client.get_core_engine_for_request(_make_request("long-0", 4096))
    second_engine = client.get_core_engine_for_request(_make_request("long-1", 4096))

    assert first_engine == client.core_engines[0]
    assert second_engine == client.core_engines[1]
    assert client._ascend_engine_prefill_tokens[first_engine] == 4096
    assert client._ascend_engine_prefill_tokens[second_engine] == 4096


def test_first_output_zeros_local_prefill_debt_as_compatibility_fallback():
    client = _make_client()
    request = _make_request("request-0", 2048)
    engine = client.get_core_engine_for_request(request)

    outputs = EngineCoreOutputs(
        outputs=[EngineCoreOutput(request.request_id, [10])],
    )
    asyncio.run(client.process_engine_outputs(client, outputs))

    assert engine not in client._ascend_engine_prefill_tokens
    assert client._ascend_request_prefill_tokens[request.request_id] == 0


def test_engine_progress_updates_partial_debt_and_is_not_forwarded():
    client = _make_client()
    request = _make_request("request-0", 2048)
    engine = client.get_core_engine_for_request(request)
    outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                PREFILL_PROGRESS_OUTPUT_ID,
                [],
                kv_transfer_params={PREFILL_PROGRESS_PAYLOAD_KEY: [[request.request_id, 768]]},
            )
        ]
    )

    asyncio.run(client.process_engine_outputs(client, outputs))

    assert client._ascend_engine_prefill_tokens[engine] == 768
    assert client._ascend_request_prefill_tokens[request.request_id] == 768
    assert outputs.outputs == []


def test_engine_progress_can_restore_debt_after_preemption():
    client = _make_client()
    request = _make_request("request-0", 2048)
    engine = client.get_core_engine_for_request(request)

    zero_progress = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                PREFILL_PROGRESS_OUTPUT_ID,
                [],
                kv_transfer_params={PREFILL_PROGRESS_PAYLOAD_KEY: [[request.request_id, 0]]},
            )
        ]
    )
    reset_progress = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                PREFILL_PROGRESS_OUTPUT_ID,
                [],
                kv_transfer_params={PREFILL_PROGRESS_PAYLOAD_KEY: [[request.request_id, 2048]]},
            )
        ]
    )

    asyncio.run(client.process_engine_outputs(client, zero_progress))
    asyncio.run(client.process_engine_outputs(client, reset_progress))

    assert client._ascend_engine_prefill_tokens[engine] == 2048


def test_finished_request_releases_local_prefill_debt():
    client = _make_client()
    request = _make_request("request-0", 2048)
    engine = client.get_core_engine_for_request(request)

    outputs = EngineCoreOutputs(finished_requests={request.request_id})
    asyncio.run(client.process_engine_outputs(client, outputs))

    assert engine not in client._ascend_engine_prefill_tokens
    assert client.engine_inflight[engine] == 0


def test_engine_appends_absolute_chunk_progress_for_originating_client():
    request = SimpleNamespace(
        request_id="request-0",
        client_index=3,
        sampling_params=object(),
        num_prompt_tokens=4096,
        num_computed_tokens=3072,
        num_in_flight_tokens=1024,
    )
    scheduler = SimpleNamespace(
        requests={request.request_id: request},
        _ascend_reported_prefill_remaining={},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: 1024},
        preempted_req_ids=None,
    )
    engine_outputs = {}

    _append_prefill_progress(scheduler, scheduler_output, engine_outputs)

    progress_output = engine_outputs[3].outputs[0]
    assert progress_output.request_id == PREFILL_PROGRESS_OUTPUT_ID
    assert progress_output.kv_transfer_params == {PREFILL_PROGRESS_PAYLOAD_KEY: [[request.request_id, 2048]]}


def test_engine_reports_increased_prefill_debt_after_preemption():
    request = SimpleNamespace(
        request_id="request-0",
        client_index=0,
        sampling_params=object(),
        num_prompt_tokens=4096,
        num_computed_tokens=4096,
        num_in_flight_tokens=0,
    )
    scheduler = SimpleNamespace(
        requests={request.request_id: request},
        _ascend_reported_prefill_remaining={},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={request.request_id: 1},
        preempted_req_ids=None,
    )

    first_outputs = {}
    _append_prefill_progress(scheduler, scheduler_output, first_outputs)
    request.num_computed_tokens = 0
    preempted_outputs = {}
    preemption_output = SimpleNamespace(
        num_scheduled_tokens={},
        preempted_req_ids={request.request_id},
    )
    _append_prefill_progress(scheduler, preemption_output, preempted_outputs)

    payload = preempted_outputs[0].outputs[0].kv_transfer_params
    assert payload == {PREFILL_PROGRESS_PAYLOAD_KEY: [[request.request_id, 4096]]}


def test_engine_forgets_aborted_request_without_scheduled_output():
    scheduler = SimpleNamespace(
        requests={},
        _ascend_reported_prefill_remaining={"request-0": 1024},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        preempted_req_ids=None,
    )
    engine_outputs = {0: EngineCoreOutputs(finished_requests={"request-0"})}

    _append_prefill_progress(scheduler, scheduler_output, engine_outputs)

    assert scheduler._ascend_reported_prefill_remaining == {}


def test_shadow_mode_keeps_upstream_choice_and_records_disagreement():
    client = _make_client(shadow_mode=True)
    client.lb_engines = [[0, 0, 0.0], [0, 10, 0.0]]
    client._ascend_engine_prefill_tokens[client.core_engines[0]] = 81920

    chosen_engine = client.get_core_engine_for_request(_make_request("request-0", 16))

    assert chosen_engine == client.core_engines[0]
    assert client._ascend_compute_routing_shadow_disagreements == 1


def test_multimodal_request_falls_back_to_upstream_router():
    client = _make_client()
    request = _make_request("request-0", 16)
    request.mm_features = [object()]

    chosen_engine = client.get_core_engine_for_request(request)

    assert chosen_engine == client.core_engines[0]
    assert client._ascend_compute_routing_fallbacks == 1


def test_missing_config_keeps_upstream_routing_path():
    client = _make_client()
    del client._ascend_compute_aware_routing_config

    chosen_engine = client.get_core_engine_for_request(_make_request("request-0", 16))

    assert chosen_engine == client.core_engines[0]
    assert client.lb_engines[0][0] == 1
