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

from vllm_ascend.core.compute_aware_routing import (
    estimate_engine_score,
    remaining_prefill_tokens,
    select_engine_index,
)


def test_local_prefill_breaks_equal_request_count_tie():
    selected = select_engine_index(
        [[0, 2, 0.1], [0, 2, 0.1]],
        local_inflight_counts=[2, 2],
        local_prefill_tokens=[8192, 512],
        client_count=1,
        max_num_batched_tokens=4096,
        scan_start=0,
    )

    assert selected == 1


def test_client_count_scales_local_prefill_estimate():
    one_client = estimate_engine_score([0, 0, 0], 0, 4096, 1, 4096)
    four_clients = estimate_engine_score([0, 0, 0], 0, 4096, 4, 4096)

    assert one_client == 1
    assert four_clients == 4


def test_upstream_request_count_remains_a_load_floor():
    lightly_loaded = estimate_engine_score([0, 1, 0], 0, 0, 1, 4096)
    shared_loaded = estimate_engine_score([4, 4, 0], 0, 0, 1, 4096)

    assert lightly_loaded == 1
    assert shared_loaded == 8


def test_upstream_kv_pressure_penalty_is_preserved():
    low_kv = estimate_engine_score([2, 1, 0.2], 0, 0, 1, 4096)
    high_kv = estimate_engine_score([2, 1, 0.9], 0, 0, 1, 4096)

    assert low_kv is not None
    assert high_kv is not None
    assert high_kv > low_kv


def test_invalid_snapshot_requests_upstream_fallback():
    assert (
        select_engine_index(
            [[0, 1], [0, 1, 0]],
            [0, 0],
            [0, 0],
            1,
            4096,
            0,
        )
        is None
    )
    assert (
        select_engine_index(
            [[0, 1, float("nan")]],
            [0],
            [0],
            1,
            4096,
            0,
        )
        is None
    )


def test_rotating_start_controls_equal_score_tie():
    snapshots = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    assert select_engine_index(snapshots, [0] * 3, [0] * 3, 1, 4096, 2) == 2


def test_remaining_prefill_excludes_unfinished_async_steps():
    assert remaining_prefill_tokens(4096, 3072, 1024) == 2048
    assert remaining_prefill_tokens(4096, 4096, 0) == 0
    assert remaining_prefill_tokens(4096, 0, 0) == 4096
