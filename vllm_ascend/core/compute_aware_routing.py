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

from __future__ import annotations

import math
from collections.abc import Sequence

# EngineCoreOutput has no plugin extension field. The Engine uses this reserved
# output and payload key to carry progress over the existing Engine -> API
# channel; the API-side patch consumes the output before normal processing.
PREFILL_PROGRESS_OUTPUT_ID = "__vllm_ascend_prefill_progress_v1__"
PREFILL_PROGRESS_PAYLOAD_KEY = "vllm_ascend.prefill_progress.v1"


def remaining_prefill_tokens(
    num_prompt_tokens: int,
    num_computed_tokens: int,
    num_in_flight_tokens: int,
) -> int:
    """Return prompt tokens not yet confirmed complete by EngineCore."""
    confirmed_tokens = max(0, num_computed_tokens - num_in_flight_tokens)
    confirmed_prompt_tokens = min(num_prompt_tokens, confirmed_tokens)
    return max(0, num_prompt_tokens - confirmed_prompt_tokens)


def estimate_engine_score(
    engine_snapshot: Sequence[int | float],
    local_inflight_count: int,
    local_prefill_tokens: int,
    client_count: int,
    max_num_batched_tokens: int,
) -> float | None:
    """Estimate load using upstream safeguards plus outstanding prefill work."""
    if len(engine_snapshot) < 3 or client_count <= 0 or max_num_batched_tokens <= 0:
        return None

    try:
        waiting = float(engine_snapshot[0])
        running = float(engine_snapshot[1])
        kv_cache_usage = float(engine_snapshot[2])
        local_inflight = float(local_inflight_count)
        local_prefill = float(local_prefill_tokens)
    except (TypeError, ValueError):
        return None

    values = (
        waiting,
        running,
        kv_cache_usage,
        local_inflight,
        local_prefill,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if waiting < 0 or running < 0 or not 0 <= kv_cache_usage <= 1 or local_inflight < 0 or local_prefill < 0:
        return None

    # Preserve upstream's request-count floor and KV-pressure guardrail.
    score = max(client_count * local_inflight, waiting + running)
    if waiting:
        score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)

    # Coordinator snapshots stay unchanged. Scale this API server's local
    # prompt debt using the same multi-client approximation as vLLM's local
    # in-flight request-count floor.
    estimated_prefill = client_count * local_prefill
    return score + estimated_prefill / max_num_batched_tokens


def select_engine_index(
    engine_snapshots: Sequence[Sequence[int | float]],
    local_inflight_counts: Sequence[int],
    local_prefill_tokens: Sequence[int],
    client_count: int,
    max_num_batched_tokens: int,
    scan_start: int,
) -> int | None:
    """Select the lowest-score engine while preserving rotating tie breaks."""
    num_engines = len(engine_snapshots)
    if num_engines == 0 or len(local_inflight_counts) != num_engines or len(local_prefill_tokens) != num_engines:
        return None

    selected_index: int | None = None
    selected_score = math.inf
    for offset in range(num_engines):
        engine_index = (scan_start + offset) % num_engines
        score = estimate_engine_score(
            engine_snapshots[engine_index],
            local_inflight_counts[engine_index],
            local_prefill_tokens[engine_index],
            client_count,
            max_num_batched_tokens,
        )
        if score is None:
            return None
        if score < selected_score:
            selected_index = engine_index
            selected_score = score
    return selected_index
