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

"""Install Prefill-aware internal DP routing and Engine progress reports."""

from __future__ import annotations

from collections import defaultdict
from functools import wraps
from typing import Any

from vllm.logger import init_logger
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.core.compute_aware_routing import (
    PREFILL_PROGRESS_OUTPUT_ID,
    PREFILL_PROGRESS_PAYLOAD_KEY,
    remaining_prefill_tokens,
    select_engine_index,
)

logger = init_logger(__name__)

_SHADOW_SUMMARY_INTERVAL = 1000

_original_init = DPLBAsyncMPClient.__init__
_original_add_request_async = DPLBAsyncMPClient.add_request_async
_original_abort_requests_async = DPLBAsyncMPClient.abort_requests_async
_original_get_core_engine_for_request = DPLBAsyncMPClient.get_core_engine_for_request
_original_process_engine_outputs = DPLBAsyncMPClient.process_engine_outputs


def _append_prefill_progress(
    scheduler: Any,
    scheduler_output: Any,
    engine_core_outputs: dict[int, EngineCoreOutputs],
) -> None:
    """Append absolute Prefill progress to each request's API output."""
    reported_remaining: dict[str, int] = scheduler._ascend_reported_prefill_remaining
    request_ids = set(scheduler_output.num_scheduled_tokens)
    request_ids.update(getattr(scheduler_output, "preempted_req_ids", None) or ())

    finished_request_ids: set[str] = set()
    for outputs in engine_core_outputs.values():
        finished_request_ids.update(outputs.finished_requests or ())
        finished_request_ids.update(output.request_id for output in outputs.outputs if output.finished)
    for request_id in finished_request_ids:
        reported_remaining.pop(request_id, None)
    request_ids.difference_update(finished_request_ids)

    progress_by_client: dict[int, list[list[str | int]]] = defaultdict(list)
    for request_id in request_ids:
        request = scheduler.requests.get(request_id)
        if request is None or request.sampling_params is None or getattr(request, "mm_features", None):
            continue

        remaining = remaining_prefill_tokens(
            request.num_prompt_tokens,
            request.num_computed_tokens,
            request.num_in_flight_tokens,
        )
        if reported_remaining.get(request_id) == remaining:
            continue

        reported_remaining[request_id] = remaining
        progress_by_client[request.client_index].append([request_id, remaining])

    for client_index, progress in progress_by_client.items():
        outputs = engine_core_outputs.setdefault(client_index, EngineCoreOutputs())
        outputs.outputs.append(
            EngineCoreOutput(
                request_id=PREFILL_PROGRESS_OUTPUT_ID,
                new_token_ids=[],
                kv_transfer_params={PREFILL_PROGRESS_PAYLOAD_KEY: progress},
            )
        )


def _install_engine_progress_hook(engine_core: EngineCore) -> None:
    """Wrap the concrete Scheduler instance for sync and async step paths."""
    routing_config = init_ascend_config(engine_core.vllm_config).scheduler_config.compute_aware_routing_config
    if not routing_config.enabled:
        return

    scheduler = engine_core.scheduler
    scheduler._ascend_prefill_progress_enabled = True
    scheduler._ascend_reported_prefill_remaining = {}
    logger.info("Direct Engine-to-API Prefill progress reporting enabled.")
    scheduler_cls = type(scheduler)
    current_update_from_output = scheduler_cls.update_from_output
    if getattr(current_update_from_output, "_ascend_prefill_progress_hooked", False):
        return

    original_update_from_output = current_update_from_output

    @wraps(original_update_from_output)
    def _update_from_output(self, scheduler_output, model_output):
        engine_core_outputs = original_update_from_output(self, scheduler_output, model_output)
        if getattr(self, "_ascend_prefill_progress_enabled", False):
            _append_prefill_progress(self, scheduler_output, engine_core_outputs)
        return engine_core_outputs

    _update_from_output._ascend_prefill_progress_hooked = True
    scheduler_cls.update_from_output = _update_from_output


def _apply_engine_core_patch() -> None:
    """Install the Engine hook once in both parent and spawned processes."""
    if getattr(EngineCore, "_ascend_prefill_progress_patched", False):
        return

    original_engine_core_init = EngineCore.__init__

    @wraps(original_engine_core_init)
    def _engine_core_init(self, *args, **kwargs):
        original_engine_core_init(self, *args, **kwargs)
        _install_engine_progress_hook(self)

    EngineCore.__init__ = _engine_core_init
    EngineCore._ascend_prefill_progress_patched = True


_apply_engine_core_patch()

# Multiprocessing spawn starts a fresh interpreter. Wrapping the subprocess
# entry point makes importing this module re-apply the EngineCore patch before
# an EngineCore instance is constructed. Preserve earlier wrappers (notably
# balance scheduling) by delegating to the method visible at import time.
_original_run_engine_core = EngineCoreProc.run_engine_core


def _compute_aware_run_engine_core(*args, **kwargs):
    _apply_engine_core_patch()
    return _original_run_engine_core(*args, **kwargs)


EngineCoreProc.run_engine_core = staticmethod(_compute_aware_run_engine_core)


def _routing_enabled(client: DPLBAsyncMPClient) -> bool:
    config = getattr(client, "_ascend_compute_aware_routing_config", None)
    return bool(config is not None and config.enabled)


def _request_prefill_tokens(request: Any) -> int | None:
    """Get text-prompt length without inspecting device data."""
    if getattr(request, "sampling_params", None) is None or getattr(request, "mm_features", None):
        return None
    try:
        return length_from_prompt_token_ids_or_embeds(
            getattr(request, "prompt_token_ids", None),
            getattr(request, "prompt_embeds", None),
        )
    except (TypeError, ValueError):
        return None


def _finish_prefill_debt(client: DPLBAsyncMPClient, request_id: str) -> None:
    prompt_tokens = client._ascend_request_prefill_tokens.pop(request_id, None)
    engine = client._ascend_request_prefill_engine.pop(request_id, None)
    if prompt_tokens is None or engine is None:
        return

    remaining = client._ascend_engine_prefill_tokens.get(engine, 0) - prompt_tokens
    if remaining <= 0:
        client._ascend_engine_prefill_tokens.pop(engine, None)
    else:
        client._ascend_engine_prefill_tokens[engine] = remaining


def _update_prefill_debt(client: DPLBAsyncMPClient, request_id: str, remaining_tokens: int) -> None:
    old_remaining = client._ascend_request_prefill_tokens.get(request_id)
    engine = client._ascend_request_prefill_engine.get(request_id)
    if old_remaining is None or engine is None:
        return

    # Engine messages are absolute, which makes duplicate delivery idempotent.
    # Keep the zero-valued request entry until completion so a later preemption
    # can restore Prefill debt for a request that must recompute its prompt.
    remaining_tokens = max(0, int(remaining_tokens))
    client._ascend_request_prefill_tokens[request_id] = remaining_tokens
    engine_remaining = client._ascend_engine_prefill_tokens.get(engine, 0) + remaining_tokens - old_remaining
    if engine_remaining <= 0:
        client._ascend_engine_prefill_tokens.pop(engine, None)
    else:
        client._ascend_engine_prefill_tokens[engine] = engine_remaining


def _track_prefill_debt(
    client: DPLBAsyncMPClient,
    request_id: str,
    engine: Any,
    prompt_tokens: int | None,
) -> None:
    if prompt_tokens is None:
        return
    _finish_prefill_debt(client, request_id)
    client._ascend_request_prefill_tokens[request_id] = prompt_tokens
    client._ascend_request_prefill_engine[request_id] = engine
    client._ascend_engine_prefill_tokens[engine] = client._ascend_engine_prefill_tokens.get(engine, 0) + prompt_tokens


def _drop_removed_engine_debt(client: DPLBAsyncMPClient) -> None:
    live_engines = set(client.core_engines)
    removed_engines = set(client._ascend_engine_prefill_tokens) - live_engines
    for engine in removed_engines:
        client._ascend_engine_prefill_tokens.pop(engine, None)
    if not removed_engines:
        return
    for request_id, engine in list(client._ascend_request_prefill_engine.items()):
        if engine in removed_engines:
            client._ascend_request_prefill_engine.pop(request_id, None)
            client._ascend_request_prefill_tokens.pop(request_id, None)


@wraps(_original_init)
def _patched_init(self: DPLBAsyncMPClient, *args, **kwargs) -> None:
    _original_init(self, *args, **kwargs)

    vllm_config = kwargs.get("vllm_config")
    if vllm_config is None and args:
        vllm_config = args[0]
    if vllm_config is None:
        raise TypeError("DPLBAsyncMPClient requires vllm_config")

    self._ascend_compute_aware_routing_config = init_ascend_config(
        vllm_config
    ).scheduler_config.compute_aware_routing_config
    if not self._ascend_compute_aware_routing_config.enabled:
        return

    self._ascend_max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    self._ascend_request_prefill_tokens: dict[str, int] = {}
    self._ascend_request_prefill_engine: dict[str, Any] = {}
    self._ascend_engine_prefill_tokens: dict[Any, int] = {}
    self._ascend_compute_routing_decisions = 0
    self._ascend_compute_routing_fallbacks = 0
    self._ascend_compute_routing_shadow_disagreements = 0
    logger.info(
        "Prefill-aware DP routing enabled (shadow_mode=%s).",
        self._ascend_compute_aware_routing_config.shadow_mode,
    )


@wraps(_original_get_core_engine_for_request)
def _patched_get_core_engine_for_request(self: DPLBAsyncMPClient, request):
    if not _routing_enabled(self):
        return _original_get_core_engine_for_request(self, request)

    prompt_tokens = _request_prefill_tokens(request)
    # Preserve explicit-rank, pooling/late-interaction, and unsupported
    # multimodal behavior from upstream.
    if request.data_parallel_rank is not None or prompt_tokens is None:
        chosen_engine = _original_get_core_engine_for_request(self, request)
        _track_prefill_debt(
            self,
            request.request_id,
            chosen_engine,
            prompt_tokens,
        )
        if prompt_tokens is None and request.data_parallel_rank is None:
            self._ascend_compute_routing_fallbacks += 1
        return chosen_engine

    _drop_removed_engine_debt(self)
    local_inflight_counts = [self.engine_inflight[engine] for engine in self.core_engines]
    local_prefill_tokens = [self._ascend_engine_prefill_tokens.get(engine, 0) for engine in self.core_engines]
    computed_index = select_engine_index(
        self.lb_engines,
        local_inflight_counts,
        local_prefill_tokens,
        self.client_count,
        self._ascend_max_num_batched_tokens,
        self.eng_start_index,
    )
    if computed_index is None:
        self._ascend_compute_routing_fallbacks += 1
        chosen_engine = _original_get_core_engine_for_request(self, request)
        _track_prefill_debt(
            self,
            request.request_id,
            chosen_engine,
            prompt_tokens,
        )
        return chosen_engine

    computed_engine = self.core_engines[computed_index]
    config = self._ascend_compute_aware_routing_config
    if config.shadow_mode:
        chosen_engine = _original_get_core_engine_for_request(self, request)
        self._ascend_compute_routing_decisions += 1
        if computed_engine != chosen_engine:
            self._ascend_compute_routing_shadow_disagreements += 1
        logger.debug(
            "API-local Prefill-aware DP routing shadow decision: request=%s current=%s prefill_aware=%s",
            request.request_id,
            chosen_engine,
            computed_engine,
        )
        if self._ascend_compute_routing_decisions % _SHADOW_SUMMARY_INTERVAL == 0:
            logger.info(
                "API-local Prefill-aware routing shadow summary: decisions=%d, disagreements=%d, fallbacks=%d.",
                self._ascend_compute_routing_decisions,
                self._ascend_compute_routing_shadow_disagreements,
                self._ascend_compute_routing_fallbacks,
            )
    else:
        # Reuse upstream explicit-rank handling for request and abort routing.
        request.data_parallel_rank = computed_index
        try:
            chosen_engine = _original_get_core_engine_for_request(self, request)
        finally:
            request.data_parallel_rank = None

        # The explicit-rank path skips upstream's optimistic count and tie
        # rotation; reproduce them for automatic Prefill-aware decisions.
        self.lb_engines[computed_index][0] += self.client_count
        self.eng_start_index = (self.eng_start_index + 1) % len(self.core_engines)
        self._ascend_compute_routing_decisions += 1

    _track_prefill_debt(
        self,
        request.request_id,
        chosen_engine,
        prompt_tokens,
    )
    return chosen_engine


@wraps(_original_add_request_async)
async def _patched_add_request_async(self: DPLBAsyncMPClient, request) -> None:
    try:
        await _original_add_request_async(self, request)
    except Exception:
        if _routing_enabled(self):
            _finish_prefill_debt(self, request.request_id)
            engine = self.reqs_in_flight.pop(request.request_id, None)
            if engine is not None:
                self.engine_inflight[engine] -= 1
        raise


@wraps(_original_abort_requests_async)
async def _patched_abort_requests_async(self: DPLBAsyncMPClient, request_ids: list[str]) -> None:
    await _original_abort_requests_async(self, request_ids)
    if _routing_enabled(self):
        for request_id in request_ids:
            _finish_prefill_debt(self, request_id)


@wraps(_original_process_engine_outputs)
async def _patched_process_engine_outputs(self: DPLBAsyncMPClient, outputs: EngineCoreOutputs) -> None:
    if _routing_enabled(self):
        regular_outputs: list[EngineCoreOutput] = []
        for output in outputs.outputs:
            progress_payload = None
            if output.request_id == PREFILL_PROGRESS_OUTPUT_ID and output.kv_transfer_params is not None:
                progress_payload = output.kv_transfer_params.get(PREFILL_PROGRESS_PAYLOAD_KEY)
            if progress_payload is not None:
                if isinstance(progress_payload, (list, tuple)):
                    for item in progress_payload:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            request_id, remaining_tokens = item
                            if isinstance(request_id, str) and isinstance(remaining_tokens, int):
                                _update_prefill_debt(self, request_id, remaining_tokens)
                continue

            regular_outputs.append(output)
            if output.new_token_ids:
                # Compatibility fallback for Engine versions that do not emit
                # the direct progress message.
                _update_prefill_debt(self, output.request_id, 0)

        outputs.outputs = regular_outputs

        finished_requests = set(outputs.finished_requests or ())
        finished_requests.update(output.request_id for output in outputs.outputs if output.finished)
        for request_id in finished_requests:
            _finish_prefill_debt(self, request_id)

    await _original_process_engine_outputs(self, outputs)


if not getattr(DPLBAsyncMPClient, "_ascend_compute_aware_routing_patched", False):
    DPLBAsyncMPClient.__init__ = _patched_init
    DPLBAsyncMPClient.add_request_async = _patched_add_request_async
    DPLBAsyncMPClient.abort_requests_async = _patched_abort_requests_async
    DPLBAsyncMPClient.get_core_engine_for_request = _patched_get_core_engine_for_request
    DPLBAsyncMPClient.process_engine_outputs = staticmethod(_patched_process_engine_outputs)
    DPLBAsyncMPClient._ascend_compute_aware_routing_patched = True
