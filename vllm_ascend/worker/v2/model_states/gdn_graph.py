# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata, UniformDecodeInputs


class GDNGraphMetadata(TargetDeviceMetadata):
    """Stable inputs and a graph-side metadata prologue for uniform GLM decode."""

    def __init__(self, accepted_state: torch.Tensor, query_len: int):
        super().__init__(wait_before_forward=True)
        batch = torch.empty(accepted_state.numel() + 1, dtype=torch.int64, device=accepted_state.device)
        self.inputs = UniformDecodeInputs(batch, accepted_state, query_len)
        # Keep asynchronous pinned uploads alive until their copy has completed.
        self._uploads: list[tuple[torch.Tensor, torch.npu.Event]] = []
        self._sources: dict[tuple[int, int], tuple] = {}

    def eligible(self, batch, scheduled_drafts, uniform_token_count: int | None) -> bool:
        if uniform_token_count != self.inputs.query_len or not batch.decode_graph_eligible:
            return False
        return (
            batch.num_tokens == len(batch.req_ids) * self.inputs.query_len
            and all(len(scheduled_drafts.get(req_id, ())) == self.inputs.query_len - 1 for req_id in batch.req_ids)
            and bool(np.all(batch.idx_mapping_np >= 0))
            and bool(np.all(batch.idx_mapping_np < self.inputs.accepted_state.numel()))
        )

    def stage(self, input_batch, block_tables, slot_mappings: torch.Tensor, *, for_capture: bool) -> None:
        if self.executor.submission_in_flight or self._tasks:
            raise RuntimeError("Previous GDN metadata must be retired before staging inputs")
        count = input_batch.num_reqs
        if not 0 < count < self.inputs.batch.numel():
            raise ValueError("GDN graph request count exceeds its persistent input capacity")
        if (
            input_batch.num_tokens != count * self.inputs.query_len
            or input_batch.num_tokens_after_padding != input_batch.num_reqs_after_padding * self.inputs.query_len
        ):
            raise ValueError("GDN graph requires the captured uniform decode layout")
        key = (input_batch.num_reqs_after_padding, input_batch.num_tokens_after_padding)
        sources = (
            input_batch.query_start_loc,
            input_batch.seq_lens,
            input_batch.positions,
            *block_tables,
            slot_mappings,
        )
        signature = tuple((tensor.data_ptr(), tensor.stride()) for tensor in sources)
        if for_capture:
            self._sources[key] = signature
        elif self._sources.get(key) != signature:
            raise RuntimeError("GDN graph inputs must retain their captured addresses and strides")
        host = torch.empty(self.inputs.batch.shape, dtype=torch.int64, device="cpu", pin_memory=True)
        values = host.numpy()
        values[0] = count
        values[1:] = -1
        values[1 : count + 1] = input_batch.idx_mapping_np[:count]
        self._uploads = [(buffer, event) for buffer, event in self._uploads if not event.query()]
        self.inputs.batch.copy_(host, non_blocking=True)
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream())
        self._uploads.append((host, ready))


def create_gdn_graph_metadata(config, model_state, attn_groups) -> GDNGraphMetadata | None:
    """Enable only the target layout whose producers share the padded contract."""
    # Lazy imports avoid the model-state / metadata-builder import cycle.
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadataBuilder,
        AscendIndexerKPoolTailMetadataBuilder,
    )
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder

    parallel = config.parallel_config
    speculative = config.speculative_config
    if (
        config.model_config.architecture != "Glm5NextForCausalLM"
        or config.cache_config.mamba_cache_mode != "align"
        or config.num_speculative_tokens <= 0
        or config.kv_transfer_config is not None
        or not get_ascend_config().enable_sparse_sfa_c8
        or not config.compilation_config.cudagraph_mode.separate_routine()
        or config.compilation_config.cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
        or any(
            size != 1
            for size in (
                parallel.prefill_context_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
            )
        )
        or speculative.enable_adaptive_verification
        or speculative.uses_dynamic_speculative_decoding()
    ):
        return None
    builders = [group.get_metadata_builder(0) for groups in attn_groups for group in groups]
    supported = (
        AscendGDNHostMetadataBuilder,
        AscendSFAMetadataBuilder,
        AscendIndexerKPoolMetadataBuilder,
        AscendIndexerKPoolTailMetadataBuilder,
    )
    if not builders or not all(type(builder) in supported for builder in builders):
        return None
    gdns = [builder for builder in builders if isinstance(builder, AscendGDNHostMetadataBuilder)]
    for builder in gdns:
        builder._ensure_conv_index_buffers()
        if builder._spec_conv_indices is None or builder.kv_cache_spec.num_speculative_blocks != builder.num_spec:
            return None
    if not gdns or any(
        isinstance(builder, AscendSFAMetadataBuilder)
        and not builder.nope
        or isinstance(builder, AscendIndexerKPoolMetadataBuilder)
        and builder.use_dsa_cp
        for builder in builders
    ):
        return None
    state = GDNGraphMetadata(model_state.num_accepted_tokens_gpu, config.num_speculative_tokens + 1)
    for builder in builders:
        builder.graph_decode_inputs = state.inputs
    return state
