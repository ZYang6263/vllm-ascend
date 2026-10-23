# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from copy import deepcopy
from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode

from tests.ut.ops.test_gdn_attn_builder import BatchSpec, _make_builder, create_common_attn_metadata
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder


def _assert_metadata_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif is_dataclass(expected):
        assert vars(actual).keys() == vars(expected).keys()
        for name, value in vars(expected).items():
            _assert_metadata_equal(getattr(actual, name), value)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "query_len,num_spec,projection_size", [(1, 0, 0), (1, 0, 11), (1, 3, 11), (2, 1, 11), (4, 3, 11), (5, 4, 11)]
)
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_fused_decode_matches_generic_metadata(query_len, num_spec, projection_size, graph_mode):
    device = torch.device("npu:0")
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_spec,
        builder_cls=AscendGDNHostMetadataBuilder,
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
        cudagraph_mode=graph_mode,
    )
    builder.vllm_config.model_config.dtype = torch.bfloat16
    builder.compilation_config.static_forward_context = {
        "layer0": SimpleNamespace(
            use_contiguous_conv_indices=True,
            use_shared_conv_output=projection_size > 0,
            local_projection_size=projection_size,
        )
    }
    # Exercise cache-block boundaries, strided inputs and shrinking batches.
    for count in (4, 1, 3):
        rows = 4 if graph_mode == CUDAGraphMode.FULL_DECODE_ONLY else count
        common = create_common_attn_metadata(BatchSpec([15, 16, 17, 32][:rows], [query_len] * rows), 16, device)
        common.is_prefilling.zero_()
        common.num_actual_tokens = count * query_len
        common.block_table_tensor = torch.arange(rows * 64, dtype=torch.int32, device=device).view(rows, 64)[:, ::2]
        common.seq_lens = torch.stack((common.seq_lens, common.seq_lens), dim=1)[:, 0]
        accepted = torch.tensor([query_len, 1, 1, 1, query_len, 1, 1, 1], dtype=torch.int32, device=device)[::2][:rows]
        drafts = torch.full((rows,), num_spec if query_len > 1 else -1, dtype=torch.int32) if num_spec else None
        if query_len == 1:
            accepted = None
        # The generic implementation is the oracle for every metadata field.
        with patch.object(builder, "_uniform_decode_query_len", return_value=None):
            generic = builder.build(0, common, accepted, drafts, num_actual_reqs=count)
        expected = deepcopy(generic)
        for name in (
            "spec_state_indices_tensor",
            "non_spec_state_indices_tensor",
            "spec_sequence_masks",
            "spec_query_start_loc",
            "non_spec_query_start_loc",
            "spec_actual_seq_lengths",
            "non_spec_actual_seq_lengths",
            "num_accepted_tokens",
            "spec_token_indx",
            "_spec_conv_indices",
        ):
            getattr(builder, name).fill_(12345)
        if builder._conv_output_buffer is not None:
            builder._conv_output_buffer.fill_(float("nan"))
        with patch.object(builder, "_build_uniform_decode", wraps=builder._build_uniform_decode) as fused:
            actual = builder.build(0, common, accepted, drafts, num_actual_reqs=count)
        fused.assert_called_once()
        _assert_metadata_equal(actual, expected)
        conv = (
            actual.spec_decode_metadata.spec_causal_conv1d
            if query_len > 1
            else actual.non_spec_decode_metadata.causal_conv1d
        )
        assert conv.output_buffer is builder._conv_output_buffer
        if graph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
            for name, value in vars(generic).items():
                if isinstance(value, torch.Tensor):
                    assert getattr(actual, name).data_ptr() == value.data_ptr(), name
        if query_len == 1 and num_spec and graph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
            assert not torch.any(builder.spec_actual_seq_lengths[: rows + 1])
            assert not torch.any(builder.num_accepted_tokens[:rows])
