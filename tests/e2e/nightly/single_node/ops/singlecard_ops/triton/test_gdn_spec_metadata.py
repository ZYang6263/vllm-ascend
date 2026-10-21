# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from tests.gdn_metadata_utils import assert_metadata_equal, snapshot_metadata, spec_buffer_addresses
from tests.ut.ops.test_gdn_metadata_wrapper import make_uniform_case
from vllm_ascend.ops import gdn_attn_builder as gdn
from vllm_ascend.ops.triton.v2.mamba.gdn_metadata import prepare_gdn_decode_metadata


@pytest.mark.parametrize("query_len", [2, 4, 5])
@pytest.mark.parametrize("actual,graph", [(1, 1), (1, 4), (3, 4), (4, 4)])
def test_gdn_spec_metadata(query_len, actual, graph):
    device = "npu"
    # A nontrivial row stride and sequences on either side of an align boundary.
    table = torch.arange(graph * 64, device=device, dtype=torch.int32).reshape(graph, 64)[:, :32]
    lengths = torch.tensor([15, 16, 17, 32][:graph], device=device, dtype=torch.int32)
    accepted = torch.tensor([query_len, 1, 2, query_len][:graph], device=device, dtype=torch.int32)
    states = torch.empty((graph, query_len), device=device, dtype=torch.int32)
    conv = torch.empty(graph, device=device, dtype=torch.int32)
    masks = torch.empty(graph, device=device, dtype=torch.bool)
    query = torch.empty(graph + 1, device=device, dtype=torch.int32)
    actual_lengths = torch.empty_like(query)
    accepted_out = torch.empty_like(conv)
    tokens = torch.empty(graph * query_len, device=device, dtype=torch.int32)
    addresses = [x.data_ptr() for x in (states, conv, masks, query, actual_lengths, accepted_out, tokens)]
    # Reuse capture buffers after a larger batch, including poisoned padding.
    for count in (graph, actual):
        for buffer in (states, conv, query, actual_lengths, accepted_out, tokens):
            buffer.fill_(12345)
        prepare_gdn_decode_metadata[(graph,)](
            table,
            lengths,
            accepted,
            states,
            conv,
            masks,
            query,
            actual_lengths,
            accepted_out,
            tokens,
            NUM_ACTUAL_REQS=count,
            NUM_GRAPH_REQS=graph,
            QUERY_LEN=query_len,
            BLOCK_SIZE=16,
            BLOCK_TABLE_STRIDE=table.stride(0),
            PAD_BLOCK_ID=0,
            BLOCK=1 << (query_len - 1).bit_length(),
        )
        start = ((lengths[:count].cpu() - 1) // 16).clamp_min(0)
        expected_states = torch.zeros((graph, query_len), dtype=torch.int32)
        expected_states[:count] = table[:count].cpu().gather(1, (start[:, None] + torch.arange(query_len)).long())
        torch.testing.assert_close(states.cpu(), expected_states)
        torch.testing.assert_close(conv.cpu(), expected_states[:, 0])
        torch.testing.assert_close(masks.cpu(), torch.arange(graph) < count)
        torch.testing.assert_close(query.cpu(), torch.arange(graph + 1, dtype=torch.int32).clamp_max(count) * query_len)
        torch.testing.assert_close(
            actual_lengths.cpu(), torch.tensor([0] + [query_len] * count + [0] * (graph - count), dtype=torch.int32)
        )
        expected_accepted = torch.ones(graph, dtype=torch.int32)
        expected_accepted[:count] = accepted[:count].cpu()
        torch.testing.assert_close(accepted_out.cpu(), expected_accepted)
        torch.testing.assert_close(
            tokens[: count * query_len].cpu(), torch.arange(count * query_len, dtype=torch.int32)
        )
        assert addresses == [x.data_ptr() for x in (states, conv, masks, query, actual_lengths, accepted_out, tokens)]


@pytest.mark.parametrize("actual,graph", [(4, 4), (1, 4), (3, 4), (1, 1)])
@pytest.mark.parametrize("boundary", [15, 16, 17, 31, 32, 33])
def test_builder_schema_equivalence(actual, graph, boundary):
    builder, common, accepted, drafts = make_uniform_case(torch.device("npu:0"), actual, graph, boundary)
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=actual)
    expected = snapshot_metadata(captured)
    addresses = spec_buffer_addresses(captured)
    with patch.object(gdn, "prepare_gdn_decode_metadata", wraps=gdn.prepare_gdn_decode_metadata) as kernel:
        # MagicMock cannot wrap Triton's __getitem__; delegate the real launch.
        kernel.__getitem__.side_effect = prepare_gdn_decode_metadata.__getitem__
        runtime = builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_spec_decode=True)
        kernel.__getitem__.assert_called_once_with((graph,))
    assert_metadata_equal(snapshot_metadata(runtime), expected)
    assert spec_buffer_addresses(runtime) == addresses


def test_graph_consumer_observes_reused_buffers_after_phase_transitions():
    builder, common, accepted, drafts = make_uniform_case(torch.device("npu:0"), actual=4)
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=4)
    addresses = spec_buffer_addresses(captured)
    conv = captured.spec_decode_metadata.spec_causal_conv1d
    inputs = (
        captured.spec_state_indices_tensor,
        conv.cache_indices,
        captured.spec_sequence_masks,
        captured.spec_query_start_loc,
        captured.spec_decode_metadata.actual_seq_lengths,
        captured.num_accepted_tokens,
        captured.spec_token_indx,
    )
    graph = torch.npu.NPUGraph()
    # Warm up the consumer before capture, independently of metadata writers.
    outputs = [tensor.clone() for tensor in inputs]
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        outputs = [tensor.clone() for tensor in inputs]
    for step, actual in enumerate((1, 3, 2, 4)):
        common.num_actual_tokens = actual * 4
        common.block_table_tensor = common.block_table_tensor.flip(0) + 7
        common.seq_lens.add_(1)
        accepted = accepted.roll(1)
        generic = builder.build(0, common, accepted, drafts, num_actual_reqs=actual)
        expected = snapshot_metadata(generic)
        runtime = builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_spec_decode=True)
        assert_metadata_equal(snapshot_metadata(runtime), expected)
        assert spec_buffer_addresses(runtime) == addresses
        graph.replay()
        for output, source in zip(outputs[:-1], inputs[:-1]):
            torch.testing.assert_close(output.cpu(), source.cpu())
        torch.testing.assert_close(outputs[-1][: actual * 4].cpu(), runtime.spec_token_indx.cpu())
        if step == 1:
            # No-spec decode must neutralize the previously captured spec path.
            decode = common.unpadded(4, 4)
            decode.query_start_loc = torch.arange(5, device="npu:0", dtype=torch.int32)
            decode.query_start_loc_cpu = torch.arange(5, dtype=torch.int32)
            decode.max_query_len = 1
            ordinary = builder.build(0, decode, accepted, torch.full((4,), -1, dtype=torch.int32))
            assert ordinary.spec_decode_metadata is None
            graph.replay()
            torch.testing.assert_close(outputs[3].cpu(), torch.zeros(5, dtype=torch.int32))
            torch.testing.assert_close(outputs[4].cpu(), torch.zeros(5, dtype=torch.int32))
            torch.testing.assert_close(outputs[5].cpu(), torch.zeros(4, dtype=torch.int32))
