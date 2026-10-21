# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from tests.gdn_metadata_utils import (
    assert_metadata_equal,
    decode_buffer_addresses,
    poison_decode_buffers,
    snapshot_metadata,
)
from tests.ut.ops.test_gdn_metadata_wrapper import make_uniform_case
from vllm_ascend.ops import gdn_attn_builder as gdn
from vllm_ascend.ops.triton.v2.mamba.gdn_metadata import prepare_gdn_decode_metadata


def test_gdn_graph_replay_reads_live_counts_and_request_mapping():
    builder, common, _, _ = make_uniform_case(torch.device("npu:0"), actual=4, graph=4)
    builder._ensure_conv_index_buffers()
    batch = torch.tensor([4, 3, 1, 0, 2], dtype=torch.int64, device="npu")
    accepted = torch.tensor([1, 4, 2, 3], dtype=torch.int32, device="npu")

    def prepare():
        prepare_gdn_decode_metadata[(4,)](
            common.block_table_tensor,
            common.seq_lens,
            accepted,
            builder.spec_state_indices_tensor,
            builder._spec_conv_indices,
            builder.spec_sequence_masks,
            builder.spec_query_start_loc,
            builder.spec_actual_seq_lengths,
            builder.num_accepted_tokens,
            builder.spec_token_indx,
            NUM_ACTUAL_REQS=0,
            NUM_GRAPH_REQS=4,
            QUERY_LEN=4,
            BLOCK_SIZE=16,
            BLOCK_TABLE_STRIDE=common.block_table_tensor.stride(0),
            PAD_BLOCK_ID=0,
            BLOCK=4,
            batch=batch,
            GRAPH_INPUTS=True,
        )

    prepare()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        prepare()
    for count, mapping in ((1, [2]), (3, [1, 0, 3]), (4, [3, 2, 1, 0])):
        batch.copy_(torch.tensor([count, *mapping, *([-1] * (4 - count))], dtype=torch.int64, device="npu"))
        builder.spec_token_indx[:16].fill_(12345)
        graph.replay()
        expected = torch.ones(4, dtype=torch.int32)
        expected[:count] = accepted.cpu()[mapping]
        torch.testing.assert_close(builder.num_accepted_tokens[:4].cpu(), expected)
        torch.testing.assert_close(builder.spec_token_indx[:16].cpu(), torch.arange(16, dtype=torch.int32))
        torch.testing.assert_close(
            builder.spec_query_start_loc[:5].cpu(), torch.arange(5, dtype=torch.int32).clamp_max(count) * 4
        )


@pytest.mark.parametrize("query_len", [1, 2, 4, 5])
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
        masks.fill_(True)
        prepare_gdn_decode_metadata[(graph,)](
            table,
            lengths,
            accepted if query_len > 1 else None,
            states,
            conv if query_len > 1 else None,
            masks if query_len > 1 else None,
            query,
            actual_lengths,
            accepted_out if query_len > 1 else None,
            tokens if query_len > 1 else None,
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
        torch.testing.assert_close(query.cpu(), torch.arange(graph + 1, dtype=torch.int32).clamp_max(count) * query_len)
        torch.testing.assert_close(
            actual_lengths.cpu(), torch.tensor([0] + [query_len] * count + [0] * (graph - count), dtype=torch.int32)
        )
        if query_len > 1:
            torch.testing.assert_close(conv.cpu(), expected_states[:, 0])
            torch.testing.assert_close(masks.cpu(), torch.arange(graph) < count)
            expected_accepted = torch.ones(graph, dtype=torch.int32)
            expected_accepted[:count] = accepted[:count].cpu()
            torch.testing.assert_close(accepted_out.cpu(), expected_accepted)
            torch.testing.assert_close(
                tokens[: count * query_len].cpu(), torch.arange(count * query_len, dtype=torch.int32)
            )
        else:
            # Optional speculative outputs must remain untouched in Q=1 mode.
            for buffer in (conv, accepted_out, tokens):
                assert torch.all(buffer.cpu() == 12345)
        assert addresses == [x.data_ptr() for x in (states, conv, masks, query, actual_lengths, accepted_out, tokens)]


@pytest.mark.parametrize("actual,graph", [(4, 4), (1, 4), (3, 4), (1, 1)])
@pytest.mark.parametrize("boundary", [15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("query_len,num_spec", [(1, 0), (1, 3), (4, 3)])
def test_builder_schema_equivalence(actual, graph, boundary, query_len, num_spec):
    builder, common, accepted, drafts = make_uniform_case(
        torch.device("npu:0"), actual, graph, boundary, query_len=query_len, num_spec=num_spec
    )
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=actual)
    expected = snapshot_metadata(captured)
    addresses = decode_buffer_addresses(captured)
    poison_decode_buffers(captured)
    with patch.object(gdn, "prepare_gdn_decode_metadata", wraps=gdn.prepare_gdn_decode_metadata) as kernel:
        # MagicMock cannot wrap Triton's __getitem__; delegate the real launch.
        kernel.__getitem__.side_effect = prepare_gdn_decode_metadata.__getitem__
        runtime = builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_decode_query_len=query_len)
        kernel.__getitem__.assert_called_once_with((graph,))
    assert_metadata_equal(snapshot_metadata(runtime), expected)
    assert decode_buffer_addresses(runtime) == addresses


def test_graph_consumer_observes_reused_buffers_after_phase_transitions():
    builder, common, accepted, drafts = make_uniform_case(torch.device("npu:0"), actual=4)
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=4)
    addresses = decode_buffer_addresses(captured)
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
        runtime = builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_decode_query_len=4)
        assert_metadata_equal(snapshot_metadata(runtime), expected)
        assert decode_buffer_addresses(runtime) == addresses
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
            ordinary = builder.build(
                0,
                decode,
                accepted,
                torch.full((4,), -1, dtype=torch.int32),
                num_actual_reqs=4,
                uniform_decode_query_len=1,
            )
            assert ordinary.spec_decode_metadata is None
            graph.replay()
            torch.testing.assert_close(outputs[3].cpu(), torch.zeros(5, dtype=torch.int32))
            torch.testing.assert_close(outputs[4].cpu(), torch.zeros(5, dtype=torch.int32))
            torch.testing.assert_close(outputs[5].cpu(), torch.zeros(4, dtype=torch.int32))


@pytest.mark.parametrize("num_spec", [0, 3])
def test_ordinary_graph_consumer_observes_reused_buffers(num_spec):
    builder, common, accepted, drafts = make_uniform_case(
        torch.device("npu:0"), actual=4, query_len=1, num_spec=num_spec
    )
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=4)
    addresses = decode_buffer_addresses(captured)
    inputs = (
        captured.non_spec_state_indices_tensor,
        captured.non_spec_query_start_loc,
        captured.non_spec_decode_metadata.actual_seq_lengths,
    )
    outputs = [tensor.clone() for tensor in inputs]
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [tensor.clone() for tensor in inputs]
    for actual in (1, 3, 2, 4):
        common.num_actual_tokens = actual
        common.block_table_tensor = common.block_table_tensor.flip(0) + 7
        common.seq_lens.add_(1)
        expected = snapshot_metadata(builder.build(0, common, accepted, drafts, num_actual_reqs=actual))
        runtime = builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_decode_query_len=1)
        assert_metadata_equal(snapshot_metadata(runtime), expected)
        assert decode_buffer_addresses(runtime) == addresses
        graph.replay()
        for output, source in zip(outputs, inputs):
            torch.testing.assert_close(output.cpu(), source.cpu())
