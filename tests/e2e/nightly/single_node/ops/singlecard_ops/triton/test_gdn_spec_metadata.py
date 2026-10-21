# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.mamba.gdn_metadata import prepare_gdn_spec_metadata


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
        prepare_gdn_spec_metadata[(graph,)](
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
