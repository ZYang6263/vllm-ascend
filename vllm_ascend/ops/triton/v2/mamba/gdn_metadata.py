# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.triton_utils import tl, triton


@triton.jit
def prepare_gdn_decode_metadata(
    block_table,
    seq_lens,
    accepted,
    state_indices,
    conv_indices,
    sequence_masks,
    query_start_loc,
    actual_seq_lengths,
    accepted_out,
    token_indices,
    NUM_ACTUAL_REQS: tl.constexpr,
    NUM_GRAPH_REQS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    PAD_BLOCK_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Prepare metadata tensors for every token in a uniform decode batch.

    The current caller supports speculative decode with align cache mode and
    stable graph buffers. Each query includes its normal decode token and all
    draft tokens. Ordinary non-speculative decode needs its own metadata wrapper.

    Launch one program per graph request. Inputs are int32 seq_lens/accepted
    vectors and a block table with unit column stride and enough columns for
    the align start plus QUERY_LEN. Outputs are builder-owned, contiguous:
    state_indices [NUM_GRAPH_REQS, QUERY_LEN], conv_indices/accepted_out
    [NUM_GRAPH_REQS], query_start_loc/actual_seq_lengths [NUM_GRAPH_REQS + 1]
    (all int32), sequence_masks [NUM_GRAPH_REQS] (bool), and token_indices
    [at least NUM_ACTUAL_REQS * QUERY_LEN] (int32). BLOCK >= QUERY_LEN.

    All request rows are overwritten. Padding gets PAD_BLOCK_ID, False,
    accepted=1 and zero query length. Only the live token-index prefix is
    written or exposed by the caller. Storage and stream ordering belong to
    the builder; Python assembly must not copy or recompute these outputs.
    """
    row = tl.program_id(0)
    active = row < NUM_ACTUAL_REQS
    seq_len = tl.load(seq_lens + row, active, 0)
    start = tl.maximum((seq_len - 1) // BLOCK_SIZE, 0)
    cols = tl.arange(0, BLOCK)
    indices = tl.load(
        block_table + row * BLOCK_TABLE_STRIDE + start + cols,
        active & (cols < QUERY_LEN),
        PAD_BLOCK_ID,
    )
    tl.store(state_indices + row * QUERY_LEN + cols, indices, cols < QUERY_LEN)
    conv_index = tl.load(block_table + row * BLOCK_TABLE_STRIDE + start, active, PAD_BLOCK_ID)
    tl.store(conv_indices + row, conv_index)
    tl.store(sequence_masks + row, active)
    count = tl.load(accepted + row, active, 1)
    tl.store(accepted_out + row, count)
    tl.store(query_start_loc + row, tl.minimum(row, NUM_ACTUAL_REQS) * QUERY_LEN)
    tl.store(actual_seq_lengths + row + 1, tl.where(active, QUERY_LEN, 0))
    tl.store(token_indices + row * QUERY_LEN + cols, row * QUERY_LEN + cols, active & (cols < QUERY_LEN))
    if row == 0:
        tl.store(query_start_loc + NUM_GRAPH_REQS, NUM_ACTUAL_REQS * QUERY_LEN)
        tl.store(actual_seq_lengths, 0)
