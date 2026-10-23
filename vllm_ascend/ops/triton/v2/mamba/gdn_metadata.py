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
    conv_output,
    NUM_ACTUAL_REQS: tl.constexpr,
    NUM_REQS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    SEQ_LEN_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    PAD_BLOCK_ID: tl.constexpr,
    BLOCK: tl.constexpr,
    CONV_OUTPUT_NUMEL: tl.constexpr,
    CLEAR_BLOCK: tl.constexpr,
):
    """Write align-cache decode metadata into the builder's persistent buffers.

    One program per request, including padding. QUERY_LEN=1 is ordinary decode;
    larger widths include one decode token and the configured draft tokens.
    Padded rows have no query tokens and use a safe state slot and accepted=1.
    Speculative outputs are unused for QUERY_LEN=1. Additional programs clear
    the entire shared convolution output, including previously active padding.
    """
    row = tl.program_id(0)
    if row < NUM_REQS:
        active = row < NUM_ACTUAL_REQS
        seq_len = tl.load(seq_lens + row * SEQ_LEN_STRIDE, active, 0)
        start = tl.maximum((seq_len - 1) // BLOCK_SIZE, 0)
        cols = tl.arange(0, BLOCK)
        indices = tl.load(
            block_table + row * TABLE_ROW_STRIDE + (start + cols) * TABLE_COL_STRIDE,
            active & (cols < QUERY_LEN),
            PAD_BLOCK_ID,
        )
        tl.store(state_indices + row * QUERY_LEN + cols, indices, cols < QUERY_LEN)
        if QUERY_LEN > 1:
            conv_index = tl.load(block_table + row * TABLE_ROW_STRIDE + start * TABLE_COL_STRIDE, active, PAD_BLOCK_ID)
            tl.store(conv_indices + row, conv_index)
            tl.store(sequence_masks + row, active)
            tl.store(accepted_out + row, tl.load(accepted + row * ACCEPTED_STRIDE, active, 1))
            tl.store(token_indices + row * QUERY_LEN + cols, row * QUERY_LEN + cols, active & (cols < QUERY_LEN))
        tl.store(query_start_loc + row, tl.minimum(row, NUM_ACTUAL_REQS) * QUERY_LEN)
        tl.store(actual_seq_lengths + row + 1, tl.where(active, QUERY_LEN, 0))
        if row == 0:
            tl.store(query_start_loc + NUM_REQS, NUM_ACTUAL_REQS * QUERY_LEN)
            tl.store(actual_seq_lengths, 0)
    elif CONV_OUTPUT_NUMEL:
        offsets = (row - NUM_REQS) * CLEAR_BLOCK + tl.arange(0, CLEAR_BLOCK)
        tl.store(conv_output + offsets, 0, offsets < CONV_OUTPUT_NUMEL)
