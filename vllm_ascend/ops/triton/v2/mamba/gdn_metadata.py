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
    batch=None,
    GRAPH_INPUTS: tl.constexpr = False,
):
    """Prepare metadata tensors for every token in a uniform decode batch.

    The caller supports align cache mode and stable graph buffers. Ordinary
    decode has QUERY_LEN=1; speculative decode includes the normal decode token
    and every draft token. QUERY_LEN selects the accepted-token semantics.

    Launch one program per graph request. Inputs are int32 seq_lens/accepted
    vectors and a block table with unit column stride and enough columns for
    the align start plus QUERY_LEN. Outputs are builder-owned, contiguous:
    state_indices [NUM_GRAPH_REQS, QUERY_LEN], conv_indices/accepted_out
    [NUM_GRAPH_REQS], query_start_loc/actual_seq_lengths [NUM_GRAPH_REQS + 1]
    (all int32), sequence_masks [NUM_GRAPH_REQS] (bool), and token_indices
    [at least NUM_ACTUAL_REQS * QUERY_LEN] (int32). BLOCK >= QUERY_LEN.

    State indices, query boundaries and lengths are overwritten in both modes.
    Only speculative mode accesses accepted, conv_indices, sequence_masks,
    accepted_out and token_indices; these pointers may be None in ordinary
    mode. Padding gets PAD_BLOCK_ID, False, accepted=1 and zero query length.
    Eager preparation writes only the live token-index prefix. FULL preparation
    reads the live count and request mapping from batch, and overwrites the
    entire captured token-index extent, including padding. Storage and stream
    ordering belong to the builder; Python assembly only wraps these outputs.
    """
    row = tl.program_id(0)
    if GRAPH_INPUTS:
        actual_reqs = tl.load(batch).to(tl.int32)
    else:
        actual_reqs = NUM_ACTUAL_REQS
    active = row < actual_reqs
    seq_len = tl.load(seq_lens + row, active, 0)
    start = tl.maximum((seq_len - 1) // BLOCK_SIZE, 0)
    cols = tl.arange(0, BLOCK)
    indices = tl.load(
        block_table + row * BLOCK_TABLE_STRIDE + start + cols,
        active & (cols < QUERY_LEN),
        PAD_BLOCK_ID,
    )
    tl.store(state_indices + row * QUERY_LEN + cols, indices, cols < QUERY_LEN)
    if QUERY_LEN > 1:
        conv_index = tl.load(block_table + row * BLOCK_TABLE_STRIDE + start, active, PAD_BLOCK_ID)
        tl.store(conv_indices + row, conv_index)
        tl.store(sequence_masks + row, active)
        if GRAPH_INPUTS:
            state_row = tl.load(batch + 1 + row, active, 0)
        else:
            state_row = row
        count = tl.load(accepted + state_row, active, 1)
        tl.store(accepted_out + row, count)
        tl.store(
            token_indices + row * QUERY_LEN + cols,
            row * QUERY_LEN + cols,
            (active | GRAPH_INPUTS) & (cols < QUERY_LEN),
        )
    tl.store(query_start_loc + row, tl.minimum(row, actual_reqs) * QUERY_LEN)
    tl.store(actual_seq_lengths + row + 1, tl.where(active, QUERY_LEN, 0))
    if row == 0:
        tl.store(query_start_loc + NUM_GRAPH_REQS, actual_reqs * QUERY_LEN)
        tl.store(actual_seq_lengths, 0)
