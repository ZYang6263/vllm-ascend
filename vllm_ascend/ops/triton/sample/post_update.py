# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _post_update_kernel(
    mapping,
    computed,
    last,
    bins,
    samples,
    counts,
    rejected,
    cu_query,
    history,
    total,
    drafts,
    drafts_out,
    num_reqs,
    sample_stride: tl.constexpr,
    sample_col_stride: tl.constexpr,
    history_stride: tl.constexpr,
    bin_stride: tl.constexpr,
    sample_width: tl.constexpr,
    draft_width: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    # Each lane owns one request. Keep skipped PP slots and a partial tile
    # masked throughout, including gathers from the sampled-token buffer.
    for sweep in range(TILES_PER_PROGRAM):
        tile = tl.program_id(0) + sweep * tl.num_programs(0)
        rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        slots = tl.load(mapping + rows, rows < num_reqs, other=-1).to(tl.int64)
        live = slots >= 0
        slots = tl.maximum(slots, 0)
        count = tl.load(counts + rows, live, other=0)
        start = tl.load(total + slots, live, other=0)
        for col in range(sample_width):
            accepted = live & (col < count)
            token = tl.load(samples + rows * sample_stride + col * sample_col_stride, accepted, other=0)
            tl.store(history + slots * history_stride + start + col, token, accepted)
            tl.store(last + slots, token, accepted & (col + 1 == count))
            if bins is not None:
                # A request may accept the same token more than once in this step.
                tl.atomic_add(bins + slots * bin_stride + token, 1, mask=accepted, sem="relaxed")
        tl.store(total + slots, start + count, live & (count > 0))

        delta = -tl.load(rejected + rows, live, other=0)
        if cu_query is not None:
            delta += tl.load(cu_query + rows + 1, live, other=0) - tl.load(cu_query + rows, live, other=0)
        previous = tl.load(computed + slots, live, other=0)
        tl.store(computed + slots, previous + delta, live & (delta != 0))

        if drafts is not None:
            for col in range(draft_width):
                token = tl.load(drafts + rows * draft_width + col, live, other=0)
                tl.store(drafts_out + slots * draft_width + col, token, live)


def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_tokens_out: torch.Tensor | None = None,
) -> None:
    num_reqs = idx_mapping.numel()
    if num_reqs == 0:
        return
    if draft_tokens is not None:
        assert draft_tokens_out is not None and draft_tokens.shape[1] == draft_tokens_out.shape[1]
        assert draft_tokens.is_contiguous() and draft_tokens_out.is_contiguous()
    if sampled_tokens.ndim == 1:
        sampled_tokens = sampled_tokens.unsqueeze(1)
    block_rows = 32
    num_tiles = triton.cdiv(num_reqs, block_rows)
    # On Ascend960 the driver reports 64 cores, but this kernel can miss writes
    # beyond program 35. Bound the grid and let each program visit more tiles.
    num_programs = min(num_tiles, 32)
    _post_update_kernel[(num_programs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
        draft_tokens,
        draft_tokens_out,
        num_reqs,
        sampled_tokens.stride(0),
        sampled_tokens.stride(1),
        all_token_ids.stride(0),
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens.shape[1],
        draft_tokens.shape[1] if draft_tokens is not None else 0,
        BLOCK_ROWS=block_rows,
        TILES_PER_PROGRAM=triton.cdiv(num_tiles, num_programs),
    )
