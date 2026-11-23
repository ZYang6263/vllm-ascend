# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.hardware_profile import get_current_hardware_profile

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().requires_tiled_post_update, reason="requires the A6 MRv2 post-update path"
)


def test_resample_mixed_batch_keeps_rejection_positions():
    from vllm_ascend.ops.triton.spec_decode.resample import resample

    # One decode row verifies three drafts. The other 47 prefill rows have one
    # logit each. All rows need a greedy bonus and must count it exactly once.
    n, width, vocab = 48, 4, 154880
    lengths = torch.ones(n, dtype=torch.int32)
    lengths[0] = width
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
    rows = torch.arange(n, dtype=torch.int32).repeat_interleave(lengths.long())
    expected_ids = 150000 + torch.arange(n)
    logits = torch.full((rows.numel(), vocab), -2.0)
    logits[torch.arange(rows.numel()), expected_ids[rows.long()]] = 3.0
    positions = lengths - 1
    counts = positions.npu()
    sampled = torch.full((n, width), -1, dtype=torch.int64, device="npu")
    zeros = torch.zeros(n, device="npu")
    resample(
        sampled,
        counts,
        logits.npu(),
        zeros,
        None,
        zeros,
        cu.npu(),
        rows.npu(),
        torch.zeros(rows.numel(), dtype=torch.int64, device="npu"),
        zeros,
        torch.arange(n, dtype=torch.int64, device="npu"),
        torch.arange(rows.numel(), device="npu"),
    )
    expected = torch.full((n, width), -1, dtype=torch.int64)
    expected[torch.arange(n), positions.long()] = expected_ids
    torch.testing.assert_close(counts.cpu(), lengths)
    torch.testing.assert_close(sampled.cpu(), expected)


@pytest.mark.parametrize(
    ("num_reqs", "width", "use_bins", "use_query", "use_drafts"),
    [
        (0, 1, False, True, False),
        (1, 1, False, True, False),
        (35, 2, False, True, False),
        (36, 2, True, True, False),
        (37, 2, False, True, False),
        (47, 4, True, True, False),
        (48, 1, False, True, False),
        (48, 2, True, True, False),
        (48, 4, True, True, False),
        (48, 4, False, False, True),
        (97, 4, True, True, True),
        (1185, 4, False, True, False),
        (2305, 2, True, True, False),
    ],
)
def test_post_update(num_reqs, width, use_bins, use_query, use_drafts):
    from vllm.v1.worker.gpu import input_batch, model_runner

    import vllm_ascend.patch.worker.patch_input_batch  # noqa: F401
    from vllm_ascend.ops.triton.sample.post_update import post_update

    assert input_batch.post_update is model_runner.post_update is post_update
    capacity, history_width, vocab = num_reqs + 7, 25, 29
    rows = torch.arange(num_reqs, dtype=torch.int32)
    generator = torch.Generator().manual_seed(713)
    expected = {
        "computed": torch.arange(capacity, dtype=torch.int32) + 20,
        "last": torch.full((capacity, 1), -2, dtype=torch.int64),
        "history": torch.full((capacity, history_width), -3, dtype=torch.int32),
        "total": history_width - 3 * width - torch.arange(capacity, dtype=torch.int32) % 3,
    }
    if use_bins:
        expected["bins"] = torch.ones((capacity, vocab), dtype=torch.int32)
    if use_drafts:
        expected["drafts"] = torch.full((capacity, width - 1), -5, dtype=torch.int32)
    actual = {name: value.npu() for name, value in expected.items()}

    for step in range(3):
        slots = torch.randperm(capacity, generator=generator)[:num_reqs].int()
        counts = torch.full_like(rows, width)
        if step:
            slots[::9] = -1  # PP rows freed before this update.
            counts = (rows + step) % (width + 1)  # Includes chunked prefills with zero samples.
        rejected = torch.where(counts > 0, width - counts, 0)
        qlens = torch.where(counts > 0, width, 5 + rows % 7)
        cu = torch.cat((torch.zeros(1, dtype=torch.int32), qlens.cumsum(0, dtype=torch.int32))) + 11
        # Columns 0 and 1 repeat the same token. Slice on device to retain the stride.
        samples = (rows[:, None] + step + torch.arange(width)[None, :] // 2) % vocab
        samples[torch.arange(width)[None, :] >= counts[:, None]] = -1
        samples_npu = samples.repeat_interleave(2, dim=1).npu()[:, ::2]
        drafts = (samples[:, 0:1] + torch.arange(width - 1)) % vocab if use_drafts else None
        args = (drafts.int().npu(), actual["drafts"]) if drafts is not None else ()
        model_runner.post_update(
            slots.npu(),
            actual["computed"],
            actual["last"],
            actual.get("bins"),
            samples_npu[:, 0] if width == 1 else samples_npu,
            counts.npu(),
            rejected.npu(),
            cu.npu() if use_query else None,
            actual["history"],
            actual["total"],
            *args,
        )
        for row, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            n = counts[row].item()
            expected["computed"][slot] += (qlens[row] if use_query else 0) - rejected[row]
            if n:
                start = expected["total"][slot].item()
                expected["history"][slot, start : start + n] = samples[row, :n]
                expected["last"][slot, 0] = samples[row, n - 1]
                expected["total"][slot] += n
                if use_bins:
                    for token in samples[row, :n].tolist():
                        expected["bins"][slot, token] += 1
            if use_drafts:
                expected["drafts"][slot] = drafts[row]
        for name, tensor in actual.items():
            torch.testing.assert_close(tensor.cpu(), expected[name], msg=f"{name} after step {step}")
