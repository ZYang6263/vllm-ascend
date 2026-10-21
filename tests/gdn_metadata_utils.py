# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields, is_dataclass

import torch


def snapshot_metadata(value):
    """Freeze all fields, including dynamically attached phase metadata.

    Do not whitelist names: a new field must participate in equivalence checks.
    Clone tensors because the next build overwrites graph-owned storage.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        names = {field.name for field in fields(value)} | set(vars(value))
        return {name: snapshot_metadata(getattr(value, name)) for name in sorted(names)}
    if isinstance(value, dict):
        return {key: snapshot_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot_metadata(item) for item in value)
    return value


def assert_metadata_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_metadata_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert_metadata_equal(got, want)
    else:
        assert actual == expected


def _decode_buffers(metadata):
    spec = metadata.spec_decode_metadata
    if spec is not None:
        return (
            metadata.spec_state_indices_tensor,
            spec.spec_causal_conv1d.cache_indices,
            metadata.spec_sequence_masks,
            metadata.spec_query_start_loc,
            spec.actual_seq_lengths,
            metadata.num_accepted_tokens,
            metadata.spec_token_indx,
        )
    return (
        metadata.non_spec_state_indices_tensor,
        metadata.non_spec_decode_metadata.causal_conv1d.cache_indices,
        metadata.non_spec_query_start_loc,
        metadata.non_spec_decode_metadata.actual_seq_lengths,
    )


def decode_buffer_addresses(metadata):
    return tuple(buffer.data_ptr() for buffer in _decode_buffers(metadata))


def poison_decode_buffers(metadata):
    """Require every fused output to replace stale capture data."""
    for buffer in _decode_buffers(metadata):
        buffer.fill_(True if buffer.dtype == torch.bool else 12345)
