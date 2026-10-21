# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.config.compilation import CUDAGraphMode

from tests.gdn_metadata_utils import assert_metadata_equal, decode_buffer_addresses, snapshot_metadata
from tests.ut.ops.test_gdn_attn_builder import BatchSpec, _make_builder, create_common_attn_metadata
from vllm_ascend.ops import gdn_attn_builder as gdn


def make_uniform_case(
    device,
    actual=3,
    graph=4,
    boundary=17,
    *,
    query_len=4,
    num_spec=3,
    mamba_cache_mode="align",
    num_speculative_blocks=None,
):
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_spec,
        builder_cls=gdn.AscendGDNHostMetadataBuilder,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_spec if num_speculative_blocks is None else num_speculative_blocks,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    builder.compilation_config.static_forward_context = {"layer0": SimpleNamespace(use_contiguous_conv_indices=True)}
    common = create_common_attn_metadata(BatchSpec([boundary] * graph, [query_len] * graph), 16, device)
    common.block_table_tensor = torch.arange(graph * 32, dtype=torch.int32, device=device).view(graph, 32)
    common.num_actual_tokens = actual * query_len
    common.is_prefilling.zero_()
    accepted = torch.tensor([4, 1, 2, 3][:graph], dtype=torch.int32, device=device)
    drafts = torch.full((graph,), num_spec if query_len > 1 else -1, dtype=torch.int32) if num_spec else None
    if query_len == 1:
        accepted = None
    return builder, common, accepted, drafts


class ViewsOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func is torch.ops.aten.slice.Tensor, f"Assembly introduced tensor computation: {func}"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("phase", ["spec", "decode", "prefill", "mixed"])
def test_common_wrapper_only_wraps_prepared_tensors(phase):
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"))
    if phase != "spec":
        query_lens = {"decode": [1, 1], "prefill": [8, 8], "mixed": [1, 8]}[phase]
        common = create_common_attn_metadata(BatchSpec([32, 32], query_lens), 16, torch.device("cpu"))
        common.block_table_tensor = torch.arange(64, dtype=torch.int32).view(2, 32)
        accepted, drafts = None, None
    wrapper = builder._wrap_metadata

    def check_wrapper(metadata, **tensors):
        # Neither arithmetic, copies, allocations nor scalar extraction is allowed.
        with ViewsOnly():
            wrapped = wrapper(metadata, **tensors)
        assert wrapped is metadata
        assert wrapped.non_spec_prefill_metadata is None
        assert wrapped.non_spec_decode_metadata is None
        if metadata.spec_sequence_masks is not None:
            spec = wrapped.spec_decode_metadata
            assert spec.actual_seq_lengths is tensors["actual_seq_lengths"]
            assert spec.spec_causal_conv1d.cache_indices is tensors["cache_indices"]
        else:
            assert wrapped.spec_decode_metadata is None
        return wrapped

    with patch.object(builder, "_wrap_metadata", side_effect=check_wrapper) as wrap:
        metadata = builder.build(0, common, accepted, drafts, num_actual_reqs=3 if phase == "spec" else None)
    wrap.assert_called_once()
    # The ordinary phase helpers still attach their metadata after wrapping.
    assert (metadata.non_spec_prefill_metadata is not None) == (phase in ("prefill", "mixed"))
    assert (metadata.non_spec_decode_metadata is not None) == (phase in ("decode", "mixed"))


@pytest.mark.parametrize("query_len", [1, 4])
def test_fused_path_uses_common_schema_without_generic_preparation(monkeypatch, query_len):
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"), query_len=query_len)
    # Materialize the capture buffers on CPU; the kernel is tested on NPU.
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=3)
    expected = snapshot_metadata(captured)
    addresses = decode_buffer_addresses(captured)
    wrapper = builder._wrap_metadata

    def extend_schema(metadata, **kwargs):
        metadata = wrapper(metadata, **kwargs)
        metadata.test_host_field = ("extension", metadata.num_actual_tokens)
        return metadata

    monkeypatch.setattr(builder, "_wrap_metadata", extend_schema)
    generic = builder.build(0, common, accepted, drafts, num_actual_reqs=3)
    kernel = MagicMock()
    monkeypatch.setattr(gdn, "prepare_gdn_decode_metadata", kernel)
    with (
        patch.object(builder, "_attach_spec_decode_metadata", side_effect=AssertionError("generic preparation")),
        patch.object(builder, "_pad_spec_decode_metadata", side_effect=AssertionError("extra padding")),
        patch.object(builder, "_attach_non_spec_decode_metadata", side_effect=AssertionError("generic preparation")),
        patch.object(builder, "_pad_decode_metadata", side_effect=AssertionError("extra padding")),
        patch.object(builder, "_prepare_conv_indices", side_effect=AssertionError("extra conv copy")),
        patch.object(builder, "_reset_spec_decode_graph_inputs") as reset,
        ViewsOnly(),
    ):
        fused = builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_decode_query_len=query_len)
    kernel.__getitem__.assert_called_once_with((4,))
    kernel.__getitem__.return_value.assert_called_once()
    expected["test_host_field"] = ("extension", 3 * query_len)
    assert_metadata_equal(snapshot_metadata(generic), expected)
    assert_metadata_equal(snapshot_metadata(fused), expected)
    assert decode_buffer_addresses(fused) == addresses
    launch = kernel.__getitem__.return_value.call_args
    assert launch.kwargs["QUERY_LEN"] == query_len
    if query_len > 1:
        reset.assert_not_called()
        conv = fused.spec_decode_metadata.spec_causal_conv1d
        assert conv.query_start_loc is fused.spec_query_start_loc
        assert conv.num_accepted_tokens.data_ptr() == fused.num_accepted_tokens.data_ptr()
    else:
        reset.assert_called_once_with(4)
        conv = fused.non_spec_decode_metadata.causal_conv1d
        assert conv.query_start_loc is fused.non_spec_query_start_loc
        assert conv.cache_indices.data_ptr() == fused.non_spec_state_indices_tensor.data_ptr()
        assert fused.num_accepted_tokens is None


@pytest.mark.parametrize(
    "reason",
    [
        "not_uniform",
        "no_conv",
        "non_align",
        "all_cache",
        "blocks",
        "variable_width",
        "capacity",
    ],
)
def test_ineligible_batches_keep_generic_dispatch(reason):
    builder, common, accepted, drafts = make_uniform_case(
        torch.device("cpu"),
        mamba_cache_mode={"non_align": "none", "all_cache": "all"}.get(reason, "align"),
        num_speculative_blocks=2 if reason == "blocks" else 3,
    )
    actual, uniform = 3, 4
    builder._ensure_conv_index_buffers()
    if reason == "not_uniform":
        uniform = None
    elif reason == "no_conv":
        builder._spec_conv_indices = None
        builder.compilation_config.static_forward_context = {}
    elif reason == "variable_width":
        common.num_actual_tokens = 11
    elif reason == "capacity":
        # The final FIA request extent must fit the request buffers.
        builder.decode_cudagraph_max_bs = common.num_reqs - 1
    expected = SimpleNamespace(spec_decode_metadata=None, non_spec_prefill_metadata=None, non_spec_decode_metadata=None)
    with (
        patch.object(gdn.AscendGDNAttentionMetadataBuilder, "build", return_value=expected) as generic,
        patch.object(builder, "_build_uniform_decode", side_effect=AssertionError("ineligible fast path")),
    ):
        assert (
            builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_decode_query_len=uniform)
            is expected
        )
    generic.assert_called_once()


@pytest.mark.parametrize("num_speculative_blocks", [0, 2])
def test_ordinary_decode_only_requires_one_state_slot(num_speculative_blocks, monkeypatch):
    builder, common, accepted, drafts = make_uniform_case(
        torch.device("cpu"), query_len=1, num_speculative_blocks=num_speculative_blocks
    )
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=3)
    expected = snapshot_metadata(captured)
    addresses = decode_buffer_addresses(captured)
    kernel = MagicMock()
    monkeypatch.setattr(gdn, "prepare_gdn_decode_metadata", kernel)
    with patch.object(
        gdn.AscendGDNAttentionMetadataBuilder, "build", side_effect=AssertionError("generic preparation")
    ):
        metadata = builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_decode_query_len=1)
    kernel.__getitem__.return_value.assert_called_once()
    assert_metadata_equal(snapshot_metadata(metadata), expected)
    assert decode_buffer_addresses(metadata) == addresses


@pytest.mark.parametrize("num_spec", [0, 3])
def test_empty_pcp_placeholder_keeps_generic_dispatch(num_spec):
    # PCP retains one zero-token placeholder and can preserve global Q=1.
    # The final token-count check must reject this otherwise eligible input.
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"), actual=1, query_len=1, num_spec=num_spec)
    common.num_actual_tokens = 0
    common.seq_lens.zero_()
    with patch.object(
        builder, "_build_uniform_decode", side_effect=AssertionError("empty placeholder entered fusion")
    ) as fused:
        metadata = builder.build(0, common, accepted, drafts, num_actual_reqs=1, uniform_decode_query_len=1)
    fused.assert_not_called()
    assert metadata.spec_decode_metadata is None
    assert metadata.num_decode_tokens == 0
    assert not torch.any(metadata.non_spec_decode_metadata.actual_seq_lengths)


@pytest.mark.parametrize("missing", ["accepted", "actual"])
def test_uniform_decode_requires_runner_inputs(missing):
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"))
    with pytest.raises(AssertionError):
        builder.build(
            0,
            common,
            None if missing == "accepted" else accepted,
            drafts,
            num_actual_reqs=None if missing == "actual" else 3,
            uniform_decode_query_len=4,
        )


def test_fused_dispatch_waits_for_lazy_conv_buffers():
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"))
    context = builder.compilation_config.static_forward_context
    builder.compilation_config.static_forward_context = {}
    expected = SimpleNamespace(spec_decode_metadata=None, non_spec_prefill_metadata=None, non_spec_decode_metadata=None)
    with (
        patch.object(gdn.AscendGDNAttentionMetadataBuilder, "build", return_value=expected) as generic,
        patch.object(builder, "_build_uniform_decode", return_value=expected) as fused,
    ):
        builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_decode_query_len=4)
        generic.assert_called_once()
        fused.assert_not_called()
        builder.compilation_config.static_forward_context = context
        builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_decode_query_len=4)
        fused.assert_called_once()
        generic.assert_called_once()


@pytest.mark.parametrize("num_spec", [0, 3])
def test_ordinary_fused_decode_neutralizes_captured_spec_inputs(num_spec, monkeypatch):
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"), query_len=1, num_spec=num_spec)
    builder._ensure_conv_index_buffers()
    if num_spec:
        for buffer in (
            builder.spec_state_indices_tensor,
            builder.spec_query_start_loc,
            builder.spec_actual_seq_lengths,
            builder.num_accepted_tokens,
            builder._spec_conv_indices,
        ):
            buffer.fill_(12345)
    monkeypatch.setattr(gdn, "prepare_gdn_decode_metadata", MagicMock())
    with patch.object(
        builder, "_reset_spec_decode_graph_inputs", wraps=builder._reset_spec_decode_graph_inputs
    ) as reset:
        metadata = builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_decode_query_len=1)
    assert metadata.spec_decode_metadata is None
    assert metadata.non_spec_decode_metadata is not None
    if num_spec:
        reset.assert_called_once_with(4)
        assert torch.all(builder.spec_state_indices_tensor[:4] == gdn.PAD_SLOT_ID)
        assert not torch.any(builder.spec_query_start_loc[:5])
        assert not torch.any(builder.spec_actual_seq_lengths[:5])
        assert not torch.any(builder.num_accepted_tokens[:4])
        assert torch.all(builder._spec_conv_indices[:4] == gdn.PAD_SLOT_ID)
    else:
        reset.assert_not_called()
