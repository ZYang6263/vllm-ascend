# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.config.compilation import CUDAGraphMode

from tests.gdn_metadata_utils import assert_metadata_equal, snapshot_metadata, spec_buffer_addresses
from tests.ut.ops.test_gdn_attn_builder import BatchSpec, _make_builder, create_common_attn_metadata
from vllm_ascend.ops import gdn_attn_builder as gdn


def make_uniform_case(device, actual=3, graph=4, boundary=17, *, mamba_cache_mode="align", num_speculative_blocks=3):
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=3,
        builder_cls=gdn.AscendGDNHostMetadataBuilder,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    builder.compilation_config.static_forward_context = {"layer0": SimpleNamespace(use_contiguous_conv_indices=True)}
    common = create_common_attn_metadata(BatchSpec([boundary] * graph, [4] * graph), 16, device)
    common.block_table_tensor = torch.arange(graph * 32, dtype=torch.int32, device=device).view(graph, 32)
    common.num_actual_tokens = actual * 4
    common.is_prefilling.zero_()
    accepted = torch.tensor([4, 1, 2, 3][:graph], dtype=torch.int32, device=device)
    drafts = torch.full((graph,), 3, dtype=torch.int32)
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
            assert spec.actual_seq_lengths is tensors["spec_actual_seq_lengths"]
            assert spec.spec_causal_conv1d.cache_indices is tensors["spec_cache_indices"]
        else:
            assert wrapped.spec_decode_metadata is None
        return wrapped

    with patch.object(builder, "_wrap_metadata", side_effect=check_wrapper) as wrap:
        metadata = builder.build(0, common, accepted, drafts, num_actual_reqs=3 if phase == "spec" else None)
    wrap.assert_called_once()
    # The ordinary phase helpers still attach their metadata after wrapping.
    assert (metadata.non_spec_prefill_metadata is not None) == (phase in ("prefill", "mixed"))
    assert (metadata.non_spec_decode_metadata is not None) == (phase in ("decode", "mixed"))


def test_fused_path_uses_common_schema_without_generic_preparation(monkeypatch):
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"))
    # Materialize the capture buffers on CPU; the kernel is tested on NPU.
    captured = builder.build(0, common, accepted, drafts, num_actual_reqs=3)
    expected = snapshot_metadata(captured)
    addresses = spec_buffer_addresses(captured)
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
        patch.object(builder, "_prepare_conv_indices", side_effect=AssertionError("extra conv copy")),
        ViewsOnly(),
    ):
        fused = builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_spec_decode=True)
    kernel.__getitem__.assert_called_once_with((4,))
    kernel.__getitem__.return_value.assert_called_once()
    expected["test_host_field"] = ("extension", 12)
    assert_metadata_equal(snapshot_metadata(generic), expected)
    assert_metadata_equal(snapshot_metadata(fused), expected)
    assert spec_buffer_addresses(fused) == addresses
    conv = fused.spec_decode_metadata.spec_causal_conv1d
    assert conv.query_start_loc is fused.spec_query_start_loc
    assert conv.num_accepted_tokens.data_ptr() == fused.num_accepted_tokens.data_ptr()


@pytest.mark.parametrize(
    "reason",
    [
        "not_uniform",
        "no_conv",
        "non_align",
        "blocks",
        "variable_width",
        "stride",
        "capacity",
    ],
)
def test_ineligible_batches_keep_generic_dispatch(reason):
    builder, common, accepted, drafts = make_uniform_case(
        torch.device("cpu"),
        mamba_cache_mode="none" if reason == "non_align" else "align",
        num_speculative_blocks=2 if reason == "blocks" else 3,
    )
    actual, uniform = 3, True
    builder._ensure_conv_index_buffers()
    if reason == "not_uniform":
        uniform = False
    elif reason == "no_conv":
        builder._spec_conv_indices = None
        builder.compilation_config.static_forward_context = {}
    elif reason == "variable_width":
        common.num_actual_tokens = 11
    elif reason == "stride":
        common.block_table_tensor = common.block_table_tensor[:, ::2]
    elif reason == "capacity":
        builder.decode_cudagraph_max_bs = 8
    expected = SimpleNamespace(spec_decode_metadata=None, non_spec_prefill_metadata=None, non_spec_decode_metadata=None)
    with (
        patch.object(gdn.AscendGDNAttentionMetadataBuilder, "build", return_value=expected) as generic,
        patch.object(builder, "_build_uniform_spec_decode", side_effect=AssertionError("ineligible fast path")),
    ):
        assert (
            builder.build(0, common, accepted, drafts, num_actual_reqs=actual, uniform_spec_decode=uniform) is expected
        )
    generic.assert_called_once()


@pytest.mark.parametrize("missing", ["accepted", "actual"])
def test_uniform_spec_decode_requires_runner_inputs(missing):
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"))
    with pytest.raises(AssertionError):
        builder.build(
            0,
            common,
            None if missing == "accepted" else accepted,
            drafts,
            num_actual_reqs=None if missing == "actual" else 3,
            uniform_spec_decode=True,
        )


def test_fused_dispatch_waits_for_lazy_conv_buffers():
    builder, common, accepted, drafts = make_uniform_case(torch.device("cpu"))
    context = builder.compilation_config.static_forward_context
    builder.compilation_config.static_forward_context = {}
    expected = SimpleNamespace(spec_decode_metadata=None, non_spec_prefill_metadata=None, non_spec_decode_metadata=None)
    with (
        patch.object(gdn.AscendGDNAttentionMetadataBuilder, "build", return_value=expected) as generic,
        patch.object(builder, "_build_uniform_spec_decode", return_value=expected) as fused,
    ):
        builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_spec_decode=True)
        generic.assert_called_once()
        fused.assert_not_called()
        builder.compilation_config.static_forward_context = context
        builder.build(0, common, accepted, drafts, num_actual_reqs=3, uniform_spec_decode=True)
        fused.assert_called_once()
        generic.assert_called_once()
