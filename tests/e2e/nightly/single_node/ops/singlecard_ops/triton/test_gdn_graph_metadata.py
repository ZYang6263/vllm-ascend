# SPDX-License-Identifier: Apache-2.0
"""Joint producer-stream capture, live counts and padded metadata regression."""

from functools import partial
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from tests.ut.ops.test_gdn_metadata_wrapper import make_uniform_case
from vllm_ascend.attention.sfa_v1 import SparseMLAMetadataState
from vllm_ascend.ops.triton.glm5_next_cann_keypool import build_keypool_metadata, build_tail_inputs
from vllm_ascend.worker.device_metadata import (
    DeferredMetadataBuilder,
    DeviceMetadataStage,
    TargetDeviceMetadata,
    UniformDecodeInputs,
)


def _pool_buffers(device):
    return SimpleNamespace(
        **{
            name: torch.empty(17, dtype=torch.int64, device=device)
            for name in (
                "positions",
                "slot_mapping",
                "seq_lens",
                "raw_seq_lens",
                "query_start_loc",
                "cum_query_lens",
                "start_pos",
                "pool_tail",
                "start_offsets",
                "pooled_slot_mapping",
            )
        }
    )


def test_joint_full_graph_metadata_prologue_replays_without_python_builders():
    device = torch.device("npu:0")
    cases = [make_uniform_case(device, actual=4, graph=4) for _ in range(3)]
    accepted = torch.tensor([1, 4, 2, 3, 2, 1, 4, 3], dtype=torch.int32, device=device)
    batch = torch.tensor([4, 3, 1, 0, 2], dtype=torch.int64, device=device)
    inputs = UniformDecodeInputs(batch, accepted, 4)
    extra = DeferredMetadataBuilder()
    extra.init_device_metadata()
    extra.graph_decode_inputs = inputs
    for builder, _, _, _ in cases:
        builder.graph_decode_inputs = inputs
        builder._ensure_conv_index_buffers()
    builders = [case[0] for case in cases] + [extra]
    groups = [[SimpleNamespace(get_metadata_builder=lambda _, builder=builder: builder)] for builder in builders]
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=16,
        query_start_loc=torch.arange(5, dtype=torch.int32, device=device) * 4,
        seq_lens=torch.full((4,), 16, dtype=torch.int32, device=device),
        positions=torch.arange(16, dtype=torch.int64, device=device),
        slot_mapping=torch.arange(16, dtype=torch.int64, device=device),
        block_table_tensor=torch.arange(32, dtype=torch.int32, device=device).reshape(4, 8),
    )
    buffers, reference = _pool_buffers(device), _pool_buffers(device)
    tail = {
        name: torch.empty(4, 4, dtype=torch.int64, device=device) for name in ("history", "writes", "state", "retained")
    }
    tail_reference = {name: torch.empty_like(value) for name, value in tail.items()}
    sfa = object.__new__(SparseMLAMetadataState)
    sfa.split, sfa.table_stride, sfa.block_size, sfa.use_smla = 1, 1, 128, False
    sfa.block_table_buffer = torch.empty_like(common.block_table_tensor)
    state = TargetDeviceMetadata(wait_before_forward=True)
    checksum = torch.empty((), dtype=torch.int64, device=device)

    def prepare():
        with state.build(groups, True):
            metadata = [builder.build_for_cudagraph_capture(value) for builder, value, _, _ in cases]
            pool = build_keypool_metadata(
                common,
                buffers,
                16,
                16,
                128,
                4,
                graph_inputs=inputs,
                defer=partial(extra.run_device_metadata, DeviceMetadataStage.INDEXER),
            )
            tail_inputs = build_tail_inputs(
                common,
                4,
                8,
                80,
                4,
                tail,
                graph_inputs=inputs,
                defer=partial(extra.run_device_metadata, DeviceMetadataStage.INDEXER),
            )
            sfa.prepare(
                SimpleNamespace(block_table=common.block_table_tensor),
                defer=partial(extra.run_device_metadata, DeviceMetadataStage.ATTENTION),
            )
        return metadata, pool, tail_inputs

    def forward():
        state.begin_forward()
        checksum.copy_(cases[0][0].num_accepted_tokens[:4].sum() + sfa.block_table_buffer.sum())
        state.finish()

    prepare()
    forward()
    torch.npu.synchronize()
    metadata, pool, tail_inputs = prepare()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    for step, count in enumerate((1, 3, 4, 2)):
        mapping = [(7 - i + step) % 8 for i in range(count)]
        batch.copy_(torch.tensor([count, *mapping, *([-1] * (4 - count))], dtype=torch.int64, device=device))
        common.query_start_loc.copy_(torch.arange(5, dtype=torch.int32, device=device).clamp_max(count) * 4)
        common.seq_lens[count:].zero_()
        common.seq_lens[:count].fill_(16 + step)
        common.block_table_tensor.add_(1)
        sfa.block_table_buffer.fill_(-99)
        for builder, _, _, _ in cases:
            builder.spec_token_indx[:16].fill_(9999)
        graph.replay()
        common.num_actual_tokens = count * 4
        expected_pool = build_keypool_metadata(common, reference, 16, 16, 128, 4)
        expected_tail = build_tail_inputs(common, 4, 8, 80, 4, tail_reference)
        for got, want in zip(pool, expected_pool):
            torch.testing.assert_close(got.cpu(), want.cpu())
        for field in ("history_indices", "write_indices", "retained_indices"):
            torch.testing.assert_close(getattr(tail_inputs, field).cpu(), getattr(expected_tail, field).cpu())
        expected_accepted = torch.ones(4, dtype=torch.int32)
        expected_accepted[:count] = accepted.cpu()[mapping]
        for value in metadata:
            torch.testing.assert_close(value.num_accepted_tokens.cpu(), expected_accepted)
            torch.testing.assert_close(value.spec_token_indx.cpu(), torch.arange(16, dtype=torch.int32))
        torch.testing.assert_close(sfa.block_table_buffer.cpu(), common.block_table_tensor.cpu())
        assert checksum.cpu().item() == (expected_accepted.sum() + common.block_table_tensor.cpu().sum()).item()
