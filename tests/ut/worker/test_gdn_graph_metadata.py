# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker import device_metadata
from vllm_ascend.worker.v2.model_states import gdn_graph


@pytest.fixture
def graph_state(monkeypatch):
    class Executor:
        submission_in_flight = False

        def __init__(self, **kwargs):
            self.calls = []

        def submit(self, tasks):
            self.calls.append("submit")
            self.submission_in_flight = True

        def wait(self, stage, group):
            self.calls.append("wait")

        def release(self):
            self.calls.append("release")
            self.submission_in_flight = False

    class Event:
        done = False

        def record(self, stream):
            pass

        def query(self):
            return self.done

    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: empty(*args, **{key: value for key, value in kwargs.items() if key != "pin_memory"}),
    )
    monkeypatch.setattr(device_metadata, "DeviceMetadataExecutor", Executor)
    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: None)
    return gdn_graph.GDNGraphMetadata(torch.tensor([1, 4, 2, 3, 1, 2, 4, 1], dtype=torch.int32), 4)


def make_batch(count=4):
    return SimpleNamespace(
        num_reqs=count,
        num_tokens=count * 4,
        num_reqs_after_padding=4,
        num_tokens_after_padding=16,
        req_ids=[f"req-{i}" for i in range(count)],
        idx_mapping_np=np.arange(count, dtype=np.int64),
        decode_graph_eligible=True,
        query_start_loc=torch.arange(5, dtype=torch.int32) * 4,
        seq_lens=torch.full((4,), 32, dtype=torch.int32),
        positions=torch.arange(16, dtype=torch.int64),
    )


def test_staging_reuses_device_input_and_retains_pending_host_uploads(graph_state):
    batch = make_batch()
    tables, slots = (torch.zeros(4, 8, dtype=torch.int32),), torch.zeros(2, 16, dtype=torch.int32)
    address = graph_state.inputs.batch.data_ptr()
    graph_state.stage(batch, tables, slots, for_capture=True)
    first_host, first_copy = graph_state._uploads[0]
    batch.num_reqs, batch.num_tokens = 1, 4
    batch.idx_mapping_np = np.array([6], dtype=np.int64)
    graph_state.stage(batch, tables, slots, for_capture=False)
    assert graph_state.inputs.batch.data_ptr() == address
    assert graph_state.inputs.batch.tolist() == [1, 6, -1, -1, -1, -1, -1, -1, -1]
    assert first_host.tolist()[:5] == [4, 0, 1, 2, 3]
    assert len(graph_state._uploads) == 2
    first_copy.done = True
    graph_state.stage(batch, tables, slots, for_capture=False)
    assert all(host is not first_host for host, _ in graph_state._uploads)


@pytest.mark.parametrize("reason", ["address", "stride", "tokens", "capacity", "in_flight", "unretired"])
def test_invalid_graph_inputs_are_rejected_before_upload(graph_state, reason):
    batch = make_batch()
    tables, slots = (torch.zeros(4, 8, dtype=torch.int32),), torch.zeros(2, 16, dtype=torch.int32)
    graph_state.stage(batch, tables, slots, for_capture=True)
    if reason == "address":
        tables = (tables[0].clone(),)
    elif reason == "stride":
        tables = (tables[0].as_strided((4, 4), (8, 2)),)
    elif reason == "tokens":
        batch.num_tokens -= 1
    elif reason == "capacity":
        batch.num_reqs = 9
    elif reason == "in_flight":
        graph_state.executor.submission_in_flight = True
    else:
        graph_state._tasks = (object(),)
    with pytest.raises((ValueError, RuntimeError)):
        graph_state.stage(batch, tables, slots, for_capture=False)
    assert len(graph_state._uploads) == 1


@pytest.mark.parametrize("reason", [None, "width", "prefill", "drafts", "tokens", "sentinel", "index"])
def test_batch_eligibility_excludes_unsafe_full_replay(graph_state, reason):
    batch = make_batch()
    drafts = {req_id: [0, 0, 0] for req_id in batch.req_ids}
    width = 4
    if reason == "width":
        width = None
    elif reason == "prefill":
        batch.decode_graph_eligible = False
    elif reason == "drafts":
        drafts[batch.req_ids[0]] = []
    elif reason == "tokens":
        batch.num_tokens -= 1
    elif reason == "sentinel":
        batch.idx_mapping_np[0] = -1
    elif reason == "index":
        batch.idx_mapping_np[0] = graph_state.inputs.accepted_state.numel()
    assert graph_state.eligible(batch, drafts, width) == (reason is None)


def test_graph_prologue_joins_before_model_and_does_not_resubmit_on_replay(graph_state):
    class Builder(device_metadata.DeferredMetadataBuilder):
        pass

    builder = Builder()
    builder.init_device_metadata()
    builder.graph_decode_inputs = graph_state.inputs
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]
    output = torch.empty(4)
    with graph_state.build(groups, True):
        builder.run_device_metadata(device_metadata.DeviceMetadataStage.ATTENTION, output, lambda: None)
    assert graph_state.executor.calls == []
    graph_state.begin_forward()
    assert graph_state.executor.calls == ["submit", "wait"]
    graph_state.finish()
    assert graph_state.executor.calls[-1] == "release"
    before = list(graph_state.executor.calls)
    with graph_state.build(groups, True):
        builder.run_device_metadata(device_metadata.DeviceMetadataStage.ATTENTION, output, lambda: None)
    graph_state.finish_replay()
    assert graph_state.executor.calls == before
    calls = []
    builder.run_device_metadata(device_metadata.DeviceMetadataStage.ATTENTION, output, lambda: calls.append("eager"))
    assert calls == ["eager"]


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "architecture",
        "cache",
        "no_spec",
        "connector",
        "cp",
        "adaptive",
        "dynamic",
        "c8",
        "backend",
        "conv",
        "blocks",
        "rope",
    ],
)
def test_factory_only_enables_the_supported_target_topology(graph_state, monkeypatch, reason):
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder

    gdn = object.__new__(AscendGDNHostMetadataBuilder)
    gdn._ensure_conv_index_buffers = lambda: None
    gdn._spec_conv_indices = torch.empty(4)
    gdn.num_spec = 3
    gdn.kv_cache_spec = SimpleNamespace(num_speculative_blocks=3)
    sfa = object.__new__(AscendSFAMetadataBuilder)
    sfa.nope = True
    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="Glm5NextForCausalLM"),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        num_speculative_tokens=3,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(separate_routine=lambda: True, decode_mode=lambda: CUDAGraphMode.FULL)
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=False, uses_dynamic_speculative_decoding=lambda: False
        ),
    )
    flags = SimpleNamespace(enable_sparse_sfa_c8=True)
    builders = [gdn, sfa]
    if reason == "architecture":
        config.model_config.architecture = "OtherModel"
    elif reason == "cache":
        config.cache_config.mamba_cache_mode = "none"
    elif reason == "no_spec":
        config.num_speculative_tokens = 0
    elif reason == "connector":
        config.kv_transfer_config = object()
    elif reason == "cp":
        config.parallel_config.decode_context_parallel_size = 2
    elif reason == "adaptive":
        config.speculative_config.enable_adaptive_verification = True
    elif reason == "dynamic":
        config.speculative_config.uses_dynamic_speculative_decoding = lambda: True
    elif reason == "c8":
        flags.enable_sparse_sfa_c8 = False
    elif reason == "backend":
        builders.append(object())
    elif reason == "conv":
        gdn._spec_conv_indices = None
    elif reason == "blocks":
        gdn.kv_cache_spec.num_speculative_blocks = 2
    elif reason == "rope":
        sfa.nope = False
    monkeypatch.setattr(gdn_graph, "get_ascend_config", lambda: flags)
    monkeypatch.setattr(gdn_graph, "GDNGraphMetadata", lambda *args: graph_state)
    groups = [[SimpleNamespace(get_metadata_builder=lambda _, builder=builder: builder)] for builder in builders]
    state = gdn_graph.create_gdn_graph_metadata(
        config, SimpleNamespace(num_accepted_tokens_gpu=graph_state.inputs.accepted_state), groups
    )
    if reason is None:
        assert state is graph_state
        assert gdn.graph_decode_inputs is sfa.graph_decode_inputs is state.inputs
    else:
        assert state is None


def test_model_state_capture_and_replay_use_the_same_graph_preparation(graph_state, monkeypatch):
    from vllm_ascend.worker.v2.model_states import mamba_hybrid

    calls = []
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", lambda **kwargs: calls.append(kwargs) or {})
    state = SimpleNamespace(
        device_metadata=graph_state,
        vllm_config=SimpleNamespace(num_speculative_tokens=3, parallel_config=None),
        max_model_len=1024,
    )
    batch = make_batch()
    batch.is_prefilling_np = np.zeros(4, dtype=bool)
    batch.prefill_runs_as_decode_np = None
    batch.uniform_token_count = 4
    batch.num_scheduled_tokens = np.full(4, 4, dtype=np.int32)
    batch.query_start_loc_np = np.arange(5, dtype=np.int32) * 4
    batch.seq_lens_np = np.full(4, 32, dtype=np.int32)
    batch.dcp_local_seq_lens, batch.attn_state = None, None
    tables, slots = (torch.ones(4, 8, dtype=torch.int32),), torch.zeros(2, 16, dtype=torch.int32)
    mamba_hybrid.AscendMambaHybridModelState.prepare_attn(
        state, batch, CUDAGraphMode.NONE, tables, slots, [], None, for_capture=True
    )
    batch.num_reqs, batch.num_tokens = 1, 4
    batch.idx_mapping_np = np.array([6], dtype=np.int64)
    batch.is_prefilling_np = np.zeros(1, dtype=bool)
    batch.num_scheduled_tokens = np.full(1, 4, dtype=np.int32)
    mamba_hybrid.AscendMambaHybridModelState.prepare_attn(state, batch, CUDAGraphMode.FULL, tables, slots, [], None)
    assert len(calls) == 2
    for call in calls:
        assert call["model_specific_attn_metadata"].num_accepted_tokens is None
    assert calls[0]["for_cudagraph_capture"]
    assert calls[1]["full_graph_mode"]
    assert calls[1]["num_actual_reqs"] == 1 and calls[1]["num_reqs"] == 4
    assert graph_state.inputs.batch[:2].tolist() == [1, 6]


@pytest.mark.parametrize("width", [None, 1, 4])
def test_runner_filters_unsupported_batches_before_graph_dispatch(graph_state, monkeypatch, width):
    from collections import namedtuple

    from vllm_ascend.worker.v2 import model_runner

    batch = make_batch()
    fields = namedtuple("BatchState", "req_ids num_tokens idx_mapping_np decode_graph_eligible")
    value = fields(batch.req_ids, batch.num_tokens, batch.idx_mapping_np, True)
    monkeypatch.setattr(model_runner.GPUModelRunner, "gather_batch_req_state", lambda *args: (value, width))
    runner = object.__new__(model_runner.NPUModelRunner)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.model_state = SimpleNamespace(device_metadata=graph_state)
    schedule = SimpleNamespace(scheduled_spec_decode_tokens={req_id: [0, 0, 0] for req_id in batch.req_ids})
    result, actual_width = runner.gather_batch_req_state(schedule, False)
    assert result.decode_graph_eligible == (width == 4)
    assert actual_width == (4 if width == 4 else None)
