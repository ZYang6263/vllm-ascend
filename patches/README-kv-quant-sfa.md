# Experimental KV-quant Sparse Flash Attention integration

This patch set is a guarded bring-up plan for replacing the existing SFA `npu_sparse_flash_attention` path with CANN/PTA `npu_kv_quant_sparse_flash_attention` in the vLLM-Ascend SFA backend.

Base used for the patch: upstream `vllm-project/vllm-ascend` tag `v0.19.1rc1` (`da421afad7192dac64e39ae1d32305d57344f3cf`).

## Files intended to be changed

- `vllm_ascend/patch/platform/patch_kv_cache_interface.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/attention/sfa_v1.py`

## Enable flag

The implementation is intentionally guarded by an env var:

```bash
export VLLM_ASCEND_ENABLE_KV_QUANT_SFA=1
export VLLM_ASCEND_DEBUG_KVQSFA=1
```

With the flag disabled, the existing SFA path should keep using `_C_ascend.npu_sparse_flash_attention`.

## Apply locally

From a checkout based on `v0.19.1rc1`:

```bash
git apply patches/0001-experimental-kv-quant-sfa.patch
```

Then run formatting and targeted tests/builds in your CANN/PTA environment.

## Important limitations

This is a draft integration patch generated from code inspection. It was not compiled inside a CANN/torch_npu runtime. Before production use, validate:

1. `torch_npu.npu_mla_prolog_v3` signature in your installed PTA.
2. `torch_npu.npu_kv_quant_sparse_flash_attention` signature in your installed PTA.
3. Packed KV cache layout: `[num_blocks, block_size, 1, kv_lora_rank + 2 * qk_rope_head_dim + 16]`.
4. SFA output layout before `_v_up_proj`; vLLM-Ascend expects TND here, so do not blindly copy recipe's `transpose(0, 1)`.
5. CP / PD-disaggregation paths are intentionally guarded out in the first bring-up.
