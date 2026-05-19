#!/usr/bin/env python3
"""Apply experimental KV-quant SFA integration edits.

Run from repository root after checking out a v0.19.1rc1-based tree:

    python patches/apply_kv_quant_sfa_patch.py

The script intentionally uses conservative string patches and aborts if an
expected anchor is not found. Review the generated diff before committing.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, data: str) -> None:
    (ROOT / path).write_text(data, encoding="utf-8")


def replace_once(data: str, old: str, new: str, *, label: str) -> str:
    count = data.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return data.replace(old, new, 1)


def insert_after(data: str, anchor: str, text: str, *, label: str) -> str:
    idx = data.find(anchor)
    if idx < 0:
        raise RuntimeError(f"{label}: anchor not found")
    idx += len(anchor)
    return data[:idx] + text + data[idx:]


def patch_kv_cache_interface() -> None:
    path = "vllm_ascend/patch/platform/patch_kv_cache_interface.py"
    data = read(path)

    data = replace_once(
        data,
        "    c8_k_cache_dtype: torch.dtype = torch.int8\n"
        "    c8_k_scale_cache_dtype: torch.dtype = torch.float16\n",
        "    c8_k_cache_dtype: torch.dtype = torch.int8\n"
        "    c8_k_scale_cache_dtype: torch.dtype = torch.float16\n"
        "\n"
        "    # Experimental: enable packed KV-cache layout required by\n"
        "    # torch_npu.npu_kv_quant_sparse_flash_attention. In this mode,\n"
        "    # kv_cache[0] stores ckv/kr/scale repo data and kv_cache[1]\n"
        "    # is a fake empty kr cache. kv_cache[2]/[3] remain the DSA\n"
        "    # lightning-indexer cache and optional scale cache.\n"
        "    cache_kv_quant: bool = False\n"
        "    kv_quant_tile_size: int = 128\n"
        "    kv_quant_extra_dim: int = 16\n",
        label="kv-cache-interface-fields",
    )

    data = replace_once(
        data,
        "    @property\n"
        "    def page_size_bytes(self) -> int:\n"
        "        if self.cache_sparse_c8:\n",
        "    @property\n"
        "    def kv_quant_packed_last_dim(self) -> int:\n"
        "        assert self.sparse_head_dim is not None\n"
        "        kv_lora_rank, qk_rope_head_dim, _ = self.sparse_head_dim\n"
        "        return kv_lora_rank + 2 * qk_rope_head_dim + self.kv_quant_extra_dim\n"
        "\n"
        "    @property\n"
        "    def page_size_bytes(self) -> int:\n"
        "        if self.cache_kv_quant:\n"
        "            assert self.sparse_head_dim is not None\n"
        "            assert len(self.sparse_head_dim) == 3\n"
        "            num_heads_per_page = self.block_size * self.num_kv_heads\n"
        "            _, _, index_head_dim = self.sparse_head_dim\n"
        "\n"
        "            packed_bytes = (\n"
        "                num_heads_per_page\n"
        "                * self.kv_quant_packed_last_dim\n"
        "                * get_dtype_size(self.dtype)\n"
        "            )\n"
        "            if self.cache_sparse_c8:\n"
        "                indexer_k_bytes = (\n"
        "                    num_heads_per_page\n"
        "                    * index_head_dim\n"
        "                    * get_dtype_size(self.c8_k_cache_dtype)\n"
        "                )\n"
        "                indexer_k_scale_bytes = (\n"
        "                    num_heads_per_page\n"
        "                    * get_dtype_size(self.c8_k_scale_cache_dtype)\n"
        "                )\n"
        "            else:\n"
        "                indexer_k_bytes = (\n"
        "                    num_heads_per_page\n"
        "                    * index_head_dim\n"
        "                    * get_dtype_size(self.dtype)\n"
        "                )\n"
        "                indexer_k_scale_bytes = 0\n"
        "            return packed_bytes + indexer_k_bytes + indexer_k_scale_bytes\n"
        "\n"
        "        if self.cache_sparse_c8:\n",
        label="kv-cache-interface-page-size",
    )

    data = replace_once(
        data,
        "            cache_sparse_c8=cache_sparse_c8_set.pop(),\n"
        "        )\n",
        "            cache_sparse_c8=cache_sparse_c8_set.pop(),\n"
        "            cache_kv_quant=specs[0].cache_kv_quant,\n"
        "            kv_quant_tile_size=specs[0].kv_quant_tile_size,\n"
        "            kv_quant_extra_dim=specs[0].kv_quant_extra_dim,\n"
        "        )\n",
        label="kv-cache-interface-merge",
    )

    write(path, data)


def patch_model_runner() -> None:
    path = "vllm_ascend/worker/model_runner_v1.py"
    data = read(path)

    data = replace_once(
        data,
        "import math\nimport sys\nimport time\n",
        "import math\nimport os\nimport sys\nimport time\n",
        label="model-runner-import-os",
    )

    data = replace_once(
        data,
        "                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(\n"
        "                        block_size=self.block_size,\n"
        "                        num_kv_heads=1,\n"
        "                        head_size=sum(self.sparse_head_dim),\n"
        "                        sparse_head_dim=self.sparse_head_dim,\n"
        "                        dtype=self.kv_cache_dtype,\n"
        "                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,\n"
        "                        cache_sparse_c8=self.ascend_config.is_sparse_c8_layer(layer_name),\n"
        "                    )\n",
        "                    enable_kv_quant_sfa = os.getenv(\"VLLM_ASCEND_ENABLE_KV_QUANT_SFA\", \"0\") == \"1\"\n"
        "                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(\n"
        "                        block_size=self.block_size,\n"
        "                        num_kv_heads=1,\n"
        "                        head_size=sum(self.sparse_head_dim),\n"
        "                        sparse_head_dim=self.sparse_head_dim,\n"
        "                        dtype=self.kv_cache_dtype,\n"
        "                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,\n"
        "                        cache_sparse_c8=self.ascend_config.is_sparse_c8_layer(layer_name),\n"
        "                        cache_kv_quant=enable_kv_quant_sfa,\n"
        "                        kv_quant_tile_size=128,\n"
        "                    )\n"
        "                    if enable_kv_quant_sfa:\n"
        "                        logger.warning(\n"
        "                            \"[KVQSFA][spec] layer=%s sparse_head_dim=%s dtype=%s \"\n"
        "                            \"cache_sparse_c8=%s packed_last_dim=%s page_size_bytes=%s\",\n"
        "                            layer_name,\n"
        "                            self.sparse_head_dim,\n"
        "                            self.kv_cache_dtype,\n"
        "                            self.ascend_config.is_sparse_c8_layer(layer_name),\n"
        "                            kv_cache_spec[layer_name].kv_quant_packed_last_dim,\n"
        "                            kv_cache_spec[layer_name].page_size_bytes,\n"
        "                        )\n",
        label="model-runner-spec-enable",
    )

    marker = "                    if self.use_sparse:\n                        # for deepseek v3.2, we split the kv cache according to the corresponding ratio\n                        kv_cache_spec = layer_kv_cache_spec[layer_name]\n                        current_sparse_c8 = kv_cache_spec_uses_sparse_c8(kv_cache_spec)\n"
    data = replace_once(
        data,
        marker,
        marker + "                        current_kv_quant = getattr(kv_cache_spec, \"cache_kv_quant\", False)\n",
        label="model-runner-current-kv-quant",
    )

    data = replace_once(
        data,
        "                        sparse_kv_cache_ratio = kv_cache_spec.sparse_kv_cache_ratio\n"
        "                        k_tensor_split_factor = sparse_kv_cache_ratio[0]\n"
        "                        v_tensor_split_factor = sparse_kv_cache_ratio[1]\n"
        "                        dsa_k_tensor_split_factor = sparse_kv_cache_ratio[2]\n"
        "                        dsa_k_scale_tensor_split_factor = sparse_kv_cache_ratio[3] if current_sparse_c8 else None\n",
        "                        if current_kv_quant:\n"
        "                            packed_last_dim = kv_cache_spec.kv_quant_packed_last_dim\n"
        "                            _, _, index_head_dim = kv_cache_spec.sparse_head_dim\n"
        "                            page_tokens = kv_cache_spec.block_size * kv_cache_spec.num_kv_heads\n"
        "                            estimated_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes\n"
        "                            k_tensor_split_factor = kv_cache_tensor.size / (\n"
        "                                estimated_blocks * page_tokens * packed_last_dim * get_dtype_size(kv_cache_spec.dtype)\n"
        "                            )\n"
        "                            # kv_cache[1] is a fake empty kr cache for repo-mode kv quant.\n"
        "                            v_tensor_split_factor = float(\"inf\")\n"
        "                            if current_sparse_c8:\n"
        "                                dsa_k_tensor_split_factor = kv_cache_tensor.size / (\n"
        "                                    estimated_blocks * page_tokens * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)\n"
        "                                )\n"
        "                                dsa_k_scale_tensor_split_factor = kv_cache_tensor.size / (\n"
        "                                    estimated_blocks * page_tokens * get_dtype_size(self.c8_k_scale_cache_dtype)\n"
        "                                )\n"
        "                            else:\n"
        "                                dsa_k_tensor_split_factor = kv_cache_tensor.size / (\n"
        "                                    estimated_blocks * page_tokens * index_head_dim * get_dtype_size(kv_cache_spec.dtype)\n"
        "                                )\n"
        "                                dsa_k_scale_tensor_split_factor = None\n"
        "                        else:\n"
        "                            sparse_kv_cache_ratio = kv_cache_spec.sparse_kv_cache_ratio\n"
        "                            k_tensor_split_factor = sparse_kv_cache_ratio[0]\n"
        "                            v_tensor_split_factor = sparse_kv_cache_ratio[1]\n"
        "                            dsa_k_tensor_split_factor = sparse_kv_cache_ratio[2]\n"
        "                            dsa_k_scale_tensor_split_factor = sparse_kv_cache_ratio[3] if current_sparse_c8 else None\n",
        label="model-runner-split-factor",
    )

    data = replace_once(
        data,
        "                    k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)\n"
        "                    v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)\n",
        "                    k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)\n"
        "                    v_tensor_size = 0 if (self.use_sparse and current_kv_quant) else int(kv_cache_tensor.size // v_tensor_split_factor)\n",
        label="model-runner-fake-kr-size",
    )

    data = replace_once(
        data,
        "                    if self.use_sparse:\n                        dsa_k_cache_shape = (\n                            num_blocks,\n                            current_kv_cache_spec.block_size,\n                            current_kv_cache_spec.num_kv_heads,\n                            self.model_config.hf_text_config.index_head_dim,\n                        )\n",
        "                    if self.use_sparse:\n                        current_kv_quant = getattr(current_kv_cache_spec, \"cache_kv_quant\", False)\n                        if current_kv_quant:\n                            packed_shape = (\n                                num_blocks,\n                                current_kv_cache_spec.block_size,\n                                current_kv_cache_spec.num_kv_heads,\n                                current_kv_cache_spec.kv_quant_packed_last_dim,\n                            )\n                            fake_kr_shape = (\n                                num_blocks,\n                                current_kv_cache_spec.block_size,\n                                current_kv_cache_spec.num_kv_heads,\n                                0,\n                            )\n                            k_cache = raw_k_tensor.view(current_kv_cache_spec.dtype).view(packed_shape)\n                            v_cache = raw_v_tensor.view(current_kv_cache_spec.dtype).view(fake_kr_shape)\n                            logger.warning(\n                                \"[KVQSFA][reshape] layer=%s packed=%s/%s fake_kr=%s/%s\",\n                                layer_name, tuple(k_cache.shape), k_cache.dtype,\n                                tuple(v_cache.shape), v_cache.dtype,\n                            )\n\n                        dsa_k_cache_shape = (\n                            num_blocks,\n                            current_kv_cache_spec.block_size,\n                            current_kv_cache_spec.num_kv_heads,\n                            self.model_config.hf_text_config.index_head_dim,\n                        )\n",
        label="model-runner-reshape-packed",
    )

    write(path, data)


def patch_sfa_v1() -> None:
    path = "vllm_ascend/attention/sfa_v1.py"
    data = read(path)

    data = replace_once(
        data,
        "from dataclasses import dataclass\nfrom typing import TYPE_CHECKING, TypeVar\n",
        "from dataclasses import dataclass\nimport os\nfrom typing import TYPE_CHECKING, TypeVar\n",
        label="sfa-import-os",
    )

    data = insert_after(
        data,
        "BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024\n",
        "\nKVQSFA_DEBUG = os.getenv(\"VLLM_ASCEND_DEBUG_KVQSFA\", \"0\") == \"1\"\n\n\ndef _kvqsfa_debug_tensor_meta(name: str, tensor: torch.Tensor) -> None:\n    if not KVQSFA_DEBUG:\n        return\n    logger.warning(\n        \"[KVQSFA][tensor] %s shape=%s dtype=%s device=%s stride=%s contiguous=%s ptr=%s\",\n        name, tuple(tensor.shape), tensor.dtype, tensor.device, tensor.stride(),\n        tensor.is_contiguous(), hex(tensor.data_ptr()),\n    )\n\n\ndef _kvqsfa_debug_tensor_value(name: str, tensor: torch.Tensor, limit: int = 16) -> None:\n    if not KVQSFA_DEBUG:\n        return\n    if hasattr(torch.ops._C_ascend, \"device_print\") and hasattr(torch.ops._C_ascend, \"device_print_tensor\"):\n        torch.ops._C_ascend.device_print(f\"[KVQSFA][value] {name}\")\n        x = tensor.flatten()[:limit]\n        if x.dtype in (torch.float16, torch.bfloat16):\n            x = x.to(torch.float32)\n        torch.ops._C_ascend.device_print_tensor(x)\n\n",
        label="sfa-debug-helpers",
    )

    data = insert_after(
        data,
        "        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO\n",
        "\n        self.enable_kv_quant_sfa = os.getenv(\"VLLM_ASCEND_ENABLE_KV_QUANT_SFA\", \"0\") == \"1\"\n        self.kv_quant_tile_size = 128\n        if self.enable_kv_quant_sfa:\n            logger.warning(\"[KVQSFA] enabled for layer=%s\", kwargs.get(\"layer_name\"))\n",
        label="sfa-init-enable",
    )

    data = insert_after(
        data,
        "        if self.use_sparse_c8_indexer and AscendSFAImpl.k_hadamard is None:\n            AscendSFAImpl.k_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device=\"npu\") / (\n                128**0.5\n            )\n",
        "\n        if self.enable_kv_quant_sfa:\n            self.ckv_a_alpha = torch.nn.Parameter(\n                torch.ones(1, dtype=torch.float32, device=self.W_UK_T.device),\n                requires_grad=False,\n            )\n            if not hasattr(torch_npu, \"npu_kv_quant_sparse_flash_attention\"):\n                raise RuntimeError(\n                    \"VLLM_ASCEND_ENABLE_KV_QUANT_SFA=1 requires \"\n                    \"torch_npu.npu_kv_quant_sparse_flash_attention\"\n                )\n",
        label="sfa-process-alpha",
    )

    data = insert_after(
        data,
        "    def exec_kv(\n        self,\n        kv_no_split: torch.Tensor,\n        cos: torch.Tensor,\n        sin: torch.Tensor,\n        kv_cache: tuple,\n        slots: torch.Tensor,\n        attn_metadata: M,\n    ):\n",
        "",
        label="noop-anchor",
    )

    helper_anchor = "    # Return `ql_nope`, `q_pe`\n    def _q_proj_and_k_up_proj(self, x):\n"
    data = replace_once(
        data,
        helper_anchor,
        "    def _debug_kvqsfa_inputs(\n"
        "        self,\n"
        "        query: torch.Tensor,\n"
        "        packed_cache: torch.Tensor,\n"
        "        topk_indices: torch.Tensor,\n"
        "        block_table: torch.Tensor,\n"
        "        actual_seq_lengths_query: torch.Tensor,\n"
        "        actual_seq_lengths_key: torch.Tensor,\n"
        "    ) -> None:\n"
        "        if not KVQSFA_DEBUG:\n"
        "            return\n"
        "        _kvqsfa_debug_tensor_meta(\"query_cat_qnope_qpe\", query)\n"
        "        _kvqsfa_debug_tensor_meta(\"packed_cache\", packed_cache)\n"
        "        _kvqsfa_debug_tensor_meta(\"topk_indices\", topk_indices)\n"
        "        _kvqsfa_debug_tensor_meta(\"block_table\", block_table)\n"
        "        _kvqsfa_debug_tensor_meta(\"actual_seq_lengths_query\", actual_seq_lengths_query)\n"
        "        _kvqsfa_debug_tensor_meta(\"actual_seq_lengths_key\", actual_seq_lengths_key)\n"
        "        _kvqsfa_debug_tensor_value(\"topk_indices[:16]\", topk_indices)\n"
        "        _kvqsfa_debug_tensor_value(\"block_table[:2]\", block_table[:2])\n"
        "        _kvqsfa_debug_tensor_value(\"actual_seq_lengths_query\", actual_seq_lengths_query)\n"
        "        _kvqsfa_debug_tensor_value(\"actual_seq_lengths_key\", actual_seq_lengths_key)\n"
        "\n"
        "    # Return `ql_nope`, `q_pe`\n    def _q_proj_and_k_up_proj(self, x):\n",
        label="sfa-debug-method",
    )

    data = replace_once(
        data,
        "        block_table = attn_metadata.block_table\n"
        "        kv = kv_cache[0]\n"
        "        key_rope = kv_cache[1]\n\n"
        "        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(\n"
        "            query=ql_nope,\n"
        "            key=kv,\n"
        "            value=kv,\n"
        "            sparse_indices=topk_indices,\n"
        "            scale_value=self.scale,\n"
        "            sparse_block_size=1,\n"
        "            block_table=block_table,\n"
        "            actual_seq_lengths_query=actual_seq_lengths_query,\n"
        "            actual_seq_lengths_kv=actual_seq_lengths_key,\n"
        "            query_rope=q_pe,\n"
        "            key_rope=key_rope,\n"
        "            layout_query=\"TND\",\n"
        "            layout_kv=\"PA_BSND\",\n"
        "            sparse_mode=3,\n"
        "        )\n"
        "        return attn_output\n",
        "        block_table = attn_metadata.block_table\n"
        "        if self.enable_kv_quant_sfa:\n"
        "            packed_cache = kv_cache[0]\n"
        "            query = torch.cat([ql_nope, q_pe], dim=-1).contiguous()\n"
        "            actual_seq_lengths_query = actual_seq_lengths_query.to(torch.int32)\n"
        "            actual_seq_lengths_key = actual_seq_lengths_key.to(torch.int32)\n"
        "            self._debug_kvqsfa_inputs(\n"
        "                query, packed_cache, topk_indices, block_table,\n"
        "                actual_seq_lengths_query, actual_seq_lengths_key,\n"
        "            )\n"
        "            return torch_npu.npu_kv_quant_sparse_flash_attention(\n"
        "                query=query,\n"
        "                key=packed_cache,\n"
        "                value=packed_cache,\n"
        "                sparse_indices=topk_indices,\n"
        "                scale_value=self.scale,\n"
        "                sparse_block_size=1,\n"
        "                block_table=block_table,\n"
        "                actual_seq_lengths_query=actual_seq_lengths_query,\n"
        "                actual_seq_lengths_kv=actual_seq_lengths_key,\n"
        "                layout_query=\"TND\",\n"
        "                layout_kv=\"PA_BSND\",\n"
        "                sparse_mode=3,\n"
        "                key_quant_mode=2,\n"
        "                value_quant_mode=2,\n"
        "                attention_mode=2,\n"
        "                quant_scale_repo_mode=1,\n"
        "                tile_size=self.kv_quant_tile_size,\n"
        "                rope_head_dim=self.qk_rope_head_dim,\n"
        "                key_dequant_scale=None,\n"
        "                value_dequant_scale=None,\n"
        "            )\n\n"
        "        kv = kv_cache[0]\n"
        "        key_rope = kv_cache[1]\n"
        "\n"
        "        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(\n"
        "            query=ql_nope,\n"
        "            key=kv,\n"
        "            value=kv,\n"
        "            sparse_indices=topk_indices,\n"
        "            scale_value=self.scale,\n"
        "            sparse_block_size=1,\n"
        "            block_table=block_table,\n"
        "            actual_seq_lengths_query=actual_seq_lengths_query,\n"
        "            actual_seq_lengths_kv=actual_seq_lengths_key,\n"
        "            query_rope=q_pe,\n"
        "            key_rope=key_rope,\n"
        "            layout_query=\"TND\",\n"
        "            layout_kv=\"PA_BSND\",\n"
        "            sparse_mode=3,\n"
        "        )\n"
        "        return attn_output\n",
        label="sfa-attention-call",
    )

    write(path, data)


def main() -> None:
    patch_kv_cache_interface()
    patch_model_runner()
    patch_sfa_v1()
    print("Applied experimental KV-quant SFA edits. Review `git diff` before committing.")


if __name__ == "__main__":
    main()
