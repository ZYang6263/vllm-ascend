import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig


_ORIGINAL_DEEPSEEK_V2_ATTENTION_INIT = deepseek_v2.DeepseekV2Attention.__init__


def _should_skip_indexer_init(
    config: DeepseekV2Config | DeepseekV3Config,
    prefix: str,
) -> bool:
    if not hasattr(config, "index_topk"):
        return False

    layer_id = deepseek_v2.extract_layer_index(prefix)
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if index_topk_pattern is None:
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        skip_topk = (
            max(layer_id - index_skip_topk_offset + 1, 0)
            % index_topk_freq
            != 0
        )
    elif 0 <= layer_id < len(index_topk_pattern):
        skip_topk = index_topk_pattern[layer_id] == "S"
    else:
        skip_topk = False

    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    is_mtp_layer = num_hidden_layers is not None and layer_id >= num_hidden_layers
    return skip_topk and not is_mtp_layer


def _deepseek_v2_attention_init(
    self,
    vllm_config: VllmConfig,
    config: DeepseekV2Config | DeepseekV3Config,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    max_position_embeddings: int = 8192,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    topk_indices_buffer=None,
    prefix: str = "",
) -> None:
    if not _should_skip_indexer_init(config, prefix):
        return _ORIGINAL_DEEPSEEK_V2_ATTENTION_INIT(
            self,
            vllm_config,
            config,
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            max_position_embeddings,
            cache_config,
            quant_config,
            topk_indices_buffer,
            prefix,
        )

    original_indexer = deepseek_v2.Indexer
    deepseek_v2.Indexer = _skip_indexer_init
    try:
        _ORIGINAL_DEEPSEEK_V2_ATTENTION_INIT(
            self,
            vllm_config,
            config,
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            max_position_embeddings,
            cache_config,
            quant_config,
            topk_indices_buffer,
            prefix,
        )
    finally:
        deepseek_v2.Indexer = original_indexer

    self.indexer_rope_emb = None
    self.indexer = None


def _skip_indexer_init(*args, **kwargs):
    return None


deepseek_v2.DeepseekV2Attention.__init__ = _deepseek_v2_attention_init
