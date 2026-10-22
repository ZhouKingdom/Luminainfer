import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1  # 专家并行（EP）：MoE 专家按卡分布，>1 时要求 tensor_parallel_size==1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "auto"  # "auto"=bf16, "fp8"=fp8 e4m3 KV cache

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.expert_parallel_size <= 8
        # 纯 EP 模式（非 MoE 层 replicated）暂不支持与 TP 叠加，要求 TP==1
        if self.expert_parallel_size > 1:
            assert self.tensor_parallel_size == 1, "EP>1 时暂不支持 TP 叠加（2D 并行未实现）"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
