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
    scheduling_policy: str = "alternating"  # alternating, bounded, or adaptive
    cache_schedule_metadata: bool = True  # reuse host-side sequence offsets/slots
    active_prefill_tokens: int = 512  # prefill budget while decode requests exist
    adaptive_tpot_seconds: float = .1
    adaptive_ttft_seconds: float = 2.
    adaptive_min_prefill_tokens: int = 128
    distributed_port: int = 2333  # 独立引擎进程需使用不同 rendezvous 端口
    enforce_eager: bool = False
    enable_mtp: bool = False  # opt-in Qwen3.5 MTP-1; request-private draft KV
    quantization: str | None = None  # experimental "int8_weight_only": Qwen3.5 MLP W8A16
    mtp_batch_draft: bool = False  # opt-in packed draft projection/head and one IPC per batch
    mtp_num_steps: int = 1  # recurrent top-1 draft chain, not number of checkpoint layers
    verification_state_commit: bool = True  # select saved recurrent state instead of model replay
    verification_state_budget_mb: int = 512  # workspace budget per configured request
    verification_cuda_graph: bool = False  # opt-in BF16 TP MTP-1 target replay; bounded exact layouts
    verification_fused_gdn: bool = False  # opt-in short-window recurrent verify kernel
    verification_graph_max_layouts: int = 16
    qwen35_linear_backend: str = "decode_equivalent"  # decode_equivalent | batch_invariant | batched (batch rows, accept BF16 rounding)
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_state_slots: int = -1  # linear attention 递归状态的可分配 slot 数（model_runner 按显存算）
    kv_cache_dtype: str = "auto"  # "auto"=bf16, "fp8"=fp8 e4m3 KV cache

    def __post_init__(self):
        if self.quantization not in (None, "int8_weight_only"):
            raise ValueError("quantization must be None or int8_weight_only")
        if self.quantization is not None and self.enable_mtp:
            raise ValueError("weight quantization currently supports ordinary decoding only; disable MTP")
        if type(self.mtp_num_steps) is not int or not 1 <= self.mtp_num_steps <= 8:
            raise ValueError("mtp_num_steps must be an integer between 1 and 8")
        if not 1 <= self.verification_graph_max_layouts <= 64:
            raise ValueError("verification_graph_max_layouts must be between 1 and 64")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.scheduling_policy in ("alternating", "bounded", "adaptive")
        assert self.adaptive_tpot_seconds > 0 and self.adaptive_ttft_seconds > 0
        assert self.adaptive_min_prefill_tokens > 0
        assert self.verification_state_budget_mb >= 0
        assert self.qwen35_linear_backend in ("decode_equivalent", "batch_invariant", "batched")
        assert self.active_prefill_tokens > 0
        assert 1 <= self.distributed_port <= 65535
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.expert_parallel_size <= 8
        # 纯 EP 模式（非 MoE 层 replicated）暂不支持与 TP 叠加，要求 TP==1
        if self.expert_parallel_size > 1:
            assert self.tensor_parallel_size == 1, "EP>1 时暂不支持 TP 叠加（2D 并行未实现）"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 多模态模型（如 Qwen3.5 的 Qwen3_5ForConditionalGeneration）：文本结构在 text_config 下
        if getattr(self.hf_config, "text_config", None) is not None:
            self.hf_config = self.hf_config.text_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.quantization is not None:
            import torch
            if self.hf_config.model_type != "qwen3_5_text" or self.hf_config.dtype != torch.bfloat16:
                raise ValueError("int8_weight_only requires BF16 Qwen3.5 text weights")
            if getattr(self.hf_config, "quantization_config", None):
                raise ValueError("int8_weight_only loads BF16 source checkpoints, not prequantized checkpoints")
            if self.qwen35_linear_backend != "decode_equivalent":
                raise ValueError("weight quantization cannot be combined with experimental linear backends")
        if self.hf_config.model_type == "qwen3_5_text":
            if self.tensor_parallel_size not in (1, 2, 4, 8) or self.expert_parallel_size != 1:
                raise ValueError("Qwen3.5 supports TP=1/2/4/8, EP=1")
        if self.enable_mtp:
            if not 1 <= self.max_num_seqs <= 8:
                raise ValueError("MTP requires an explicit max_num_seqs between 1 and 8")
            if (self.hf_config.model_type != "qwen3_5_text"
                    or self.expert_parallel_size != 1):
                raise ValueError("MTP requires Qwen3.5, TP=1/2/4/8 and EP=1")
            cfg = self.hf_config
            fields = ("layer_types", "linear_num_key_heads", "linear_num_value_heads",
                      "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim")
            if self.mtp_num_steps > 1 and all(hasattr(cfg, name) for name in fields):
                import torch
                itemsize = torch.empty((), dtype=cfg.dtype).element_size()
                nv = cfg.linear_num_value_heads // self.tensor_parallel_size
                nk = cfg.linear_num_key_heads // self.tensor_parallel_size
                hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
                per_step = cfg.layer_types.count("linear_attention") * (
                    nv*hk*hv*4 + (2*nk*hk+nv*hv)*(cfg.linear_conv_kernel_dim-1)*itemsize)
                minimum = ((self.mtp_num_steps+1)*per_step+1024**2-1)//1024**2
                if self.verification_state_budget_mb < minimum:
                    raise ValueError(f"MTP-{self.mtp_num_steps} needs verification_state_budget_mb >= {minimum}")
