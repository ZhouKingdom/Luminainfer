"""正确性测试共享 fixtures。

设计：
- 模型层测试（test_logits.py）用 `our_model` 直接实例化 Qwen3ForCausalLM，对比 HF 的 logits。
- 引擎层测试（test_generate.py）用 `llm`（tp=1，enforce_eager=True）走完整 generate 流程。
- 采样器单测（test_sampler.py）用 `Sampler` 类直接测 Triton 内核逻辑。

所有 fixture 都是 session 级，模型只加载一次。
"""

import os

import pytest
import torch
import torch.distributed as dist

# 测试全程离线（模型已下载到本地），避免 from_pretrained 尝试联网
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B")


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    """初始化 world_size=1 的进程组，并把 init_process_group 打补丁成 no-op。

    原因：zjhvllm 的模型层（linear/embed_head/attention）都调用 dist.get_rank()/get_world_size()，
    而引擎 ModelRunner 内部还会硬编码 init_process_group("nccl", "localhost:2333", ...)。
    这里统一 init 一次（gloo, world_size=1），并让引擎内的重复 init 变成空操作，避免冲突。
    """
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://localhost:2335", world_size=1, rank=0)
    _orig = dist.init_process_group
    dist.init_process_group = lambda *args, **kwargs: None
    yield
    dist.init_process_group = _orig


@pytest.fixture(scope="session")
def model_path():
    assert os.path.isdir(MODEL_PATH), f"模型不存在: {MODEL_PATH}（请先下载 Qwen3-0.6B）"
    return MODEL_PATH


@pytest.fixture(scope="session")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


@pytest.fixture(scope="session")
def hf_model(model_path):
    """HuggingFace 参考模型（bf16, GPU）。"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    return model


@pytest.fixture(scope="session")
def hf_config(model_path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


@pytest.fixture(scope="session")
def our_model(hf_config, model_path):
    """直接实例化 zjhvllm 的 Qwen3ForCausalLM 并加载权重（绕过引擎/调度器）。"""
    from zjhvllm.models.qwen3 import Qwen3ForCausalLM
    from zjhvllm.utils.loader import load_model

    prev_dtype = torch.get_default_dtype()
    prev_device = torch.get_default_device()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")
    try:
        model = Qwen3ForCausalLM(hf_config)
        load_model(model, model_path)
        model.eval()
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)
    return model


@pytest.fixture(scope="session")
def llm(model_path):
    """zjhvllm 引擎（tp=1，eager 模式，确定性，bf16 KV cache）。

    max_num_batched_tokens 设得较小（512），这样长 prompt 会触发 chunked prefill；
    gpu_memory_utilization 设得较小（0.1），避免测试时把整张卡显存占满。
    fp8 KV cache 的测试在 tests/test_fp8.py 里用子进程隔离跑。
    """
    from zjhvllm import LLM

    engine = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.1,
    )
    return engine
