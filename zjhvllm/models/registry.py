"""模型注册表：根据 config.model_type 分发到对应的模型实现。

新增模型家族时，在这里注册即可；ModelRunner 通过 get_model_cls 拿到模型类。
"""

from zjhvllm.models.qwen3 import Qwen3ForCausalLM
from zjhvllm.models.qwen3_moe import Qwen3MoeForCausalLM
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM


_MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
    "qwen3_5_text": Qwen3_5ForCausalLM,
}


def get_model_cls(model_type: str):
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"不支持的模型类型 '{model_type}'，已注册: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[model_type]


def register_model(model_type: str, model_cls):
    """注册一个新的模型类型（供扩展）。"""
    _MODEL_REGISTRY[model_type] = model_cls
    return model_cls
