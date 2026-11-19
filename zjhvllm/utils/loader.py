import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


_EXPERT_RE = re.compile(r"^(.+)experts\.(\d+)\.(.+)$")


def load_model(model: nn.Module, path: str, expert_start: int | None = None, expert_end: int | None = None):
    """加载权重。expert_start/expert_end 非 None 时，只加载 expert id 在 [start, end) 的
    专家（EP 模式下每卡只加载本地专家），并把专家 id 重映射到本地 index。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_params = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                original_name = weight_name  # safetensors 文件里的原始名（全局专家 id）
                # EP 专家过滤 + 重映射（只用于找参数名，读张量仍用 original_name）
                if expert_start is not None:
                    m = _EXPERT_RE.match(weight_name)
                    if m:
                        e = int(m.group(2))
                        if not (expert_start <= e < expert_end):
                            continue  # 跳过其他 rank 的专家
                        weight_name = f"{m.group(1)}experts.{e - expert_start}.{m.group(3)}"
                # 模型自定义权重名重映射（如 Qwen3.5 的 model.language_model.* 前缀剥离、
                # 跳过 mtp.* 等非文本权重）。返回 None 表示跳过该权重。
                remap = getattr(model, "remap_weight_name", None)
                if remap is not None:
                    weight_name = remap(weight_name)
                    if weight_name is None:
                        continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(original_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(original_name))
                    loaded_params.add(weight_name)
    required_mtp = {name for name, _ in model.named_parameters() if name.startswith("mtp.")}
    if missing := required_mtp - loaded_params:
        raise ValueError(f"Missing MTP checkpoint weights: {sorted(missing)}")
    if required_mtp:
        model.mtp.loaded_weight_names = frozenset(required_mtp)
