"""张量并行（TP）的全局 size/rank 状态。

把 tp_size/tp_rank 从 dist.get_world_size()/get_rank() 解耦，这样专家并行（EP）模式下
可以让非 MoE 层 replicated（tp=1）而 dist world_size 是 EP 大小（专家分布）。默认回落
到 dist 的值，纯 TP 模式行为不变。
"""

import torch.distributed as dist

_TP_SIZE = None
_TP_RANK = None


def set_tp(size: int, rank: int):
    """由 ModelRunner 在初始化时设置（EP 模式下 size=1, rank=0）。"""
    global _TP_SIZE, _TP_RANK
    _TP_SIZE = size
    _TP_RANK = rank


def get_tp_size() -> int:
    return _TP_SIZE if _TP_SIZE is not None else (dist.get_world_size() if dist.is_initialized() else 1)


def get_tp_rank() -> int:
    return _TP_RANK if _TP_RANK is not None else (dist.get_rank() if dist.is_initialized() else 0)
