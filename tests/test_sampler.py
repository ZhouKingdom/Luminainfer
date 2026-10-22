"""采样器单测：直接测 Triton 内核（greedy/top_k/repetition_penalty）与 numpy 参考对齐。

这是最容易藏静默 bug 的地方（Gumbel-max、top_k/top_p 二分阈值、重复惩罚），
用确定性（greedy）路径和 numpy 参考做精确比对。
"""

import numpy as np
import torch

from zjhvllm.layers.sampler import Sampler


def _run_sampler(logits, *, top_k=-1, top_p=1.0, penalty=1.0, history=None):
    """以 greedy 模式调用 Sampler，返回 int。"""
    logits = torch.tensor(logits, dtype=torch.float32, device="cuda").unsqueeze(0)  # [1, vocab]
    sampler = Sampler()
    temperatures = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    top_ks = torch.tensor([top_k], dtype=torch.int32, device="cuda")
    top_ps = torch.tensor([top_p], dtype=torch.float32, device="cuda")
    penalties = torch.tensor([penalty], dtype=torch.float32, device="cuda")
    seeds = torch.tensor([0], dtype=torch.int64, device="cuda")
    greedy = torch.tensor([1], dtype=torch.int32, device="cuda")

    hist_ids = hist_lens = None
    if history is not None:
        hist_ids = torch.tensor([list(history)], dtype=torch.int32, device="cuda")
        hist_lens = torch.tensor([len(history)], dtype=torch.int32, device="cuda")

    out = sampler(logits, temperatures, top_ks, top_ps, penalties, seeds, greedy, hist_ids, hist_lens)
    return int(out.item())


def _ref(logits, *, top_k=-1, penalty=1.0, history=()):
    """numpy 参考实现：温度缩放 + 重复惩罚 + top_k，然后 argmax（greedy）。"""
    x = np.array(logits, dtype=np.float64).copy()
    for t in history:
        if x[t] > 0:
            x[t] /= penalty
        else:
            x[t] *= penalty
    if 0 < top_k < len(x):
        kth = np.partition(x, -top_k)[-top_k]
        x[x < kth] = -np.inf
    return int(np.argmax(x))


LOGITS = [0.1, 2.0, 1.5, -3.0, 0.7, -1.0, 0.3, 2.5]


def test_greedy_matches_argmax():
    assert _run_sampler(LOGITS) == _ref(LOGITS)


def test_topk_matches_numpy():
    for k in [1, 2, 3, 5]:
        assert _run_sampler(LOGITS, top_k=k) == _ref(LOGITS, top_k=k), f"top_k={k}"


def test_repetition_penalty_matches_numpy():
    # 惩罚 token 7（logit=2.5，原本是 argmax）→ 2.5/1.5=1.667，argmax 变成 token 1
    history = (7,)
    assert _run_sampler(LOGITS, penalty=1.5, history=history) == _ref(LOGITS, penalty=1.5, history=history)


def test_repetition_penalty_negative_logit():
    # 惩罚一个负 logit 的 token（logit=-3.0，token 3）→ -3.0*1.5=-4.5，仍不是 argmax
    history = (3,)
    assert _run_sampler(LOGITS, penalty=1.5, history=history) == _ref(LOGITS, penalty=1.5, history=history)
