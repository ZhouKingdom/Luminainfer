"""专家并行（EP）正确性：多卡 EP 的 MoE 输出 == 单卡全量 MoE 输出。

EP 跑在子进程（tests/_ep_runner.py，world_size 卡 NCCL），父进程用单卡 reference 比对。
权重/输入都是确定性的（seed 固定），所以比对应是精确的。
"""

import json
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from zjhvllm.layers.ep_moe import LocalExpertFFN, _randn

WORLD_SIZE = 4
HIDDEN, INTER, N_EXPERTS, N_TOPK, N_TOKENS = 32, 16, 8, 2, 64


def _reference(norm_topk_prob: bool) -> torch.Tensor:
    """单卡全量 MoE 的逐 token 参考实现（确定性权重，与 EP 各 rank 一致）。"""
    torch.manual_seed(0)
    x = torch.randn(N_TOKENS, HIDDEN, device="cuda")

    gate = nn.Linear(HIDDEN, N_EXPERTS, bias=False).cuda()
    with torch.no_grad():
        gate.weight.copy_(_randn((N_EXPERTS, HIDDEN), 999_999))
    experts = [LocalExpertFFN(HIDDEN, INTER, expert_id=e).cuda() for e in range(N_EXPERTS)]

    with torch.inference_mode():
        routing = gate(x)
        w, ids = torch.topk(routing, N_TOPK, dim=-1)
        w = w.float()
        if norm_topk_prob:
            w = F.softmax(w, dim=-1)

        out = torch.zeros_like(x, dtype=torch.float32)
        for i in range(N_TOKENS):
            for k in range(N_TOPK):
                e = int(ids[i, k])
                out[i] += w[i, k] * experts[e](x[i:i + 1]).squeeze(0)
    return out


def test_ep_matches_single_gpu():
    runner = os.path.join(os.path.dirname(__file__), "_ep_runner.py")
    for norm in [False, True]:
        proc = subprocess.run([sys.executable, runner, str(WORLD_SIZE), str(int(norm))],
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, f"EP runner 失败:\n{proc.stderr}"

        result_line = next(l for l in proc.stdout.splitlines() if l.startswith("__RESULT__"))
        ep_out = torch.tensor(json.loads(result_line[len("__RESULT__"):]), device="cuda")

        ref = _reference(norm)
        max_diff = (ep_out - ref).abs().max().item()
        print(f"\n[norm_topk_prob={norm}] EP vs 单卡 max_abs_diff={max_diff:.2e}")
        assert torch.allclose(ep_out, ref, atol=1e-4), f"norm_topk_prob={norm} 下 EP 输出不一致"
