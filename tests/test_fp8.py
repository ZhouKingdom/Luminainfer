"""fp8 KV cache 正确性：fp8 引擎的 greedy 输出与 HF 参考对齐。

fp8 引擎在子进程里跑（避免与 bf16 引擎在同一进程争抢显存），父进程对比 HF。
fp8 量化引入 ~3% 误差，greedy 可能在某一步 near-tie 分叉（同 test_generate.py 的说明），
这里用公共前缀长度做鲁棒断言（真 bug 会在前几步就分叉）。
"""

import json
import os
import subprocess
import sys

import torch

MIN_COMMON_PREFIX = 8


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_fp8_greedy_matches_hf(model_path, hf_model, tokenizer):
    prompts = [
        "The capital of France is",
        "The largest planet is",
        "Water boils at 100 degrees",
    ]
    runner = os.path.join(os.path.dirname(__file__), "_fp8_runner.py")
    proc = subprocess.run([sys.executable, runner, model_path],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"fp8 runner 失败:\n{proc.stderr}"

    result_line = next(l for l in proc.stdout.splitlines() if l.startswith("__RESULT__"))
    ours = json.loads(result_line[len("__RESULT__"):])

    for prompt, tok_ids in zip(prompts, ours):
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        ref = hf_model.generate(torch.tensor([ids]).cuda(), max_new_tokens=24, do_sample=False)[0][len(ids):].tolist()
        common = _common_prefix(tok_ids, ref)
        print(f"\n[{prompt!r}] fp8 common_prefix={common}/{len(ref)}")
        assert common >= MIN_COMMON_PREFIX, (
            f"fp8 greedy 与 HF 公共前缀过短 ({common} < {MIN_COMMON_PREFIX}): {prompt!r}"
        )
