"""用 Lumina 引擎测 spec decode 接受率（比 transformers device_map 快）。

分两次跑（0.6B 和 32B 各一次，避免端口冲突），greedy 生成固定 seed 的随机 token，
保存 completion token_ids，最后 --compare 对比算接受率 p。

用法：
  python benchmarks/bench_acceptance_lumina.py <model> <out> <tp>     # 生成
  python benchmarks/bench_acceptance_lumina.py --compare <draft.npy> <target.npy>  # 对比
"""

import random
import sys

import numpy as np

from zjhvllm import LLM, SamplingParams

N_PROMPTS = 20
PROMPT_LEN = 32
MAX_TOKENS = 32
SEED = 0
VOCAB = 32000


def generate(model, out, tp):
    rng = random.Random(SEED)
    prompts = [[rng.randint(0, VOCAB - 1) for _ in range(PROMPT_LEN)] for _ in range(N_PROMPTS)]

    llm = LLM(model, enforce_eager=True, tensor_parallel_size=tp)
    sp = SamplingParams(greedy=True, max_tokens=MAX_TOKENS, ignore_eos=True)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    tokens = [o["token_ids"] for o in outputs]
    np.save(out, np.array(tokens, dtype=object), allow_pickle=True)
    print(f"已保存 {len(tokens)} 个样本的 completion token_ids 到 {out}")


def compare(draft_path, target_path):
    draft = np.load(draft_path, allow_pickle=True)
    target = np.load(target_path, allow_pickle=True)
    assert len(draft) == len(target), "样本数不一致"

    total = total_match = 0
    accept_lens = []
    for d, t in zip(draft, target):
        L = min(len(d), len(t))
        accept = 0
        for i in range(L):
            total += 1
            if d[i] == t[i]:
                total_match += 1
                accept += 1
            else:
                break
        accept_lens.append(accept)

    p = total_match / total if total else 0.0
    print(f"样本数 {len(draft)}, 总 token {total}, 匹配 {total_match}")
    print(f"接受率 p = {p:.4f} ({p:.1%})")
    if p < 1:
        print(f"期望接受长度 E[M] ≈ {p/(1-p):.2f}")
    print(f"平均接受长度 {np.mean(accept_lens):.2f}, 中位 {np.median(accept_lens):.0f}, 最大 {np.max(accept_lens)}")
    hist = {}
    for a in accept_lens:
        hist[a] = hist.get(a, 0) + 1
    print("接受长度分布:")
    for a in sorted(hist):
        print(f"  {a:>2}: {hist[a]:>3}  {'#' * hist[a]}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        model, out = sys.argv[1], sys.argv[2]
        tp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        generate(model, out, tp)
