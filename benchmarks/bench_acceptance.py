"""测 speculative decoding 的接受率：draft model vs target model 的 greedy 逐 token 匹配率。

原理：spec decode 的接受率 p =「draft 生成的候选 token 被 target 接受的概率」。对
greedy 对齐验证，它精确等于「draft 和 target 独立 greedy 生成的第一个不匹配位置」——
因为在第一个不匹配之前两者的 state 完全一致。所以直接对比两个模型的 greedy 输出，
就能精确估算 p，无需实现 spec decode 本身。

p 是 spec decode 方案的生死线：
  - p >= 0.7  → 加速比有戏（期望接受长度 p/(1-p) ≈ 2.3）
  - p <= 0.5  → 即使路线 B（不对齐）加速也有限，建议提前止损

用法（分三次跑，避免 0.6B 与 32B 同时占用显存）：
  # 1. 生成 draft 输出（0.6B，单卡）
  python benchmarks/bench_acceptance.py --model ~/huggingface/Qwen3-0.6B --out /tmp/draft.npy --num-prompts 100 --max-tokens 64
  # 2. 生成 target 输出（32B，需要 4 张空闲卡）
  python benchmarks/bench_acceptance.py --model /home/data/jhzhou/sft_prj/models/Qwen3-32B --out /tmp/target.npy --num-prompts 100 --max-tokens 64
  # 3. 对比，算接受率
  python benchmarks/bench_acceptance.py --compare /tmp/draft.npy /tmp/target.npy

prompt 用固定 seed 的随机 token（deterministic），保证两次生成的 prompt 一致。
"""

import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompts(num_prompts: int, prompt_len: int, vocab: int, seed: int):
    """固定 seed 的随机 token prompt（draft/target 两次生成完全一致）。"""
    rng = random.Random(seed)
    return [[rng.randint(0, vocab - 1) for _ in range(prompt_len)] for _ in range(num_prompts)]


def generate(model, tokenizer, prompts, max_tokens, device):
    model.eval()
    out = []
    with torch.no_grad():
        for p in prompts:
            input_ids = torch.tensor([p], dtype=torch.long, device=device)
            gen = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,          # greedy（deterministic）
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            # 只取新生成的 token（去掉 prompt 部分）
            new_tokens = gen[0, input_ids.size(1):].tolist()
            out.append(new_tokens)
    return out


def compare(draft_path, target_path):
    draft = np.load(draft_path, allow_pickle=True)
    target = np.load(target_path, allow_pickle=True)
    assert len(draft) == len(target), "两次生成的样本数不一致"

    total = 0
    total_match = 0
    accept_lens = []          # 每个样本的「接受长度」= 第一个不匹配位置
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
    print(f"样本数: {len(draft)}")
    print(f"总 token 数: {total}, 匹配 token 数: {total_match}")
    print(f"逐 token 接受率 p = {p:.4f}  (≈ {p:.1%})")
    print(f"期望接受长度 E[M] ≈ {p/(1-p):.2f}  (几何分布, 若 p<1)")
    print(f"平均接受长度: {np.mean(accept_lens):.2f}, 中位数: {np.median(accept_lens):.0f}, 最大: {np.max(accept_lens)}")
    # 接受长度分布（关键：batch 越大取 min 越惨）
    hist = {}
    for a in accept_lens:
        hist[a] = hist.get(a, 0) + 1
    print("接受长度分布（长度: 样本数）:")
    for a in sorted(hist):
        bar = "#" * int(hist[a] / max(1, len(accept_lens) / 30))
        print(f"  {a:>2}: {hist[a]:>4}  {bar}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("DRAFT", "TARGET"))
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    assert args.model and args.out, "需要 --model 和 --out（生成模式）"

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",           # 32B 自动分多卡；0.6B 单卡
        trust_remote_code=True,
    )
    device = next(model.parameters()).device

    prompts = make_prompts(args.num_prompts, args.prompt_len, tokenizer.vocab_size, args.seed)
    out = generate(model, tokenizer, prompts, args.max_tokens, device)
    # 对齐长度后存成 object array（不同样本长度不同）
    np.save(args.out, np.array(out, dtype=object), allow_pickle=True)
    print(f"已保存 {len(out)} 个样本的 token_ids 到 {args.out}")


if __name__ == "__main__":
    main()
