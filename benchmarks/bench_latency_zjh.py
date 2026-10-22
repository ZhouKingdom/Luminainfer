"""zjh-vllm 延迟 benchmark（TTFT / TPOT / p99 尾延迟）。

与 bench_latency_vllm.py 的 workload 完全一致（seed 固定），公平对比延迟。

指标定义：
    TTFT（Time To First Token）：请求提交 → 第一个 completion token 生成
    ITL（Inter-Token Latency）：相邻 completion token 的生成间隔
    p50 / p90 / p99：TTFT、ITL 的分位数（尾延迟）

用法：
    python bench_latency_zjh.py <model_path> <tensor_parallel_size>

示例：
    python bench_latency_zjh.py ~/huggingface/Qwen3-0.6B 1
    python bench_latency_zjh.py /path/to/Qwen3-32B 4
"""

import os
import sys

from random import randint, seed

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from zjhvllm import LLM, SamplingParams


def percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数（numpy.percentile 的线性插值语义）。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def report(name: str, vals: list[float]):
    vals = sorted(vals)
    if not vals:
        print(f"{name}: (empty)")
        return
    mean = sum(vals) / len(vals)
    print(
        f"{name}: mean {mean*1000:8.2f}ms   "
        f"p50 {percentile(vals, 0.50)*1000:8.2f}ms   "
        f"p90 {percentile(vals, 0.90)*1000:8.2f}ms   "
        f"p99 {percentile(vals, 0.99)*1000:8.2f}ms"
    )


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # --- workload（必须与 bench_latency_vllm.py 逐字一致，保证对比公平）---
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    llm = LLM(model, enforce_eager=False, tensor_parallel_size=tp, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # warmup（触发 CUDA graph 捕获、首次 kernel 编译），并清掉 warmup 的统计记录
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    llm.seq_map.clear()

    # 提交全部请求（同时到达，offline batch）
    for prompt, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(prompt, sp)

    # 跑到底（每步内部已同步，时间戳由 postprocess 记录）
    while not llm.is_finished():
        llm.step()

    stats = llm.latency_stats()
    ttfts, itls = stats

    print(f"zjh-vllm latency: {len(ttfts)} requests")
    report("TTFT", ttfts)
    report("ITL ", itls)


if __name__ == "__main__":
    main()
