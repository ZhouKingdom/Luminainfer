"""官方 vLLM 延迟 benchmark（对比基线，TTFT / TPOT / p99）。

与 bench_latency_zjh.py 的 workload 完全一致（seed 固定），公平对比延迟。
vLLM 0.11 中 RequestOutput.metrics 已废弃（恒为 None），延迟从 Prometheus
histogram 指标（llm.get_metrics()）读取：

    vllm:time_to_first_token_seconds      → TTFT
    vllm:inter_token_latency_seconds      → 每 token 间隔（近似 TPOT）

环境要求（vLLM 在 verl 环境）：
    conda activate verl
    export CUDA_HOME=/usr/local/cuda-13.2
    export PATH=/usr/local/cuda-13.2/bin:$PATH

用法：
    python bench_latency_vllm.py <model_path> <tensor_parallel_size>

示例：
    python bench_latency_vllm.py ~/huggingface/Qwen3-0.6B 1
    python bench_latency_vllm.py /path/to/Qwen3-32B 4
"""

import os
import sys

from random import randint, seed

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Histogram


def _parse_le(s: str) -> float:
    return float("inf") if s in ("+Inf", "Inf", "inf") else float(s)


def hist_stats(hist: Histogram):
    """从累计 histogram 提取 (mean, p50, p90, p99)。buckets 是累计计数。"""
    if hist is None or hist.count == 0:
        return None
    mean = hist.sum / hist.count
    entries = sorted((_parse_le(k), v) for k, v in hist.buckets.items())
    total = hist.count

    def quantile(p: float) -> float:
        target = p * total
        prev_le, prev_c = 0.0, 0
        for le, c in entries:
            if c >= target:
                if le == float("inf"):
                    return prev_le
                if c == prev_c:
                    return prev_le
                frac = (target - prev_c) / (c - prev_c)
                return prev_le + frac * (le - prev_le)
            prev_le, prev_c = le, c
        return prev_le

    return mean, quantile(0.50), quantile(0.90), quantile(0.99)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # --- workload（必须与 bench_latency_zjh.py 逐字一致，保证对比公平）---
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    llm = LLM(model=model, enforce_eager=False, tensor_parallel_size=tp, disable_log_stats=False)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # warmup
    llm.generate([{"prompt_token_ids": [100, 200, 300]}], SamplingParams(max_tokens=1), use_tqdm=False)

    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    llm.generate(prompts, sampling_params, use_tqdm=False)

    # 从 Prometheus histogram 读延迟指标
    ttft_hist = itl_hist = None
    for m in llm.get_metrics():
        if isinstance(m, Histogram):
            if m.name == "vllm:time_to_first_token_seconds":
                ttft_hist = m
            elif m.name == "vllm:inter_token_latency_seconds":
                itl_hist = m

    def report(name: str, stats):
        if stats is None:
            print(f"{name}: (no metrics)")
            return
        mean, p50, p90, p99 = stats
        print(f"{name}: mean {mean*1000:8.2f}ms   p50 {p50*1000:8.2f}ms   p90 {p90*1000:8.2f}ms   p99 {p99*1000:8.2f}ms")

    print(f"vLLM latency (histogram)")
    report("TTFT", hist_stats(ttft_hist))
    report("ITL ", hist_stats(itl_hist))


if __name__ == "__main__":
    main()
