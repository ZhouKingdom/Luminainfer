"""官方 vLLM 真实负载 benchmark（closed-loop，重尾长度 + 共享前缀）。

与 bench_trace_zjh.py 用同一 workload（workload.py）。vLLM 离线 LLM 只有 closed-loop
的 generate() 接口，无法像 zjh 那样用 add_request/step 模拟「请求陆续到达」，所以
这里对比的是「重尾输入 + 共享前缀」这两个 workload 要素；到达模式（open-loop）见
zjh 侧 bench_trace_zjh.py。

由于 2000 条同时到达会在单卡 OOM，默认 num_seqs=256。延迟从 Prometheus histogram
读取（需 disable_log_stats=False）。

环境要求（vLLM 在 verl 环境）：
    conda activate verl
    export CUDA_HOME=/usr/local/cuda-13.2
    export PATH=/usr/local/cuda-13.2/bin:$PATH

用法：
    python bench_trace_vllm.py <model_path> [<tp>] [--num-seqs 256] [--prefix-len 512] [--no-prefix]
"""

import argparse
import os
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Histogram
from workload import make_requests


def _parse_le(s: str) -> float:
    return float("inf") if s in ("+Inf", "Inf", "inf") else float(s)


def _hist_stats(hist):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tp", type=int, default=1, nargs="?")
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--prefix-len", type=int, default=512)
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prefix_len = 0 if args.no_prefix else args.prefix_len

    llm = LLM(model=args.model, enforce_eager=False, tensor_parallel_size=args.tp, disable_log_stats=False)

    reqs = make_requests(args.num_seqs, seed=args.seed, prefix_len=prefix_len)

    # warmup
    llm.generate([{"prompt_token_ids": [100, 200, 300]}], SamplingParams(max_tokens=1), use_tqdm=False)

    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=r.max_tokens) for r in reqs]
    prompts = [{"prompt_token_ids": r.prompt_token_ids} for r in reqs]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    total_in = sum(len(r.prompt_token_ids) for r in reqs)
    total_out = sum(r.max_tokens for r in reqs)
    throughput = total_out / wall

    ttft_hist = itl_hist = None
    for m in llm.get_metrics():
        if isinstance(m, Histogram):
            if m.name == "vllm:time_to_first_token_seconds":
                ttft_hist = m
            elif m.name == "vllm:inter_token_latency_seconds":
                itl_hist = m

    def report(name, stats):
        if stats is None:
            print(f"{name}: (no metrics)")
            return
        mean, p50, p90, p99 = stats
        print(f"{name}: mean {mean*1000:8.2f}ms   p50 {p50*1000:8.2f}ms   "
              f"p90 {p90*1000:8.2f}ms   p99 {p99*1000:8.2f}ms")

    print(f"vLLM trace: {args.num_seqs} reqs, prefix={prefix_len} tok, wall={wall:.2f}s")
    print(f"  input {total_in} tok, output {total_out} tok, throughput {throughput:.2f} tok/s")
    report("TTFT", _hist_stats(ttft_hist))
    report("ITL ", _hist_stats(itl_hist))


if __name__ == "__main__":
    main()
