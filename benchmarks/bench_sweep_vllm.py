"""官方 vLLM 并发扫描（sweep）：吞吐 / TTFT 尾延迟 vs 并发数。

closed-loop，每个并发档位跑一遍，workload 用重尾输入长度（workload.py）。

延迟从 Prometheus histogram 读，且 histogram 是累计的，故每个档位前后做「bucket
快照差」得到该档位增量的分位数。吞吐直接按 wall-clock 计时。

环境要求（vLLM 在 verl 环境）：
    conda activate verl
    export CUDA_HOME=/usr/local/cuda-13.2
    export PATH=/usr/local/cuda-13.2/bin:$PATH

用法：
    python bench_sweep_vllm.py <model_path> [<tp>] [--levels 8,16,32,64,128,256,512]
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


def _snapshot(hist):
    if hist is None:
        return None
    return (dict(hist.buckets), hist.count, hist.sum)


def _delta(prev, cur):
    """cur 相对 prev 的增量 histogram。prev 为 None 表示首个档位。"""
    if prev is None:
        return cur
    pb, pc, ps = prev
    cb, cc, cs = cur
    buckets = {k: cb.get(k, 0) - pb.get(k, 0) for k in set(pb) | set(cb)}
    return (buckets, cc - pc, cs - ps)


def _stats(snap):
    if snap is None or snap[1] == 0:
        return None
    buckets, count, total_sum = snap
    mean = total_sum / count
    entries = sorted((_parse_le(k), v) for k, v in buckets.items())

    def quantile(p: float) -> float:
        target = p * count
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

    return mean, quantile(0.50), quantile(0.99)


def _locate_hist(llm):
    ttft = itl = None
    for m in llm.get_metrics():
        if isinstance(m, Histogram):
            if m.name == "vllm:time_to_first_token_seconds":
                ttft = m
            elif m.name == "vllm:inter_token_latency_seconds":
                itl = m
    return ttft, itl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tp", type=int, default=1, nargs="?")
    ap.add_argument("--levels", type=str, default="8,16,32,64,128,256,512")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    llm = LLM(model=args.model, enforce_eager=False, tensor_parallel_size=args.tp, disable_log_stats=False)

    # warmup
    llm.generate([{"prompt_token_ids": [100, 200, 300]}], SamplingParams(max_tokens=1), use_tqdm=False)

    print(f"{'concurrency':>11}  {'throughput(tok/s)':>19}  {'TTFT p50(ms)':>13}  {'TTFT p99(ms)':>13}")
    print("-" * 66)
    for c in levels:
        reqs = make_requests(c, seed=args.seed, prefix_len=0)
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=r.max_tokens) for r in reqs]
        prompts = [{"prompt_token_ids": r.prompt_token_ids} for r in reqs]

        prev = _snapshot(_locate_hist(llm)[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        cur = _snapshot(_locate_hist(llm)[0])

        total_out = sum(r.max_tokens for r in reqs)
        throughput = total_out / wall
        ttft_stats = _stats(_delta(prev, cur))
        if ttft_stats:
            _mean, p50, p99 = ttft_stats
            print(f"{c:>11}  {throughput:>19.2f}  {p50*1000:>13.2f}  {p99*1000:>13.2f}")
        else:
            print(f"{c:>11}  {throughput:>19.2f}  {'(n/a)':>13}  {'(n/a)':>13}")


if __name__ == "__main__":
    main()
