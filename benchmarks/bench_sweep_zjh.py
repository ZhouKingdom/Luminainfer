"""zjh-vllm 并发扫描（sweep）：吞吐 / TTFT 尾延迟 vs 并发数。

closed-loop：每个并发档位同时提交 C 条请求，跑到底，测吞吐 + TTFT 的 p50/p99。
workload 用重尾输入长度（workload.py），默认不加前缀（隔离前缀缓存影响）。

注意：小并发档位总 token 少、运行时间短，吞吐数值噪声大；看趋势（吞吐随并发上升
后趋于平稳）即可。每个档位用独立的 seed 生成请求，避免跨档位命中前缀缓存。

用法：
    python bench_sweep_zjh.py <model_path> [<tp>] [--levels 8,16,32,64,128,256,512]
"""

import argparse
import os
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from zjhvllm import LLM, SamplingParams
from workload import make_requests, percentile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tp", type=int, default=1, nargs="?")
    ap.add_argument("--levels", type=str, default="8,16,32,64,128,256,512")
    ap.add_argument("--prefix-len", type=int, default=0, help="前缀长度（>0 时每个请求带 prefix_len token 的前缀）")
    ap.add_argument("--unique-prefix", action="store_true", help="每个请求唯一前缀（同长度、不共享，作 prefix cache 的 abl 基线）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp, max_model_len=4096)

    # warmup（CUDA graph 捕获、首次编译）
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

    print(f"{'concurrency':>11}  {'throughput(tok/s)':>19}  {'TTFT p50(ms)':>13}  {'TTFT p99(ms)':>13}")
    print("-" * 66)
    for c in levels:
        reqs = make_requests(c, seed=args.seed, prefix_len=args.prefix_len, shared_prefix=not args.unique_prefix)

        llm.seq_map.clear()
        t0 = time.perf_counter()
        for r in reqs:
            llm.add_request(
                r.prompt_token_ids,
                SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=r.max_tokens),
            )
        while not llm.is_finished():
            llm.step()
        wall = time.perf_counter() - t0

        total_out = sum(r.max_tokens for r in reqs)
        throughput = total_out / wall
        ttfts, _itls = llm.latency_stats()
        ttft_sorted = sorted(ttfts)
        print(f"{c:>11}  {throughput:>19.2f}  "
              f"{percentile(ttft_sorted, 0.50)*1000:>13.2f}  {percentile(ttft_sorted, 0.99)*1000:>13.2f}")


if __name__ == "__main__":
    main()
