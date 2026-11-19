"""zjh-vllm 真实负载（trace）benchmark：open-loop Poisson 到达 + 重尾长度 + 共享前缀。

与 bench_trace_vllm.py 用同一 workload（workload.py），但 zjh 侧额外用引擎原生的
add_request/step 循环实现「请求陆续到达」（open-loop），从而：
  - 消除旧 bench「256 条同时到达」导致的 TTFT 虚高；
  - 2000 条样本让 p99 尾延迟更稳定；
  - 共享前缀触发 prefix caching（--no-prefix 关掉即可 abl 对比收益）。

指标：arrival-limited 吞吐 + TTFT/ITL 的 mean/p50/p90/p99。

用法：
    python bench_trace_zjh.py <model_path> [<tp>] [--num-seqs 2000] [--rate 50] [--prefix-len 512] [--no-prefix]

示例：
    python bench_trace_zjh.py ~/huggingface/Qwen3-0.6B 1 --num-seqs 2000 --rate 50
    python bench_trace_zjh.py /path/to/Qwen3-32B 4 --num-seqs 2000 --rate 50
"""

import argparse
import os
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from zjhvllm import LLM, SamplingParams
from workload import make_requests, report_latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tp", type=int, default=1, nargs="?")
    ap.add_argument("--num-seqs", type=int, default=2000)
    ap.add_argument("--rate", type=float, default=50.0, help="Poisson 到达速率（请求/秒）")
    ap.add_argument("--prefix-len", type=int, default=512)
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prefix_len = 0 if args.no_prefix else args.prefix_len

    llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp, max_model_len=4096)

    reqs = make_requests(args.num_seqs, seed=args.seed, prefix_len=prefix_len, arrival_rate=args.rate)

    # warmup（CUDA graph 捕获、首次编译），并清掉 warmup 的延迟统计
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    llm.seq_map.clear()

    # ---- open-loop：按到达时刻陆续提交，引擎边收边跑 ----
    issued = 0
    n = len(reqs)
    t0 = time.perf_counter()
    while issued < n or not llm.is_finished():
        now = time.perf_counter() - t0
        while issued < n and reqs[issued].arrival <= now:
            llm.add_request(
                reqs[issued].prompt_token_ids,
                SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=reqs[issued].max_tokens),
            )
            issued += 1
            now = time.perf_counter() - t0
        if not llm.is_finished():
            llm.step()
        elif issued < n:
            wait = reqs[issued].arrival - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    wall = time.perf_counter() - t0
    ttfts, itls = llm.latency_stats()

    total_in = sum(len(r.prompt_token_ids) for r in reqs)
    total_out = sum(r.max_tokens for r in reqs)
    throughput = total_out / wall

    print(f"zjh-vllm trace: {args.num_seqs} reqs, rate={args.rate}/s, "
          f"prefix={prefix_len} tok, wall={wall:.2f}s")
    print(f"  input {total_in} tok, output {total_out} tok, "
          f"arrival-limited throughput {throughput:.2f} tok/s")
    print(f"  latency samples: {len(ttfts)} requests, {len(itls)} inter-token intervals")
    report_latency("TTFT", ttfts)
    report_latency("ITL ", itls)


if __name__ == "__main__":
    main()
