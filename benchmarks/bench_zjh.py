"""zjh-vllm 吞吐 benchmark。

与 bench_vllm.py 的 workload 完全一致（seed 固定），用于和官方 vLLM 公平对比吞吐。

用法：
    python bench_zjh.py <model_path> <tensor_parallel_size>

示例：
    python bench_zjh.py ~/huggingface/Qwen3-0.6B 1
    python bench_zjh.py /path/to/Qwen3-32B 4
"""

import os
import sys
import time
from random import randint, seed

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from zjhvllm import LLM, SamplingParams


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # --- workload（必须与 bench_vllm.py 逐字一致，保证对比公平）---
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    llm = LLM(model, enforce_eager=False, tensor_parallel_size=tp, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)  # warmup

    torch.cuda.synchronize()
    t = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t = time.perf_counter() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"zjh-vllm: Total {total_tokens}tok, Time {t:.2f}s, Throughput {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
