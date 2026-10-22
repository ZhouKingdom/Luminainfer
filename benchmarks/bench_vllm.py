"""官方 vLLM 吞吐 benchmark（对比基线）。

与 bench_zjh.py 的 workload 完全一致（seed 固定），用于公平对比吞吐。

环境要求（vLLM 在 verl 环境）：
    conda activate verl
    export CUDA_HOME=/usr/local/cuda-13.2   # 否则用默认 /usr/bin/nvcc(CUDA 11.5) 编译 extension 报 math.h 找不到
    export PATH=/usr/local/cuda-13.2/bin:$PATH

用法：
    python bench_vllm.py <model_path> <tensor_parallel_size>

示例：
    python bench_vllm.py ~/huggingface/Qwen3-0.6B 1
    python bench_vllm.py /path/to/Qwen3-32B 4
"""

import os
import sys
import time
from random import randint, seed

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # --- workload（必须与 bench_zjh.py 逐字一致，保证对比公平）---
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    llm = LLM(model=model, enforce_eager=False, tensor_parallel_size=tp)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # warmup
    llm.generate([{"prompt_token_ids": [100, 200, 300]}], SamplingParams(max_tokens=1), use_tqdm=False)

    torch.cuda.synchronize()
    t = time.perf_counter()
    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t = time.perf_counter() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"vLLM: Total {total_tokens}tok, Time {t:.2f}s, Throughput {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
