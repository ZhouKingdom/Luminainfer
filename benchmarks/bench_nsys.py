"""nsys profiling 用：固定 workload，跑完即退出，对比 enforce_eager=True/False。

用 `torch.cuda.profiler.start/stop` 只采集 `llm.generate` 这段（跳过模型加载、
CUDA Graph 捕获、warmup 的噪声），配合 nsys 的 `--capture-range=cudaProfilerApi`。

用法（lumina 环境）：
    # 关闭 CUDA Graph（eager）
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
        --output=/tmp/nsys_eager python benchmarks/bench_nsys.py --enforce-eager

    # 开启 CUDA Graph（decode 走 replay）
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
        --output=/tmp/nsys_graph python benchmarks/bench_nsys.py

之后用 `nsys stats /tmp/nsys_eager.nsys-rep` 对比 launch 次数 / GPU idle 等。
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from zjhvllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enforce-eager", action="store_true", help="关闭 CUDA Graph")
    ap.add_argument("--num-seqs", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=args.enforce_eager, tensor_parallel_size=1)

    sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    prompts = [prompt] * args.num_seqs

    torch.cuda.profiler.start()
    llm.generate(prompts, sp)
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
