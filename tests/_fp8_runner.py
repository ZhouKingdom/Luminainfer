"""子进程运行 fp8 KV cache 引擎，打印 greedy 输出的 token IDs（JSON）。

test_fp8.py 用它做子进程隔离，避免 fp8 引擎与 bf16 引擎在同一进程里争抢显存。
用法：python _fp8_runner.py <model_path>
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from zjhvllm import LLM, SamplingParams  # noqa: E402


PROMPTS = [
    "The capital of France is",
    "The largest planet is",
    "Water boils at 100 degrees",
]


def main():
    model_path = sys.argv[1]
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1,
              kv_cache_dtype="fp8", max_num_batched_tokens=512, gpu_memory_utilization=0.2)
    sp = SamplingParams(greedy=True, max_tokens=24)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    print("__RESULT__" + json.dumps([o["token_ids"] for o in outs]))


if __name__ == "__main__":
    main()
