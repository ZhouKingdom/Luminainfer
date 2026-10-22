import os
import time
import torch
from random import randint, seed
from zjhvllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # path = os.path.expanduser("/root/autodl-tmp/huggingface/Qwen3-8B")
    # path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    path = os.path.expanduser("/home/data/jhzhou/sft_prj/models/Qwen3-32B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096, tensor_parallel_size=4)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams(),use_tqdm=False)  # warmup

    torch.cuda.synchronize()
    t = time.perf_counter()

    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)

    torch.cuda.synchronize()
    t = (time.perf_counter() - t)

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
