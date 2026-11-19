"""Compare eager/fused state updates in the same loaded Qwen3.5 engine."""
import argparse
import time
import torch
from zjhvllm import LLM, SamplingParams
import zjhvllm.layers.gated_delta as delta


def eager(q, k, v, g, beta, cache, slots):
    state = cache[slots]
    decay = g.exp()
    residual = v - decay[..., None] * torch.einsum('bhd,bhdv->bhv', k, state)
    state = decay[..., None, None] * state + torch.einsum('bhd,bhv->bhdv', beta[..., None] * k, residual)
    result = torch.einsum('bhd,bhdv->bhv', q, state)
    cache[slots] = state
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/data/jhzhou/huggingface/Qwen3.5-27B')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    llm = LLM(args.model, enforce_eager=True, max_num_seqs=2,
              max_model_len=512, max_num_batched_tokens=128,
              gpu_memory_utilization=.9)
    fused = delta.gated_delta_decode
    params = SamplingParams(greedy=True, max_tokens=24, ignore_eos=True)
    prompts = ['Explain why the sky is blue in simple terms. ' * 6]
    expected = None
    try:
        for name, fn in [('eager', eager), ('fused', fused)]:
            delta.gated_delta_decode = fn
            # Warm the model and JIT kernels before timing.
            output = llm.generate(prompts, params, use_tqdm=False)
            tokens = output[0]['token_ids']
            if expected is None:
                expected = tokens
            else:
                assert tokens == expected, 'greedy token mismatch'
            timings = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                assert output[0]['token_ids'] == expected
            print(f'{name}: times={timings}, median={sorted(timings)[len(timings)//2]:.4f}s', flush=True)
        print('Greedy tokens identical across eager/fused runs.', flush=True)
    finally:
        delta.gated_delta_decode = fused
        llm.exit()


if __name__ == '__main__':
    main()
