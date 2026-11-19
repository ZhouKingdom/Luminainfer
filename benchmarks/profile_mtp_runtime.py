"""Profile a warmed real MTP generation; loading/JIT excluded from trace."""
import argparse
import json
from pathlib import Path
import torch
from zjhvllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--cuda-graph', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = json.loads(Path('benchmarks/qwen35_mtp_8gpu_20261106/workload.json').read_text())
    prompt = next(r['prompt_token_ids'] for r in rows if r['id'] == 2)
    engine = LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B', enable_mtp=True,
                 max_num_seqs=1, max_model_len=4096, max_num_batched_tokens=256,
                 enforce_eager=True, gpu_memory_utilization=.85, distributed_port=2381)
    engine.model_runner.config.verification_cuda_graph = args.cuda_graph
    try:
        params = SamplingParams(greedy=True, max_tokens=args.tokens)
        engine.generate_speculative(prompt, params, verification_mode='batched_experimental')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            output = engine.generate_speculative(prompt, params, verification_mode='batched_experimental')
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / 'trace.json'))
        (args.output / 'cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
        (args.output / 'cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40))
        (args.output / 'output.json').write_text(json.dumps(output, indent=2))
        print((args.output / 'cuda.txt').read_text(), flush=True)
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
