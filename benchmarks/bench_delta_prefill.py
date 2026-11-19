"""48-head DeltaNet prefill: correctness, latency, and NVTX profiling.

The torch/triton paths share the production chunk preprocessing; only scan
backend changes. Inputs mimic transposed production projections, with nonzero
initial state. Timings exclude state reset and JIT compilation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import torch
import torch.nn.functional as F
from zjhvllm.models.qwen3_5 import _chunk_gated_delta_rule


def recurrent(q, k, v, g, beta, state):
    outputs = []
    for t in range(q.shape[1]):
        decay = g[:, t].exp()
        prediction = torch.einsum('hd,hdv->hv', k[:, t], state)
        residual = v[:, t] - decay[:, None] * prediction
        state = decay[:, None, None] * state + (beta[:, t, None] * k[:, t])[:, :, None] * residual[:, None, :]
        outputs.append(torch.einsum('hd,hdv->hv', q[:, t], state))
    return torch.stack(outputs), state


def inputs(length, heads=48, dim=128):
    q, k = [F.normalize(torch.randn(length, heads, dim, device='cuda'), dim=-1).transpose(0, 1) for _ in range(2)]
    q = q / dim**.5
    v = torch.randn(length, heads, dim, device='cuda').transpose(0, 1)
    g = -torch.rand(length, heads, device='cuda').T * .5
    beta = torch.rand(length, heads, device='cuda').T
    state = torch.randn(heads, dim, dim, device='cuda') * .1
    return (q, k, v, g, beta), state


def measure(fn, state, initial, repeats):
    for _ in range(2):
        state.copy_(initial)
        fn(state)
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(repeats):
        state.copy_(initial)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        fn(state)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t) * 1000)
        gpu.append(start.elapsed_time(end))
    return {'gpu_ms': median(gpu), 'wall_ms': median(wall), 'gpu_samples_ms': gpu}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', type=int, nargs='+', default=[33, 65, 256, 1024, 4096])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', default='benchmarks/delta_prefill_baseline.json')
    parser.add_argument('--scan-warps', type=int, default=8)
    parser.add_argument('--scan-precision', choices=['ieee', 'tf32x3'], default='ieee')
    parser.add_argument('--profile', choices=['torch', 'triton'])
    args = parser.parse_args()
    torch.manual_seed(123)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'heads': 48, 'dim': 128,
              'state': 'nonzero fp32', 'scan_precision': args.scan_precision, 'scan_warps': args.scan_warps, 'repeats': args.repeats, 'source_sha256': {
                  p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                  ('zjhvllm/models/qwen3_5.py', 'zjhvllm/layers/gated_delta.py')}, 'results': []}
    for length in args.lengths:
        data, initial = inputs(length)
        state = initial.clone()
        functions = {'recurrent': lambda s: recurrent(*data, s)}
        for backend in ('torch', 'triton'):
            functions[backend] = lambda s, backend=backend: _chunk_gated_delta_rule(*data, s, scan_backend=backend, scan_precision=args.scan_precision, scan_warps=args.scan_warps)
        if args.profile:
            fn = functions[args.profile]
            for _ in range(2):
                fn(initial.clone())
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            for i in range(args.repeats):
                state.copy_(initial)
                with torch.cuda.nvtx.range(f'delta_prefill/{args.profile}/L{length}'):
                    fn(state)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            continue
        expected, final = functions['recurrent'](initial.clone())
        row = {'length': length, 'timings': {}, 'correctness': {}}
        for name, fn in functions.items():
            actual, actual_state = fn(initial.clone())
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual_state, final, atol=2e-5, rtol=2e-4)
            row['correctness'][name] = {'output_max_abs': (actual-expected).abs().max().item(),
                                        'state_max_abs': (actual_state-final).abs().max().item()}
            row['timings'][name] = measure(fn, state, initial, args.repeats)
        split = length // 2 + 1
        left, middle = _chunk_gated_delta_rule(*(x[:, :split] for x in data), initial.clone(), scan_precision=args.scan_precision, scan_warps=args.scan_warps)
        right, end = _chunk_gated_delta_rule(*(x[:, split:] for x in data), middle, scan_precision=args.scan_precision, scan_warps=args.scan_warps)
        torch.testing.assert_close(torch.cat([left, right]), expected, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(end, final, atol=2e-5, rtol=2e-4)
        row['continuation_passed'] = True
        report['results'].append(row)
        print('length=', length, {name: round(t['gpu_ms'], 3) for name,t in row['timings'].items()}, flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
