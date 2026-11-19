"""Isolate short-window recurrence; this is not an end-to-end throughput test."""
import argparse
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from zjhvllm.layers.gated_delta import gated_delta_decode, gated_delta_verify


@torch.inference_mode()
def run(capacity):
    torch.manual_seed(904)
    heads, dim, layers = 24, 128, 48
    sizes = [2] * capacity
    count = sum(sizes)
    state_pool = torch.zeros(5, layers, heads, dim, dim, device='cuda')
    slots = torch.tensor([3, 1, 4, 0][:capacity], device='cuda', dtype=torch.int32)
    state_pool[slots.long(), 17] = torch.randn(capacity, heads, dim, dim, device='cuda') * .01
    state = state_pool[:, 17]
    scratch = torch.zeros(count, layers, heads, dim, dim, device='cuda')
    history = scratch[:, 17]
    indices = torch.arange(count, device='cuda', dtype=torch.int32)
    cu = torch.arange(0, count + 1, 2, device='cuda', dtype=torch.int32)
    q, k = [torch.nn.functional.normalize(torch.randn(count, heads, dim, device='cuda'), dim=-1) for _ in range(2)]
    q *= dim ** -.5
    v = torch.randn(count, heads, dim, device='cuda')
    g, beta = -torch.rand(count, heads, device='cuda'), torch.rand(count, heads, device='cuda')
    def reference():
        outputs = []
        for request in range(capacity):
            for step in range(request * 2, request * 2 + 2):
                first = step % 2 == 0
                outputs.append(gated_delta_decode(q[step:step+1], k[step:step+1], v[step:step+1],
                    g[step:step+1], beta[step:step+1], state if first else history,
                    slots[request:request+1] if first else indices[step-1:step],
                    output_cache=history, output_slots=indices[step:step+1]))
        return torch.cat(outputs)
    def fused():
        return gated_delta_verify(q, k, v, g, beta, state, slots, cu, history, max_steps=2)
    expected = reference().clone(); expected_states = history.clone()
    actual = fused()
    assert torch.equal(actual, expected) and torch.equal(history, expected_states)
    timings = []
    for iteration in range(3):
        values = {}
        for name, fn in ([('decode_loop', reference), ('window_kernel', fused)] if iteration % 2 == 0
                         else [('window_kernel', fused), ('decode_loop', reference)]):
            values[name] = do_bench_cudagraph(fn, rep=200)
        timings.append(values)
    means = {key:sum(row[key] for row in timings)/len(timings) for key in timings[0]}
    return {'capacity':capacity, 'heads':heads, 'key_dim':dim, 'value_dim':dim,
            'layer_stride':state.stride(0), 'window':sizes, 'exact':True,
            'cuda_graph_milliseconds':means, 'three_rounds_ms':timings,
            'speedup':means['decode_loop']/means['window_kernel']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [run(count) for count in (1, 4)]
    args.output.write_text(json.dumps({'contract':'isolated recurrence only; prepared identical FP32 q/k/v/g/beta; strided 48-layer TP2-shaped state; 3 reverse-order CUDA Graph timing rounds', 'rows':results}, indent=2)+'\n')
    for row in results:print(row['capacity'], row['cuda_graph_milliseconds'], row['speedup'])
