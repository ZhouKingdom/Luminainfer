"""Tune scan launch geometry while retaining IEEE FP32 arithmetic."""
import json
from pathlib import Path
import torch
from benchmarks.bench_delta_prefill import inputs, measure
from zjhvllm.models.qwen3_5 import _chunk_gated_delta_rule
import zjhvllm.layers.gated_delta as delta


@torch.inference_mode()
def main():
    torch.manual_seed(123)
    data, initial = inputs(1024)
    original = delta.gated_delta_scan
    packed = []
    def capture(*args, **kw):
        packed[:] = [x.contiguous() for x in args[:-1]]
        return original(*args, **kw)
    delta.gated_delta_scan = capture
    try:
        expected, expected_state = _chunk_gated_delta_rule(*data, initial.clone())
    finally:
        delta.gated_delta_scan = original
    # original output layout is [heads,chunks,chunk,values].
    expected = expected.transpose(0, 1).reshape(48, 16, 64, 128).contiguous()
    rows = []
    for tile in (16, 32, 64):
        for warps in (4, 8):
            fn = lambda state: original(*packed, state, dv_tile=tile, num_warps=warps)
            try:
                output, state = fn(initial.clone())
                torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=2e-4)
                compiled = delta._gated_delta_scan_kernel.warmup(
                    *packed, initial.clone(), torch.empty_like(expected), 16,
                    Dk=128, Dv=128, CHUNK=64, DV_TILE=tile, DOT_PRECISION='ieee',
                    num_warps=warps, num_stages=1, grid=(48, 128//tile))
                compiled._init_handles()
                row = {'tile': tile, 'warps': warps, 'output_max_abs': (output-expected).abs().max().item(), 'state_max_abs': (state-expected_state).abs().max().item(), 'registers': compiled.n_regs, 'spills': compiled.n_spills, 'shared_bytes': compiled.metadata.shared, **measure(fn, initial.clone(), initial, 7)}
            except Exception as e:
                row = {'tile': tile, 'warps': warps, 'error': str(e)}
            rows.append(row)
            print(row, flush=True)
            Path('benchmarks/delta_scan_geometry.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
