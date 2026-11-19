"""Compare two ordinary decode backends from identical state and token history."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from zjhvllm import LLM, SamplingParams
from zjhvllm.engine.speculative import HybridCheckpoint
from zjhvllm.models.qwen3_5 import Qwen3_5Linear
from zjhvllm.utils.context import reset_context


def difference(a, b):
    d = a.double() - b.double()
    return {"exact": torch.equal(a, b), "different": int(torch.count_nonzero(d)),
            "elements": a.numel(), "max_abs": d.abs().max().item(),
            "rmse": d.square().mean().sqrt().item()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('require a new output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workload = json.loads(Path('benchmarks/qwen35_mtp_8gpu_20261106/workload.json').read_text())
    item = next(r for r in workload if r['id'] == 2)
    report = {"request_id": 2, "mode": "ordinary decode only, no MTP, no Graph",
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path('zjhvllm').rglob('*.py')}}
    engine = None
    try:
        engine = LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B', max_num_seqs=1,
                     max_model_len=4096, max_num_batched_tokens=256,
                     enforce_eager=True, gpu_memory_utilization=.85, distributed_port=2397)
        runner = engine.model_runner
        seq_id = engine.add_request(item['prompt_token_ids'], SamplingParams(greedy=True, max_tokens=2))
        engine.step()  # Complete the common 36-token prefill and emit token 0.
        seq = engine.seq_map[seq_id]
        assert seq.state_ready and seq.num_tokens == seq.num_cached_tokens + 1
        report['cached_tokens'] = seq.num_cached_tokens
        report['pending_token'] = seq.last_token
        checkpoint = HybridCheckpoint.capture(runner.S_cache, runner.conv_cache, seq.state_slot)
        trace, comparisons, first, handles = {}, [], {}, []
        mode = 'decode_equivalent'

        def collect(name):
            def hook(module, inputs, output):
                if not isinstance(output, torch.Tensor):
                    return
                actual = output.detach().cpu().clone()
                if mode == 'decode_equivalent':
                    trace[name] = (actual, inputs[0].detach().cpu().clone()
                                   if isinstance(module, Qwen3_5Linear) else None)
                    return
                expected, saved_input = trace[name]
                stats = difference(expected, actual)
                if not stats['exact']:
                    comparisons.append({'module': name, **stats})
                    if not first:
                        first.update({'module': name, **stats})
                        if isinstance(module, Qwen3_5Linear):
                            x = inputs[0].detach().cpu()
                            first['input_exact'] = torch.equal(saved_input, x)
                            indices = torch.nonzero(expected[0] != actual[0]).flatten()[:64]
                            w = module.weight[indices.to(module.weight.device)].double().cpu()
                            reference = w @ saved_input[0].double()
                            if module.bias is not None:
                                reference += module.bias[indices.to(module.bias.device)].double().cpu()
                            old, new = expected[0, indices].double(), actual[0, indices].double()
                            rounded = reference.bfloat16().double()
                            first['fp64_selected_channels'] = {
                                'selection': 'first up to 64 differing output channels',
                                'channels': indices.tolist(), 'reference': reference.tolist(),
                                'legacy': old.tolist(), 'invariant': new.tolist(),
                                'legacy_max_error': (old-reference).abs().max().item(),
                                'invariant_max_error': (new-reference).abs().max().item(),
                                'legacy_matches_rounded_reference': int((old==rounded).sum()),
                                'invariant_matches_rounded_reference': int((new==rounded).sum())}
            return hook

        for name, module in runner.model.named_modules():
            if name and not isinstance(module, (torch.nn.ModuleList, torch.nn.Conv1d)):
                handles.append(module.register_forward_hook(collect(name)))
        logits = {}
        try:
            for mode in ('decode_equivalent', 'batch_invariant'):
                checkpoint.restore(runner.S_cache, runner.conv_cache)
                assert torch.equal(runner.S_cache[seq.state_slot], checkpoint.state)
                assert torch.equal(runner.conv_cache[seq.state_slot], checkpoint.conv)
                for module in runner.model.modules():
                    if isinstance(module, Qwen3_5Linear):
                        module.backend = mode
                ids, positions = runner.prepare_decode([seq])
                hidden = runner.model(ids, positions)
                logits[mode] = runner.model.compute_logits(hidden).cpu()
                reset_context()
        finally:
            for handle in handles:
                handle.remove()
            checkpoint.restore(runner.S_cache, runner.conv_cache)
            for module in runner.model.modules():
                if isinstance(module, Qwen3_5Linear):
                    module.backend = 'decode_equivalent'
            reset_context()
        report['first_difference'] = first
        report['differing_modules'] = comparisons
        report['logits_difference'] = difference(logits['decode_equivalent'], logits['batch_invariant'])
        report['top5'] = {}
        for mode, value in logits.items():
            values, indices = value.float().topk(5, dim=-1)
            report['top5'][mode] = {'argmax': value.argmax(-1).item(),
                'ids': indices[0].tolist(), 'values': values[0].tolist(),
                'gap': (values[0,0]-values[0,1]).item()}
        engine.step()  # Commit ordinary reference and release scheduler resources.
        report['resources_reclaimed'] = (not engine.scheduler.running and not engine.scheduler.waiting)
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        if engine is not None:
            engine.exit()
    print(json.dumps({k: report[k] for k in ('first_difference','logits_difference','top5')}, indent=2))


if __name__ == '__main__':
    main()
