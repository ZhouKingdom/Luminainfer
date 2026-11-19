"""Real-checkpoint HF alignment on identical token histories (two GPUs).

Run from Lumina: python -m benchmarks.validate_qwen35 --output benchmarks/qwen35_alignment.json
Captures actual engine prefill/decode logits; HF then replays the engine's tokens,
so an early greedy tie cannot conceal later state errors behind divergent inputs.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
from zjhvllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/data/jhzhou/huggingface/Qwen3.5-27B')
    parser.add_argument('--hf-device', default='cuda:1')
    parser.add_argument('--output', default='benchmarks/qwen35_alignment.json')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--hf-attention', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    parser.add_argument('--max-abs', type=float, default=1.0)
    parser.add_argument('--max-rmse', type=float, default=.1)
    parser.add_argument('--tie-gap', type=float, default=.25)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The conditional-generation checkpoint contains a vision tower; the text
    # AutoModelForCausalLM mapping intentionally loads just its language model.
    print('Loading HuggingFace text reference on', args.hf_device, flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={'': args.hf_device},
        attn_implementation=args.hf_attention, local_files_only=True,
    ).eval()
    print('Loading Lumina on cuda:0', flush=True)
    llm = LLM(args.model, enforce_eager=True, max_num_seqs=3,
              max_model_len=1024, max_num_batched_tokens=128,
              gpu_memory_utilization=.9)
    runner = llm.model_runner
    original_run, original_model = runner.run, runner.run_model
    active, captured = [], {}

    def capture_run(seqs, is_prefill):
        active[:] = seqs
        return original_run(seqs, is_prefill)

    def capture_model(input_ids, positions, is_prefill):
        logits = original_model(input_ids, positions, is_prefill)
        for seq, row in zip(active, logits):
            if not is_prefill or seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                captured.setdefault(seq.seq_id, []).append(row.detach().float().cpu())
        return logits

    runner.run, runner.run_model = capture_run, capture_model
    params = SamplingParams(greedy=True, max_tokens=args.tokens, ignore_eos=True)
    short = tokenizer.encode('The capital of France is')
    base = tokenizer.encode('The quick brown fox jumps over the lazy dog. ' * 70)
    long = base[:513]
    cases = [('short_batch', [short, tokenizer.encode('计算 17 加 25 的结果：')]),
             ('chunked_batch', [base[:65], base[:257], long]),
             ('repeated_long_prefix', [long]),
             ('shared_prefix_different_suffix', [long[:-4] + tokenizer.encode(' Paris'), long[:-4] + tokenizer.encode(' London')])]
    report = {'source_sha256': hashlib.sha256(Path('zjhvllm/models/qwen3_5.py').read_bytes()).hexdigest(), 'model': args.model, 'torch': torch.__version__, 'transformers': transformers_version, 'gpu': torch.cuda.get_device_name(0),
              'thresholds': {'max_abs': args.max_abs, 'rmse': args.max_rmse, 'tie_gap': args.tie_gap},
              'hf_attention': args.hf_attention, 'comparison': 'HF logits conditioned on the same engine-generated token history', 'cases': []}
    failed = False
    try:
        for name, prompts in cases:
            captured.clear()
            first_id = set(llm.seq_map)
            outputs = llm.generate(prompts, params, use_tqdm=False)
            seq_ids = sorted(set(llm.seq_map) - first_id)
            for index, (prompt, output, seq_id) in enumerate(zip(prompts, outputs, seq_ids)):
                rows = captured[seq_id]
                assert len(rows) == args.tokens
                cache = None
                entry = {'case': name, 'index': index, 'prompt_length': len(prompt), 'tokens': output['token_ids'], 'steps': []}
                for step, ours in enumerate(rows):
                    ids = prompt if step == 0 else [output['token_ids'][step - 1]]
                    with torch.inference_mode():
                        ref = hf(input_ids=torch.tensor([ids], device=args.hf_device),
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = ref.past_key_values
                    logits = ref.logits[0, -1].float().cpu()
                    diff = ours - logits
                    top = logits.topk(2)
                    gap = (top.values[0] - top.values[1]).item()
                    match = ours.argmax().item() == logits.argmax().item()
                    result = {'step': step, 'max_abs': diff.abs().max().item(),
                              'rmse': diff.square().mean().sqrt().item(), 'argmax_match': match,
                              'hf_gap': gap, 'ours_top1': ours.argmax().item(), 'hf_top1': logits.argmax().item()}
                    result['passed'] = bool(torch.isfinite(diff).all()) and result['max_abs'] <= args.max_abs and result['rmse'] <= args.max_rmse and (match or gap <= args.tie_gap)
                    failed |= not result['passed']
                    entry['steps'].append(result)
                report['cases'].append(entry)
                print(name, index, 'length=', len(prompt), 'max_abs=', max(r['max_abs'] for r in entry['steps']),
                      'rmse=', max(r['rmse'] for r in entry['steps']), 'top1_matches=', sum(r['argmax_match'] for r in entry['steps']), '/', len(rows), flush=True)
                del cache, ref
            report['passed'] = not failed
            Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    finally:
        runner.run, runner.run_model = original_run, original_model
        llm.exit()
    if failed:
        raise SystemExit('Alignment failed; inspect ' + args.output)
    print('All alignment cases passed. Report:', args.output, flush=True)


if __name__ == '__main__':
    main()
