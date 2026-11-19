"""Compare prefill scan backends in one loaded 27B model; optional nsys capture."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import torch
from zjhvllm import LLM, SamplingParams
import zjhvllm.layers.gated_delta as delta
import zjhvllm.models.qwen3_5 as model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/data/jhzhou/huggingface/Qwen3.5-27B')
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--backends', nargs='+', choices=['ieee', 'ieee8', 'torch', 'tf32x3'], default=['ieee', 'ieee8', 'torch'])
    parser.add_argument('--hf-reference', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', default='benchmarks/qwen35_prefill.json')
    args = parser.parse_args()
    if args.backends[0] != 'ieee':
        parser.error('--backends must start with ieee for a consistent baseline')
    llm = LLM(args.model, enforce_eager=True, max_num_seqs=1,
              max_model_len=args.length + 64, max_num_batched_tokens=args.length,
              gpu_memory_utilization=.9)
    # Vary content across paragraphs rather than looping one short sentence.
    text = '\n'.join(f'Observation {i}: A language model processes a prompt, stores its state, and predicts the next token. Explain how batching affects inference performance.' for i in range(args.length))
    prompt = llm.tokenizer.encode(text)[:args.length]
    assert len(prompt) == args.length
    params = SamplingParams(greedy=True, max_tokens=1, ignore_eos=True)
    original_chunk, original_scan = model._chunk_gated_delta_rule, delta.gated_delta_scan
    selected = 'tf32x3'
    captured = []
    original_run_model = llm.model_runner.run_model

    def chunk(*a, **kw):
        kw['scan_backend'] = 'torch' if selected == 'torch' else 'triton'
        return original_chunk(*a, **kw)

    def scan(*a, **kw):
        kw['input_precision'] = 'ieee' if selected in ('ieee', 'ieee8') else 'tf32x3'
        kw['num_warps'] = 8 if selected == 'ieee8' else 4
        return original_scan(*a, **kw)

    def run_model(*a, **kw):
        logits = original_run_model(*a, **kw)
        captured.append(logits.detach().float().cpu())
        return logits

    model._chunk_gated_delta_rule, delta.gated_delta_scan = chunk, scan
    llm.model_runner.run_model = run_model
    report = {'model': args.model, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'length': args.length, 'batch': 1, 'new_tokens': 1,
              'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
              'timing': 'generate includes scheduler, state zeroing, prefill and sampling; excludes model load/warmup',
              'results': {}}
    logits_by_backend = {}
    baseline_logits = None
    expected_token = None
    try:
        # Compile/warm all variants before timing or profiler capture.
        for selected in args.backends:
            llm.generate([prompt], params, use_tqdm=False)
        if args.profile:
            torch.cuda.profiler.start()
        for selected in args.backends:
            times = []
            for _ in range(args.repeats):
                captured.clear()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.cuda.nvtx.range(f'qwen35/prefill/{selected}/L{args.length}'):
                    output = llm.generate([prompt], params, use_tqdm=False)
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                token = output[0]['token_ids']
                if baseline_logits is None:
                    baseline_logits = captured[-1]
                    expected_token = token
                logits_by_backend[selected] = captured[-1]
                difference = (captured[-1] - baseline_logits).abs()
                top2 = captured[-1][0].topk(2).values
                row = {'argmax_matches_ieee': token == expected_token, 'top1_top2_gap': (top2[0] - top2[1]).item(), 'seconds': times.copy(), 'median_seconds': median(times), 'token_ids': token,
                       'logits_max_abs_vs_ieee': difference.max().item(),
                       'logits_rmse_vs_ieee': difference.square().mean().sqrt().item()}
                report['results'][selected] = row
            print(selected, report['results'][selected], flush=True)
        if args.profile:
            torch.cuda.profiler.stop()
        if args.hf_reference:
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                    device_map={'': 'cuda:1'}, attn_implementation='flash_attention_2', local_files_only=True).eval()
            with torch.inference_mode():
                reference = hf(torch.tensor([prompt], device='cuda:1'), use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
            top2 = reference[0].topk(2)
            report['hf_reference'] = {'token': reference.argmax().item(), 'top1_top2_gap': (top2.values[0]-top2.values[1]).item()}
            for name, logits in logits_by_backend.items():
                diff = logits-reference
                report['results'][name]['hf_argmax_match'] = logits.argmax().item() == reference.argmax().item()
                report['results'][name]['hf_max_abs'] = diff.abs().max().item()
                report['results'][name]['hf_rmse'] = diff.square().mean().sqrt().item()
            print('HF reference:', report['hf_reference'], flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    finally:
        model._chunk_gated_delta_rule, delta.gated_delta_scan = original_chunk, original_scan
        llm.model_runner.run_model = original_run_model
        llm.exit()


if __name__ == '__main__':
    main()
