"""Measure HF's own bf16 sensitivity to batch size on identical histories."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    path = '/home/data/jhzhou/huggingface/Qwen3.5-27B'
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={'': 'cuda:1'}, attn_implementation='flash_attention_2').eval()
    ids = AutoTokenizer.from_pretrained(path).encode('The quick brown fox jumps over the lazy dog. ' * 70)
    report = []
    for length in (65, 257, 513):
        one = torch.tensor([ids[:length]], device='cuda:1')
        two = one.repeat(2, 1)
        cache_one = cache_two = None
        rows = []
        for step in range(8):
            single = hf(one, past_key_values=cache_one, use_cache=True, logits_to_keep=1)
            batch = hf(two, past_key_values=cache_two, use_cache=True, logits_to_keep=1)
            cache_one, cache_two = single.past_key_values, batch.past_key_values
            a, b = single.logits[0, -1].float(), batch.logits[0, -1].float()
            diff = a - b
            row = {'length': length, 'step': step, 'max_abs': diff.abs().max().item(), 'rmse': diff.square().mean().sqrt().item(), 'top1_match': a.argmax().item() == b.argmax().item()}
            rows.append(row)
            one = a.argmax().reshape(1, 1)
            two = one.repeat(2, 1)
        print('HF single vs batch', length, 'max_abs=', max(r['max_abs'] for r in rows), 'max_rmse=', max(r['rmse'] for r in rows), flush=True)
        report.extend(rows)
        del cache_one, cache_two, single, batch
    with open('benchmarks/qwen35_hf_batch_sensitivity.json', 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
