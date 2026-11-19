"""Stratified HF audit of first policy divergences on shared histories."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--cases-per-length',type=int,default=4)
    args=parser.parse_args()
    cases=json.loads((args.directory/'divergence_cases.json').read_text())
    groups=defaultdict(list)
    for case in cases:groups[case['input_tokens']].append(case)
    selected=[];rng=random.Random(29)
    for length,rows in sorted(groups.items()):
        rng.shuffle(rows);selected.extend(rows[:args.cases_per_length])
    report={'population':len(cases),'selection':f'seed29, up to{args.cases_per_length} first-divergence cases per input length',
            'scope':'sampled shared-history argmax audit; not full HF logits acceptance','cases':[]}
    if selected:
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(1)
        model=AutoModelForCausalLM.from_pretrained('/home/data/jhzhou/huggingface/Qwen3.5-27B',
            dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='flash_attention_2',local_files_only=True).eval()
        with torch.inference_mode():
            for case in selected:
                logits=model(torch.tensor([case['prefix']],device='cuda:0'),use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
                maximum=logits.max();top2=logits.topk(2).values
                row={k:v for k,v in case.items() if k!='prefix'}
                row.update({'hf_token':int(logits.argmax()),'hf_top2_gap':float(top2[0]-top2[1]),
                    'baseline_regret':float(maximum-logits[case['baseline_token']]),
                    'candidate_regret':float(maximum-logits[case['candidate_token']])})
                report['cases'].append(row);print(json.dumps(row),flush=True)
    report['candidate_worse_by_gt1']=sum(r['candidate_regret']-r['baseline_regret']>1. for r in report['cases'])
    report['baseline_worse_by_gt1']=sum(r['baseline_regret']-r['candidate_regret']>1. for r in report['cases'])
    (args.directory/'hf_divergence_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
