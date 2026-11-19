"""Measure HF's own batch/forward-geometry sensitivity on audited histories."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path


def run(job):
    gpu,cases=job
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1)
    model=AutoModelForCausalLM.from_pretrained('/home/data/jhzhou/huggingface/Qwen3.5-27B',dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='flash_attention_2',local_files_only=True).eval()
    result=[]
    with torch.inference_mode():
        for case in cases:
            prefix=case['prefix'];prompt_length=case['input_tokens']
            one=torch.tensor([prefix],device='cuda')
            full=model(one,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
            batch=model(one.repeat(2,1),use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
            cached=model(one[:,:prompt_length],use_cache=True,logits_to_keep=1)
            for position in range(prompt_length,len(prefix)):
                cached=model(one[:,position:position+1],past_key_values=cached.past_key_values,use_cache=True,logits_to_keep=1)
            incremental=cached.logits[0,-1].float().cpu()
            row={k:v for k,v in case.items() if k!='prefix'}
            row['hf_self_max_abs']={'full_vs_batch2':float((full-batch).abs().max()),
                                     'full_vs_incremental':float((full-incremental).abs().max())}
            row['references']={}
            for name,logits in [('full_batch1',full),('full_batch2',batch),('incremental_batch1',incremental)]:
                top=logits.topk(2).values
                row['references'][name]={'token':int(logits.argmax()),'top2_gap':float(top[0]-top[1]),
                    'baseline_regret':float(logits.max()-logits[case['baseline_token']]),
                    'candidate_regret':float(logits.max()-logits[case['candidate_token']])}
            result.append(row);print('AUDITED',gpu,case['request'],flush=True)
            del cached
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--gpus',type=int,nargs='+',default=[2,3]);args=parser.parse_args()
    sampled=json.loads((args.directory/'hf_divergence_audit.json').read_text())['cases']
    lookup={(r['mode'],r['trial'],r['request']):r for r in json.loads((args.directory/'divergence_cases.json').read_text())}
    cases=[lookup[(r['mode'],r['trial'],r['request'])] for r in sampled]
    # Balance the number of forced decode steps across independent HF workers.
    groups=[[] for _ in args.gpus];loads=[0]*len(groups)
    for case in sorted(cases,key=lambda r:r['first_divergence'],reverse=True):
        index=min(range(len(groups)),key=lambda i:loads[i]);groups[index].append(case);loads[index]+=case['first_divergence']+8
    with mp.get_context('spawn').Pool(len(groups)) as pool:
        rows=[row for output in pool.map(run,list(zip(args.gpus,groups))) for row in output]
    report={'scope':'16 sampled actual shared histories; HF bf16 reference sensitivity, not alignment acceptance',
            'cases':rows,'reference_top1_disagreements':sum(len({r['token'] for r in row['references'].values()})>1 for row in rows),
            'candidate_worse_gt1_all_references':sum(all(r['candidate_regret']-r['baseline_regret']>1 for r in row['references'].values()) for row in rows)}
    (args.directory/'hf_self_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
