"""HF audit at first greedy divergence, conditioned on the shared history."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main():
    root=Path(__file__).resolve().parent
    a=root/'scheduling_pilot_alternating';b=root/'scheduling_pilot_mixed'
    manifest={r['id']:r for r in json.loads((a/'manifest.json').read_text())}
    hf=AutoModelForCausalLM.from_pretrained('/home/data/jhzhou/huggingface/Qwen3.5-27B',
            dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='flash_attention_2',local_files_only=True).eval()
    rows=[]
    for phase in ['closed_r1_c16','open_r1_rate1']:
        old={r['id']:r for r in map(json.loads,(a/(phase+'.jsonl')).read_text().splitlines())}
        new={r['id']:r for r in map(json.loads,(b/(phase+'.jsonl')).read_text().splitlines())}
        for req_id in sorted(old):
            x,y=old[req_id]['token_ids'],new[req_id]['token_ids']
            if x==y:continue
            d=next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v)
            prefix=manifest[req_id]['prompt_token_ids']+x[:d]
            logits=hf(torch.tensor([prefix],device='cuda:0'),use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
            top=logits.topk(2)
            row={'phase':phase,'request':req_id,'first_divergence':d,'old_token':x[d],'new_token':y[d],
                 'hf_token':logits.argmax().item(),'hf_top2_gap':(top.values[0]-top.values[1]).item(),
                 'hf_old_regret':(logits.max()-logits[x[d]]).item(),'hf_new_regret':(logits.max()-logits[y[d]]).item()}
            rows.append(row)
            print(row,flush=True)
    (root/'scheduling_pilot_hf_audit.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
