"""Locate first BF16 batch-dependent arithmetic and audit greedy divergence."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from zjhvllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    model='/home/jhzhou/huggingface/Qwen3-0.6B'
    llm=LLM(model,enforce_eager=True,gpu_memory_utilization=.2,max_model_len=1024,max_num_seqs=16)
    prompts=['Hello world','The weather is nice','Machine learning is']
    logs=[]; projections=[]; operations=[]; active=None
    original=llm.model_runner.run_model
    def capture(ids,pos,prefill):
        nonlocal active
        active={}
        logits=original(ids,pos,prefill)
        operations.append(active)
        active=None
        logs.append(logits.float().cpu())
        return logits
    llm.model_runner.run_model=capture
    from zjhvllm.layers.linear import LinearBase
    from zjhvllm.layers.layernorm import RMSNorm
    from zjhvllm.layers.attention import Attention
    from zjhvllm.layers.activation import SiluAndMul
    handles=[]
    def op_hook(name):
        def hook(module,args,output):
            if active is not None:
                inputs=[x.detach().cpu() for x in args if isinstance(x,torch.Tensor)]
                outputs=[x.detach().cpu() for x in (output if isinstance(output,tuple) else (output,)) if isinstance(x,torch.Tensor)]
                active[name]=(inputs,outputs)
        return hook
    for name,module in llm.model_runner.model.named_modules():
        if isinstance(module,(LinearBase,RMSNorm,Attention,SiluAndMul)):
            handles.append(module.register_forward_hook(op_hook(name)))
    first=llm.model_runner.model.model.layers[0].self_attn.qkv_proj
    def projection_hook(module,args,output):
        if len(projections)<4:projections.append((args[0].cpu(),output.cpu()))
    # Capture prefill only (decode has different token layout).
    handle=None
    def generate(items):
        nonlocal handle
        handle=first.register_forward_hook(projection_hook)
        # Remove after the first forward so only prefill is captured.
        one=handle
        def remove_after(module,args,output):one.remove()
        remover=first.register_forward_hook(remove_after)
        logs.clear()
        try:out=llm.generate(items,SamplingParams(greedy=True,max_tokens=16),use_tqdm=False)
        finally:one.remove();remover.remove()
        return out,list(logs)
    try:
        batch,batched=generate(prompts)
        batch_ops=list(operations);operations.clear()
        individual=[];single_logs=[];single_ops=[]
        for prompt in prompts:
            out,rows=generate([prompt]);individual+=out;single_logs.append(rows)
            single_ops.append(list(operations));operations.clear()
        x=torch.cat([p[0] for p in projections[1:]])
        y=torch.cat([p[1] for p in projections[1:]])
        report={'first_qkv_input_max_abs':(projections[0][0]-x).abs().max().item(),
                'first_qkv_output_max_abs':(projections[0][1]-y).abs().max().item(),
                'first_qkv_changed_elements':int((projections[0][1]!=y).sum()),'divergences':[]}
        differences=[]
        for step,bops in enumerate(batch_ops[:9]):
            for name,(inputs,outputs) in bops.items():
                diffs=[]
                for side,items in enumerate((inputs,outputs)):
                    ds=[]
                    for j,item in enumerate(items):
                        reference=torch.cat([o[step][name][side][j] for o in single_ops])
                        ds.append(float((item.float()-reference.float()).abs().max()))
                    diffs.append(ds)
                if any(diffs[1]):
                    differences.append({'step':step,'operation':name,'input_max_abs':diffs[0],'output_max_abs':diffs[1]})
                    break
        report['first_different_operation_by_step']=differences
        # Replay exactly the same projection inputs with float32 accumulation.
        x=x.cuda();weight=first.weight.float()
        full=F.linear(x.float(),weight).bfloat16()
        sizes=[p[0].shape[0] for p in projections[1:]]
        split=torch.cat([F.linear(part.float(),weight).bfloat16() for part in x.split(sizes)])
        report['fp32_projection_batch_max_abs']=(full-split).abs().max().item()
        hf=AutoModelForCausalLM.from_pretrained(model,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').cuda().eval()
        for i,(a,b) in enumerate(zip(batch,individual)):
            a,b=a['token_ids'],b['token_ids']
            if a==b:continue
            d=next(j for j,(u,v) in enumerate(zip(a,b)) if u!=v)
            before=batched[d][i];after=single_logs[i][d][0]
            prefix=llm.tokenizer.encode(prompts[i])+a[:d]
            ref=hf(torch.tensor([prefix],device='cuda')).logits[0,-1].float().cpu()
            report['divergences'].append({'prompt':prompts[i],'position':d,'batch_token':a[d],'single_token':b[d],
                'logits_max_abs':(before-after).abs().max().item(),'batch_top2_gap':float(before.topk(2).values.diff().abs()[0]),
                'single_top2_gap':float(after.topk(2).values.diff().abs()[0]),'hf_top2_gap':float(ref.topk(2).values.diff().abs()[0]),
                'hf_batch_regret':float(ref.max()-ref[a[d]]),'hf_single_regret':float(ref.max()-ref[b[d]])})
        target=Path(__file__).resolve().parent/'batch_consistency_audit.json'
        target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    finally:llm.exit()

if __name__=='__main__':main()
