"""Real checkpoint TP MTP vs same-engine ordinary greedy; output IDs retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from zjhvllm import LLM, SamplingParams
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
import torch.distributed as dist

# Installed at module import so spawned workers perform the same finite check.
_original_model_init = Qwen3_5ForCausalLM.__init__
def _checked_model_init(self, *args, **kwargs):
    _original_model_init(self, *args, **kwargs)
    self.lm_head.finite_checks = 0
    def check(module, inputs, logits):
        valid = torch.isfinite(logits).all().to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        module.finite_checks += 1
        if not valid.item():
            raise FloatingPointError("non-finite logits on a TP rank; benchmark rejected")
    self.lm_head.register_forward_hook(check)
Qwen3_5ForCausalLM.__init__ = _checked_model_init


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tp',type=int,required=True)
    p.add_argument('--port',type=int,default=2470);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=32);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:os.setsid()
    except PermissionError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for name in ('zjhvllm','tests'):
        shutil.copytree(name,args.output/'source'/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__,args.output/'source/benchmark.py')
    report={'tp':args.tp,'pid':os.getpid(),'pgid':os.getpgrp(),'complete':False,'phases':[],
        'contract':'same-engine/same-TP/default numerical backend; greedy ignore EOS; E2E output-only rate includes prefill; no Graph',
        'sha256':{str(p.relative_to(args.output/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.output/'source').rglob('*.py')}}
    def save():(args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();engine=None
    try:
        engine=LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B',tensor_parallel_size=args.tp,
            enable_mtp=True,max_num_seqs=4,max_model_len=4096,max_num_batched_tokens=512,
            enforce_eager=True,gpu_memory_utilization=.85,distributed_port=args.port)
        report['workers']=[p.pid for p in engine.ps];report['ranks']=engine.model_runner.call('parallel_metadata');save()
        assert all(r['mtp']['loaded_groups']==15 for r in report['ranks'])
        items=json.loads(Path('benchmarks/qwen35_mtp_8gpu_20261106/workload.json').read_text())
        indices=[2,39,76,127];prompts=[items[i]['prompt_token_ids'] for i in indices]
        report['workload']=[items[i] for i in indices]
        pages=len(engine.scheduler.block_manager.free_block_ids);slots=len(engine.scheduler.free_state_slots)
        param=SamplingParams(greedy=True,max_tokens=args.tokens,ignore_eos=True)
        engine.generate([[7,11,13]],SamplingParams(greedy=True,max_tokens=4,ignore_eos=True),use_tqdm=False)
        engine.generate_speculative([7,11,13],SamplingParams(greedy=True,max_tokens=4,ignore_eos=True),verification_mode='batched_experimental')
        for c in (1,2,4):
            reference=None
            for mode in (('ordinary','serial','batched') if c==1 else ('ordinary','batched')):
                results=[];torch.cuda.synchronize();start=perf_counter()
                for offset in range(0,len(prompts),c):
                    batch=prompts[offset:offset+c]
                    if mode=='ordinary':results.extend(engine.generate(batch,param,use_tqdm=False))
                    elif c==1:results.append(engine.generate_speculative(batch[0],param,verification_mode='serial' if mode=='serial' else 'batched_experimental'))
                    else:results.extend(engine.generate_speculative_batch(batch,param))
                torch.cuda.synchronize();seconds=perf_counter()-start
                if reference is None:reference=[r['token_ids'] for r in results]
                exact=sum(r['token_ids']==ref for r,ref in zip(results,reference))
                rounds=[round for r in results for round in r.get('rounds',[])]
                phase={'concurrency':c,'mode':mode,'seconds':seconds,'output_tok_s':sum(len(r['token_ids']) for r in results)/seconds,
                    'exact':exact,'paired':len(results),'output_ids':[r['token_ids'] for r in results],
                    'proposed':sum(r.get('proposed',0) for r in rounds),'accepted':sum(r.get('accepted',0) for r in rounds),
                    'reclaimed':len(engine.scheduler.block_manager.free_block_ids)==pages and len(engine.scheduler.free_state_slots)==slots,
                    'rounds':rounds}
                report['phases'].append(phase);save()
                assert exact==len(results) and phase['reclaimed']
                assert all(len(r['token_ids'])==args.tokens for r in results)
                print(f'TP{args.tp} C{c} {mode}: exact={exact}/{len(results)} tok/s={phase["output_tok_s"]:.3f}',flush=True)
                engine.seq_map.clear()
        report['finite_checks_rank0']=engine.model_runner.model.lm_head.finite_checks
        report['complete']=True;save()
    finally:
        if engine is not None:engine.exit()


if __name__=='__main__':main()
