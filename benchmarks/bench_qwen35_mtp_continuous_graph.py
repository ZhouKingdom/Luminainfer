"""Real TP continuous MTP with Graph/eager parity, finite logits and refill trace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import torch.distributed as dist
from zjhvllm import LLM,SamplingParams
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.engine.model_runner import ModelRunner


def check_finite(logits):
    valid=torch.isfinite(logits).all().to(torch.int32)
    dist.all_reduce(valid,op=dist.ReduceOp.MIN)
    if not valid.item():raise FloatingPointError('non-finite logits on a TP rank')


original_init=Qwen3_5ForCausalLM.__init__
def checked_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.lm_head.finite_checks=0
    def hook(module,inputs,logits):
        if not torch.cuda.is_current_stream_capturing():
            check_finite(logits);module.finite_checks+=1
    self.lm_head.register_forward_hook(hook)
Qwen3_5ForCausalLM.__init__=checked_init
original_verification=ModelRunner.run_verification
def checked_verification(self,*args):
    result=original_verification(self,*args)
    if self.last_verification_graph_replay:
        check_finite(self.verification_graph.logits)
        self.model.lm_head.finite_checks+=1
    return result
ModelRunner.run_verification=checked_verification


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tp',type=int,required=True)
    p.add_argument('--port',type=int,default=2483);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:os.setsid()
    except PermissionError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for name in ('zjhvllm','tests'):
        shutil.copytree(name,args.output/'source'/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__,args.output/'source/benchmark.py')
    report={'tp':args.tp,'pid':os.getpid(),'pgid':os.getpgrp(),'complete':False,'phases':[],
        'contract':'same engine/TP/default numerical backend; output-only E2E includes prefill and all-rank finite checks; graph warmup reported separately',
        'source_sha256':{str(f.relative_to(args.output/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.output/'source').rglob('*.py')}}
    def save():(args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();engine=None
    try:
        engine=LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B',tensor_parallel_size=args.tp,
            enable_mtp=True,max_num_seqs=4,max_model_len=1024,max_num_batched_tokens=256,
            enforce_eager=True,gpu_memory_utilization=.85,distributed_port=args.port)
        report['workers']=[worker.pid for worker in engine.ps]
        report['initial_ranks']=engine.model_runner.call('parallel_metadata');save()
        items=json.loads(Path('benchmarks/qwen35_mtp_8gpu_20261106/workload.json').read_text())
        ids=[2,6,39,76,119,127,119,127];lengths=[36,25,106,128,64,96,255,256]
        budgets=[8,48,16,32,12,24,9,17]
        prompts=[items[index]['prompt_token_ids'][:length] for index,length in zip(ids,lengths)]
        params=[SamplingParams(greedy=True,max_tokens=budget,ignore_eos=True) for budget in budgets]
        report['workload']=[{'input_ids':prompt,'budget':budget} for prompt,budget in zip(prompts,budgets)]
        pages=len(engine.scheduler.block_manager.free_block_ids);slots=len(engine.scheduler.free_state_slots)
        original_step=engine.step_speculative
        trace=[]
        def step():
            finished,count,stream=original_step()
            trace.append({'active':[seq.seq_id for seq in engine.scheduler.running],
                'waiting':[seq.seq_id for seq in engine.scheduler.waiting],
                'finished':[seq_id for seq_id,_ in finished],
                'stream':stream,'stats':engine.last_speculative_stats})
            return finished,count,stream
        engine.step_speculative=step
        reference=None;cohort_reference=None
        for mode in ('ordinary','continuous_eager','cohort_ordinary','cohort_eager','graph_warmup','continuous_graph','continuous_graph_repeat'):
            trace.clear()
            engine.model_runner.call('set_verification_graph',mode.startswith('continuous_graph') or mode=='graph_warmup')
            torch.cuda.synchronize();start=perf_counter()
            if mode=='ordinary':results=engine.generate(prompts,params,use_tqdm=False)
            elif mode=='cohort_ordinary':
                results=[]
                for offset in range(0,len(prompts),4):
                    results.extend(engine.generate(prompts[offset:offset+4],params[offset:offset+4],use_tqdm=False))
            else:results=engine.generate_speculative_batch(prompts,params,continuous=mode!='cohort_eager')
            torch.cuda.synchronize();elapsed=perf_counter()-start
            output_ids=[r['token_ids'] for r in results]
            if reference is None:reference=output_ids
            seqs=list(engine.seq_map.values())
            if mode=='cohort_ordinary':cohort_reference=output_ids
            expected=cohort_reference if mode.startswith('cohort') else reference
            exact=sum(a==b for a,b in zip(output_ids,expected))
            refill=seqs[4].first_token_time<max(s.finished_time for s in seqs[:4])
            phase={'mode':mode,'seconds':elapsed,'output_tok_s':sum(map(len,output_ids))/elapsed,
                'exact':exact,'paired':len(results),'reference':'cohort_ordinary' if mode.startswith('cohort') else 'ordinary',
                'same_as_continuous_ordinary':sum(a==b for a,b in zip(output_ids,reference)),'output_ids':output_ids,'refill_before_initial_cohort_finished':refill,
                'reclaimed':len(engine.scheduler.block_manager.free_block_ids)==pages and len(engine.scheduler.free_state_slots)==slots,
                'graph_rounds':sum(r.get('verification_graph',False) for result in results for r in result.get('rounds',[])),
                'rounds':[r.get('rounds',[]) for r in results],'trace':list(trace),
                'ranks':engine.model_runner.call('parallel_metadata')}
            report['phases'].append(phase);save()
            print(f'TP{args.tp} {mode}: exact={exact}/8 refill={refill} graph_rounds={phase["graph_rounds"]} tok/s={phase["output_tok_s"]:.3f}',flush=True)
            assert exact==len(results) and phase['reclaimed']
            assert all(len(out)==budget for out,budget in zip(output_ids,budgets))
            if mode.startswith('continuous') or mode=='graph_warmup':assert refill
            if mode.startswith('continuous_graph'):assert phase['graph_rounds']>0
            engine.seq_map.clear()
        report['finite_checks_rank0']=engine.model_runner.model.lm_head.finite_checks
        report['complete']=True;save()
    finally:
        if engine is not None:engine.exit()


if __name__=='__main__':main()
