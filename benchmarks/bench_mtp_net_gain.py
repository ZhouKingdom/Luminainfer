"""Paired MTP net gain against BOTH no-draft verification and native decode Graph.

All instrumentation is benchmark-local; production runtime files are unchanged.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from functools import wraps
from time import perf_counter
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import torch.distributed as dist
from zjhvllm import LLM,SamplingParams
from zjhvllm.engine.model_runner import ModelRunner
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.utils.context import get_context,reset_context
import zjhvllm.engine.model_runner as runner_module


def finite(logits):
    valid=torch.isfinite(logits).all().to(torch.int32)
    if dist.is_initialized():dist.all_reduce(valid,op=dist.ReduceOp.MIN)
    if not valid.item():raise FloatingPointError('non-finite logits; measurement rejected')


def add_count(r,name,value=1):
    if hasattr(r,'net_counts'):r.net_counts[name]=r.net_counts.get(name,0)+value


def oracle_proposal(seq, references):
    """Diagnostic only: replay a known continuation without running a drafter.

    A divergence invalidates the optimistic bound: never keep feeding tokens
    from a different prefix. The real verification/commit path is unchanged.
    """
    expected = references[tuple(seq.token_ids[:seq.num_prompt_tokens])]
    completed = seq.num_completion_tokens
    if list(seq.completion_token_ids) != expected[:completed]:
        raise AssertionError('oracle prefix diverged from ordinary Graph reference')
    if completed >= len(expected):
        raise ValueError('oracle reference exhausted')
    return [expected[completed]], {'draft_source': 'offline_oracle_not_deployable'}


_original_init=Qwen3_5ForCausalLM.__init__
def checked_init(self,*args,**kwargs):
    _original_init(self,*args,**kwargs)
    def hook(module,inputs,logits):
        if not torch.cuda.is_current_stream_capturing():finite(logits)
    self.lm_head.register_forward_hook(hook)

_original_run=ModelRunner.run_model
def checked_run(self,ids,positions,is_prefill):
    previous=getattr(self,'graph_vars',None)
    if not is_prefill and not self.enforce_eager and hasattr(self,'net_native_vars'):
        self.graph_vars=self.net_native_vars[len(ids)]
    try:
        result=_original_run(self,ids,positions,is_prefill)
        if not is_prefill:
            add_count(self,'native_target_forwards');add_count(self,'native_target_rows',len(ids))
        else:add_count(self,'prefill_forwards')
        return result
    finally:
        if previous is not None:self.graph_vars=previous

_original_verify=ModelRunner.run_verification
def checked_verify(self,*args):
    start=perf_counter();result=_original_verify(self,*args)
    if self.last_verification_graph_replay:finite(self.verification_graph.logits)
    else:raise AssertionError('verification fell back to eager in Graph control')
    add_count(self,'verification_forwards');add_count(self,'verification_rows',len(args[0]))
    add_count(self,'verification_seconds',perf_counter()-start)
    return result

_original_draft=ModelRunner.draft_mtp
@wraps(_original_draft)
def checked_draft(self,*args):
    start=perf_counter();result=_original_draft(self,*args)
    add_count(self,'draft_forwards');add_count(self,'draft_seconds',perf_counter()-start)
    add_count(self,'draft_extend_pairs',result[1]['draft_extend_tokens'])
    return result


_original_draft_batch=ModelRunner.draft_mtp_batch
@wraps(_original_draft_batch)
def checked_draft_batch(self,seqs):
    start=perf_counter();result=_original_draft_batch(self,seqs)
    add_count(self,'draft_forwards');add_count(self,'draft_batch_calls')
    add_count(self,'draft_requests',len(seqs))
    add_count(self,'draft_seconds',perf_counter()-start)
    add_count(self,'draft_extend_pairs',sum(info['draft_extend_tokens'] for _,info in result))
    return result


_original_draft_k=ModelRunner.draft_mtp_k_batch
@wraps(_original_draft_k)
def checked_draft_k(self,seqs,limits):
    start=perf_counter();result=_original_draft_k(self,seqs,limits)
    add_count(self,'draft_forwards',max(limits));add_count(self,'draft_batch_calls')
    add_count(self,'draft_requests',len(seqs));add_count(self,'draft_seconds',perf_counter()-start)
    add_count(self,'draft_extend_pairs',sum(info['draft_extend_tokens']+info['draft_recursive_pairs'] for _,info in result))
    return result


def prepare_native_control(self):
    """Native hidden-state Graph, same C1 arithmetic as the MTP target.

    Capture B3 too, so an unsupported bucket cannot handicap the no-MTP
    baseline. Separate capture buffers are selected by checked_run above.
    """
    original=runner_module.set_context
    maximum=self.config.max_num_seqs
    def set_decode_context(*args,**kwargs):
        original(*args,**kwargs)
        if not get_context().is_prefill:
            get_context().decode_equivalent=True
            get_context().decode_batched=(getattr(self.config,'qwen35_linear_backend','decode_equivalent')=='batched')
    runner_module.set_context=set_decode_context
    try:
        self.capture_cudagraph()
        graphs=dict(self.graphs);variables={b:self.graph_vars for b in graphs}
        pools=[self.graph_pool]
        if maximum>=3 and 3 not in graphs:
            self.config.max_num_seqs=3
            self.capture_cudagraph()
            graphs[3]=self.graphs[3];variables[3]=self.graph_vars;pools.append(self.graph_pool)
        self.graphs=graphs;self.graph_bs=sorted(graphs)
        self.net_native_vars=variables;self.net_native_pools=pools
        self.graph_vars=variables[maximum]
    finally:
        self.config.max_num_seqs=maximum;runner_module.set_context=original;reset_context()
    if self.world_size>1:dist.barrier()


def net_reset(self,native,batch_draft=False,fused_gdn=False,mtp_steps=1):
    self.net_counts={};self.enforce_eager=not native
    self.config.mtp_batch_draft=batch_draft
    self.config.verification_fused_gdn=fused_gdn
    self.config.mtp_num_steps=mtp_steps
    self.config.verification_cuda_graph=True
    if self.world_size>1:dist.barrier()


def net_metrics(self):
    local={'rank':self.rank,'counts':dict(getattr(self,'net_counts',{})),
        'native_replays':getattr(self,'decode_graph_replays',0),
        'graphs':{str(k):g.replays for k,g in getattr(self,'verification_graphs',{}).items()}}
    if self.world_size==1:return [local]
    result=[None]*self.world_size;dist.all_gather_object(result,local);return result

_original_exit=ModelRunner.exit
def clean_exit(self):
    for name in ('net_native_vars','net_native_pools'):
        if hasattr(self,name):delattr(self,name)
    gc.collect();return _original_exit(self)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tp',type=int,required=True,choices=[1,2,4,8])
    p.add_argument('--capacity',type=int,default=4,choices=[1,4])
    p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--port',type=int,default=2510)
    p.add_argument('--backend',type=str,default='decode_equivalent',
                   choices=['decode_equivalent','batched','batch_invariant'])
    p.add_argument('--oracle',action='store_true',
                   help='add offline correct-draft diagnostic; requires exact default arithmetic')
    p.add_argument('--batch-draft',action='store_true',help='add packed MTP draft comparison')
    p.add_argument('--compare-fused-gdn',action='store_true',help='compare short-window GDN verify fusion in the same engine')
    p.add_argument('--mtp-steps',nargs='+',type=int,choices=range(1,9),help='sweep recurrent draft K with fused verification')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.oracle and args.backend!='decode_equivalent':
        p.error('oracle bound currently requires decode_equivalent; batched prefixes may diverge')
    if args.compare_fused_gdn and not args.batch_draft:
        p.error('--compare-fused-gdn requires --batch-draft')
    if args.mtp_steps and (args.oracle or args.compare_fused_gdn):
        p.error('--mtp-steps is a separate sweep; omit --oracle and --compare-fused-gdn')
    args.output.mkdir(parents=True,exist_ok=False)
    try:os.setsid()
    except PermissionError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for name in ('zjhvllm','tests'):
        shutil.copytree(name,args.output/'source'/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__,args.output/'source/benchmark.py')
    report={'args':vars(args)|{'output':str(args.output)},'pid':os.getpid(),'pgid':os.getpgrp(),
        'complete':False,'phases':[],'warmups':[],
        'contract':f'{args.backend}; one resident MTP-enabled engine; same fixed cohorts/prefill; output-only E2E + post-prefill decode; all-rank finite checks; native control captures B3 and matching backend arithmetic; warm every mode then reverse order in repeat1; mtp-steps uses recurrent draft hidden and truncates temporary KV',
        'oracle_contract':('offline ordinary-Graph continuation; zero draft model/IPC cost, real verification and commit; must accept every proposal and preserve reference prefix; optimistic diagnostic only' if args.oracle else None),
        'source_sha256':{str(f.relative_to(args.output/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.output/'source').rglob('*.py')}}
    def save():(args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();engine=None
    try:
        engine=LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B',tensor_parallel_size=args.tp,
            enable_mtp=True,max_num_seqs=args.capacity,max_model_len=2048,max_num_batched_tokens=1024,
            verification_cuda_graph=True,enforce_eager=True,gpu_memory_utilization=.85,distributed_port=args.port,
            qwen35_linear_backend=args.backend,mtp_num_steps=max(args.mtp_steps or [1]),
            verification_state_budget_mb=1024 if args.mtp_steps else 512,
            verification_graph_max_layouts=64 if args.mtp_steps else 16)
        runner=engine.model_runner
        report['workers']=[w.pid for w in engine.ps];report['ranks']=runner.call('parallel_metadata');save()
        assert all(r['mtp']['loaded_groups']==15 for r in report['ranks'])
        runner.call('prepare_native_control')
        report['native_buckets']=list(runner.graphs);save()
        items=json.loads(Path('benchmarks/qwen35_mtp_8gpu_20261106/workload.json').read_text())
        indices=[2,6,39,76,119,127,119,127];lengths=[36,25,106,128,64,96,255,256]
        prompts=[items[i]['prompt_token_ids'][:n] for i,n in zip(indices,lengths)]
        report['workload']=[{'source_id':i,'input_ids':ids,'max_tokens':args.tokens} for i,ids in zip(indices,prompts)]
        params=SamplingParams(greedy=True,max_tokens=args.tokens,ignore_eos=True)
        pages=len(engine.scheduler.block_manager.free_block_ids);slots=len(engine.scheduler.free_state_slots)
        mode_state={'value':None};original_call=runner.call;oracle_references={}
        def dispatch(name,*call_args):
            if name=='draft_mtp' and mode_state['value']=='no_draft_graph':
                return [],{'draft_source':'disabled_control'}
            if name=='draft_mtp' and mode_state['value']=='oracle_graph':
                return oracle_proposal(call_args[0],oracle_references)
            return original_call(name,*call_args)
        runner.call=dispatch

        def execute(mode):
            mtp_steps=int(mode.split('_')[1][1:]) if mode.startswith('mtp_k') else 1
            canonical='mtp_batched_graph' if mode.startswith('mtp_k') else mode.replace('_fused_graph','_graph')
            fused_gdn=canonical!=mode
            mode_state['value']=canonical
            all_ids=[];rounds=[];prefill_seconds=decode_seconds=0.;decode_tokens=0;groups=[]
            before=runner.call('net_metrics')
            for offset in range(0,len(prompts),args.capacity):
                # ALL modes use ordinary eager for the SAME one-batch prefill.
                runner.call('net_reset',False)
                torch.cuda.synchronize();begin=perf_counter()
                ids=[engine.add_request(prompt,params) for prompt in prompts[offset:offset+args.capacity]]
                seqs=[engine.seq_map[i] for i in ids]
                engine.step()
                assert all(s.num_completion_tokens==1 and s.num_cached_tokens==s.num_prompt_tokens for s in seqs)
                torch.cuda.synchronize();prefill_seconds+=perf_counter()-begin
                runner.call('net_reset',canonical=='native_graph',canonical=='mtp_batched_graph',fused_gdn,mtp_steps)
                completed_before=sum(s.num_completion_tokens for s in seqs)
                native_before=getattr(runner,'decode_graph_replays',0)
                group_rounds=[]
                torch.cuda.synchronize();begin=perf_counter()
                while not engine.is_finished():
                    if canonical=='native_graph':engine.step()
                    else:
                        engine.step_speculative()
                        group_rounds.extend(engine.last_speculative_stats.values())
                torch.cuda.synchronize();decode_seconds+=perf_counter()-begin
                decode_tokens+=sum(s.num_completion_tokens for s in seqs)-completed_before
                after=runner.call('net_metrics')
                if canonical not in ('mtp_graph','mtp_batched_graph'):
                    assert all(rank['counts'].get('draft_forwards',0)==0 for rank in after)
                if canonical=='native_graph':
                    assert after[0]['native_replays']-native_before==after[0]['counts']['native_target_forwards']
                groups.append({'outputs':[list(s.completion_token_ids) for s in seqs],
                    'rounds':group_rounds,'ranks':after})
                all_ids.extend(list(s.completion_token_ids) for s in seqs);rounds.extend(group_rounds)
                engine.seq_map.clear()
                assert len(engine.scheduler.block_manager.free_block_ids)==pages and len(engine.scheduler.free_state_slots)==slots
            outputs=sum(map(len,all_ids))
            result={'mode':mode,'prefill_seconds':prefill_seconds,'decode_seconds':decode_seconds,
                'e2e_seconds':prefill_seconds+decode_seconds,'output_tokens':outputs,'decode_output_tokens':decode_tokens,
                'e2e_tok_s':outputs/(prefill_seconds+decode_seconds),'decode_tok_s':decode_tokens/decode_seconds,
                'outputs':all_ids,'groups':groups,'reclaimed':True,
                'proposed':sum(r.get('proposed',0) for r in rounds),'accepted':sum(r.get('accepted',0) for r in rounds),
                'target_forwards':sum(sum(r['counts'].get(k,0) for k in ('native_target_forwards','verification_forwards')) for g in groups for r in g['ranks'] if r['rank']==0),
                'target_rows':sum(sum(r['counts'].get(k,0) for k in ('native_target_rows','verification_rows')) for g in groups for r in g['ranks'] if r['rank']==0),
                'draft_forwards':sum(r['counts'].get('draft_forwards',0) for g in groups for r in g['ranks'] if r['rank']==0)}
            if canonical=='no_draft_graph':assert result['proposed']==result['accepted']==result['draft_forwards']==0
            if canonical=='oracle_graph':
                assert result['proposed']==result['accepted']>0,'oracle is not fully accepted'
                assert result['draft_forwards']==0
            if canonical!='native_graph':assert all(s['verification_graph'] for s in rounds)
            else:
                replays=sum(g['ranks'][0]['counts'].get('native_target_forwards',0) for g in groups)
                assert replays>0
            return result

        modes=['native_graph','no_draft_graph','mtp_graph']
        if args.oracle:modes.append('oracle_graph')
        if args.batch_draft:modes.append('mtp_batched_graph')
        if args.compare_fused_gdn:
            modes.remove('mtp_graph')
            modes.extend(['no_draft_fused_graph','mtp_batched_fused_graph'])
            if args.oracle:modes.append('oracle_fused_graph')
        if args.mtp_steps:
            modes=['native_graph','no_draft_fused_graph']+[f'mtp_k{k}_graph' for k in dict.fromkeys(args.mtp_steps)]
        reference=None
        strict=args.backend=='decode_equivalent'
        for mode in modes:
            result=execute(mode)
            if reference is None:
                reference=result['outputs']
                for prompt,output in zip(prompts,reference):
                    key=tuple(prompt)
                    if key in oracle_references:assert oracle_references[key]==output
                    oracle_references[key]=output
            result['exact']=sum(a==b for a,b in zip(result['outputs'],reference));report['warmups'].append(result);save()
            if strict:assert result['exact']==len(prompts),'warmup output mismatch'
            print(f'TP{args.tp} warm {mode}: exact={result["exact"]}/8 decode={result["decode_tok_s"]:.3f}',flush=True)
        baseline_layouts={k:tuple(r['graphs']) for k,r in enumerate(runner.call('net_metrics'))}
        for repeat in range(args.repeats):
            for mode in modes if repeat%2==0 else reversed(modes):
                result=execute(mode);result['repeat']=repeat
                result['exact']=sum(a==b for a,b in zip(result['outputs'],reference));report['phases'].append(result);save()
                if strict:assert result['exact']==len(prompts),'measured output mismatch'
                layouts={k:tuple(r['graphs']) for k,r in enumerate(runner.call('net_metrics'))}
                assert layouts==baseline_layouts,'new Graph capture inside timed repeat'
                print(f'TP{args.tp} repeat{repeat} {mode}: exact={result["exact"]}/8 decode={result["decode_tok_s"]:.3f} e2e={result["e2e_tok_s"]:.3f} accepted={result["accepted"]}/{result["proposed"]}',flush=True)
        report['complete']=True;save()
    finally:
        if engine is not None:engine.exit()
    report['processes_exited']=all(not w.is_alive() for w in engine.ps)
    save()


def install_hooks():
    Qwen3_5ForCausalLM.__init__=checked_init
    ModelRunner.run_model=checked_run
    ModelRunner.run_verification=checked_verify
    ModelRunner.draft_mtp=checked_draft
    ModelRunner.draft_mtp_batch=checked_draft_batch
    ModelRunner.draft_mtp_k_batch=checked_draft_k
    ModelRunner.prepare_native_control=prepare_native_control
    ModelRunner.net_reset=net_reset
    ModelRunner.net_metrics=net_metrics
    ModelRunner.exit=clean_exit


if __name__ in ('__main__','__mp_main__'):
    install_hooks()
if __name__=='__main__':main()
