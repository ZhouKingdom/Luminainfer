"""Native TP capacity benchmark, separating E2E output rate and full-batch decode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from zjhvllm import LLM,SamplingParams


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp',type=int,default=1,choices=[1,2,4,8])
    parser.add_argument('--concurrency',type=int,nargs='+',default=[1,32,128])
    parser.add_argument('--input-tokens',type=int,default=128)
    parser.add_argument('--output-tokens',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--port',type=int,default=2401)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--eager',action='store_true')
    parser.add_argument('--check-eager',action='store_true',help='strict same-engine Graph/eager warm-output parity')
    args=parser.parse_args()
    if args.output.exists():parser.error('require new output directory')
    args.output.mkdir(parents=True)
    try:os.setsid()
    except PermissionError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for name in ('zjhvllm','tests'):
        shutil.copytree(name,args.output/'source'/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__,args.output/'source/benchmark.py')
    report={'args':vars(args)|{'output':str(args.output)},'pid':os.getpid(),'pgid':os.getpgrp(),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'source_sha256':{str(p.relative_to(args.output/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.output/'source').rglob('*.py')},
            'complete':False,'phases':[],
            'contract':'output-only rates; fixed lengths, ignore EOS; E2E includes prefill; steady window requires all requests prefilled and active'}
    def save():(args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();engine=None
    try:
        engine=LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B',tensor_parallel_size=args.tp,
                   max_num_seqs=max(args.concurrency),max_model_len=2048,max_num_batched_tokens=2048,
                   enforce_eager=args.eager,gpu_memory_utilization=.85,distributed_port=args.port)
        report['ranks']=engine.model_runner.call('parallel_metadata')
        report['workers']=[p.pid for p in engine.ps];save()
        original_run_model=engine.model_runner.run_model
        finite_checks=0
        def checked_model(*inputs):
            nonlocal finite_checks
            logits=original_run_model(*inputs)
            finite_checks+=1
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError('non-finite target logits; throughput rejected')
            return logits
        engine.model_runner.run_model=checked_model
        items=json.loads(Path('benchmarks/qwen35_mtp_8gpu_20261106/workload.json').read_text())
        filler=engine.tokenizer.encode(' Explain the reasoning and provide a concrete example.')
        prompts=[]
        for index in range(max(args.concurrency)):
            ids=items[index%len(items)]['prompt_token_ids']
            prompts.append((ids+filler*(1+args.input_tokens//len(filler)))[:args.input_tokens])
        (args.output/'prompts.json').write_text(json.dumps(prompts)+'\n')
        pages=len(engine.scheduler.block_manager.free_block_ids)
        slots=len(engine.scheduler.free_state_slots)
        for concurrency in args.concurrency:
            warm=engine.generate(prompts[:concurrency],SamplingParams(greedy=True,max_tokens=8,ignore_eos=True),use_tqdm=False)
            if args.check_eager:
                engine.model_runner.call('set_decode_eager',True)
                eager=engine.generate(prompts[:concurrency],SamplingParams(greedy=True,max_tokens=8,ignore_eos=True),use_tqdm=False)
                engine.model_runner.call('set_decode_eager',args.eager)
                exact=sum(a['token_ids']==b['token_ids'] for a,b in zip(warm,eager))
                report.setdefault('eager_parity',[]).append({'concurrency':concurrency,'exact':exact,'paired':len(warm)})
                save()
                if exact!=len(warm):raise AssertionError('Graph/eager outputs differ; throughput rejected')
        engine.seq_map.clear()
        for repeat in range(args.repeats):
            for concurrency in args.concurrency if repeat%2==0 else reversed(args.concurrency):
                torch.cuda.synchronize();start=perf_counter()
                prior_replays=getattr(engine.model_runner,'decode_graph_replays',0)
                seq_ids=[engine.add_request(prompt,SamplingParams(greedy=True,max_tokens=args.output_tokens,ignore_eos=True))
                         for prompt in prompts[:concurrency]]
                seqs=[engine.seq_map[i] for i in seq_ids]
                steady_seconds=steady_tokens=steady_steps=0
                decode_batches=[]
                while any(not s.is_finished for s in seqs):
                    eligible=all(not s.is_finished and s.num_cached_tokens>=s.num_prompt_tokens for s in seqs)
                    before=sum(s.num_completion_tokens for s in seqs)
                    begin=perf_counter();engine.step();elapsed=perf_counter()-begin
                    emitted=sum(s.num_completion_tokens for s in seqs)-before
                    if not engine.last_step_info['is_prefill']:
                        decode_batches.append(emitted)
                        if eligible:
                            steady_seconds+=elapsed;steady_tokens+=emitted;steady_steps+=1
                torch.cuda.synchronize();seconds=perf_counter()-start
                phase={'repeat':repeat,'concurrency':concurrency,'seconds':seconds,
                       'graph_replays':getattr(engine.model_runner,'decode_graph_replays',0)-prior_replays,
                       'output_tokens':sum(s.num_completion_tokens for s in seqs),
                       'e2e_output_tok_s':sum(s.num_completion_tokens for s in seqs)/seconds,
                       'steady_output_tok_s':steady_tokens/steady_seconds if steady_seconds else None,
                       'steady_tokens':steady_tokens,'steady_seconds':steady_seconds,'steady_steps':steady_steps,
                       'max_decode_batch':max(decode_batches,default=0),
                       'mean_decode_batch':sum(decode_batches)/len(decode_batches),
                       'resources_reclaimed':len(engine.scheduler.block_manager.free_block_ids)==pages
                                              and len(engine.scheduler.free_state_slots)==slots,
                       'outputs':[list(s.completion_token_ids) for s in seqs]}
                report['phases'].append(phase);save();engine.seq_map.clear()
                print(f"TP{args.tp} C{concurrency} repeat{repeat}: E2E {phase['e2e_output_tok_s']:.2f}, "
                      f"steady {phase['steady_output_tok_s']}, max batch {phase['max_decode_batch']}",flush=True)
        report['complete']=True
        report['finite_checks']=finite_checks
        report['all_logits_finite']=True
        report['all_resources_reclaimed']=all(p['resources_reclaimed'] for p in report['phases'])
        report['lengths_passed']=all(all(len(row)==args.output_tokens for row in p['outputs']) for p in report['phases'])
        report['same_config_repeat_exact']=all(p['outputs']==next(q['outputs'] for q in report['phases']
            if q['repeat']==0 and q['concurrency']==p['concurrency']) for p in report['phases'])
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        save()
        if engine is not None:engine.exit()


if __name__=='__main__':main()
