"""Single-GPU engine replicas: exact latency records and capacity scaling.

No TP/EP extension. Each worker binds one GPU and a separate rendezvous port.
Closed-loop maintains concurrency by replacing completions; open-loop counts
scheduled-arrival lag in TTFT. Parent orchestrates phases; no HTTP overhead.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import socket
import time
import traceback

from benchmarks.service_metrics import summarize


def worker(connection, gpu, model, port, options):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from zjhvllm import LLM, SamplingParams
    engine = None
    try:
        engine = LLM(model, distributed_port=port, enforce_eager=True,
                     scheduling_policy=options['policy'], active_prefill_tokens=options['prefill_tokens'],
                     cache_schedule_metadata=options['host_metadata'],
                     adaptive_tpot_seconds=options['tpot'],adaptive_ttft_seconds=options['ttft'],
                     max_model_len=4096, max_num_batched_tokens=options['token_budget'],
                     max_num_seqs=options['max_seqs'], gpu_memory_utilization=.9)
        original_policy=engine.scheduler.scheduling_policy
        engine.scheduler.scheduling_policy='adaptive'
        engine.generate(['Warmup inference.'], SamplingParams(greedy=True, max_tokens=4, ignore_eos=True), use_tqdm=False)
        for prompt in options.get('warmups',[]):
            engine.generate([prompt]*8,SamplingParams(greedy=True,max_tokens=4,ignore_eos=True),use_tqdm=False)
        engine.scheduler.scheduling_policy=original_policy
        engine.seq_map.clear()
        connection.send({'type': 'ready', 'gpu': gpu, 'state_slots': engine.model_runner.config.num_state_slots,
                         'kv_blocks': engine.model_runner.config.num_kvcache_blocks,
                         'model_bytes': sum(p.numel()*p.element_size() for p in engine.model_runner.model.parameters()),
                         'allocated_bytes': torch.cuda.memory_allocated(), 'gpu_name': torch.cuda.get_device_name()})
        while True:
            job = connection.recv()
            if job is None:
                break
            engine.scheduler.scheduling_policy=job.get('policy',options['policy'])
            records = []
            steps = []
            issued = 0
            requests = job['requests']
            start = job['start_at']
            mode = job['mode']
            concurrency = job['concurrency']
            peak_running = 0
            progress_at = time.perf_counter()
            def submit(item, scheduled):
                seq_id = engine.add_request(item['prompt_token_ids'], SamplingParams(greedy=True, max_tokens=item['max_tokens'], ignore_eos=True))
                seq = engine.seq_map[seq_id]
                submitted = seq.arrival_time
                seq.arrival_time = scheduled
                metadata[seq_id] = (item['id'], scheduled, submitted)
            metadata = {}
            while time.perf_counter() < start:
                time.sleep(max(0., min(.01, start-time.perf_counter())))
            torch.cuda.reset_peak_memory_stats()
            while issued < len(requests) or not engine.is_finished():
                now = time.perf_counter()
                while issued < len(requests):
                    item = requests[issued]
                    if mode in ('open','burst'):
                        scheduled = start+item['arrival']
                        if scheduled > now:
                            break
                    else:
                        if len(metadata) >= concurrency:
                            break
                        scheduled = time.perf_counter()
                    submit(item, scheduled)
                    issued += 1
                    now = time.perf_counter()
                peak_running = max(peak_running, len(engine.scheduler.running))
                if not engine.is_finished():
                    outputs, _, _ = engine.step()
                    steps.append(dict(engine.last_step_info,at=time.perf_counter()))
                    for seq_id, tokens in outputs:
                        seq = engine.seq_map.pop(seq_id)
                        req_id, scheduled, submitted = metadata.pop(seq_id)
                        records.append({'id': req_id, 'gpu': gpu, 'scheduled_at': scheduled, 'submitted_at': submitted,
                                        'finished_at': seq.finished_time, 'input_tokens': seq.num_prompt_tokens,
                                        'output_tokens': len(tokens), 'token_times': seq.token_times, 'token_ids': tokens,
                                        'output_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
                elif issued < len(requests):
                    time.sleep(max(0., min(.005, start+requests[issued]['arrival']-time.perf_counter())))
                if time.perf_counter()-progress_at > 30:
                    connection.send({'type': 'progress', 'phase': job['name'], 'gpu': gpu, 'issued': issued, 'completed': len(records)})
                    progress_at = time.perf_counter()
            connection.send({'type': 'result', 'gpu': gpu, 'records': records,
                             'steps':steps,'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_running': peak_running})
    except BaseException:
        connection.send({'type': 'error', 'gpu': gpu, 'traceback': traceback.format_exc()})
    finally:
        if engine is not None:
            engine.exit()
        connection.close()


def free_ports(count):
    sockets = []
    for _ in range(count):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        sockets.append(sock)
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    return ports


def make_workload(model, count, input_tokens, output_tokens, seed, input_lengths=None, output_lengths=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rng = random.Random(seed)
    requests = []
    for i in range(count):
        length = rng.choice(input_lengths) if input_lengths else input_tokens
        completion = rng.choice(output_lengths) if output_lengths else output_tokens
        text = '\n'.join(f'Case {i}, fact {j}: Inference processes incoming requests, maintains model state, and generates useful answers. Compare scheduling policies and describe their tradeoffs.' for j in range(length//8+8))
        ids = tokenizer.encode(text)[:length]
        assert len(ids)==length
        requests.append({'id': i, 'prompt_token_ids': ids, 'max_tokens': completion, 'arrival': 0.})
    rng.shuffle(requests)
    return requests


def make_arrivals(selected, mode, rate, seed, burst_size=0, burst_interval=2.):
    rng=random.Random(seed+1000)
    arrival=0.
    result=[]
    for i,item in enumerate(selected):
        if mode=='burst':arrival=(i//burst_size)*burst_interval
        elif mode=='open':arrival+=rng.expovariate(rate)
        result.append(dict(item,arrival=arrival))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--disable-host-metadata',action='store_true')
    parser.add_argument('--policy',choices=['alternating','bounded','adaptive'],default='alternating')
    parser.add_argument('--compare-policies',choices=['alternating','bounded','adaptive'],nargs='+')
    parser.add_argument('--active-prefill-tokens',type=int,default=512)
    parser.add_argument('--adaptive-tpot',type=float,default=.1)
    parser.add_argument('--adaptive-ttft',type=float,default=2.)
    parser.add_argument('--input-lengths',type=int,nargs='+')
    parser.add_argument('--output-lengths',type=int,nargs='+')
    parser.add_argument('--burst-size',type=int,default=0)
    parser.add_argument('--burst-interval',type=float,default=2.)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--model',default='/home/data/jhzhou/huggingface/Qwen3.5-27B')
    parser.add_argument('--gpus',type=int,nargs='+',default=[0])
    parser.add_argument('--replicas',type=int,nargs='+',default=[1])
    parser.add_argument('--concurrencies',type=int,nargs='+',default=[1,4,8,16])
    parser.add_argument('--extra-concurrencies',type=int,nargs='*',default=[])
    parser.add_argument('--requests-per-replica',type=int,default=32)
    parser.add_argument('--input-tokens',type=int,default=128)
    parser.add_argument('--output-tokens',type=int,default=64)
    parser.add_argument('--token-budget',type=int,default=1024)
    parser.add_argument('--max-seqs',type=int,default=32)
    parser.add_argument('--rates',type=float,nargs='*',default=[])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output-dir',default='benchmarks/service_pilot')
    args=parser.parse_args()
    policies=args.compare_policies or [args.policy]
    if len(set(policies))!=len(policies):parser.error('duplicate policies')
    if max(args.replicas)*len(policies)>len(args.gpus) or min(args.replicas)<1 or args.requests_per_replica<1:
        parser.error('insufficient GPUs for simultaneous policy groups')
    if args.repeats<1 or args.burst_interval<=0 or args.burst_size<0 or any(rate<=0 for rate in args.rates):
        parser.error('invalid arrival/repeat parameters')
    lengths=(args.input_lengths or [args.input_tokens])+(args.output_lengths or [args.output_tokens])
    if min(lengths)<1:parser.error('token lengths must be positive')
    directory=Path(args.output_dir)
    if (directory/'summary.json').exists():parser.error('output directory already contains results')
    directory.mkdir(parents=True,exist_ok=True)
    manifest=make_workload(args.model,args.requests_per_replica*max(args.replicas),
                           args.input_tokens,args.output_tokens,args.seed,args.input_lengths,args.output_lengths)
    if any(r['max_tokens']<2 or len(r['prompt_token_ids'])+r['max_tokens']>4096 for r in manifest):
        parser.error('requests must have >=2 output tokens and fit model length 4096')
    source_root=Path(__file__).resolve().parents[1]
    hashes={}
    sources=list((source_root/'zjhvllm').rglob('*.py'))+[Path(__file__).resolve(),source_root/'benchmarks/service_metrics.py']
    for source in sources:
        relative=source.relative_to(source_root);data=source.read_bytes()
        target=directory/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[str(relative)]=hashlib.sha256(data).hexdigest()
    (directory/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    warmups=[]
    for length in sorted({len(r['prompt_token_ids']) for r in manifest}|{128,256}):
        source=max(manifest,key=lambda r:len(r['prompt_token_ids']))['prompt_token_ids']
        warmups.append((source*((length+len(source)-1)//len(source)))[:length])
    (directory/'manifest.json').write_text(json.dumps(manifest)+'\n')
    ctx=mp.get_context('spawn')
    processes=[];connections=[];ready=[]
    report={'config':vars(args),'scope':'engine-native independent single-GPU replicas; no HTTP or model parallelism',
            'manifest_sha256':hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),'workers':ready,'phases':[]}
    def receive(pending,wanted):
        results=[]
        while pending:
            for pipe in wait(pending,timeout=5):
                message=pipe.recv()
                if message['type']=='error':raise RuntimeError(message['traceback'])
                if message['type']==wanted:
                    results.append((pipe,message));pending.remove(pipe)
                else:print(json.dumps(message),flush=True)
            for process,pipe in zip(processes,connections):
                if pipe in pending and process.exitcode is not None:raise RuntimeError(f'Worker exited: {process.exitcode}')
        return results
    try:
        options={'token_budget':args.token_budget,'max_seqs':args.max_seqs,'policy':args.policy,
                 'prefill_tokens':args.active_prefill_tokens,'host_metadata':not args.disable_host_metadata,
                 'tpot':args.adaptive_tpot,'ttft':args.adaptive_ttft,'warmups':warmups}
        for gpu,port in zip(args.gpus,free_ports(len(args.gpus))):
            parent,child=ctx.Pipe()
            process=ctx.Process(target=worker,args=(child,gpu,args.model,port,options))
            process.start();child.close();processes.append(process);connections.append(parent)
        ready.extend(message for _,message in receive(connections.copy(),'ready'))
        print('ALL READY',json.dumps(ready),flush=True)
        for replica_count in args.replicas:
            levels=args.concurrencies+(args.extra_concurrencies if replica_count==max(args.replicas) else [])
            phases=[('closed',c,None) for c in levels]
            if replica_count==max(args.replicas):
                phases.extend(('open',args.max_seqs,rate) for rate in args.rates)
                if args.burst_size:phases.append(('burst',args.max_seqs,args.burst_size/args.burst_interval))
            for trial in range(args.repeats):
                for mode,concurrency,rate in phases:
                    stem=f'{mode}_r{replica_count}_c{concurrency}' if rate is None else f'{mode}_r{replica_count}_rate{rate:g}'
                    start=time.perf_counter()+2
                    selected=make_arrivals(manifest[:args.requests_per_replica*replica_count],mode,rate,args.seed,args.burst_size,args.burst_interval)
                    groups=[];pending=[]
                    for index,policy in enumerate(policies):
                        slot=(index+trial)%len(policies)
                        pipes=connections[slot*replica_count:(slot+1)*replica_count]
                        name=stem+(f'_{policy}' if args.compare_policies else '')+(f'_trial{trial}' if args.repeats>1 else '')
                        jobs=[selected[i::replica_count] for i in range(replica_count)]
                        for pipe,requests in zip(pipes,jobs):
                            pipe.send({'name':name,'mode':mode,'policy':policy,'concurrency':concurrency,'start_at':start,'requests':requests})
                        pending.extend(pipes);groups.append((policy,name,pipes))
                    all_outputs=dict(receive(pending,'result'))
                    for policy,name,pipes in groups:
                        outputs=[all_outputs[pipe] for pipe in pipes]
                        rows=[r for output in outputs for r in output['records']]
                        assert len(rows)==len(selected) and len({r['id'] for r in rows})==len(selected)
                        assert sum(r['output_tokens'] for r in rows)==sum(r['max_tokens'] for r in selected)
                        summary=summarize(rows,start,replica_count)
                        summary['by_input_length']={str(length):summarize([r for r in rows if r['input_tokens']==length],start,replica_count) for length in sorted({r['input_tokens'] for r in rows})}
                        summary.update({'trial':trial,'policy':policy,'name':name,'mode':mode,'concurrency_per_replica':concurrency,
                                        'offered_requests_per_second':rate,'peaks':[{'gpu':o['gpu'],'allocated_bytes':o['peak_allocated_bytes'],'running':o['peak_running']} for o in outputs]})
                        with (directory/(name+'.jsonl')).open('w') as f:
                            for row in rows:f.write(json.dumps(row)+'\n')
                        with (directory/(name+'_steps.jsonl')).open('w') as f:
                            for output in outputs:
                                for step in output['steps']:f.write(json.dumps(dict(step,gpu=output['gpu']))+'\n')
                        report['phases'].append(summary)
                        (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
                        print('PHASE',json.dumps({k:v for k,v in summary.items() if k!='by_input_length'}),flush=True)
    finally:
        for pipe,process in zip(connections,processes):
            if process.is_alive():
                try:pipe.send(None)
                except (BrokenPipeError,EOFError):pass
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():process.terminate();process.join()


if __name__=='__main__':main()
