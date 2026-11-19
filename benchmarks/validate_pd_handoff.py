"""Two-GPU P→D correctness prototype using trusted local host relay.

Compares complete token/logit traces against the colocated engine, not HF.
Instrumentation and CPU/pickle transfer make these timings unsuitable for
throughput/SLO claims. No receiver prefill is permitted.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import time
import traceback

from benchmarks.bench_replica_service import free_ports


def worker(pipe,gpu,port,model,role):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(1)
    from zjhvllm import LLM,SamplingParams
    from zjhvllm.engine.sequence import Sequence
    engine=None
    try:
        engine=LLM(model,distributed_port=port,enforce_eager=True,gpu_memory_utilization=.12,
                   max_num_seqs=8,max_num_batched_tokens=128,max_model_len=1024)
        engine.scheduler.block_manager.enable_prefix_caching=False
        original=engine.model_runner.run_model
        last=[None]
        def capture(*args):
            logits=original(*args)
            last[0]=hashlib.sha256(logits.detach().float().cpu().numpy().tobytes()).hexdigest()
            return logits
        engine.model_runner.run_model=capture
        def advance(seq_id,one=False):
            hashes=[];kinds=[]
            while not engine.is_finished():
                _,_,stream=engine.step()
                kinds.append(engine.last_step_info['is_prefill'])
                if stream:hashes.append(last[0])
                if one and engine.seq_map[seq_id].num_completion_tokens==1:break
            return hashes,kinds
        pipe.send({'ready':True,'gpu':gpu,'role':role})
        dummy=None
        while True:
            job=pipe.recv()
            if job['cmd']=='exit':break
            cmd=job['cmd']
            if cmd in ('reference','prefill'):
                # Exact token lengths including physical page boundaries.
                text=('Explain batching, cache ownership, resource allocation, and error recovery. '*100)
                ids=engine.tokenizer.encode(text)[:job['length']]
                params=SamplingParams(**job['sampling'])
                seq_id=engine.add_request(ids,params)
                hashes,kinds=advance(seq_id,one=cmd=='prefill')
                if cmd=='reference':
                    seq=engine.seq_map.pop(seq_id)
                    pipe.send({'tokens':seq.completion_token_ids,'hashes':hashes,'steps':kinds})
                else:
                    packet=engine.export_prefill(seq_id)
                    pipe.send({'packet':pickle.dumps(packet),'transfer_id':packet.transfer_id,
                        'hashes':hashes,'nbytes':packet.nbytes,'steps':kinds,
                        'source_blocks':list(engine.seq_map[seq_id].block_table)})
            elif cmd=='accept':
                packet=pickle.loads(job['packet'])
                # Occupy pages to prove that import relocates physical IDs.
                dummy=Sequence([0]*512)
                engine.scheduler.block_manager.allocate(dummy,0)
                seq_id=engine.accept_prefill(packet)
                assert engine.accept_prefill(packet)==seq_id
                pipe.send({'seq_id':seq_id,'destination_blocks':list(engine.seq_map[seq_id].block_table)})
            elif cmd=='ack':
                engine.finish_export(job['transfer_id'],accepted=True)
                assert engine.is_finished() and not engine.seq_map
                assert not engine.scheduler.block_manager.used_block_ids
                pipe.send({'released':True})
            elif cmd=='decode':
                hashes,kinds=advance(job['seq_id'])
                assert not any(kinds),'D worker unexpectedly recomputed prefill'
                seq=engine.seq_map.pop(job['seq_id'])
                engine.scheduler.block_manager.deallocate(dummy);dummy=None
                assert not engine.scheduler.block_manager.used_block_ids
                pipe.send({'tokens':seq.completion_token_ids,'hashes':hashes,'steps':kinds})
            else:raise ValueError(cmd)
    except BaseException:
        pipe.send({'error':traceback.format_exc()})
    finally:
        if engine is not None:engine.exit()
        pipe.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='/home/jhzhou/huggingface/Qwen3-0.6B')
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    parser.add_argument('--output',type=Path,default=Path('benchmarks/pd_handoff/result.json'))
    args=parser.parse_args()
    if len(set(args.gpus))!=2:
        parser.error("P-D validation requires two distinct GPUs")
    # Fail before loading either model on a shared server. Each worker reserves
    # 12% of one device plus headroom; never disturb unrelated workloads.
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total',
                                    '--format=csv,noheader,nounits'],text=True)
    devices={int(row[0]):(int(row[1]),int(row[2]))
             for line in memory.splitlines() if (row:=line.split(','))}
    for gpu in args.gpus:
        free,total=devices[gpu]
        if free<total*.12+2048:
            raise RuntimeError(f"GPU {gpu}: only {free} MiB free; P-D validation needs >{total*.12+2048:.0f} MiB")
    context=mp.get_context('spawn');processes=[];pipes=[]
    def receive(pipe):
        if not pipe.poll(300):raise TimeoutError('P-D worker response timed out')
        result=pipe.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def call(pipe,cmd,**kwargs):pipe.send(dict(cmd=cmd,**kwargs));return receive(pipe)
    source=Path(__file__).resolve().parents[1]
    directory=args.output.parent;directory.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in ['zjhvllm/engine/handoff.py','zjhvllm/engine/scheduler.py','zjhvllm/engine/llm_engine.py','benchmarks/validate_pd_handoff.py']:
        data=(source/name).read_bytes();target=directory/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    report={'scope':'two-GPU host-staged correctness prototype; no performance or HF acceptance claim',
        'model':args.model,'gpus':args.gpus,'prefix_caching':False,'source_sha256':hashes,'cases':[]}
    try:
        for gpu,port,role in zip(args.gpus,free_ports(2),['prefill','decode']):
            parent,child=context.Pipe();proc=context.Process(target=worker,args=(child,gpu,port,args.model,role))
            proc.start();child.close();processes.append(proc);pipes.append(parent)
        report['workers']=[receive(pipe) for pipe in pipes]
        p,d=pipes
        for length,stochastic in [(64,False),(128,False),(256,False),(257,False),(513,False),(128,True)]:
            params={'max_tokens':16,'ignore_eos':True,'greedy':not stochastic,'seed':42,
                    'temperature':.8,'top_k':50,'top_p':.9,'repetition_penalty':1.1 if stochastic else 1.}
            reference=call(p,'reference',length=length,sampling=params)
            start=time.perf_counter();export=call(p,'prefill',length=length,sampling=params)
            receipt=call(d,'accept',packet=export['packet'])
            ack=call(p,'ack',transfer_id=export['transfer_id'])
            result=call(d,'decode',seq_id=receipt['seq_id'])
            row={'input_tokens':length,'stochastic':stochastic,'sampling':params,
                'tokens_equal':reference['tokens']==result['tokens'],
                'logits_bitwise_equal':reference['hashes']==export['hashes']+result['hashes'],
                'source_pages_released':ack['released'],'decode_prefill_steps':sum(result['steps']),
                'payload_bytes':export['nbytes'],'source_blocks':export['source_blocks'],
                'destination_blocks':receipt['destination_blocks'],
                'instrumented_wall_seconds':time.perf_counter()-start,
                'reference_tokens':reference['tokens'],'pd_tokens':result['tokens']}
            report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        assert all(r['tokens_equal'] and r['logits_bitwise_equal'] for r in report['cases'])
    finally:
        for pipe,proc in zip(pipes,processes):
            if proc.is_alive():
                try:pipe.send({'cmd':'exit'})
                except (BrokenPipeError,EOFError):pass
        for proc in processes:
            proc.join(timeout=20)
            if proc.is_alive():proc.terminate();proc.join()

if __name__=='__main__':main()
