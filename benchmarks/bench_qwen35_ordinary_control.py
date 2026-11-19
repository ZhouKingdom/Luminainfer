"""Actual MTP-disabled ordinary Graph, keeping its normal batched arithmetic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from zjhvllm import LLM, SamplingParams
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from benchmarks.bench_mtp_net_gain import checked_init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--port', type=int, default=2539)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    source = json.loads(args.workload.read_text())
    workload = source['workload']
    tokens = source['args']['tokens']; capacity = source['args']['capacity']
    for folder in ('zjhvllm', 'tests'):
        shutil.copytree(folder, args.output/'source'/folder, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__, args.output/'source/benchmark.py')
    shutil.copyfile('benchmarks/bench_mtp_net_gain.py', args.output/'source/checked_init_source.py')
    result = {'complete':False, 'args':vars(args)|{'workload':str(args.workload),'output':str(args.output)},
              'contract':'enable_mtp=False; actual ordinary batch arithmetic; native decode Graph; same prompts/prefill/greedy output budget; NOT rowwise numerical control',
              'warmups':[], 'phases':[],
              'source_sha256':{str(p.relative_to(args.output/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.output/'source').rglob('*.py')}}
    def save():
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save(); engine = None
    try:
        engine = LLM('/home/data/jhzhou/huggingface/Qwen3.5-27B', tensor_parallel_size=args.tp,
            enable_mtp=False, max_num_seqs=capacity, max_model_len=2048, max_num_batched_tokens=1024,
            enforce_eager=False, gpu_memory_utilization=.85, distributed_port=args.port)
        result['ranks'] = engine.model_runner.call('parallel_metadata')
        assert all(capacity in rank['graph_buckets'] for rank in result['ranks'])
        params = SamplingParams(greedy=True, max_tokens=tokens, ignore_eos=True)
        pages = len(engine.scheduler.block_manager.free_block_ids)
        slots = len(engine.scheduler.free_state_slots)
        reference = None
        for iteration in range(3):
            outputs = []; prefill_seconds = decode_seconds = 0.; decode_tokens = 0; groups = []
            for offset in range(0,len(workload),capacity):
                torch.cuda.synchronize(); start = perf_counter()
                seqs = [engine.seq_map[engine.add_request(item['input_ids'], params)] for item in workload[offset:offset+capacity]]
                engine.step()
                assert all(seq.num_completion_tokens==1 and seq.num_cached_tokens==seq.num_prompt_tokens for seq in seqs)
                torch.cuda.synchronize(); prefill_seconds += perf_counter()-start
                before = getattr(engine.model_runner,'decode_graph_replays',0)
                torch.cuda.synchronize(); start = perf_counter()
                steps = 0
                while not engine.is_finished():
                    engine.step(); steps += 1
                torch.cuda.synchronize(); decode_seconds += perf_counter()-start
                assert engine.model_runner.decode_graph_replays-before==steps
                ids = [list(seq.completion_token_ids) for seq in seqs]
                outputs.extend(ids); decode_tokens += sum(map(len,ids))-len(seqs)
                groups.append({'decode_graph_replays':steps,'outputs':ids})
                engine.seq_map.clear()
                assert len(engine.scheduler.block_manager.free_block_ids)==pages
                assert len(engine.scheduler.free_state_slots)==slots
            phase = {'outputs':outputs,'groups':groups,'reclaimed':True,
                'decode_output_tokens':decode_tokens,'output_tokens':sum(map(len,outputs)),
                'prefill_seconds':prefill_seconds,'decode_seconds':decode_seconds,
                'e2e_seconds':prefill_seconds+decode_seconds,
                'decode_tok_s':decode_tokens/decode_seconds,
                'e2e_tok_s':sum(map(len,outputs))/(prefill_seconds+decode_seconds),
                'same_as_rowwise_reference':sum(a==b for a,b in zip(outputs,source['warmups'][0]['outputs']))}
            if reference is None:reference=outputs
            assert outputs==reference, 'ordinary Graph repeat mismatch'
            if iteration==0:result['warmups'].append(phase)
            else:phase['repeat']=iteration-1; result['phases'].append(phase)
            save();print(f"ordinary warm={iteration==0} decode={phase['decode_tok_s']:.3f} e2e={phase['e2e_tok_s']:.3f} same_rowwise={phase['same_as_rowwise_reference']}/8",flush=True)
        result['complete']=True; save()
    finally:
        if engine is not None:engine.exit()
    result['processes_exited']=all(not worker.is_alive() for worker in engine.ps)
    save()


if __name__ in ('__main__', '__mp_main__'):
    Qwen3_5ForCausalLM.__init__ = checked_init
if __name__=='__main__':main()
