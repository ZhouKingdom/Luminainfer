"""Distributed verification Graph replay, dynamic requests, and rollback."""
import argparse
import hashlib
import gc
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from zjhvllm.engine.model_runner import ModelRunner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.engine.speculative import HybridStateHistory
from zjhvllm.engine.verification_graph import VerificationGraph
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.models.qwen3_5_mtp import Qwen3_5MTP
from zjhvllm.utils.tp import set_tp
from zjhvllm.utils.context import reset_context
from zjhvllm.sampling_params import SamplingParams


def worker(rank, world, port, output, fused_gdn=False, mtp_steps=1):
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world,
                            timeout=timedelta(minutes=2))
    try:
        torch.manual_seed(918);set_tp(world,rank)
        cfg=SimpleNamespace(model_type='qwen3_5_text',hidden_size=256,intermediate_size=512,hidden_act='silu',
            num_hidden_layers=2,layer_types=['linear_attention','full_attention'],linear_num_key_heads=8,
            linear_num_value_heads=24,linear_key_head_dim=128,linear_value_head_dim=128,
            linear_conv_kernel_dim=4,rms_norm_eps=1e-6,num_attention_heads=8,num_key_value_heads=2,
            head_dim=64,max_position_embeddings=512,attention_bias=False,
            rope_parameters={'rope_theta':10000000,'partial_rotary_factor':.25},vocab_size=128,
            tie_word_embeddings=False,mtp_num_hidden_layers=1,mtp_use_dedicated_embeddings=False,dtype=torch.bfloat16)
        r=ModelRunner.__new__(ModelRunner)
        r.rank,r.world_size,r.tp_size,r.enforce_eager,r.has_linear_state=rank,world,world,True,True
        r.block_size=256
        r.config=SimpleNamespace(hf_config=cfg,enable_mtp=True,max_num_seqs=4,max_model_len=512,
            cache_schedule_metadata=True,eos=-1,verification_state_commit=True,verification_state_budget_mb=512,verification_cuda_graph=True)
        r.config.verification_fused_gdn=fused_gdn
        r.config.mtp_num_steps=mtp_steps
        r.model=Qwen3_5ForCausalLM(cfg).cuda().bfloat16().eval()
        r.model.mtp=Qwen3_5MTP(cfg,512,slots=4).cuda().bfloat16().eval()
        r.S_cache=torch.zeros(4,1,24//world,128,128,device='cuda')
        r.conv_cache=torch.zeros(4,1,5120//world,3,device='cuda',dtype=torch.bfloat16)
        r.kv_cache=torch.zeros(2,1,4,256,max(1,2//world),64,device='cuda',dtype=torch.bfloat16)
        r.target_hidden=torch.zeros(4,512,256,device='cuda',dtype=torch.bfloat16)
        linear=r.model.model.layers[0].linear_attn
        linear.S_cache,linear.conv_cache=r.S_cache[:,0],r.conv_cache[:,0]
        attn=r.model.model.layers[1].self_attn.attn
        attn.k_cache,attn.v_cache=r.kv_cache[:,0]
        seqs=[]
        with torch.inference_mode():
            for slot,ids in [(3,[3,7,11]),(1,[9,13,17,21])]:
                seq=Sequence(ids,SamplingParams(greedy=True,ignore_eos=True,max_tokens=8))
                seq.state_slot,seq.state_ready,seq.num_scheduled_tokens=slot,False,len(ids)
                seq.block_table=[slot]
                x,p=r.prepare_prefill([seq]);logits=r.run_model(x,p,True)
                seq.num_cached_tokens=len(ids);seq.append_token(r._root_predictions(logits.argmax(-1)).item())
                seq.state_ready=True;seq.is_prefill=False;seqs.append(seq)
                reset_context()
            buffers=[r.S_cache,r.conv_cache,r.kv_cache,r.target_hidden]
            initial=[x.clone() for x in buffers]
            def restore():
                for dest,src in zip(buffers,initial):dest.copy_(src)
                reset_context()
            correct=[]
            for seq in seqs:
                out,_=r._speculative_transaction(r.verify_greedy,(seq,[],1,'serial'))
                correct.append(out[0])
            restore()
            drafts=[[correct[0]],[(correct[1]+1)%128]]
            expected=[]
            for seq,draft in zip(seqs,drafts):
                out,_=r._speculative_transaction(r.verify_greedy,(seq,draft,3,'serial'))
                expected.append(out)
            expected_state=[r.S_cache.clone(),r.conv_cache.clone()]
            restore()
            actual,stats=r._speculative_transaction(r.verify_greedy_batch,(seqs,drafts,[3,3]))
            assert actual==expected and [s['accepted'] for s in stats]==[1,0]
            assert all(s['verification_graph'] for s in stats)
            restore()
            swapped,_=r._speculative_transaction(r.verify_greedy_batch,(list(reversed(seqs)),list(reversed(drafts)),[3,3]))
            assert swapped==list(reversed(expected))
            for actual_buffer,reference in zip(buffers,expected_state):
                torch.testing.assert_close(actual_buffer,reference,rtol=0,atol=0)
            restore()
            for seq in seqs:
                draft,_=r._speculative_transaction(r.draft_mtp,(seq,),draft_only=True)
                assert len(draft)==1
            lengths=list(r.model.mtp.cache.lengths)
            # Root decisions are authoritative even if another rank proposes a different value.
            assert r._root_predictions(torch.tensor([rank+17],device='cuda')).item()==17
            fault_results=[]
            for fault in ('commit','prepare','draft','graph_allocate'):
                restore()
                original_commit=HybridStateHistory.commit_many
                original_prepare=r.prepare_verification_history
                original_update=VerificationGraph.update
                if fault=='commit':
                    def commit(history,*args):
                        original_commit(history,*args)
                        if rank==world-1:raise RuntimeError('injected rank-local commit fault')
                    HybridStateHistory.commit_many=commit
                elif fault=='prepare':
                    def prepare(*args):
                        if rank==world-1:raise MemoryError('injected rank-local workspace fault')
                        return original_prepare(*args)
                    r.prepare_verification_history=prepare
                elif fault=='graph_allocate':
                    def update(graph,*args):
                        if rank==world-1:raise MemoryError('injected graph buffer fault')
                        return original_update(graph,*args)
                    VerificationGraph.update=update
                else:
                    r.model.mtp.cache.set_length(3,seqs[0].num_cached_tokens if rank==world-1 else 0)
                before=[x.clone() for x in buffers[:2]]
                before_lengths=list(r.model.mtp.cache.lengths)
                failed=False
                try:
                    if fault=='draft':r._speculative_transaction(r.draft_mtp,(seqs[0],),draft_only=True)
                    else:r._speculative_transaction(r.verify_greedy_batch,(seqs,[drafts[0],[]] if fault=='graph_allocate' else drafts,[3,3]))
                except (RuntimeError,MemoryError,ValueError):failed=True
                assert failed==(rank==0)
                for actual_buffer,reference in zip(buffers,before):assert torch.equal(actual_buffer,reference)
                assert r.model.mtp.cache.lengths==before_lengths
                fault_results.append(fault)
                HybridStateHistory.commit_many=original_commit
                r.prepare_verification_history=original_prepare
                VerificationGraph.update=original_update
                r.restore_mtp_lengths(dict(enumerate(lengths)))
                # Workers remain usable after rollback.
                restore();out,_=r._speculative_transaction(r.verify_greedy_batch,(seqs,drafts,[3,3]))
                assert out==expected
            batch_faults=[]
            for fault in ('none','prepare','prediction'):
                restore()
                for seq in seqs:r.model.mtp.cache.set_length(seq.state_slot,0)
                if fault=='prepare' and rank==world-1:
                    r.model.mtp.cache.set_length(seqs[-1].state_slot,seqs[-1].num_cached_tokens)
                before_lengths=list(r.model.mtp.cache.lengths)
                original_predictions=r._root_predictions
                if fault=='prediction':
                    def fail_after_predictions(*args):
                        tokens=original_predictions(*args)
                        if rank==world-1:raise RuntimeError('injected packed draft prediction fault')
                        return tokens
                    r._root_predictions=fail_after_predictions
                failed=False
                try:
                    packed=r._speculative_transaction(r.draft_mtp_batch,(seqs,),draft_only=True)
                except (ValueError,RuntimeError):failed=True
                finally:r._root_predictions=original_predictions
                assert failed==(fault!='none' and rank==0)
                if fault=='none':
                    assert len(packed)==len(seqs) and all(len(draft)==1 for draft,_ in packed)
                    for seq in seqs:assert r.model.mtp.cache.get_length(seq.state_slot)==seq.num_cached_tokens
                    proposals=[None]*world;dist.all_gather_object(proposals,packed)
                    assert all(value==proposals[0] for value in proposals)
                else:
                    assert r.model.mtp.cache.lengths==before_lengths
                    batch_faults.append(fault)
                assert r.model.mtp.cache.batch_layout is None
                for actual_buffer,reference in zip(buffers[:2],initial[:2]):assert torch.equal(actual_buffer,reference)
                restore();out,_=r._speculative_transaction(r.verify_greedy_batch,(seqs,drafts,[3,3]))
                assert out==expected
            recursive_faults=[]
            if mtp_steps>1:
                limits=[mtp_steps,max(1,mtp_steps-1)]
                for fault in ('none','prepare','prediction'):
                    restore()
                    for seq in seqs:r.model.mtp.cache.set_length(seq.state_slot,0)
                    if fault=='prepare' and rank==world-1:
                        r.model.mtp.cache.set_length(seqs[-1].state_slot,seqs[-1].num_cached_tokens)
                    before_lengths=list(r.model.mtp.cache.lengths)
                    original_predictions=r._root_predictions;prediction_calls=[0]
                    if fault=='prediction':
                        def fail_at_final_prediction(*args):
                            tokens=original_predictions(*args);prediction_calls[0]+=1
                            if prediction_calls[0]==mtp_steps and rank==world-1:
                                raise RuntimeError('injected final recursive prediction fault')
                            return tokens
                        r._root_predictions=fail_at_final_prediction
                    failed=False
                    try:packed=r._speculative_transaction(r.draft_mtp_k_batch,(seqs,limits),draft_only=True)
                    except (ValueError,RuntimeError):failed=True
                    finally:r._root_predictions=original_predictions
                    assert failed==(fault!='none' and rank==0)
                    if fault=='none':
                        assert [len(draft) for draft,_ in packed]==limits
                        assert all(r.model.mtp.cache.get_length(seq.state_slot)==seq.num_cached_tokens for seq in seqs)
                        proposals=[None]*world;dist.all_gather_object(proposals,packed)
                        assert all(value==proposals[0] for value in proposals)
                    else:
                        assert r.model.mtp.cache.lengths==before_lengths;recursive_faults.append(fault)
                    assert r.model.mtp.cache.batch_layout is None
                    for actual_buffer,reference in zip(buffers[:2],initial[:2]):assert torch.equal(actual_buffer,reference)
                restore()
                wide=[[correct[0]]+[0]*(mtp_steps-1),[(correct[1]+1)%128]*mtp_steps]
                wide_expected=[]
                for seq,draft in zip(seqs,wide):
                    out,_=r._speculative_transaction(r.verify_greedy,(seq,draft,mtp_steps+1,'serial'))
                    wide_expected.append(out)
                wide_state=[r.S_cache.clone(),r.conv_cache.clone()]
                restore();out,wide_stats=r._speculative_transaction(r.verify_greedy_batch,(seqs,wide,[mtp_steps+1]*2))
                assert out==wide_expected and all(s['verification_graph'] for s in wide_stats)
                for actual_buffer,reference in zip(buffers[:2],wide_state):assert torch.equal(actual_buffer,reference)
            result={'rank':rank,'passed':True,'accepted':[s['accepted'] for s in stats],
                'mtp_steps':mtp_steps,'recursive_rollback_faults':recursive_faults,
                'verification_fused_gdn':fused_gdn,
                'packed_draft':True,'packed_draft_rollback_faults':batch_faults,
                'rollback_faults':fault_results,'draft_heads':r.model.mtp.cache._keys.shape[2],
                'graphs':[{'sizes':list(key),'replays':graph.replays} for key,graph in r.verification_graphs.items()]}
            gathered=[None]*world;dist.all_gather_object(gathered,result)
            if rank==0:Path(output).write_text(json.dumps({'tp':world,'passed':True,'ranks':gathered,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('zjhvllm').rglob('*.py')}},indent=2)+'\n')
    finally:
        if 'r' in locals():
            # Exercise production shutdown; only the synthetic IPC handle is
            # stubbed because this runner calls distributed methods directly.
            r.shm=SimpleNamespace(close=lambda:None,unlink=lambda:None)
            r.exit()
        else:
            dist.destroy_process_group()



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tp',type=int,required=True)
    p.add_argument('--port',type=int,default=2460);p.add_argument('--output',required=True)
    p.add_argument('--fused-gdn',action='store_true')
    p.add_argument('--mtp-steps',type=int,default=1,choices=range(1,9));args=p.parse_args()
    mp.spawn(worker,args=(args.tp,args.port,args.output,args.fused_gdn,args.mtp_steps),nprocs=args.tp,join=True)
    report=json.loads(Path(args.output).read_text())
    report['processes_exited']=True
    report['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
