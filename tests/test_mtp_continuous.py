"""Real scheduler admission, streaming, EOS and capacity fallback under MTP."""
from types import SimpleNamespace
import pytest
from zjhvllm.engine.llm_engine import LLMEngine
from zjhvllm.engine.scheduler import Scheduler
from zjhvllm.engine.sequence import Sequence
from zjhvllm.models.qwen3_5_mtp import MTPAttention
from zjhvllm.sampling_params import SamplingParams
import torch

@pytest.fixture(scope='session',autouse=True)
def _init_dist():
    yield


def engine_stub(pages=12, capacity=2):
    cfg=SimpleNamespace(max_num_seqs=capacity,max_num_batched_tokens=64,eos=5,
        kvcache_block_size=256,num_kvcache_blocks=pages,num_state_slots=capacity,
        max_model_len=1024,hf_config=SimpleNamespace(vocab_size=128),verification_state_commit=True)
    e=LLMEngine.__new__(LLMEngine)
    e.scheduler,e.seq_map,e._pending_exports=Scheduler(cfg),{},{}
    e.tokenizer=SimpleNamespace(decode=str)
    cache=MTPAttention(1024,1,1,torch.float32,slots=capacity)
    r=SimpleNamespace(config=cfg,world_size=1,model=SimpleNamespace(mtp=SimpleNamespace(cache=cache)))
    e.model_runner=r
    e.trace=[];e.resets=[];e.fallbacks=0;e.fail_verify=False;e.fault_verify=False
    def run_scheduled(seqs,is_prefill):
        stream=[]
        for seq in seqs:
            if not seq.state_ready:
                e.resets.append((seq.seq_id,seq.state_slot,cache.get_length(seq.state_slot)))
                cache.set_length(seq.state_slot,0);seq.state_ready=True
        stream=e.scheduler.postprocess(seqs,[(seq.last_token+1)%128 for seq in seqs],is_prefill)
        if not is_prefill:e.fallbacks+=1
        return [(s.seq_id,s.completion_token_ids) for s in seqs if s.is_finished],len(stream),stream
    e._run_scheduled=run_scheduled
    def call(name,*args):
        if name=='draft_mtp':
            seq=args[0]
            assert cache.get_length(seq.state_slot)<seq.num_cached_tokens
            cache.set_length(seq.state_slot,seq.num_cached_tokens)
            return [((seq.last_token+1) if seq.prompt_token_ids[-1]%2 else 99)%128],{'draft_source':'mtp'}
        assert name=='verify_greedy_batch'
        if e.fail_verify:
            e.fail_verify=False
            raise MemoryError('injected workspace pressure')
        if e.fault_verify:raise RuntimeError('injected target fault')
        seqs,drafts,budgets=args;outputs=[];stats=[]
        for seq,draft,budget in zip(seqs,drafts,budgets):
            next_token=(seq.last_token+1)%128
            accepted=int(bool(draft) and draft[0]==next_token)
            out=[next_token]
            if accepted and budget>1 and (seq.ignore_eos or next_token!=cfg.eos):out.append((next_token+1)%128)
            outputs.append(out);stats.append({'accepted':accepted,'proposed':len(draft)})
        e.trace.append([s.seq_id for s in seqs])
        return outputs,stats
    r.call=call
    return e


@pytest.mark.parametrize('continuous',[False,True])
def test_heterogeneous_budgets_eos_and_refill(continuous):
    e=engine_stub()
    prompts=[[11],[20],[31],[40],[2]]
    params=[SamplingParams(greedy=True,max_tokens=n,ignore_eos=ignore)
            for n,ignore in [(17,True),(2,True),(5,True),(1,True),(10,False)]]
    result=e.generate_speculative_batch(prompts,params,continuous=continuous)
    assert [r['token_ids'] for r in result]==[list(range(12,29)),[21,22],list(range(32,37)),[41],[3,4,5]]
    seqs=list(e.seq_map.values())
    # Third request joins while the first remains live; fixed cohorts wait.
    assert any(seqs[0].seq_id in group and seqs[2].seq_id in group for group in e.trace)==continuous
    assert [seq.seq_id for seq in seqs]==[seq.seq_id for seq in e.seq_map.values()]
    assert all(seq.is_finished for seq in seqs)
    assert len(e.scheduler.free_state_slots)==2
    assert len(e.scheduler.block_manager.free_block_ids)==12
    assert len(e.resets)==5
    assert any(old_length>0 for _,_,old_length in e.resets[2:])


def test_online_arrival_and_streamed_tokens_match_published_history():
    e=engine_stub()
    first=e.add_request([11],SamplingParams(greedy=True,max_tokens=12,ignore_eos=True))
    collected={first:[]}
    e.step_speculative();collected[first]=list(e.seq_map[first].completion_token_ids)
    e.step_speculative();collected[first]=list(e.seq_map[first].completion_token_ids)
    second=e.add_request([31],SamplingParams(greedy=True,max_tokens=3,ignore_eos=True));collected[second]=[]
    while not e.is_finished():
        _,_,stream=e.step_speculative()
        for seq_id,token in stream:collected[seq_id].append(token)
    assert all(collected[seq_id]==e.seq_map[seq_id].completion_token_ids for seq_id in collected)
    assert e.seq_map[second].first_token_time<e.seq_map[first].finished_time
    assert e.step_speculative()==([],0,[])


def test_workspace_pressure_falls_back_without_rescheduling_reserved_batch():
    e=engine_stub();e.fail_verify=True
    result=e.generate_speculative_batch([[11],[31]],SamplingParams(greedy=True,max_tokens=7,ignore_eos=True))
    assert [r['token_ids'] for r in result]==[list(range(12,19)),list(range(32,39))]
    assert e.fallbacks==1
    assert len(e.scheduler.block_manager.free_block_ids)==12


def test_target_fault_restores_draft_boundaries_and_ordinary_step_can_continue():
    e=engine_stub();e.fault_verify=True
    ids=[e.add_request([token],SamplingParams(greedy=True,max_tokens=7,ignore_eos=True)) for token in (11,31)]
    e.step_speculative()
    with pytest.raises(RuntimeError,match='injected'):e.step_speculative()
    assert e.model_runner.model.mtp.cache.lengths==[0,0]
    assert all(e.seq_map[i].num_cached_tokens+1==len(e.seq_map[i]) for i in ids)
    while not e.is_finished():e.step()
    assert [e.seq_map[i].completion_token_ids for i in ids]==[list(range(12,19)),list(range(32,39))]


def test_page_pressure_uses_decode_then_preempts_and_recomputes():
    e=engine_stub(pages=2)
    events=[]
    original=e.scheduler.preempt
    def preempt(seq):
        events.append(seq.seq_id);original(seq)
    e.scheduler.preempt=preempt
    result=e.generate_speculative_batch([[11]*255,[31]*255],SamplingParams(greedy=True,max_tokens=5,ignore_eos=True))
    assert [r['token_ids'] for r in result]==[list(range(12,17)),list(range(32,37))]
    assert e.fallbacks>0 and events
    assert len(e.scheduler.free_state_slots)==2 and len(e.scheduler.block_manager.free_block_ids)==2
