"""Bound active prefill stalls without changing decode execution semantics."""
from types import SimpleNamespace
import pytest
from zjhvllm.engine.scheduler import Scheduler
from zjhvllm.engine.sequence import Sequence
from zjhvllm.sampling_params import SamplingParams


@pytest.fixture(scope='session',autouse=True)
def _init_dist():
    yield


def scheduler(policy='bounded', cap=256):
    return Scheduler(SimpleNamespace(max_num_seqs=8,max_num_batched_tokens=1024,eos=-1,
        kvcache_block_size=256,num_kvcache_blocks=64,num_state_slots=8,
        scheduling_policy=policy,active_prefill_tokens=cap))


def request(length,tokens=16):
    return Sequence(list(range(length)),SamplingParams(greedy=True,max_tokens=tokens,ignore_eos=True))


def test_cold_prefill_keeps_full_token_budget():
    s=scheduler()
    items=[request(128) for _ in range(8)]
    for item in items:s.add(item)
    batch,prefill=s.schedule()
    assert prefill and batch==items
    assert sum(x.num_scheduled_tokens for x in batch)==1024


@pytest.mark.parametrize('policy,expected',[('alternating',1024),('bounded',256)])
def test_only_active_prefill_is_capped(policy,expected):
    s=scheduler(policy=policy)
    running=request(128)
    s.add(running)
    batch,prefill=s.schedule();s.postprocess(batch,[7],prefill)
    waiting=request(1024)
    s.add(waiting)
    batch,prefill=s.schedule()
    assert not prefill and batch==[running]
    s.postprocess(batch,[7],prefill)
    batch,prefill=s.schedule()
    assert prefill and batch==[waiting]
    assert waiting.num_scheduled_tokens==expected
    assert running.num_scheduled_tokens==0


def test_both_queues_make_progress_and_release_resources():
    s=scheduler(cap=128)
    first=request(128,tokens=24)
    second=request(1024,tokens=2)
    s.add(first)
    batch,prefill=s.schedule();s.postprocess(batch,[7],prefill)
    s.add(second)
    seen_prefill=0
    steps=0
    while not s.is_finished():
        batch,prefill=s.schedule()
        if prefill and second in batch:
            seen_prefill+=1
            if not second.num_completion_tokens:
                assert second.num_scheduled_tokens<=128
        s.postprocess(batch,[7]*len(batch),prefill)
        steps+=1
        assert steps<50
    assert seen_prefill==8
    assert first.num_completion_tokens==24 and second.num_completion_tokens==2
    assert len(s.free_state_slots)==8
    assert not s.block_manager.used_block_ids


@pytest.mark.skipif(not __import__('torch').cuda.is_available(),reason='CUDA required')
def test_host_metadata_keeps_prefill_output_and_state_bitwise_equal():
    import torch
    from zjhvllm.models.qwen3_5 import GatedDeltaNet
    from zjhvllm.utils.context import set_context,reset_context
    torch.manual_seed(33)
    with torch.inference_mode():
        layer=GatedDeltaNet(256,2,4,128,128,4,1e-6).cuda().bfloat16()
        layer.norm.weight.fill_(1)
        initial=torch.randn(4,4,128,128,device='cuda')*.01
        conv=torch.randn(4,layer.conv_dim,3,device='cuda',dtype=torch.bfloat16)*.01
        layer.S_cache=initial.clone();layer.conv_cache=conv.clone()
        cu=[0,65,193,226];slots=[3,0,2]
        x=torch.randn(226,256,device='cuda',dtype=torch.bfloat16)*.1
        cuda_cu=torch.tensor(cu,device='cuda',dtype=torch.int32)
        cuda_slots=torch.tensor(slots,device='cuda',dtype=torch.int32)
        try:
            set_context(True,cu_seqlens_q=cuda_cu,state_slots=cuda_slots)
            expected=layer(x)
            expected_s=layer.S_cache.clone();expected_c=layer.conv_cache.clone()
            layer.S_cache.copy_(initial);layer.conv_cache.copy_(conv)
            set_context(True,cu_seqlens_q=cuda_cu,state_slots=cuda_slots,cu_seqlens_q_cpu=cu,state_slots_cpu=slots)
            actual=layer(x)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            torch.testing.assert_close(layer.S_cache,expected_s,atol=0,rtol=0)
            torch.testing.assert_close(layer.conv_cache,expected_c,atol=0,rtol=0)
        finally:
            reset_context()


def test_adaptive_defers_then_services_oldest_without_starvation(monkeypatch):
    from zjhvllm.engine.adaptive_prefill import AdaptivePrefill
    s=scheduler(policy='adaptive')
    now=[10.]
    monkeypatch.setattr('zjhvllm.engine.scheduler.perf_counter',lambda:now[0])
    s.adaptive=AdaptivePrefill(.1,2.,128,minimum_running=1)
    s.adaptive.observe(True,1024,1,0,.8)
    s.adaptive.observe(False,1,1,0,.06)
    first=request(128,tokens=100)
    s.add(first)
    batch,prefill=s.schedule();s.postprocess(batch,[7],prefill)
    first.first_token_time=10.
    second=request(1024,tokens=2);s.add(second)
    batch,prefill=s.schedule();assert not prefill
    s.postprocess(batch,[7],prefill)
    now[0]=10.06
    batch,prefill=s.schedule()
    assert not prefill
    assert s.adaptive.last_decision['reason']=='earn_decode_credit'
    s.postprocess(batch,[7],prefill)
    now[0]=11.5
    batch,prefill=s.schedule()
    assert prefill and batch==[second]
    assert second.num_scheduled_tokens==1024
    assert s.adaptive.last_decision['reason']=='ttft_guard'


def test_adaptive_uses_earned_credit_and_observed_cost(monkeypatch):
    from zjhvllm.engine.adaptive_prefill import AdaptivePrefill
    s=scheduler(policy='adaptive')
    s.adaptive=AdaptivePrefill(.1,2.,128,minimum_running=1)
    s.adaptive.observe(True,1024,1,0,.8)
    s.adaptive.observe(False,1,1,0,.06)
    first=request(128,tokens=100);s.add(first)
    batch,prefill=s.schedule();s.postprocess(batch,[7],prefill)
    first.first_token_time=10.
    for _ in range(10):first.append_token(7)
    monkeypatch.setattr('zjhvllm.engine.scheduler.perf_counter',lambda:10.8)
    second=request(1024);s.add(second)
    s._prefill_turn=True
    batch,prefill=s.schedule()
    assert prefill and batch==[second]
    assert second.num_scheduled_tokens==256
    assert s.adaptive.last_decision['reason']=='credit_fit'


def test_adaptive_fills_small_decode_batch_without_waiting(monkeypatch):
    s=scheduler(policy='adaptive')
    first=request(128);s.add(first)
    batch,prefill=s.schedule();s.postprocess(batch,[7],prefill)
    first.first_token_time=10.
    monkeypatch.setattr('zjhvllm.engine.scheduler.perf_counter',lambda:10.01)
    s.add(request(1024));s._prefill_turn=True
    batch,prefill=s.schedule()
    assert prefill and sum(x.num_scheduled_tokens for x in batch)==1024
    assert s.adaptive.last_decision['reason']=='fill_decode_batch'


def test_adaptive_rejects_expensive_fragmentation():
    from zjhvllm.engine.adaptive_prefill import AdaptivePrefill
    c=AdaptivePrefill(.1,2.,128,minimum_running=1)
    c.observe(True,1024,1,0,.7);c.observe(True,128,1,0,.35)
    c.observe(False,1,1,0,.06)
    active=request(128);active.first_token_time=10.
    for _ in range(6):active.append_token(7)
    waiting=request(1024);waiting.arrival_time=10.14
    assert c.choose([waiting],[active],1024,10.14)==0
    assert c.last_decision['reason']=='earn_decode_credit'
