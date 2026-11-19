"""P→D cache relocation and transactional resource ownership."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from zjhvllm.engine.handoff import capture_prefill,restore_prefill
from zjhvllm.engine.llm_engine import LLMEngine
from zjhvllm.engine.scheduler import Scheduler
from zjhvllm.engine.sequence import Sequence
from zjhvllm.sampling_params import SamplingParams


@pytest.fixture(scope='session',autouse=True)
def _init_dist():yield


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(Sequence,'block_size',4)
    config=SimpleNamespace(model='/same/checkpoint',hf_config=SimpleNamespace(to_dict=lambda:{'model_type':'qwen3'}),
        kv_cache_dtype='auto',max_model_len=32,max_num_seqs=4,max_num_batched_tokens=16,
        eos=-1,kvcache_block_size=4,num_kvcache_blocks=8,num_state_slots=0)
    def runner():
        return SimpleNamespace(config=config,has_linear_state=False,world_size=1,tp_size=1,block_size=4,
            kv_cache=torch.full((2,2,8,4,1,2),-999.,dtype=torch.float32))
    return config,runner


def packet_from(runner,length=7):
    seq=Sequence(list(range(length)),SamplingParams(max_tokens=4,seed=42,greedy=False))
    seq.block_table=[4,2][:seq.num_blocks]
    seq.num_cached_tokens=length
    seq.append_token(11)
    seq.first_token_time=2.;seq.arrival_time=1.;seq.last_ran_seed=42
    runner.kv_cache.copy_(torch.arange(runner.kv_cache.numel()).reshape_as(runner.kv_cache))
    return seq,capture_prefill(runner,seq)


def test_valid_tokens_only_and_physical_page_relocation(setup):
    config,make_runner=setup
    source=make_runner();seq,packet=packet_from(source)
    target=make_runner();s=Scheduler(config)
    incoming=Sequence(packet.prompt_token_ids,packet.sampling)
    s.admit_prefilled(incoming,packet.first_token,lambda seq:restore_prefill(target,seq,packet))
    assert incoming.block_table==[0,1] and seq.block_table==[4,2]
    restored=target.kv_cache.flatten(2,3)[:,:,:8]
    torch.testing.assert_close(restored[:,:,:7],packet.kv,atol=0,rtol=0)
    assert torch.all(restored[:,:,7]==-999)  # never copy the source's invalid page tail
    assert incoming.num_cached_tokens==7 and incoming.last_token==11


def test_exact_page_boundary_defers_pending_token_page_to_decode(setup):
    config,make_runner=setup
    _,packet=packet_from(make_runner(),length=4)
    target=make_runner();s=Scheduler(config)
    seq=Sequence(packet.prompt_token_ids,packet.sampling)
    s.admit_prefilled(seq,packet.first_token,lambda seq:restore_prefill(target,seq,packet))
    assert len(seq.block_table)==1
    batch,prefill=s.schedule()
    assert not prefill and batch==[seq] and len(seq.block_table)==2


def test_failed_import_restores_capacity_and_does_not_publish(setup):
    config,_=setup;s=Scheduler(config)
    seq=Sequence([1]*7,SamplingParams(max_tokens=4))
    def fail(_):raise RuntimeError('injected copy failure')
    with pytest.raises(RuntimeError,match='injected'):
        s.admit_prefilled(seq,11,fail)
    assert not s.running and not s.waiting and not seq.block_table
    assert len(s.block_manager.free_block_ids)==8 and not s.block_manager.used_block_ids


def test_insufficient_capacity_rejected_before_restore(setup):
    config,_=setup;config.num_kvcache_blocks=1;s=Scheduler(config)
    seq=Sequence([1]*4,SamplingParams(max_tokens=4))
    with pytest.raises(RuntimeError,match='capacity'):
        s.admit_prefilled(seq,11,lambda _:pytest.fail('copy should not start'))
    assert len(s.block_manager.free_block_ids)==1 and not s.running


@pytest.mark.parametrize('kind',['signature','shape','dtype','version','hybrid','fp8','sharded'])
def test_incompatible_payload_rejected(setup,kind):
    _,make_runner=setup;source=make_runner();_,packet=packet_from(source)
    target=make_runner()
    if kind=='signature':packet=replace(packet,signature='wrong')
    if kind=='shape':packet=replace(packet,kv=packet.kv[:,:,:6])
    if kind=='dtype':packet=replace(packet,kv=packet.kv.double())
    if kind=='version':packet=replace(packet,version=2)
    if kind=='hybrid':target.has_linear_state=True
    if kind=='fp8':target.config.kv_cache_dtype='fp8'
    if kind=='sharded':target.world_size=2
    with pytest.raises((ValueError,NotImplementedError)):packet.validate(target)


def test_receipt_retry_keeps_one_request_and_preserves_sampling(setup):
    config,make_runner=setup;_,packet=packet_from(make_runner())
    engine=object.__new__(LLMEngine)
    engine.model_runner=make_runner();engine.scheduler=Scheduler(config)
    engine.seq_map={};engine._accepted_prefills={};engine._pending_exports={}
    first=engine.accept_prefill(packet);second=engine.accept_prefill(packet)
    assert first==second and len(engine.scheduler.running)==1
    seq=engine.seq_map[first]
    assert seq.last_ran_seed==42 and seq.seed==42 and seq.num_completion_tokens==1
    assert seq.arrival_time==1. and seq.token_times==[2.]
    # Avoid LLMEngine.exit expecting a genuine ModelRunner in fake-engine tests.
    del engine.model_runner


def test_source_holds_pages_until_ack_and_abort_resumes(setup):
    config,make_runner=setup;engine=object.__new__(LLMEngine)
    engine.model_runner=make_runner();engine.scheduler=Scheduler(config)
    engine.seq_map={};engine._pending_exports={};engine._accepted_prefills={}
    seq=Sequence([1]*7,SamplingParams(max_tokens=4))
    seq.arrival_time=1.;seq.first_token_time=2.;seq.token_times=[2.]
    engine.scheduler.admit_prefilled(seq,11,lambda _:None)
    engine.seq_map[seq.seq_id]=seq
    free=len(engine.scheduler.block_manager.free_block_ids)
    packet=engine.export_prefill(seq.seq_id)
    assert len(engine.scheduler.block_manager.free_block_ids)==free
    with pytest.raises(RuntimeError,match='paused'):engine.step()
    engine.finish_export(packet.transfer_id,accepted=False)
    assert seq in engine.scheduler.running and seq.block_table
    packet=engine.export_prefill(seq.seq_id)
    engine.finish_export(packet.transfer_id,accepted=True)
    assert not engine.seq_map and not engine.scheduler.running
    assert len(engine.scheduler.block_manager.free_block_ids)==8
    del engine.model_runner
