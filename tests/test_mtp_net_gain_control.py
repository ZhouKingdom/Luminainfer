"""Native no-draft Graph baseline must preserve the MTP target arithmetic."""
import pytest
import torch
from test_mtp_batch import tiny_runner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.sampling_params import SamplingParams
from zjhvllm.utils.context import reset_context
from benchmarks.bench_mtp_net_gain import prepare_native_control, checked_run
from zjhvllm.engine.model_runner import ModelRunner

@pytest.fixture(scope='session',autouse=True)
def _init_dist():
    yield


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('count',[1,2,3,4])
@torch.inference_mode()
def test_native_control_dynamic_slots_and_all_buckets_match_eager(count,monkeypatch):
    monkeypatch.setattr(ModelRunner,"run_model",checked_run)
    torch.manual_seed(919)
    r=tiny_runner(torch.bfloat16);r.config.max_num_seqs=4;r.config.num_state_slots=5
    r.rank=0;r.net_counts={}
    prepare_native_control(r)
    assert set(r.graphs)=={1,2,3,4}
    buffers=[r.S_cache,r.conv_cache,r.kv_cache,r.target_hidden]
    initial=[b.clone() for b in buffers]
    seqs=[]
    for cached,slot,pages in [(255,3,[5,2]),(256,1,[4,1]),(257,4,[0,3]),(254,0,[2])]:
        seq=Sequence([0]*cached+[7],SamplingParams(greedy=True,ignore_eos=True))
        seq.num_cached_tokens,seq.state_slot,seq.state_ready=cached,slot,True
        seq.block_table=pages;seqs.append(seq)
    for eager in (True,False):
        for dest,source in zip(buffers,initial):dest.copy_(source)
        r.enforce_eager=eager
        ids,positions=r.prepare_decode(seqs[:count]);logits=r.run_model(ids,positions,False)
        snapshot=[b.clone() for b in buffers]+[logits.clone()]
        reset_context()
        if eager:reference=snapshot
        else:
            for actual,expected in zip(snapshot,reference):assert torch.equal(actual,expected)
