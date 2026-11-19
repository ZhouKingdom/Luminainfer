"""Verification Graph replay must update slots/pages/positions and exact layout."""
import pytest
import torch
from test_mtp_batch import tiny_runner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.sampling_params import SamplingParams
from zjhvllm.utils.context import reset_context

@pytest.fixture(scope='session',autouse=True)
def _init_dist():
    yield


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('fused_gdn',[False,True])
@torch.inference_mode()
def test_graph_dynamic_slots_page_crossings_and_ragged_layout(fused_gdn):
    torch.manual_seed(113)
    r=tiny_runner(torch.bfloat16);r.config.verification_cuda_graph=True
    r.config.verification_fused_gdn=fused_gdn
    buffers=[r.S_cache,r.conv_cache,r.kv_cache,r.target_hidden]
    initial=[b.clone() for b in buffers]
    def restore():
        for dest,source in zip(buffers,initial):dest.copy_(source)
        reset_context()
    graphs={}
    for iteration,layout in enumerate(([2,2],[2,1],[2,2])):
        slots=[3,1] if iteration==0 else [1,3]
        seqs=[]
        for b,slot in enumerate(slots):
            cached=255+iteration+b
            seq=Sequence([0]*cached+[7+iteration+b],SamplingParams(greedy=True,max_tokens=8,ignore_eos=True))
            seq.num_cached_tokens,seq.state_slot,seq.state_ready=cached,slot,True
            seq.block_table=([5,2] if b==0 else [4,1]) if iteration==0 else ([0,3] if b==0 else [2,5])
            seqs.append(seq)
        restore();correct=[]
        for seq in seqs:
            out,_=r.verify_greedy(seq,[],1,'serial');correct.append(out[0])
        drafts=[[correct[0]],[(correct[1]+1)%128] if layout[1]==2 else []]
        snapshots=[]
        for graph in (False,True):
            restore();r.config.verification_cuda_graph=graph
            original=r.model.forward
            if graph and tuple(layout) in graphs:
                def forbidden(*args):raise AssertionError('replay called eager model')
                r.model.forward=forbidden
            try:
                outputs,stats=r.verify_greedy_batch(seqs,drafts,[3,3])
                snapshot=[b.clone() for b in buffers]
            finally:r.model.forward=original
            if not graph:reference=(outputs,snapshot)
            else:
                assert outputs==reference[0]
                assert all(stat['verification_graph'] for stat in stats)
                for actual,expected in zip(snapshot,reference[1]):assert torch.equal(actual,expected)
                graphs[tuple(layout)]=r.verification_graph
    assert len(r.verification_graphs)==2
    key=('fused_gdn',2,2) if fused_gdn else (2,2)
    assert r.verification_graphs[key].replays==2
    # Bound layout cache: an unseen layout falls back, an existing one replays.
    r.verification_graphs.update({('unused',i):None for i in range(14)})
    restore();r.config.verification_cuda_graph=True
    result,stats=r.verify_greedy_batch(seqs,[[],[]],[1,1])
    assert not any(stat['verification_graph'] for stat in stats)
