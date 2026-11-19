"""Recursive MTP chains must not publish approximate hidden-state KV."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from test_qwen35_mtp import initialized_mtp, config
from test_mtp_continuous import engine_stub
from zjhvllm.models.qwen3_5_mtp import Qwen3_5MTP
from zjhvllm.engine.model_runner import ModelRunner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.engine.speculative import greedy_prefix
from zjhvllm.sampling_params import SamplingParams


@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    yield


def recursive_runner():
    torch.manual_seed(1201)
    runner=ModelRunner.__new__(ModelRunner)
    mtp=Qwen3_5MTP(config(),32,slots=3)
    for parameter in mtp.parameters():
        parameter.data.copy_(torch.randn_like(parameter)*.02)
    mtp.cache._keys.normal_(0,.01);mtp.cache._values.normal_(0,.01)
    mtp.cache.lengths[:]=[0,0,1]
    runner.target_hidden=torch.randn(3,32,16)
    runner.model=SimpleNamespace(mtp=mtp,model=SimpleNamespace(embed_tokens=torch.nn.Embedding(64,16)),
                                 lm_head=torch.nn.Linear(16,64,bias=False))
    seqs=[Sequence([10,11,12,13,14],SamplingParams(greedy=True,max_tokens=16)),
          Sequence([20,21,22],SamplingParams(greedy=True,max_tokens=16))]
    for seq,slot,cached in zip(seqs,[2,0],[4,2]):
        seq.state_slot,seq.state_ready,seq.num_cached_tokens=slot,True,cached
    return runner,seqs


@torch.inference_mode()
def test_recursive_first_token_matches_mtp1_and_suffix_is_unpublished():
    runner,seqs=recursive_runner(); reference=deepcopy(runner)
    expected=reference.draft_mtp_batch(seqs)
    target=runner.target_hidden.clone()
    actual=runner.draft_mtp_k_batch(seqs,[4,2])
    assert [tokens[0] for tokens,_ in actual]==[tokens[0] for tokens,_ in expected]
    assert [len(tokens) for tokens,_ in actual]==[4,2]
    assert runner.model.mtp.cache.lengths==reference.model.mtp.cache.lengths==[2,0,4]
    assert torch.equal(target,runner.target_hidden)
    # Only committed KV is retained; future bytes may exist but stay hidden.
    for slot,end in [(2,4),(0,2)]:
        torch.testing.assert_close(runner.model.mtp.cache._keys[slot,:end],reference.model.mtp.cache._keys[slot,:end])
    # Simulate a partial acceptance, then rebuild with true target hiddens.
    for seq,(tokens,_) in zip(seqs,actual):
        seq.append_token(tokens[0]);seq.append_token(7);seq.num_cached_tokens+=2
    expected=reference.draft_mtp_batch(seqs)
    rebuilt=runner.draft_mtp_k_batch(seqs,[1,1])
    assert [t for t,_ in rebuilt]==[t for t,_ in expected]
    for slot,end in [(2,6),(0,4)]:
        torch.testing.assert_close(runner.model.mtp.cache._keys[slot,:end],reference.model.mtp.cache._keys[slot,:end])


@torch.inference_mode()
def test_recursive_conditioning_uses_previous_draft_hidden(monkeypatch):
    runner,seqs=recursive_runner()
    original=runner.model.mtp.forward_batch
    trace=[]
    def forward(embeddings,hidden,slots,sizes):
        output=original(embeddings,hidden,slots,sizes)
        trace.append((hidden.clone(),output.clone(),slots,sizes))
        return output
    monkeypatch.setattr(runner.model.mtp,'forward_batch',forward)
    runner.draft_mtp_k_batch(seqs,[4,2])
    assert len(trace)==4
    assert trace[1][2:]==([2,0],[1,1])
    torch.testing.assert_close(trace[1][0],trace[0][1][torch.tensor([2,4])],rtol=0,atol=0)
    assert trace[2][2:]==([2],[1])
    torch.testing.assert_close(trace[2][0],trace[1][1][:1],rtol=0,atol=0)
    torch.testing.assert_close(trace[3][0],trace[2][1],rtol=0,atol=0)


@torch.inference_mode()
def test_recursive_failure_rolls_back_every_committed_extension(monkeypatch):
    runner,seqs=recursive_runner();cache=runner.model.mtp.cache;before=list(cache.lengths)
    original=runner.model.lm_head.forward;calls=[]
    def head(hidden):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('injected recursive head failure')
        return original(hidden)
    monkeypatch.setattr(runner.model.lm_head,'forward',head)
    with pytest.raises(RuntimeError,match='recursive head failure'):
        runner.draft_mtp_k_batch(seqs,[4,2])
    assert cache.lengths==before and cache.batch_layout is None


@pytest.mark.parametrize('continuous',[False,True])
def test_recursive_engine_heterogeneous_budgets_eos_and_refill(continuous):
    engine=engine_stub();engine.model_runner.config.mtp_num_steps=4
    original=engine.model_runner.call;seen=[]
    def call(name,*args):
        if name=='draft_mtp_k_batch':
            seqs,limits=args;seen.append(limits)
            result=[]
            for seq,limit in zip(seqs,limits):
                assert limit<seq.max_tokens-seq.num_completion_tokens
                engine.model_runner.model.mtp.cache.set_length(seq.state_slot,seq.num_cached_tokens)
                chain=[(seq.last_token+i+1)%128 for i in range(limit)]
                if seq.prompt_token_ids[-1]%2==0:chain[0]=99
                elif seq.prompt_token_ids[-1]==11 and len(chain)>1:chain[1]=99
                result.append((chain,{'draft_steps':limit}))
            return result
        if name=='verify_greedy_batch':
            seqs,drafts,budgets=args;outputs=[];stats=[]
            for seq,draft,budget in zip(seqs,drafts,budgets):
                target=[(seq.last_token+i+1)%128 for i in range(len(draft)+1)]
                output,accepted=greedy_prefix(draft,target,budget,None if seq.ignore_eos else 5)
                outputs.append(output);stats.append({'accepted':accepted,'proposed':len(draft)})
            return outputs,stats
        return original(name,*args)
    engine.model_runner.call=call
    params=[SamplingParams(greedy=True,max_tokens=n,ignore_eos=ignore) for n,ignore in [(13,True),(2,True),(9,True),(1,True),(10,False)]]
    outputs=engine.generate_speculative_batch([[11],[20],[31],[40],[2]],params,continuous=continuous)
    assert [o['token_ids'] for o in outputs]==[list(range(12,25)),[21,22],list(range(32,41)),[41],[3,4,5]]
    assert any(4 in limits for limits in seen)
    assert len(engine.scheduler.free_state_slots)==2


def test_recursive_ipc_retains_limits_and_full_sequences():
    import pickle
    runner=ModelRunner.__new__(ModelRunner);runner.world_size,runner.rank=2,0
    runner.shm=SimpleNamespace(buf=bytearray(8192));runner.event=[SimpleNamespace(set=lambda:None)]
    _,seqs=recursive_runner()
    runner.write_shm('draft_mtp_k_batch',seqs,[4,2])
    size=int.from_bytes(runner.shm.buf[:4],'little')
    name,restored,limits=pickle.loads(runner.shm.buf[4:4+size])
    assert name=='draft_mtp_k_batch' and limits==[4,2]
    assert [s.__dict__ for s in restored]==[s.__dict__ for s in seqs]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('steps',[2,4,8])
@torch.inference_mode()
def test_wide_graph_commits_independent_prefixes_like_serial(steps):
    from copy import copy
    from test_mtp_batch import tiny_runner
    from zjhvllm.utils.context import reset_context
    torch.manual_seed(1209)
    r=tiny_runner(torch.bfloat16);r.config.verification_fused_gdn=True
    buffers=[r.S_cache,r.conv_cache,r.kv_cache,r.target_hidden]
    initial=[x.clone() for x in buffers]
    def restore():
        for dst,src in zip(buffers,initial):dst.copy_(src)
        reset_context()
    seqs=[]
    for cached,slot,pages in [(254,3,[5,2]),(257,1,[4,0])]:
        seq=Sequence([0]*cached+[7],SamplingParams(greedy=True,max_tokens=16,ignore_eos=True))
        seq.num_cached_tokens,seq.state_slot,seq.state_ready=cached,slot,True
        seq.block_table=pages;seqs.append(seq)
    chains=[]
    for seq in seqs:
        restore();trial=copy(seq);trial.token_ids=list(seq.token_ids);chain=[]
        for _ in range(steps):
            trial.block_table=seq.block_table[:(len(trial)+255)//256]
            ids,pos=r.prepare_decode([trial]);token=int(r.run_model(ids,pos,False).argmax(-1))
            chain.append(token);trial.num_cached_tokens+=1;trial.append_token(token)
        chains.append(chain)
    drafts=[chains[0],list(chains[1])]
    drafts[1][steps//2]=(drafts[1][steps//2]+1)%128
    budgets=[steps+1,steps+1]
    restore();expected=[]
    for seq,draft,budget in zip(seqs,drafts,budgets):
        output,_=r.verify_greedy(seq,draft,budget,'serial');expected.append(output)
    reference=[r.S_cache.clone(),r.conv_cache.clone()]
    try:
        for graph in (False,True):
            restore();r.config.verification_cuda_graph=graph
            actual,stats=r.verify_greedy_batch(seqs,drafts,budgets)
            assert actual==expected and len(actual[0])==steps+1 and len(actual[1])==steps//2+1
            assert all(s['verification_graph']==graph for s in stats)
            assert all(torch.equal(a,b) for a,b in zip(buffers[:2],reference))
    finally:reset_context()


def test_k_workspace_configuration_fails_before_model_runner(tmp_path,monkeypatch):
    from zjhvllm.config import Config,AutoConfig
    cfg=SimpleNamespace(model_type='qwen3_5_text',max_position_embeddings=4096,
        layer_types=['linear_attention']*48,linear_num_key_heads=16,linear_num_value_heads=48,
        linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4,dtype=torch.bfloat16)
    monkeypatch.setattr(AutoConfig,'from_pretrained',lambda _:cfg)
    with pytest.raises(ValueError,match='verification_state_budget_mb'):
        Config(str(tmp_path),enable_mtp=True,max_num_seqs=4,mtp_num_steps=4,
               tensor_parallel_size=1,verification_state_budget_mb=512)
    assert Config(str(tmp_path),enable_mtp=True,max_num_seqs=4,mtp_num_steps=4,
                  tensor_parallel_size=1,verification_state_budget_mb=1024).mtp_num_steps==4
    with pytest.raises(ValueError,match='mtp_num_steps'):
        Config(str(tmp_path),enable_mtp=True,max_num_seqs=4,mtp_num_steps=0)
