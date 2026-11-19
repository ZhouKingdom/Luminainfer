"""Speculation IPC must retain the complete decode frontier and sampling policy."""
import pickle
from types import SimpleNamespace
import pytest
from zjhvllm.engine.model_runner import _spec_packet, ModelRunner
from zjhvllm.engine.sequence import Sequence, SequenceStatus
from zjhvllm.sampling_params import SamplingParams

@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    yield


def test_full_speculative_ipc_preserves_decode_metadata():
    seq=Sequence([3,5,7],SamplingParams(greedy=True,ignore_eos=True,max_tokens=19))
    seq.append_token(11)
    seq.is_prefill=False
    seq.status=SequenceStatus.RUNNING
    seq.state_slot,seq.state_ready,seq.num_cached_tokens=3,True,3
    seq.block_table=[9,2]
    for args in ([seq,[13],7,'serial'], [[seq],[[13]],[7]]):
        restored=pickle.loads(pickle.dumps([_spec_packet(arg) for arg in args]))
        actual=restored[0][0] if isinstance(restored[0],list) else restored[0]
        assert actual.__dict__==seq.__dict__
    assert pickle.loads(pickle.dumps(seq)).token_ids==[]


def test_shared_memory_overflow_rejected_before_notifying_workers():
    runner=ModelRunner.__new__(ModelRunner)
    runner.world_size,runner.rank=2,0
    runner.shm=SimpleNamespace(buf=bytearray(32))
    class Event:
        def set(self):
            raise AssertionError('oversized command must not wake workers')
    runner.event=[Event()]
    with pytest.raises(ValueError,match='capacity'):
        runner.write_shm('verify_greedy',Sequence([3]*100),[7],2,'serial')


@pytest.mark.parametrize('world',[1,2,4,8])
def test_mtp_config_accepts_tensor_parallel(tmp_path,monkeypatch,world):
    from zjhvllm.config import Config, AutoConfig
    cfg=SimpleNamespace(model_type='qwen3_5_text',max_position_embeddings=4096)
    monkeypatch.setattr(AutoConfig,'from_pretrained',lambda _:cfg)
    assert Config(str(tmp_path),enable_mtp=True,max_num_seqs=4,tensor_parallel_size=world).tensor_parallel_size==world


@pytest.mark.parametrize('world',[2,4,8])
def test_mtp_private_kv_follows_local_and_replicated_heads(world):
    import torch
    from zjhvllm.utils import tp
    from zjhvllm.models.qwen3_5_mtp import Qwen3_5MTP
    from test_qwen35_tp import tiny_config
    old=tp._TP_SIZE,tp._TP_RANK
    try:
        cfg=tiny_config()
        cfg.mtp_num_hidden_layers,cfg.mtp_use_dedicated_embeddings,cfg.dtype=1,False,torch.float32
        tp.set_tp(1,0);full=Qwen3_5MTP(cfg,12,slots=3)
        source=dict(full.named_parameters())
        for rank in range(world):
            tp.set_tp(world,rank);local=Qwen3_5MTP(cfg,12,slots=3)
            for name,param in local.named_parameters():
                with torch.no_grad():
                    getattr(param,'weight_loader',lambda p,w:p.copy_(w))(param,source[name])
            assert local.cache._keys.shape==(3,12,max(1,2//world),64)
            attention=local.layers[0].self_attn
            kv_rank=rank//max(1,world//2)
            assert torch.equal(attention.k_proj.weight,full.layers[0].self_attn.k_proj.weight.chunk(min(world,2))[kv_rank])
    finally:tp._TP_SIZE,tp._TP_RANK=old


@pytest.mark.parametrize('bias',[False,True])
def test_verification_row_projection_uses_decode_collective_shapes(monkeypatch,bias):
    import torch
    from zjhvllm.models.qwen3_5 import Qwen3_5ParallelLinear
    from zjhvllm.utils import tp
    from zjhvllm.utils.context import get_context,reset_context
    old=tp._TP_SIZE,tp._TP_RANK
    calls=[]
    def reduce(tensor):
        calls.append(tuple(tensor.shape))
        tensor.mul_(2)
    monkeypatch.setattr('zjhvllm.models.qwen3_5.dist.all_reduce',reduce)
    try:
        tp.set_tp(2,0)
        layer=Qwen3_5ParallelLinear(16,32,bias=bias)
        x=torch.randn(4,8)
        get_context().decode_equivalent=True
        y=layer(x)
        assert calls==[(1,32)]*4
        calls.clear()
        expected=torch.cat([layer(row) for row in x.split(1)])
        assert torch.equal(y,expected)
    finally:
        tp._TP_SIZE,tp._TP_RANK=old
        reset_context()
