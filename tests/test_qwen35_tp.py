"""Checkpoint head partitioning, including replicated KV when TP>KV heads."""
from types import SimpleNamespace
import pytest
import torch
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.utils import tp


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def tiny_config():
    return SimpleNamespace(hidden_size=256, intermediate_size=512, hidden_act="silu",
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=8, linear_num_value_heads=24, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
        num_attention_heads=8, num_key_value_heads=2, head_dim=64,
        max_position_embeddings=512, attention_bias=True,
        rope_parameters={"rope_theta":10000000,"partial_rotary_factor":.25},
        vocab_size=128, tie_word_embeddings=False)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_load_packed_heads_vectors_and_replicated_kv(world):
    old = tp._TP_SIZE, tp._TP_RANK
    try:
        tp.set_tp(1,0)
        full = Qwen3_5ForCausalLM(tiny_config())
        source = dict(full.named_parameters())
        for rank in range(world):
            tp.set_tp(world,rank)
            model = Qwen3_5ForCausalLM(tiny_config())
            for name, param in model.named_parameters():
                getattr(param,"weight_loader",lambda p,w:p.data.copy_(w))(param, source[name])
            linear = model.model.layers[0].linear_attn
            assert linear.num_k_heads == 8//world and linear.num_v_heads == 24//world
            expected = torch.cat([x.chunk(world)[rank] for x in
                full.model.layers[0].linear_attn.in_proj_qkv.weight.split([1024,1024,3072])])
            assert torch.equal(linear.in_proj_qkv.weight, expected)
            conv = torch.cat([x.chunk(world)[rank] for x in
                full.model.layers[0].linear_attn.conv1d.weight.split([1024,1024,3072])])
            assert torch.equal(linear.conv1d.weight, conv)
            assert torch.equal(linear.A_log, full.model.layers[0].linear_attn.A_log.chunk(world)[rank])
            attention = model.model.layers[1].self_attn
            partitions = min(world,2)
            kv_rank = rank // max(1,world//2)
            assert torch.equal(attention.k_proj.weight,
                full.model.layers[1].self_attn.k_proj.weight.chunk(partitions)[kv_rank])
            assert torch.equal(attention.o_proj.bias, full.model.layers[1].self_attn.o_proj.bias)
            # Vocab is column/row sharded across ranks (was previously replicated).
            assert torch.equal(model.model.embed_tokens.weight,
                               full.model.embed_tokens.weight.chunk(world)[rank])
            assert torch.equal(model.lm_head.weight, full.lm_head.weight.chunk(world)[rank])
    finally:
        tp._TP_SIZE, tp._TP_RANK = old
