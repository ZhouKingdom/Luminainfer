"""GPU checks against the explicit write-before-read recurrence."""
import pytest
import torch
from zjhvllm.models.qwen3_5 import _chunk_gated_delta_rule


@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    yield


def reference(q, k, v, g, beta, state):
    outputs = []
    for t in range(q.shape[1]):
        decay = g[:, t].exp()
        prediction = torch.einsum('hd,hdv->hv', k[:, t], state)
        delta = v[:, t] - decay[:, None] * prediction
        state = decay[:, None, None] * state + (beta[:, t, None] * k[:, t])[:, :, None] * delta[:, None, :]
        outputs.append(torch.einsum('hd,hdv->hv', q[:, t], state))
    return torch.stack(outputs), state


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('out_of_place', [False, True])
@torch.inference_mode()
def test_decode_state_offsets_above_signed_int32(out_of_place):
    from zjhvllm.layers.gated_delta import gated_delta_decode
    torch.manual_seed(39)
    # Production TP2: [128 slots, 48 layers, 24 heads, 128, 128].
    # A layer view's highest slot exceeds 2**31 FP32 elements from its base.
    stride = 48*24*128*128
    cache = torch.empty_strided((128,24,128,128),(stride,128*128,128,1),device='cuda')
    slots = torch.tensor([0,113,114,127],device='cuda',dtype=torch.int32)
    dest = torch.tensor([126,125,124,123],device='cuda',dtype=torch.int32) if out_of_place else slots
    initial = torch.randn(4,24,128,128,device='cuda')*.01
    cache[slots.long()] = initial
    cache[2].fill_(7)
    q,k = [torch.nn.functional.normalize(torch.randn(4,24,128,device='cuda'),dim=-1) for _ in range(2)]
    v = torch.randn(4,24,128,device='cuda')
    g = -torch.rand(4,24,device='cuda')
    beta = torch.rand(4,24,device='cuda')
    decay = g.exp()
    prediction = torch.einsum('bhd,bhdv->bhv',k,initial)
    delta = v-decay[...,None]*prediction
    expected = decay[...,None,None]*initial+(beta[...,None]*k)[...,None]*delta[...,None,:]
    output = torch.einsum('bhd,bhdv->bhv',q,expected)
    actual = gated_delta_decode(q,k,v,g,beta,cache,slots,output_cache=cache,output_slots=dest)
    torch.testing.assert_close(actual,output,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(cache[dest.long()],expected,atol=2e-6,rtol=2e-5)
    assert torch.all(cache[2]==7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('length', [33, 64, 65, 129])
def test_chunk_matches_recurrence(length):
    torch.manual_seed(4)
    # Production prefill uses transposed [length, heads, dim] projections.
    q, k = [torch.nn.functional.normalize(torch.randn(length, 4, 32, device='cuda'), dim=-1).transpose(0, 1) for _ in range(2)]
    q = q / 32**0.5
    v = torch.randn(length, 4, 32, device='cuda').transpose(0, 1)
    g = -torch.rand(length, 4, device='cuda').T
    beta = torch.rand(length, 4, device='cuda').T
    initial = torch.randn(4, 32, 32, device='cuda') * .1
    expected, state = reference(q, k, v, g, beta, initial.clone())
    actual, final = _chunk_gated_delta_rule(q, k, v, g, beta, initial.clone())
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(final, state, atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch', [1, 4, 16])
def test_decode_cache_slots(batch):
    from zjhvllm.layers.gated_delta import gated_delta_decode
    torch.manual_seed(12)
    heads, dk, dv = 48, 128, 128
    # Non-contiguous per-layer view, permuted slots, untouched spare slot.
    storage = torch.randn(batch + 1, 2, heads, dk, dv, device='cuda') * .1
    cache = storage[:, 1]
    expected = storage.clone()
    slots = torch.randperm(batch, device='cuda')
    for _ in range(3):
        q, k = [torch.nn.functional.normalize(torch.randn(batch, heads, dk, device='cuda'), dim=-1) for _ in range(2)]
        q = q / dk**.5
        v = torch.randn(batch, heads, dv, device='cuda')
        g = -torch.rand(batch, heads, device='cuda')
        beta = torch.rand(batch, heads, device='cuda')
        state = expected[:, 1][slots]
        decay = g.exp()
        delta = v - decay[..., None] * torch.einsum('bhd,bhdv->bhv', k, state)
        state = decay[..., None, None] * state + (beta[..., None] * k)[..., None] * delta[:, :, None, :]
        output = torch.einsum('bhd,bhdv->bhv', q, state)
        expected[slots, 1] = state
        actual = gated_delta_decode(q, k, v, g, beta, cache, slots)
        torch.testing.assert_close(actual, output, atol=2e-6, rtol=2e-4)
        torch.testing.assert_close(storage, expected, atol=2e-6, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_layer_prefill_continuation_and_decode():
    from zjhvllm.models.qwen3_5 import GatedDeltaNet
    torch.manual_seed(8)
    layer = GatedDeltaNet(256, 2, 4, 128, 128, 4, 1e-6).cuda().to(torch.bfloat16)
    layer.norm.weight.fill_(1)
    layer.S_cache = torch.zeros(3, 4, 128, 128, device='cuda')
    layer.conv_cache = torch.zeros(3, layer.conv_dim, 3, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(2, 70, 256, device='cuda', dtype=torch.bfloat16) * .1
    states, convs = [], []
    for b in range(2):
        initial = torch.zeros_like(layer.S_cache[0])
        conv = torch.zeros_like(layer.conv_cache[0])
        full, state, history = layer._recurrence(x[b], initial.clone(), conv.clone())
        left, state_part, history_part = layer._recurrence(x[b, :35], initial.clone(), conv.clone())
        right, state_part, history_part = layer._recurrence(x[b, 35:], state_part, history_part)
        torch.testing.assert_close(torch.cat([left, right]), full, atol=3e-4, rtol=.02)
        torch.testing.assert_close(state_part, state, atol=3e-5, rtol=.002)
        torch.testing.assert_close(history_part, history)
        states.append(state)
        convs.append(history)
    slots = torch.tensor([2, 0], device='cuda')
    layer.S_cache[slots] = torch.stack(states)
    layer.conv_cache[slots] = torch.stack(convs)
    token = torch.randn(2, 256, device='cuda', dtype=torch.bfloat16) * .1
    expected = [layer._recurrence(token[b:b+1], states[b].clone(), convs[b].clone()) for b in range(2)]
    actual = layer._forward_decode(token, slots)
    torch.testing.assert_close(actual, torch.cat([row[0] for row in expected]), atol=3e-4, rtol=.02)
    torch.testing.assert_close(layer.S_cache[slots], torch.stack([row[1] for row in expected]), atol=3e-5, rtol=.002)
    torch.testing.assert_close(layer.conv_cache[slots], torch.stack([row[2] for row in expected]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_gated_norm_matches_transformers_bf16():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
    from zjhvllm.models.qwen3_5 import GatedDeltaNet
    torch.manual_seed(19)
    layer = GatedDeltaNet(32, 1, 1, 128, 128, 4, 1e-6).cuda().bfloat16()
    ref = Qwen3_5RMSNormGated(128, eps=1e-6).cuda().bfloat16()
    ref.weight.copy_(torch.randn_like(ref.weight))
    layer.norm.weight.copy_(ref.weight)
    values, gate = [torch.randn(17, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    torch.testing.assert_close(layer._z_gate(values, gate), ref(values, gate), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_partial_rope_matches_transformers_bf16():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from zjhvllm.models.qwen3_5 import _apply_qwen35_rope
    from zjhvllm.layers.rotary_embedding import RotaryEmbedding
    torch.manual_seed(22)
    rotary = RotaryEmbedding(256, 64, 1024, 10000000).cuda()
    positions = torch.tensor([0, 1, 64, 257, 513], device='cuda')
    q = torch.randn(5, 24, 256, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(5, 4, 256, device='cuda', dtype=torch.bfloat16)
    cos, sin = rotary.cos_sin_cache[positions, 0].chunk(2, dim=-1)
    cos, sin = [torch.cat((v, v), dim=-1).bfloat16()[None] for v in (cos, sin)]
    expected = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    actual = _apply_qwen35_rope(positions, q, k, rotary)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e[0].transpose(0, 1), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('decay_scale', [.3, 1., 3.])
@torch.inference_mode()
def test_production_heads_tf32x3_matches_torch_scan(decay_scale):
    # Includes nonzero state, column-major triangular-solve outputs, and
    # denormal-scale decayed keys at the strongest decay.
    torch.manual_seed(23)
    length, heads, dim = 257, 48, 128
    q, k = [torch.nn.functional.normalize(torch.randn(length, heads, dim, device='cuda'), dim=-1).transpose(0, 1) for _ in range(2)]
    q = q / dim**.5
    v = torch.randn(length, heads, dim, device='cuda').transpose(0, 1)
    g = -torch.rand(length, heads, device='cuda').T * decay_scale
    beta = torch.rand(length, heads, device='cuda').T
    initial = torch.randn(heads, dim, dim, device='cuda') * .1
    expected, state = _chunk_gated_delta_rule(q, k, v, g, beta, initial.clone(), scan_backend='torch')
    ieee4, state4 = _chunk_gated_delta_rule(q, k, v, g, beta, initial.clone(), scan_warps=4)
    ieee8, state8 = _chunk_gated_delta_rule(q, k, v, g, beta, initial.clone(), scan_warps=8)
    torch.testing.assert_close(ieee8, ieee4, atol=0, rtol=0)
    torch.testing.assert_close(state8, state4, atol=0, rtol=0)
    actual, final = _chunk_gated_delta_rule(q, k, v, g, beta, initial.clone(), scan_precision="tf32x3")
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(final, state, atol=2e-5, rtol=2e-4)
