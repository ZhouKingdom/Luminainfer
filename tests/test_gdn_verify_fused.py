"""Short-window verify must match every FP32 decode boundary exactly."""
import pytest
import torch
from zjhvllm.layers.gated_delta import gated_delta_decode, gated_delta_verify
from zjhvllm.models.qwen3_5 import GatedDeltaNet
from zjhvllm.engine.speculative import HybridStateHistory
from zjhvllm.utils.context import get_context, reset_context, set_context


@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('layout', [[2], [2, 1, 2, 1], [1, 2, 1, 2], [3, 1, 5, 2], [9]])
@pytest.mark.parametrize('heads', [6, 24, 48])
@torch.inference_mode()
def test_fused_recurrence_matches_every_decode_boundary(layout, heads):
    torch.manual_seed(902)
    rows = sum(layout)
    storage = torch.randn(6, 2, heads, 128, 128, device='cuda') * .01
    state = storage[:, 1]
    original = storage.clone()
    expected = torch.zeros(rows + 1, 2, heads, 128, 128, device='cuda')
    actual = torch.zeros_like(expected)
    slots = torch.tensor([5, 1, 4, 0][:len(layout)], device='cuda', dtype=torch.int32)
    cu = [0]
    for size in layout:
        cu.append(cu[-1] + size)
    boundaries = torch.tensor(cu, device='cuda', dtype=torch.int32)
    q, k = [torch.nn.functional.normalize(torch.randn(rows, heads, 128, device='cuda'), dim=-1) for _ in range(2)]
    q *= 128 ** -.5
    v = torch.randn(rows, heads, 128, device='cuda')
    g, beta = -torch.rand(rows, heads, device='cuda'), torch.rand(rows, heads, device='cuda')
    expected_outputs = []
    for request, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        for step in range(start, end):
            first = step == start
            expected_outputs.append(gated_delta_decode(q[step:step+1], k[step:step+1], v[step:step+1],
                g[step:step+1], beta[step:step+1], state if first else expected[:, 1],
                slots[request:request+1] if first else torch.tensor([step-1], device='cuda', dtype=torch.int32),
                output_cache=expected[:, 1], output_slots=torch.tensor([step], device='cuda', dtype=torch.int32)))
    output = gated_delta_verify(q, k, v, g, beta, state, slots, boundaries, actual[:, 1], max_steps=max(layout))
    assert torch.equal(output, torch.cat(expected_outputs))
    assert torch.equal(actual, expected)  # includes untouched tail/layer
    assert torch.equal(storage, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('layout', [[2, 2], [2, 1], [1, 2], [3, 5], [9, 1]])
@torch.inference_mode()
def test_fused_layer_matches_original_history_bitwise(layout, batched):
    torch.manual_seed(903)
    layer = GatedDeltaNet(256, 2, 6, 128, 128, 4, 1e-6).cuda().bfloat16()
    layer.norm.weight.fill_(1)
    state = torch.randn(5, 2, 6, 128, 128, device='cuda') * .01
    conv = torch.randn(5, 2, layer.conv_dim, 3, device='cuda', dtype=torch.bfloat16) * .01
    layer.S_cache, layer.conv_cache = state[:, 1], conv[:, 1]
    original = [state.clone(), conv.clone()]
    hidden = torch.randn(sum(layout), 256, device='cuda', dtype=torch.bfloat16)
    slots = torch.tensor([3, 1], device='cuda', dtype=torch.int32)
    cu = [0, layout[0], sum(layout)]
    snapshots = []
    try:
        for fused in (False, True):
            history = HybridStateHistory(state, conv, sum(layout))
            history.states.zero_(); history.convs.zero_()
            history.bind([object(), layer], sum(layout), cu)
            set_context(True, state_slots=slots, cu_seqlens_q=torch.tensor(cu, device='cuda', dtype=torch.int32),
                        cu_seqlens_q_cpu=cu)
            get_context().decode_equivalent = True
            get_context().decode_batched = batched
            get_context().verification_fused_gdn = fused
            output = history.forward_layer(layer, hidden, slots)
            snapshots.append([output.clone(), history.states.clone(), history.convs.clone()])
            assert torch.equal(state, original[0]) and torch.equal(conv, original[1])
        for actual, expected in zip(snapshots[1], snapshots[0]):
            assert torch.equal(actual, expected), f'layer mismatch max_abs={(actual.float()-expected.float()).abs().max().item()}'
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('scale', [.01, 1., 30.])
@pytest.mark.parametrize('channels', [1280, 5120])
@torch.inference_mode()
def test_short_window_convolution_matches_cudnn_bf16(channels, scale):
    from zjhvllm.layers.gated_delta import causal_conv_verify
    torch.manual_seed(907)
    cu = [0, 5, 8, 17, 18]
    slots = torch.tensor([4, 1, 3, 0], device='cuda', dtype=torch.int32)
    bounds = torch.tensor(cu, device='cuda', dtype=torch.int32)
    state = torch.randn(5, 3, channels, 3, device='cuda', dtype=torch.bfloat16) * scale
    initial = state.clone()
    history = torch.zeros(19, 3, channels, 3, device='cuda', dtype=torch.bfloat16)
    expected_history = history.clone()
    mixed = torch.randn(18, channels, device='cuda', dtype=torch.bfloat16) * scale
    weight = torch.randn(channels, 1, 4, device='cuda', dtype=torch.bfloat16) * .1
    expected = []
    for request, slot in enumerate([4, 1, 3, 0]):
        conv = state[slot:slot+1, 1]
        for row in range(cu[request], cu[request+1]):
            concat = torch.cat([conv, mixed[row:row+1, :, None]], dim=2)
            expected.append(torch.nn.functional.conv1d(concat, weight, groups=channels).squeeze(-1))
            conv = concat[:, :, -3:]
            expected_history[row, 1].copy_(conv[0])
    actual = causal_conv_verify(mixed, weight, state[:, 1], slots, bounds, history[:, 1], max_steps=9)
    reference = torch.cat(expected)
    assert torch.equal(history, expected_history) and torch.equal(state, initial)
    assert torch.equal(actual, reference), f'conv mismatch count={(actual!=reference).sum().item()} max_abs={(actual.float()-reference.float()).abs().max().item()}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_fusion_toggle_cannot_reuse_the_other_arithmetic_graph():
    from test_mtp_batch import tiny_runner
    from zjhvllm.engine.sequence import Sequence
    from zjhvllm.sampling_params import SamplingParams
    torch.manual_seed(909)
    runner = tiny_runner(torch.bfloat16)
    runner.config.verification_cuda_graph = True
    seq = Sequence([0]*255+[7], SamplingParams(greedy=True, ignore_eos=True))
    seq.num_cached_tokens, seq.state_slot, seq.state_ready = 255, 3, True
    seq.block_table = [5, 2]
    buffers = [runner.S_cache, runner.conv_cache, runner.kv_cache, runner.target_hidden]
    initial = [x.clone() for x in buffers]
    expected = None
    try:
        for fused in (False, True, False, True):
            for actual, before in zip(buffers, initial):actual.copy_(before)
            reset_context();runner.config.verification_fused_gdn = fused
            output, stats = runner.verify_greedy_batch([seq], [[8]], [2])
            assert stats[0]['verification_graph']
            snapshot = [x.clone() for x in buffers]
            if expected is None:expected=(output,snapshot)
            else:
                assert output == expected[0]
                assert all(torch.equal(a,b) for a,b in zip(snapshot,expected[1]))
        assert set(runner.verification_graphs) == {(2,), ('fused_gdn',2)}
        assert all(graph.replays == 2 for graph in runner.verification_graphs.values())
    finally:
        reset_context()
