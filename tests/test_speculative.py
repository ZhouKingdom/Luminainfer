"""Hybrid rejection semantics: acceptance, private pages, rollback and replay."""
from types import SimpleNamespace

import pytest
import torch

from zjhvllm.engine.block_manager import BlockManager
from zjhvllm.engine.sequence import Sequence
from zjhvllm.engine.speculative import HybridCheckpoint, greedy_prefix, prompt_lookup
from zjhvllm.sampling_params import SamplingParams
from zjhvllm.utils.context import get_context, set_context


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


@pytest.mark.parametrize("reject", range(5))
def test_greedy_acceptance(reject):
    draft = [10, 11, 12, 13]
    target = draft + [99]
    if reject < 4:
        target[reject] = 77
    output, accepted = greedy_prefix(draft, target, 5, None)
    assert output == draft[:reject] + [99 if reject == 4 else 77]
    assert accepted == reject


def test_eos_and_budget():
    assert greedy_prefix([10, 11], [10, 11, 12], 3, 10) == ([10], 1)
    assert greedy_prefix([10, 11], [10, 11, 12], 1, None) == ([10], 1)
    assert greedy_prefix([10, 11], [99, 11, 12], 3, 99) == ([99], 0)
    with pytest.raises(ValueError):
        greedy_prefix([1], [1], 2, None)


def test_lookup_uses_observed_history_only():
    assert prompt_lookup([1, 2, 3, 4, 1, 2], 2, 2) == [3, 4]
    assert prompt_lookup([1, 2, 3], 4, 3) == []
    assert prompt_lookup([1, 2, 1, 2], 0, 1) == []
    with pytest.raises(ValueError):
        prompt_lookup([1], 4, 0)


def test_private_page_capacity_and_reclamation():
    seq = Sequence(list(range(255)))
    manager = BlockManager(3, 256, enable_prefix_caching=False)
    manager.allocate(seq, 0)
    seq.num_cached_tokens = 254
    original = list(seq.block_table)
    with pytest.raises(MemoryError):
        manager.reserve_private(seq, 1024)
    assert seq.block_table == original and len(manager.free_block_ids) == 2
    manager.reserve_private(seq, 513)
    assert len(seq.block_table) == 3
    manager.truncate_private(seq, 1)
    assert seq.block_table == original and len(manager.free_block_ids) == 2
    assert manager.used_block_ids == set(original)
    assert not manager.hash_to_block_id
    manager.deallocate(seq)
    assert len(manager.free_block_ids) == 3


def test_shared_page_rejected():
    seq = Sequence([1, 2])
    manager = BlockManager(3, 256)
    manager.allocate(seq, 0)
    with pytest.raises(ValueError):
        manager.reserve_private(seq, 300)


def fake_runner(target, fail_call=None):
    from zjhvllm.engine.model_runner import ModelRunner
    runner = ModelRunner.__new__(ModelRunner)
    runner.world_size, runner.has_linear_state = 1, True
    runner.config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_text"), eos=99)
    runner.S_cache = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    runner.conv_cache = runner.S_cache.clone().to(torch.bfloat16)
    runner.calls = []
    runner.block_size = 256

    def prepare(seqs):
        seq = seqs[0]
        ids = torch.tensor(seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens + seq.num_scheduled_tokens])
        set_context(True)
        return ids, torch.arange(len(ids))

    class Model:
        def __call__(self, ids, positions):
            runner.calls.append(ids.tolist())
            runner.S_cache[1].add_(ids.sum())
            runner.conv_cache[1].add_(ids.sum())
            if len(runner.calls) == fail_call:
                raise RuntimeError("injected forward failure")
            return torch.zeros(len(ids), 2)

        def compute_logits(self, hidden, *, all_tokens):
            assert all_tokens
            return torch.nn.functional.one_hot(torch.tensor(target), 128).float()

    runner.prepare_prefill, runner.model = prepare, Model()

    def decode(seqs):
        seq = seqs[0]
        assert len(seq.block_table) == (len(seq) + 255) // 256
        set_context(False)
        return torch.tensor([seq.last_token]), torch.tensor([len(seq) - 1])

    def run_model(ids, positions, is_prefill):
        assert not is_prefill
        runner.model(ids, positions)
        token = target[len(runner.calls) - 1]
        return torch.nn.functional.one_hot(torch.tensor([token]), 128).float()

    runner.prepare_decode, runner.run_model = decode, run_model
    seq = Sequence([7, 8, 9], SamplingParams(ignore_eos=True))
    seq.num_cached_tokens, seq.state_slot, seq.state_ready = 2, 1, True
    return runner, seq


@pytest.mark.parametrize("reject", range(5))
def test_serial_reference_uses_decode_only(reject):
    draft = [10, 11, 12, 13]
    target = draft + [77]
    if reject < 4:
        target[reject] = 66
    runner, seq = fake_runner(target)
    seq.block_table = [0]
    original = runner.S_cache.clone()
    output, stats = runner.verify_greedy(seq, draft, 5)
    assert output == target[:reject + 1]
    assert runner.calls == [[token] for token in [9] + draft[:reject]]
    original[1].add_(sum([9] + draft[:reject]))
    torch.testing.assert_close(runner.S_cache, original, rtol=0, atol=0)
    assert stats["target_forwards"] == reject + 1 and stats["replay_tokens"] == 0
    assert stats["verification_mode"] == "serial"


@pytest.mark.parametrize("reject", range(5))
def test_verify_restores_both_states_and_replays_prefix(reject):
    draft = [10, 11, 12, 13]
    target = draft + [77]
    if reject < 4:
        target[reject] = 66
    runner, seq = fake_runner(target)
    states, convs = runner.S_cache.clone(), runner.conv_cache.clone()
    output, stats = runner.verify_greedy(seq, draft, 5, "batched_experimental")
    committed_inputs = [9] + draft[:reject]
    states[1].add_(sum(committed_inputs))
    convs[1].add_(sum(committed_inputs))
    torch.testing.assert_close(runner.S_cache, states, rtol=0, atol=0)
    torch.testing.assert_close(runner.conv_cache, convs, rtol=0, atol=0)
    assert len(output) == reject + 1
    assert runner.calls == [[9] + draft] + ([committed_inputs] if reject < 4 else [])
    assert stats["accepted"] == reject
    assert seq.token_ids == [7, 8, 9] and seq.num_cached_tokens == 2
    assert not get_context().is_prefill


@pytest.mark.parametrize("fail_call", [1, 2])
def test_verify_exception_restores_original_state(fail_call):
    runner, seq = fake_runner([77, 11, 12], fail_call)
    states, convs = runner.S_cache.clone(), runner.conv_cache.clone()
    with pytest.raises(RuntimeError, match="injected"):
        runner.verify_greedy(seq, [10, 11], 3, "batched_experimental")
    torch.testing.assert_close(runner.S_cache, states, rtol=0, atol=0)
    torch.testing.assert_close(runner.conv_cache, convs, rtol=0, atol=0)
    assert not get_context().is_prefill
    assert not get_context().decode_equivalent
    assert get_context().verification_context_lens is None


@pytest.mark.parametrize("length", [255, 256, 257])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("verification_mode", ["serial", "batched_experimental"])
def test_engine_transaction_across_page_boundary(length, failure, verification_mode):
    from zjhvllm.engine.llm_engine import LLMEngine
    from zjhvllm.engine.scheduler import Scheduler
    from zjhvllm.engine.sequence import SequenceStatus
    runner, _ = fake_runner([66, 11, 12, 13, 77], 1 if failure else None)
    runner.config.max_model_len = 1024
    runner.config.hf_config.vocab_size = 128
    runner.call = lambda name, *args: getattr(runner, name)(*args)
    scheduler = Scheduler(SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=128, eos=99,
                                         kvcache_block_size=256, num_kvcache_blocks=4, num_state_slots=3))
    seq = Sequence([7] * length, SamplingParams(max_tokens=8, ignore_eos=True))
    scheduler.block_manager.allocate(seq, 0)
    seq.append_token(9)
    seq.num_cached_tokens, seq.state_slot, seq.state_ready = length, 1, True
    seq.status, seq.is_prefill = SequenceStatus.RUNNING, False
    scheduler.free_state_slots.remove(1)
    scheduler.running.append(seq)
    engine = LLMEngine.__new__(LLMEngine)
    engine.model_runner, engine.scheduler = runner, scheduler
    engine.seq_map, engine._pending_exports = {seq.seq_id: seq}, {}
    # This test runner owns no distributed resources and needs no exit().
    original_tokens, original_pages = list(seq.token_ids), list(seq.block_table)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            engine.speculative_step(seq.seq_id, [10, 11, 12, 13], verification_mode=verification_mode)
        assert seq.token_ids == original_tokens and seq.num_cached_tokens == length
        assert seq.block_table == original_pages
    else:
        output, stats = engine.speculative_step(seq.seq_id, [10, 11, 12, 13], verification_mode=verification_mode)
        assert output == [66] and stats["accepted"] == 0
        assert seq.token_ids == original_tokens + [66]
        assert seq.num_cached_tokens == length + 1
        assert len(seq.block_table) == (length + 256) // 256
    manager = scheduler.block_manager
    assert len(manager.free_block_ids) + len(manager.used_block_ids) == 4
    assert manager.used_block_ids == set(seq.block_table)
    assert all(manager.blocks[b].ref_count == 1 for b in seq.block_table)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_real_linear_state_rejection_and_slot_isolation():
    from zjhvllm.models.qwen3_5 import GatedDeltaNet
    torch.manual_seed(109)
    layer = GatedDeltaNet(256, 2, 4, 128, 128, 4, 1e-6).cuda().to(torch.bfloat16)
    layer.norm.weight.fill_(1)
    # Non-contiguous per-layer views reproduce the real engine cache layout.
    states = torch.randn(3, 2, 4, 128, 128, device="cuda") * .01
    convs = torch.randn(3, 2, layer.conv_dim, 3, device="cuda", dtype=torch.bfloat16) * .01
    layer.S_cache, layer.conv_cache = states[:, 1], convs[:, 1]
    checkpoint = HybridCheckpoint.capture(states, convs, 1)
    initial_states, initial_convs = states.clone(), convs.clone()
    tokens = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16) * .1
    for prefix in [1, 3, 5, 1, 4]:
        checkpoint.restore(states, convs)
        # Write the speculative suffix, then reject it and replay the prefix.
        _, state, conv = layer._recurrence(tokens, layer.S_cache[1], layer.conv_cache[1])
        layer.S_cache[1].copy_(state)
        layer.conv_cache[1].copy_(conv)
        checkpoint.restore(states, convs)
        _, state, conv = layer._recurrence(tokens[:prefix], layer.S_cache[1], layer.conv_cache[1])
        layer.S_cache[1].copy_(state)
        layer.conv_cache[1].copy_(conv)
        expected = layer._recurrence(tokens[:prefix], initial_states[1, 1], initial_convs[1, 1])
        torch.testing.assert_close(layer.S_cache[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(layer.conv_cache[1], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(states[[0, 2]], initial_states[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(convs[[0, 2]], initial_convs[[0, 2]], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_linear_decode_long_horizon_fp32_reference():
    from zjhvllm.layers.gated_delta import gated_delta_decode
    torch.manual_seed(818)
    batch, heads, dim = 2, 48, 128
    storage = torch.randn(3, 2, heads, dim, dim, device="cuda") * .01
    expected = storage.clone()
    slots = torch.tensor([2, 0], device="cuda")
    for step in range(128):
        q, k = [torch.nn.functional.normalize(torch.randn(batch, heads, dim, device="cuda"), dim=-1)
                for _ in range(2)]
        q = q / dim**.5
        v = torch.randn(batch, heads, dim, device="cuda") * .1
        # Slow decay retains long-lived state; nonzero initial state and
        # permuted strided slots detect drift and cross-request contamination.
        g = -torch.rand(batch, heads, device="cuda") * .03
        beta = torch.rand(batch, heads, device="cuda")
        state = expected[slots, 1]
        decay = g.exp()
        delta = v - decay[..., None] * torch.einsum("bhd,bhdv->bhv", k, state)
        state = decay[..., None, None] * state + (beta[..., None] * k)[..., None] * delta[:, :, None, :]
        output = torch.einsum("bhd,bhdv->bhv", q, state)
        expected[slots, 1] = state
        actual = gated_delta_decode(q, k, v, g, beta, storage[:, 1], slots)
        if step % 32 == 31:
            torch.testing.assert_close(actual, output, atol=5e-6, rtol=5e-4)
            torch.testing.assert_close(storage, expected, atol=5e-6, rtol=5e-4)
