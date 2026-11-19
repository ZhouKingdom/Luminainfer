"""Independent acceptance, request-private draft caches and hybrid commits."""
from types import SimpleNamespace

import pytest
import torch

from zjhvllm.engine.model_runner import ModelRunner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.engine.speculative import greedy_prefix
from zjhvllm.layers.speculative_accept import greedy_accept_batch
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.models.qwen3_5_mtp import MTPAttention
from zjhvllm.sampling_params import SamplingParams
from zjhvllm.utils.context import get_context, reset_context


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


@torch.inference_mode()
def test_request_private_draft_kv_and_exception_selection():
    torch.manual_seed(73)
    cache = MTPAttention(12, 1, 8, torch.float32, slots=4)
    query = torch.randn(5, 2, 8)
    key, value = [torch.randn(5, 1, 8) for _ in range(2)]
    refs = [MTPAttention(12, 1, 8, torch.float32) for _ in range(2)]
    with cache.for_slot(3):
        assert torch.equal(cache(query[:2], key[:2], value[:2]), refs[0](query[:2], key[:2], value[:2]))
    with cache.for_slot(1):
        assert torch.equal(cache(query[:3], key[:3], value[:3]), refs[1](query[:3], key[:3], value[:3]))
    with cache.for_slot(3):
        assert torch.equal(cache(query[2:], key[2:], value[2:]), refs[0](query[2:], key[2:], value[2:]))
    assert cache.lengths == [0, 3, 0, 5]
    assert cache.active_slot == 0
    with pytest.raises(RuntimeError):
        with cache.for_slot(1):
            raise RuntimeError("injected")
    assert cache.active_slot == 0
    cache.set_length(3, 0)
    assert cache.get_length(1) == 3


@pytest.mark.parametrize("pages", [3, 4])
def test_engine_reserves_and_commits_pages_per_request(pages):
    from zjhvllm.engine.llm_engine import LLMEngine
    from zjhvllm.engine.scheduler import Scheduler
    from zjhvllm.engine.sequence import SequenceStatus
    scheduler = Scheduler(SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=128, eos=99,
                                         kvcache_block_size=256, num_kvcache_blocks=pages, num_state_slots=3))
    seqs = []
    for slot in [2, 0]:
        seq = Sequence([7] * 255, SamplingParams(greedy=True, max_tokens=8, ignore_eos=True))
        scheduler.block_manager.allocate(seq, 0)
        seq.append_token(9)
        seq.num_cached_tokens, seq.state_slot, seq.state_ready = 255, slot, True
        seq.status, seq.is_prefill = SequenceStatus.RUNNING, False
        scheduler.free_state_slots.remove(slot)
        scheduler.running.append(seq)
        seqs.append(seq)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler, engine._pending_exports = scheduler, {}
    engine.seq_map = {seq.seq_id: seq for seq in seqs}
    calls = []
    def call(name, requests, drafts, budgets):
        calls.append(name)
        assert all(len(seq.block_table) == 2 for seq in requests)
        return [[10, 77], [66]], [{"accepted": 1}, {"accepted": 0}]
    engine.model_runner = SimpleNamespace(call=call, config=SimpleNamespace(
        max_model_len=1024, hf_config=SimpleNamespace(vocab_size=128)))
    before = [list(seq.block_table) for seq in seqs]
    if pages == 3:
        with pytest.raises(MemoryError):
            engine.speculative_step_batch([seq.seq_id for seq in seqs], [[10], [11]])
        assert not calls
        assert [seq.block_table for seq in seqs] == before
        assert len(scheduler.block_manager.free_block_ids) == 1
    else:
        engine.speculative_step_batch([seq.seq_id for seq in seqs], [[10], [11]])
        assert [seq.num_cached_tokens for seq in seqs] == [257, 256]
        assert [len(seq.block_table) for seq in seqs] == [2, 1]
        assert len(scheduler.block_manager.free_block_ids) == 1


def test_batch_generation_restores_every_draft_length_on_target_fault():
    from zjhvllm.engine.llm_engine import LLMEngine
    from zjhvllm.engine.sequence import SequenceStatus
    engine = LLMEngine.__new__(LLMEngine)
    cache = MTPAttention(16, 1, 8, torch.float32, slots=2)
    runner = SimpleNamespace(model=SimpleNamespace(mtp=SimpleNamespace(cache=cache)), config=SimpleNamespace(
        verification_state_commit=True, max_num_seqs=2, num_state_slots=2, max_model_len=64))
    engine.model_runner, engine.seq_map, engine._pending_exports = runner, {}, {}
    engine.scheduler = SimpleNamespace(running=[])
    engine.tokenizer = SimpleNamespace(decode=str)
    engine.is_finished = lambda: True
    def add(tokens, param):
        seq = Sequence(tokens, param)
        seq.state_slot, seq.state_ready = len(engine.seq_map), True
        seq.status = SequenceStatus.RUNNING
        engine.seq_map[seq.seq_id] = seq
        engine.scheduler.running.append(seq)
        return seq.seq_id
    def step():
        for seq in engine.seq_map.values():
            seq.num_cached_tokens = len(seq)
            seq.append_token(4)
    def draft(name, seq):
        assert name == "draft_mtp"
        cache.set_length(seq.state_slot, seq.num_cached_tokens)
        return [5], {"draft_source": "mtp"}
    def fail(*args):
        assert cache.lengths == [2, 2]
        raise RuntimeError("injected target batch failure")
    engine.add_request, engine.step, runner.call = add, step, draft
    engine.speculative_step_batch = fail
    with pytest.raises(RuntimeError, match="injected"):
        engine.generate_speculative_batch([[1, 2], [2, 3]], SamplingParams(greedy=True, max_tokens=4), continuous=False)
    assert cache.lengths == [0, 0]
    assert all(seq.num_completion_tokens == 1 for seq in engine.seq_map.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [2, 5, 33])
@torch.inference_mode()
def test_gpu_acceptance_is_per_request(width):
    drafts = [list(range(10, 10 + width - 1)), list(range(30, 30 + width - 1)), [], [4]]
    targets = [drafts[0] + [90], [77] + drafts[1][1:] + [91], [88], [4, 3]]
    budgets, ignores = [width, width, 1, 1], [True, True, True, False]
    cu, ids, flat = [0], [], []
    for draft, target in zip(drafts, targets):
        ids.extend([2] + draft)
        flat.extend(target)
        cu.append(len(ids))
    outputs, accepted = greedy_accept_batch(
        torch.tensor(flat, device="cuda"), torch.tensor(ids, device="cuda"),
        torch.tensor(cu, device="cuda", dtype=torch.int32), budgets, ignores, 4, width)
    reference = [greedy_prefix(d, t, b, None if ignore else 4)
                 for d, t, b, ignore in zip(drafts, targets, budgets, ignores)]
    assert outputs == [r[0] for r in reference]
    assert accepted == [r[1] for r in reference]
    assert accepted[:2] == [width - 1, 0]  # No batch-min truncation.


def tiny_runner(kv_dtype):
    cfg = SimpleNamespace(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
                          hidden_act="silu", num_hidden_layers=4,
                          layer_types=["linear_attention", "full_attention"] * 2,
                          linear_num_key_heads=1, linear_num_value_heads=2,
                          linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                          rms_norm_eps=1e-6, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                          max_position_embeddings=1024, attention_bias=False,
                          rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": .25},
                          vocab_size=128, tie_word_embeddings=False)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=cfg, eos=-1, enable_mtp=True, max_num_seqs=5,
                                    max_model_len=1024, cache_schedule_metadata=True)
    runner.model = Qwen3_5ForCausalLM(cfg).cuda().bfloat16().eval()
    runner.has_linear_state, runner.world_size, runner.enforce_eager = True, 1, True
    runner.block_size = 256
    runner.S_cache = torch.randn(5, 2, 2, 128, 128, device="cuda") * .01
    runner.conv_cache = torch.randn(5, 2, 512, 3, device="cuda", dtype=torch.bfloat16) * .01
    runner.kv_cache = (torch.randn(2, 2, 6, 256, 2, 32, device="cuda", dtype=torch.bfloat16) * .01).to(kv_dtype)
    runner.kv_cache_scale = torch.ones(2, 2, 6, 256, device="cuda")
    runner.target_hidden = torch.zeros(5, 1024, 64, device="cuda", dtype=torch.bfloat16)
    linear, full = 0, 0
    for layer in runner.model.model.layers:
        if layer.linear_attn is not None:
            layer.linear_attn.S_cache = runner.S_cache[:, linear]
            layer.linear_attn.conv_cache = runner.conv_cache[:, linear]
            linear += 1
        else:
            attn = layer.self_attn.attn
            attn.k_cache, attn.v_cache = runner.kv_cache[:, full]
            attn.k_scale, attn.v_scale = runner.kv_cache_scale[:, full]
            full += 1
    return runner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("reverse_slots", [False, True])
@torch.inference_mode()
def test_native_decode_graph_dynamic_slots_pages_and_eager_fallback(count, reverse_slots):
    runner = tiny_runner(torch.bfloat16)
    runner.config.enable_mtp = False
    runner.config.num_state_slots = 5
    buffers = [runner.S_cache, runner.conv_cache, runner.kv_cache, runner.target_hidden]
    initial = [x.clone() for x in buffers]
    runner.capture_cudagraph()
    assert torch.count_nonzero(runner.S_cache) == 0
    assert torch.count_nonzero(runner.conv_cache) == 0
    assert 2 in runner.graphs and 3 not in runner.graphs
    slots = [3, 1, 4]
    if reverse_slots:
        slots.reverse()
    seqs = []
    for cached, slot, pages in zip([255,256,257], slots, [[5,2],[4,1],[0,3]]):
        seq = Sequence([0]*cached+[7], SamplingParams(greedy=True))
        seq.state_slot, seq.state_ready, seq.num_cached_tokens = slot, True, cached
        seq.block_table = pages
        seqs.append(seq)
    original = runner.model.forward
    try:
        for eager in (True,False):
            for dest, value in zip(buffers,initial):
                dest.copy_(value)
            runner.enforce_eager = eager
            if not eager and count == 2:
                def forbidden(*args):
                    raise AssertionError("captured bucket re-entered eager model forward")
                runner.model.forward = forbidden
            ids, positions = runner.prepare_decode(seqs[:count])
            logits = runner.run_model(ids, positions, False)
            snapshot = [x.clone() for x in buffers]+[logits.clone()]
            reset_context()
            if eager:
                reference = snapshot
            else:
                for actual, expected in zip(snapshot,reference):
                    assert torch.equal(actual,expected)
    finally:
        runner.model.forward = original
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("case", ["mixed", "ragged_budget", "eos"])
@pytest.mark.parametrize("linear_backend", ["decode_equivalent", "batch_invariant"])
@torch.inference_mode()
def test_hybrid_batch_matches_independent_serial_and_restores_fault(kv_dtype, case, linear_backend):
    torch.manual_seed(74)
    runner = tiny_runner(kv_dtype)
    from zjhvllm.models.qwen3_5 import Qwen3_5Linear
    for module in runner.model.modules():
        if isinstance(module, Qwen3_5Linear):
            module.backend = linear_backend
    seqs = []
    for cached, slot, pages, pending in [(255, 3, [4, 1], 7), (256, 1, [3, 0], 9)]:
        seq = Sequence([0] * cached + [pending], SamplingParams(greedy=True, ignore_eos=True))
        seq.num_cached_tokens, seq.state_slot, seq.state_ready = cached, slot, True
        seq.block_table = pages
        seqs.append(seq)
    buffers = [runner.S_cache, runner.conv_cache, runner.kv_cache, runner.kv_cache_scale, runner.target_hidden]
    initial = [x.clone() for x in buffers]
    def restore():
        for target, source in zip(buffers, initial):
            target.copy_(source)
        reset_context()
    try:
        correct = []
        for seq in seqs:
            output, _ = runner.verify_greedy(seq, [], 1, "serial")
            correct.append(output[0])
            restore()
        drafts, budgets = [[correct[0]], [(correct[1] + 1) % 128]], [3, 3]
        if case == "ragged_budget":
            drafts = [[correct[0], 17], []]
            budgets = [1, 3]
        if case == "eos":
            runner.config.eos = correct[0]
            seqs[0].ignore_eos = False
        expected, accepted = [], []
        for seq, draft, budget in zip(seqs, drafts, budgets):
            output, info = runner.verify_greedy(seq, draft, budget, "serial")
            expected.append(output)
            accepted.append(info["accepted"])
        final = [x.clone() for x in buffers]
        restore()
        outputs, stats = runner.verify_greedy_batch(seqs, drafts, budgets)
        assert outputs == expected
        assert [r["accepted"] for r in stats] == accepted
        if case == "mixed":
            assert [len(row) for row in outputs] == [2, 1]
        assert all(r["replay_tokens"] == 0 for r in stats)
        assert sum(r["target_forwards"] for r in stats) == 1
        assert torch.equal(buffers[0], final[0])
        assert torch.equal(buffers[1], final[1])
        for seq, output in zip(seqs, outputs):
            valid = seq.num_cached_tokens + len(output)
            indices = torch.tensor([seq.block_table[p // 256] * 256 + p % 256 for p in range(valid)], device="cuda")
            for i in (2, 3):
                a = buffers[i].flatten(2, 3).index_select(2, indices)
                b = final[i].flatten(2, 3).index_select(2, indices)
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            assert torch.equal(buffers[4][seq.state_slot, :valid], final[4][seq.state_slot, :valid])
        restore()
        history = runner.verification_history
        original = history.commit_many
        def fail(states, convs, slots, counts):
            states[slots[0]].copy_(history.states[counts[0] - 1])
            raise RuntimeError("injected multi-request partial commit")
        history.commit_many = fail
        try:
            with pytest.raises(RuntimeError, match="injected"):
                runner.verify_greedy_batch(seqs, drafts, budgets)
            assert torch.equal(buffers[0], initial[0])
            assert torch.equal(buffers[1], initial[1])
            assert get_context().verification_state is None
        finally:
            history.commit_many = original
        retry, _ = runner.verify_greedy_batch(seqs, drafts, budgets)
        assert retry == expected
    finally:
        reset_context()
