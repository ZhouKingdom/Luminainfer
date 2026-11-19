"""Packed drafting keeps private KV and atomic publication across requests."""
from copy import deepcopy
import pickle
from types import SimpleNamespace
import pytest
import torch
from test_qwen35_mtp import config
from test_mtp_continuous import engine_stub
from zjhvllm.engine.llm_engine import LLMEngine
from zjhvllm.engine.model_runner import ModelRunner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.models.qwen3_5_mtp import Qwen3_5MTP, MTPAttention
from zjhvllm.sampling_params import SamplingParams


@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    yield


@torch.inference_mode()
def test_packed_draft_incremental_ragged_slots_and_reuse_match_individual():
    torch.manual_seed(414)
    packed = Qwen3_5MTP(config(), 32, slots=4)
    for parameter in packed.parameters():
        parameter.copy_(torch.randn_like(parameter) * .02)
    serial = deepcopy(packed)
    for sizes in ([5, 3], [2, 1], [1, 2]):
        embedding, hidden = [torch.randn(sum(sizes), 16) for _ in range(2)]
        expected, offset = [], 0
        for slot, size in zip([3, 1], sizes):
            with serial.cache.for_slot(slot):
                expected.append(serial(embedding[offset:offset+size], hidden[offset:offset+size]))
            offset += size
        actual = packed.forward_batch(embedding, hidden, [3, 1], sizes)
        torch.testing.assert_close(actual, torch.cat(expected), atol=2e-6, rtol=1e-5)
        assert packed.cache.lengths == serial.cache.lengths
        for slot in (3, 1):
            end = packed.cache.get_length(slot)
            torch.testing.assert_close(packed.cache._keys[slot, :end], serial.cache._keys[slot, :end], atol=2e-6, rtol=1e-5)
        assert packed.cache.active_slot == 0 and packed.cache.batch_layout is None
    packed.cache.set_length(3, 0)
    serial.cache.set_length(3, 0)
    embedding, hidden = [torch.randn(2, 16) for _ in range(2)]
    actual = packed.forward_batch(embedding, hidden, [3], [2])
    with serial.cache.for_slot(3):
        expected = serial(embedding, hidden)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert packed.cache.get_length(1) == 6


@torch.inference_mode()
def test_packed_draft_fault_after_attention_restores_all_lengths(monkeypatch):
    model = Qwen3_5MTP(config(), 12, slots=3)
    model.cache.lengths[:] = [2, 0, 3]
    def fail(*args):
        assert model.cache.lengths == [4, 0, 4]
        raise RuntimeError('injected final norm')
    monkeypatch.setattr(model.norm, 'forward', fail)
    with pytest.raises(RuntimeError, match='final norm'):
        model.forward_batch(torch.randn(3, 16), torch.randn(3, 16), [2, 0], [1, 2])
    assert model.cache.lengths == [2, 0, 3]
    assert model.cache.batch_layout is None and model.cache.active_slot == 0


@pytest.mark.parametrize('slots,sizes', [([1, 1], [1, 1]), ([0, 2], [1, 9]), ([], [])])
def test_invalid_packed_layout_cannot_publish(slots, sizes):
    cache = MTPAttention(8, 1, 2, torch.float32, slots=3)
    cache.lengths[:] = [1, 2, 0]
    with pytest.raises(ValueError):
        with cache.packed(slots, sizes):
            raise AssertionError('invalid layout reached forward')
    assert cache.lengths == [1, 2, 0] and cache.batch_layout is None


@pytest.mark.parametrize('failure', [False, True])
@torch.inference_mode()
def test_runner_packs_shifted_pairs_and_restores_on_head_fault(failure):
    runner = ModelRunner.__new__(ModelRunner)
    runner.target_hidden = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
    cache = MTPAttention(8, 1, 2, torch.float32, slots=2)
    cache.lengths[:] = [0, 1]
    def forward_batch(embedding, hidden, slots, sizes):
        assert slots == [1, 0] and sizes == [3, 2]
        assert embedding[:, 0].tolist() == [12, 13, 14, 21, 22]
        assert torch.equal(hidden, torch.cat([runner.target_hidden[1, 1:4], runner.target_hidden[0, :2]]))
        cache.lengths[:] = [2, 4]
        return hidden
    def head(hidden):
        assert torch.equal(hidden, torch.stack([runner.target_hidden[1, 3], runner.target_hidden[0, 1]]))
        if failure:
            raise RuntimeError('injected head failure')
        return torch.tensor([[0., 9., 0.], [0., 0., 9.]])
    runner.model = SimpleNamespace(mtp=SimpleNamespace(cache=cache, forward_batch=forward_batch),
        model=SimpleNamespace(embed_tokens=lambda ids: ids[:, None].repeat(1, 2)), lm_head=head)
    seqs = [Sequence([10, 11, 12, 13, 14]), Sequence([20, 21, 22])]
    for seq, slot, cached in zip(seqs, [1, 0], [4, 2]):
        seq.state_slot, seq.state_ready, seq.num_cached_tokens = slot, True, cached
    if failure:
        with pytest.raises(RuntimeError, match='head failure'):
            runner.draft_mtp_batch(seqs)
        assert cache.lengths == [0, 1]
    else:
        result = runner.draft_mtp_batch(seqs)
        assert [draft for draft, _ in result] == [[1], [2]]
        assert [info['draft_extend_tokens'] for _, info in result] == [3, 2]
        assert cache.lengths == [2, 4]


@pytest.mark.parametrize('continuous', [False, True])
def test_engine_packed_draft_refills_and_skips_terminal_budget(continuous):
    engine = engine_stub()
    engine.model_runner.config.mtp_batch_draft = True
    original = engine.model_runner.call
    batches = []
    def call(name, *args):
        if name == 'draft_mtp_batch':
            batches.append([seq.seq_id for seq in args[0]])
            assert all(seq.max_tokens-seq.num_completion_tokens > 1 for seq in args[0])
            return [original('draft_mtp', seq) for seq in args[0]]
        assert name != 'draft_mtp', 'engine sent individual draft IPC'
        return original(name, *args)
    engine.model_runner.call = call
    result = engine.generate_speculative_batch([[11], [20], [31], [40]],
        [SamplingParams(greedy=True, max_tokens=n, ignore_eos=True) for n in [9, 2, 5, 1]], continuous=continuous)
    assert [r['token_ids'] for r in result] == [list(range(12, 21)), [21, 22], list(range(32, 37)), [41]]
    assert batches
    assert len(engine.scheduler.free_state_slots) == 2


def test_packed_draft_ipc_retains_all_request_tokens():
    runner = ModelRunner.__new__(ModelRunner)
    runner.world_size, runner.rank = 2, 0
    runner.shm = SimpleNamespace(buf=bytearray(4096))
    runner.event = [SimpleNamespace(set=lambda: None)]
    seq = Sequence([11, 12, 13])
    seq.append_token(14)
    seq.state_slot, seq.state_ready, seq.num_cached_tokens = 2, True, 3
    runner.write_shm('draft_mtp_batch', [seq])
    count = int.from_bytes(runner.shm.buf[:4], 'little')
    name, seqs = pickle.loads(runner.shm.buf[4:4+count])
    assert name == 'draft_mtp_batch' and seqs[0].__dict__ == seq.__dict__


@pytest.mark.parametrize('continuous', [False, True])
def test_engine_restores_packed_draft_frontiers_on_target_fault(continuous):
    engine = engine_stub()
    engine.model_runner.config.mtp_batch_draft = True
    original = engine.model_runner.call
    def call(name, *args):
        if name == 'draft_mtp_batch':
            return [original('draft_mtp', seq) for seq in args[0]]
        return original(name, *args)
    engine.model_runner.call = call
    engine.fault_verify = True
    with pytest.raises(RuntimeError, match='target fault'):
        engine.generate_speculative_batch([[11], [20]],
            SamplingParams(greedy=True, ignore_eos=True, max_tokens=8), continuous=continuous)
    assert engine.model_runner.model.mtp.cache.lengths == [0, 0]
    assert [seq.num_completion_tokens for seq in engine.seq_map.values()] == [1, 1]
    engine.fault_verify = False
    while not engine.is_finished():
        engine.step_speculative()
    assert [list(seq.completion_token_ids) for seq in engine.seq_map.values()] == [list(range(12, 20)), list(range(21, 29))]
    assert len(engine.scheduler.free_state_slots) == 2


def test_draft_transaction_classification_does_not_depend_on_wrapper_name(monkeypatch):
    runner = ModelRunner.__new__(ModelRunner)
    runner.world_size, runner.rank = 2, 0
    runner.write_shm = lambda *args: None
    runner.draft_mtp = lambda *args: None  # deliberately has __name__ '<lambda>'
    seen = []
    runner._speculative_transaction = lambda method, args, **kwargs: seen.append(kwargs['draft_only'])
    runner.call('draft_mtp', [])
    assert seen == [True]
