"""CPU regressions for hybrid recurrent-state scheduling; no model weights needed."""
from types import SimpleNamespace

import pytest

from zjhvllm.engine.scheduler import Scheduler
from zjhvllm.engine.sequence import Sequence
from zjhvllm.sampling_params import SamplingParams


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    # Scheduler tests do not execute distributed model layers.
    yield


def scheduler(slots=1, blocks=8, budget=256):
    return Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=budget, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=blocks,
        num_state_slots=slots,
    ))


def sequence(length=512, max_tokens=1):
    return Sequence(list(range(length)), SamplingParams(max_tokens=max_tokens))


@pytest.mark.parametrize("slots, expected_cached", [(1, 0), (-1, 256)])
def test_repeated_prefix(slots, expected_cached):
    s = scheduler(slots=slots, budget=1024)
    first = sequence()
    s.add(first)
    seqs, prefill = s.schedule()
    s.postprocess(seqs, [7], prefill)
    second = sequence()
    s.add(second)
    s.schedule()
    assert second.num_cached_tokens == expected_cached
    assert second.num_scheduled_tokens == 512 - expected_cached


def test_kv_failure_does_not_reserve_state():
    s = scheduler(blocks=1)
    seq = sequence()
    s.add(seq)
    assert s._schedule_prefills() == []
    assert seq.state_slot is None
    assert list(s.free_state_slots) == [0]


def test_chunk_continuation_and_slot_reuse():
    s = scheduler()
    first, second = sequence(), sequence(256)
    s.add(first)
    s.add(second)
    seqs, prefill = s.schedule()
    assert seqs == [first]
    first.state_ready = True
    s.postprocess(seqs, [7], prefill)
    assert first.num_cached_tokens == 256
    assert first.num_completion_tokens == 0
    assert first.state_slot == 0
    seqs, prefill = s.schedule()
    assert seqs == [first]
    assert first.state_ready
    s.postprocess(seqs, [7], prefill)
    assert first.state_slot is None
    assert not first.state_ready
    seqs, _ = s.schedule()
    assert seqs == [second]
    assert second.state_slot == 0
    assert not second.state_ready


def test_preemption_replays_full_prefix():
    s = scheduler(budget=1024)
    seq = sequence(max_tokens=4)
    s.add(seq)
    seqs, prefill = s.schedule()
    s.postprocess(seqs, [7], prefill)
    seq.state_ready = True
    s.running.remove(seq)
    s.preempt(seq)
    assert list(s.free_state_slots) == [0]
    assert not seq.state_ready
    s.schedule()
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 513
