"""Protect the diagnostic's rollback and scoped ablations without loading 27B."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks.diagnose_mtp_verification import WindowDiagnostic, interventions, top_rows
from zjhvllm.utils.context import get_context, reset_context, set_context


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def test_scoped_projection_restores_after_failure():
    linear = nn.Linear(3, 2)
    original = linear.forward
    shapes = []
    # Observe F.linear calls, including calls bypassing module forward hooks.
    def observe(x):
        shapes.append(x.shape[0])
        return original(x)
    linear.forward = observe
    runner = SimpleNamespace(model=linear)
    with pytest.raises(RuntimeError, match="injected"):
        with interventions(runner, ["projection"]):
            actual = linear(torch.ones(2, 3))
            assert actual.shape == (2, 2)
            assert shapes == [1, 1]
            raise RuntimeError("injected")
    assert linear.forward is observe
    linear(torch.ones(2, 3))
    assert shapes == [1, 1, 2]


def test_top_rows_records_argmax_independently_of_topk_ties():
    logits = torch.tensor([[4., 4., 3., 2., 1., 0.]])
    row = top_rows(logits)[0]
    assert row["argmax"] == 0
    assert row["gap"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_window_restores_state_context_and_hooks(fail):
    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Embedding(16, 2)

        def forward(self, ids, positions):
            hidden = self.model(ids)
            runner.S_cache.add_(len(ids))
            runner.conv_cache.add_(len(ids))
            runner.target_hidden[positions] = hidden
            if get_context().is_prefill and fail:
                raise RuntimeError("injected batch failure")
            return hidden

        def compute_logits(self, hidden, all_tokens=False):
            return torch.cat([hidden, hidden, hidden], dim=-1)

    runner = SimpleNamespace(model=Target(), block_size=256,
                             S_cache=torch.zeros(1, 1, 1, 1), conv_cache=torch.zeros(1, 1, 1, 1),
                             target_hidden=torch.zeros(8, 2), kv_cache=torch.zeros(2, 1, 1, 256, 1, 1),
                             verify_greedy=lambda *args: None)
    def decode(seqs):
        seq = seqs[0]
        set_context(False)
        return torch.tensor([seq.last_token]), torch.tensor([seq.num_tokens - 1])
    def prefill(seqs):
        seq = seqs[0]
        set_context(True, slot_mapping=torch.tensor([2, 3]))
        return torch.tensor(seq.token_ids[2:]), torch.tensor([2, 3])
    runner.prepare_decode, runner.prepare_prefill = decode, prefill
    seq = SimpleNamespace(num_cached_tokens=2, token_ids=[1, 2, 3], num_tokens=3,
                          state_slot=0, block_table=[0])
    diagnostic = WindowDiagnostic(runner, 1)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="injected batch failure"):
                diagnostic.compare(seq, [4])
        else:
            result = diagnostic.compare(seq, [4])
            assert result["logits"]["exact"]
            assert all(s["exact"] for s in result["states"])
            assert result["kv_committed_prefix"]["exact"]
        assert torch.count_nonzero(runner.S_cache) == 0
        assert torch.count_nonzero(runner.conv_cache) == 0
        assert torch.count_nonzero(runner.target_hidden) == 0
        assert not runner.model.model._forward_hooks
        assert not get_context().is_prefill
        assert seq.token_ids == [1, 2, 3]
    finally:
        reset_context()
