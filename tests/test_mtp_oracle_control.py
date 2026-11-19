"""The offline oracle must never hide a divergent continuation."""
import pytest
from benchmarks.bench_mtp_net_gain import oracle_proposal
from zjhvllm.engine.sequence import Sequence
from zjhvllm.sampling_params import SamplingParams


@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    # This diagnostic helper is CPU-only and requires no process group.
    yield


def sequence(completion):
    seq = Sequence([11, 12], SamplingParams(greedy=True, max_tokens=4))
    for token in completion:
        seq.append_token(token)
    return seq


def test_oracle_follows_current_completion_boundary():
    references = {(11, 12): [21, 22, 23, 24]}
    assert oracle_proposal(sequence([21]), references)[0] == [22]
    assert oracle_proposal(sequence([21, 22, 23]), references)[0] == [24]


def test_oracle_rejects_divergent_prefix():
    with pytest.raises(AssertionError, match='prefix diverged'):
        oracle_proposal(sequence([21, 99]), {(11, 12): [21, 22, 23, 24]})


def test_oracle_rejects_exhausted_reference():
    with pytest.raises(ValueError, match='exhausted'):
        oracle_proposal(sequence([21, 22]), {(11, 12): [21, 22]})
