"""Guard against flattering throughput aggregation and hidden arrival lag."""
import pytest
from benchmarks.service_metrics import summarize


@pytest.fixture(scope='session', autouse=True)
def _init_dist():
    yield


def test_cluster_uses_common_makespan_and_scheduled_arrivals():
    rows = [dict(token_times=[101.,102.], output_tokens=2, input_tokens=10,
                 scheduled_at=100., submitted_at=100., finished_at=102.),
            dict(token_times=[102.,104.], output_tokens=2, input_tokens=10,
                 scheduled_at=100.5, submitted_at=101., finished_at=104.)]
    result = summarize(rows,100.,2,ttft_slo=2.,tpot_slo=1.5)
    assert result['output_tokens_per_second'] == 1.
    assert result['output_tokens_per_second_per_gpu'] == .5
    assert result['ttft_seconds']['mean'] == 1.25
    assert result['loadgen_lag_seconds']['max'] == .5
    assert result['slo']['attainment'] == .5
    assert result['ttft_seconds']['p99'] == pytest.approx(1.495)
    assert result['p99_sample_note'].startswith('exploratory')


def test_incomplete_token_records_rejected():
    with pytest.raises(AssertionError):
        summarize([dict(token_times=[101.], output_tokens=2)], 100.,1)
