"""Exact per-request service metrics; no GPU or framework imports."""
from statistics import mean


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    def q(p):
        index = (len(values)-1)*p
        lo = int(index)
        hi = min(lo+1, len(values)-1)
        return values[lo]*(hi-index) + values[hi]*(index-lo) if hi != lo else values[lo]
    return {'count': len(values), 'mean': mean(values), 'p50': q(.5), 'p90': q(.9), 'p95': q(.95), 'p99': q(.99), 'max': values[-1]}


def summarize(rows, start, replicas, ttft_slo=2., tpot_slo=.1):
    assert rows and all(len(r['token_times']) == r['output_tokens'] > 0 for r in rows)
    ttfts = [r['token_times'][0]-r['scheduled_at'] for r in rows]
    e2e = [r['finished_at']-r['scheduled_at'] for r in rows]
    tpots = [(r['token_times'][-1]-r['token_times'][0])/(len(r['token_times'])-1) for r in rows if len(r['token_times']) > 1]
    itls = [b-a for r in rows for a,b in zip(r['token_times'], r['token_times'][1:])]
    wall = max(r['finished_at'] for r in rows)-start
    output = sum(r['output_tokens'] for r in rows)
    good = sum(r['token_times'][0]-r['scheduled_at'] <= ttft_slo and (len(r['token_times'])==1 or (r['token_times'][-1]-r['token_times'][0])/(len(r['token_times'])-1) <= tpot_slo) for r in rows)
    return {'requests': len(rows), 'replicas': replicas, 'wall_seconds': wall,
            'input_tokens': sum(r['input_tokens'] for r in rows), 'output_tokens': output,
            'output_tokens_per_second': output/wall, 'requests_per_second': len(rows)/wall,
            'output_tokens_per_second_per_gpu': output/wall/replicas,
            'ttft_seconds': quantiles(ttfts), 'tpot_seconds': quantiles(tpots),
            'itl_seconds': quantiles(itls), 'e2e_seconds': quantiles(e2e),
            'loadgen_lag_seconds': quantiles([r['submitted_at']-r['scheduled_at'] for r in rows]),
            'p99_sample_note': '>=1000 request samples' if len(rows)>=1000 else 'exploratory: <1000 request samples',
            'slo': {'ttft_seconds': ttft_slo, 'tpot_seconds': tpot_slo, 'passed_requests': good,
                    'attainment': good/len(rows), 'goodput_requests_per_second': good/wall}}
