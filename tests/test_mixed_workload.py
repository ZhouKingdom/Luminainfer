"""Load-generator invariants: exact lengths, common arrivals, burst timing."""
import pytest
from benchmarks.bench_replica_service import make_workload,make_arrivals

@pytest.fixture(scope='session',autouse=True)
def _init_dist():yield


def test_mixed_lengths_are_exact_reproducible_and_variable():
    path='/home/data/jhzhou/huggingface/Qwen3.5-27B'
    a=make_workload(path,64,128,64,29,[64,128,512,1024],[32,64,128])
    b=make_workload(path,64,128,64,29,[64,128,512,1024],[32,64,128])
    assert a==b and len({r['id'] for r in a})==64
    assert {len(r['prompt_token_ids']) for r in a}=={64,128,512,1024}
    assert {r['max_tokens'] for r in a}=={32,64,128}


def test_poisson_arrivals_and_routing_are_shared_between_policies():
    items=[{'id':i} for i in range(64)]
    first=make_arrivals(items,'open',2.,29)
    second=make_arrivals(items,'open',2.,29)
    assert first==second
    assert all(b['arrival']>a['arrival'] for a,b in zip(first,first[1:]))
    routed=[first[i::4] for i in range(4)]
    assert sorted(r['id'] for group in routed for r in group)==list(range(64))


def test_bursts_use_planned_not_submission_arrivals():
    rows=make_arrivals([{'id':i} for i in range(24)],'burst',4.,29,8,2.)
    assert [r['arrival'] for r in rows]==[0.]*8+[2.]*8+[4.]*8
