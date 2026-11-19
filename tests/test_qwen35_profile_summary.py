import importlib.util
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location("profile_summary", Path(__file__).parents[1]/"benchmarks/qwen35_profile_summary.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def test_interval_union_handles_overlapping_streams():
    assert profile.union_ns([(20, 25), (0, 10), (5, 15), (1, 2)]) == 20
    assert profile.union_ns([]) == 0
    with pytest.raises(ValueError):
        profile.union_ns([(2, 1)])


def test_nsight_summary_keeps_devices_and_overlap_separate(tmp_path):
    path = tmp_path/"trace.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
        db.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "gemm"), (2, "ncclKernel"), (3, "gated_delta_decode")])
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,demangledName INTEGER)")
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?)", [(0,10,0,1),(5,15,0,2),(20,25,0,3),(0,50,1,2)])
    result = profile.summarize(path)
    assert result["devices"]["0"]["kernel_union_ns"] == 20
    assert result["devices"]["0"]["kernel_span_ns"] == 25
    assert result["devices"]["0"]["kernel_gap_ns"] == 5
    assert sum(result["devices"]["0"]["category_sum_ns_overlapping"].values()) == 25
    assert result["devices"]["1"]["kernel_union_ns"] == 50
    assert result["kernels"][0]["device"] == 1


def test_missing_kernel_trace_is_rejected(tmp_path):
    path = tmp_path/"empty.sqlite"
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match="missing"):
        profile.summarize(path)
