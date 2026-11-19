"""Shared-GPU scheduling must fail closed and signal only owned children."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import os

import pytest

BENCH = Path(__file__).parents[1]/"benchmarks"
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("watch_qwen35", BENCH/"watch_qwen35.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
sys.path.pop(0)


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def plan(tmp_path):
    return {"source": str(tmp_path), "max_job_seconds": 10, "min_free_disk_gib": 0,
            "max_used_mib": 256, "max_util_percent": 5, "model": "model",
            "pythons": {"lumina": "python-l", "vllm": "python-v"},
            "expected_packages": {"lumina": {}, "vllm": {}}}


def gpu(index=0, **changes):
    return {"index": index, "uuid": f"GPU-{index}", "used_mib": 4, "util": 0, "pids": [], **changes}


def test_matrix_is_finite_and_covers_requested_experiments():
    cases = watch.matrix()
    assert len(cases) == 65 and len({c["id"] for c in cases}) == 65
    assert {c["tp"] for c in cases} == {1, 2, 4, 8}
    graph = [c for c in cases if c["kind"] == "graph" and c["replica"] == 0]
    assert len(graph) == 48
    assert len([c for c in cases if c["kind"] == "profile"]) == 4
    assert all(c["requests"] == 1024 for c in cases if c["kind"] == "tail")
    assert all(c["repeats"] == 3 for c in graph)


@pytest.mark.parametrize("changes", [{"pids": [123]}, {"used_mib": 300}, {"util": 50}])
def test_idle_requires_no_process_low_memory_and_low_utilization(tmp_path, changes):
    assert watch.idle(gpu(), plan(tmp_path))
    assert not watch.idle(gpu(**changes), plan(tmp_path))


def test_group_selection_requires_stable_samples_and_excludes_locks():
    gpus = [gpu(i) for i in range(4)]
    streaks = {"GPU-0": 3, "GPU-1": 4, "GPU-2": 4, "GPU-3": 4}
    assert [g["index"] for g in watch.choose_group(gpus, streaks, 2, 4)] == [1, 2]
    assert watch.choose_group(gpus, streaks, 4, 4) is None
    assert watch.choose_group(gpus, streaks, 2, 4, {"GPU-1", "GPU-2"}) is None


def test_gpu_query_parses_uuid_and_process_ownership(monkeypatch):
    def output(argv, **kw):
        if "--query-gpu=index,uuid,memory.used,utilization.gpu" in argv:
            return "0, GPU-a, 4, 0\n1, GPU-b, 75000, 99\n"
        return "GPU-b, 123\nGPU-b, 124\n"
    monkeypatch.setattr(watch.subprocess, "check_output", output)
    rows = watch.gpu_snapshot()
    assert rows[0]["pids"] == [] and rows[1]["pids"] == [123, 124]


def test_process_membership_requires_both_owned_group_and_session(monkeypatch):
    monkeypatch.setattr(watch.subprocess, "check_output", lambda *a, **kw: "10 10 10\n11 10 10\n12 10 99\n13 99 10\n")
    assert watch.process_group_pids(10) == {10, 11}


def test_foreign_gpu_process_stops_only_owned_session(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=987654321, poll=lambda: None, wait=lambda **kw: None)
    monkeypatch.setattr(watch.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(watch, "gpu_snapshot", lambda: [gpu(pids=[321, 654])])
    monkeypatch.setattr(watch, "owned_gpu_pids", lambda _: {987654321, 321})
    stopped = []
    monkeypatch.setattr(watch, "stop_owned_group", lambda p: stopped.append(p.pid))
    result = watch.execute(tmp_path, plan(tmp_path), ["fake"], {}, tmp_path/"job.log", [gpu()])
    assert result["reason"] == "foreign_gpu_process" and result["foreign_pids"] == [654]
    assert stopped == [987654321]  # foreign PID never supplied to a signal function


def test_stop_file_cleans_up_current_owned_job(tmp_path, monkeypatch):
    (tmp_path/"STOP").touch()
    process = SimpleNamespace(pid=987654321, poll=lambda: None, wait=lambda **kw: None)
    monkeypatch.setattr(watch.subprocess, "Popen", lambda *a, **kw: process)
    stopped = []
    monkeypatch.setattr(watch, "stop_owned_group", lambda p: stopped.append(p.pid))
    result = watch.execute(tmp_path, plan(tmp_path), ["fake"], {}, tmp_path/"job.log", [gpu()])
    assert result["reason"] == "stop_requested" and stopped == [987654321]


def test_missing_gpu_is_not_treated_as_idle(tmp_path, monkeypatch):
    p = plan(tmp_path)
    c = watch.matrix()[0] | {"status": "pending", "attempts": [], "order": ["lumina", "vllm"]}
    monkeypatch.setattr(watch, "package_versions", lambda _: {})
    monkeypatch.setattr(watch, "update", lambda *a, **kw: None)
    monkeypatch.setattr(watch, "gpu_snapshot", lambda: [])
    watch.run_case(tmp_path, p, c, [gpu(), gpu(1)])
    assert c["status"] == "pending" and c["attempts"][0]["reason"] == "busy_before_engine"


def test_changed_environment_prevents_launch(tmp_path, monkeypatch):
    p = plan(tmp_path)
    c = watch.matrix()[0] | {"status": "pending", "attempts": [], "order": ["lumina", "vllm"]}
    monkeypatch.setattr(watch, "package_versions", lambda _: {"torch": "changed"})
    monkeypatch.setattr(watch, "update", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="versions changed"):
        watch.run_case(tmp_path, p, c, [gpu(), gpu(1)])


def test_pid_reuse_guard_never_signals_unrelated_session(monkeypatch):
    process = SimpleNamespace(pid=os.getpid(), _watch_start_ticks="not-the-current-process")
    monkeypatch.setattr(watch, "process_group_pids", lambda _: {os.getpid()})
    signals = []
    monkeypatch.setattr(watch.os, "killpg", lambda *args: signals.append(args))
    watch.stop_owned_group(process)
    assert signals == []


def test_queue_initialization_reverses_anchor_and_freezes_sources(tmp_path, monkeypatch):
    def prepare(args):
        watch.bench.write_json(args.output, {"requests": [{"id": str(i), "input_ids": [1, 2], "max_tokens": 512}
                                                        for i in range(1024)]})
    monkeypatch.setattr(watch.bench, "prepare", prepare)
    monkeypatch.setattr(watch, "package_versions", lambda _: {"torch": "test"})
    directory = tmp_path/"queue"
    watch.init(SimpleNamespace(directory=directory, model=tmp_path))
    p = watch.json.loads((directory/"plan.json").read_text())
    anchor = [c for c in p["cases"] if c["kind"] == "graph" and c["tp"] == 2 and c["capacity"] == 32
              and c["input"] == c["output"] == 128]
    assert anchor[1]["order"] == list(reversed(anchor[0]["order"]))
    assert len(p["cases"]) == 65 and p["source_sha256"]
    for name, sha in p["source_sha256"].items():
        assert watch.bench.digest(Path(p["source"])/name) == sha


def test_profile_descendants_survive_setsid_reparent_and_pid_reuse(monkeypatch):
    process = SimpleNamespace(pid=10, _watch_token="unique", _watch_start_ticks="100")
    rows = {10: (1, "100"), 20: (10, "200"), 30: (20, "300"), 99: (1, "400")}
    monkeypatch.setattr(watch, "process_records", lambda: rows)
    monkeypatch.setattr(watch, "tagged_processes", lambda _: {})
    assert watch.owned_gpu_pids(process) == {10, 20, 30}
    rows[30] = (1, "300")  # a detached/reparented worker is still ours
    assert watch.owned_gpu_pids(process) == {10, 20, 30}
    rows[30] = (1, "900")  # PID reused by an unrelated job
    assert watch.owned_gpu_pids(process) == {10, 20}


def test_marker_recovers_detached_worker_without_trusting_other_processes(monkeypatch):
    process = SimpleNamespace(pid=10, _watch_token="unique", _watch_start_ticks="100")
    monkeypatch.setattr(watch, "process_records", lambda: {10: (1, "100"), 20: (1, "200"), 99: (1, "400")})
    monkeypatch.setattr(watch, "tagged_processes", lambda _: {20: "200"})
    assert watch.owned_gpu_pids(process) == {10, 20}
