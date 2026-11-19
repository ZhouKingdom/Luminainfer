"""CPU-only benchmark measurement and comparison contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest

spec = importlib.util.spec_from_file_location("qwen35_baseline", Path(__file__).parents[1]/"benchmarks/qwen35_baseline.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def requests():
    return [{"id": "a", "input_ids": [1, 2], "max_tokens": 3},
            {"id": "b", "input_ids": [3], "max_tokens": 1}]


def test_request_metrics_use_actual_observations_and_burst_arrival():
    times = iter([10., 11., 12., 13., 15.])
    recorder = bench.Recorder(requests(), lambda: next(times))
    recorder.observe("a", [7], False)
    recorder.observe("b", [9], True)
    recorder.observe("a", [7, 8], False)
    recorder.observe("a", [7, 8, 9], True)
    summary = recorder.summary(5.)
    assert summary["output_tok_s"] == .8
    assert summary["ttft_seconds"]["mean"] == 1.5
    assert summary["tpot_seconds"]["mean"] == 2.
    assert summary["itl_seconds"]["mean"] == 2.
    assert summary["e2e_seconds"]["mean"] == 3.5
    assert summary["tpot_seconds"]["count"] == 1


def test_multi_token_chunk_does_not_fabricate_itl():
    times = iter([0., 1., 3., 4.])
    recorder = bench.Recorder(requests(), lambda: next(times))
    recorder.observe("a", [7, 8], False)
    recorder.observe("a", [7, 8, 9], True)
    recorder.observe("b", [2], True)
    summary = recorder.summary(4.)
    assert summary["single_token_observations"] is False
    assert summary["itl_seconds"] is None and summary["tpot_seconds"] is None
    assert recorder.rows["a"]["observations"] == [{"seconds": 1., "tokens": 2}, {"seconds": 3., "tokens": 1}]


def test_reject_rewrite_early_eos_and_incomplete_run():
    recorder = bench.Recorder(requests())
    recorder.observe("a", [7], False)
    with pytest.raises(ValueError, match="cumulative"):
        recorder.observe("a", [8, 9], False)
    with pytest.raises(ValueError, match="budget"):
        recorder.observe("a", [7], True)
    with pytest.raises(ValueError, match="incomplete"):
        recorder.summary(1.)
    recorder.observe("b", [1], True)
    with pytest.raises(ValueError, match="after"):
        recorder.observe("b", [1], True)


@pytest.mark.parametrize("mutation", ["schema", "duplicate", "zero_budget", "negative_token"])
def test_invalid_workload(mutation):
    workload = {"schema": bench.SCHEMA, "arrival": "simultaneous-burst", "sampling": "greedy-ignore-eos", "requests": requests()}
    bench.validate_workload(workload)
    if mutation == "schema":
        workload["schema"] = "other"
    elif mutation == "duplicate":
        workload["requests"][1]["id"] = "a"
    elif mutation == "zero_budget":
        workload["requests"][0]["max_tokens"] = 0
    else:
        workload["requests"][0]["input_ids"] = [-1]
    with pytest.raises(ValueError):
        bench.validate_workload(workload)


def reports(tmp_path):
    base = {"complete": True, "kind": "performance", "repeat_outputs_equal": True,
            "workload_sha256": "same", "model": {"path": "same"},
            "environment": {"engine": "lumina", "cuda_visible_devices": "0,1"},
            "args": {"tp": 2, "capacity": 4, "max_model_len": 4096, "token_budget": 512,
                     "memory": .85, "eager": False, "warmups": 1, "repeats": 2},
            "phases": [{"summary": {"output_tokens": 6, "wall_seconds": 2.},
                        "requests": {"a": {"tokens": [1, 2, 3]}, "b": {"tokens": [4, 5, 6]}}},
                       {"summary": {"output_tokens": 6, "wall_seconds": 4.}}]}
    baseline = deepcopy(base)
    baseline["environment"]["engine"] = "vllm"
    baseline["phases"][0]["requests"]["b"]["tokens"] = [4, 5, 7]
    return base, baseline, SimpleNamespace(lumina=tmp_path/"l.json", vllm=tmp_path/"v.json", output=tmp_path/"comparison.json")


def test_comparison_pools_time_and_reports_output_divergence(tmp_path):
    left, right, args = reports(tmp_path)
    bench.write_json(args.lumina, left)
    bench.write_json(args.vllm, right)
    bench.compare(args)
    result = json.loads(args.output.read_text())
    assert result["lumina_output_tok_s"] == 2.  # not arithmetic mean of 3 and 1.5
    assert result["exact_sequences"] == 1 and result["sequences"] == 2
    assert result["output_pairs"][1]["common_prefix"] == 2
    with pytest.raises(FileExistsError):
        bench.compare(args)


@pytest.mark.parametrize("mutation", ["workload", "profile", "tp", "gpu", "incomplete", "unstable"])
def test_comparison_rejects_invalid_pairs(tmp_path, mutation):
    left, right, args = reports(tmp_path)
    if mutation == "workload":
        right["workload_sha256"] = "different"
    elif mutation == "profile":
        right["kind"] = "profile"
    elif mutation == "tp":
        right["args"]["tp"] = 4
    elif mutation == "gpu":
        right["environment"]["cuda_visible_devices"] = "2,3"
    elif mutation == "incomplete":
        right["complete"] = False
    else:
        right["repeat_outputs_equal"] = False
    bench.write_json(args.lumina, left)
    bench.write_json(args.vllm, right)
    with pytest.raises(ValueError):
        bench.compare(args)


def test_busy_gpu_guard_checks_selected_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    def query(argv, **kw):
        return "0, GPU-a\n1, GPU-b\n2, GPU-c\n" if "--query-gpu=index,uuid" in argv else "GPU-c, 55\n"
    monkeypatch.setattr(bench.subprocess, "check_output", query)
    bench.require_idle_gpus(2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    with pytest.raises(RuntimeError, match="existing compute"):
        bench.require_idle_gpus(2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    with pytest.raises(ValueError, match="duplicate"):
        bench.require_idle_gpus(2)


def test_lumina_adapter_preserves_prefill_and_completion_stream():
    adapter = bench.LuminaAdapter.__new__(bench.LuminaAdapter)
    adapter.ids, adapter.tokens = {10: "a", 11: "b"}, {10: [1], 11: []}
    adapter.engine = SimpleNamespace(step=lambda: ([(11, [8])], 2, [(10, 2), (11, 8)]))
    assert adapter.step() == [("a", [1, 2], False), ("b", [8], True)]


def test_vllm_adapter_keeps_cumulative_tokens():
    adapter = bench.VllmAdapter.__new__(bench.VllmAdapter)
    adapter.engine = SimpleNamespace(step=lambda: [SimpleNamespace(request_id="a", finished=True,
                                               outputs=[SimpleNamespace(token_ids=[1, 2, 3])])])
    assert adapter.step() == [("a", [1, 2, 3], True)]


def test_vllm_adapter_uses_engine_core_shutdown():
    adapter = bench.VllmAdapter.__new__(bench.VllmAdapter)
    closed = []
    adapter.engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: closed.append(True)))
    adapter.close()
    assert closed == [True]
    adapter.engine = SimpleNamespace()
    with pytest.raises(RuntimeError, match="shutdown"):
        adapter.close()


def test_runner_orchestration_warmup_exclusion_and_output_files(tmp_path, monkeypatch):
    workload = {"schema": bench.SCHEMA, "arrival": "simultaneous-burst", "sampling": "greedy-ignore-eos",
                "model": {"test": "identity"}, "requests": requests()}
    path = tmp_path/"workload.json"
    bench.write_json(path, workload)
    monkeypatch.setattr(bench, "model_identity", lambda _: workload["model"])
    monkeypatch.setattr(bench, "environment", lambda _: {"engine": "lumina"})
    monkeypatch.setattr(bench, "require_idle_gpus", lambda _: None)
    fake_torch = SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None,
        version=SimpleNamespace(cuda="test"), cuda=SimpleNamespace(synchronize=lambda: None,
        get_device_name=lambda _: "mock GPU", nvtx=SimpleNamespace(range=lambda _: nullcontext())))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    closed = []
    class Adapter:
        def __init__(self, args):
            self.engine = SimpleNamespace(model_runner=SimpleNamespace(config="test", call=lambda _: []))
        def add(self, rows):
            self.rows, self.index = rows, 0
        def step(self):
            self.index += 1
            return [(r["id"], list(range(min(self.index, r["max_tokens"]))), self.index == r["max_tokens"])
                    for r in self.rows if self.index <= r["max_tokens"]]
        def done(self):
            return self.index >= 3
        def clear(self):
            pass
        def close(self):
            closed.append(True)
    monkeypatch.setattr(bench, "LuminaAdapter", Adapter)
    args = SimpleNamespace(workload=path, model="mock", output=tmp_path/"run", engine="lumina", tp=2,
                           max_model_len=32, profile=False, warmups=1, repeats=2)
    bench.run(args)
    report = json.loads((args.output/"result.json").read_text())
    assert report["complete"] and report["shutdown_called"] and report["repeat_outputs_equal"]
    assert len(report["warmups"]) == 1 and len(report["phases"]) == 2
    assert all(p["summary"]["output_tokens"] == 4 for p in report["phases"])
    assert report["source_sha256"] and closed == [True]
