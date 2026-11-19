"""Prevent speculative mismatches from becoming accepted throughput results."""
import json

import pytest

from benchmarks.bench_mtp_replicas import summarize


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def test_paired_validation_controls_performance_acceptance(tmp_path):
    phases = []
    for repeat in range(2):
        for mode in ["baseline", "serial", "batched_experimental"]:
            name = f"r{repeat}_{mode}"
            phases.append({"name": name, "mode": mode, "repeat": repeat, "wall_seconds": 10})
            tokens = [1, 3] if mode == "batched_experimental" else [1, 2]
            row = {"id": 0, "repeat": repeat, "mode": mode, "category": "natural", "gpu": repeat,
                   "token_ids": tokens, "output_tokens": len(tokens), "resources_reclaimed": True,
                   "ttft_seconds": .2, "e2e_seconds": 1.2, "tpot_seconds": 1.,
                   "delivery_gaps_seconds": [1.], "proposed": int(mode != "baseline"),
                   "accepted": int(mode != "baseline"), "replay_tokens": 0, "target_forwards": 2}
            (tmp_path / f"{name}.gpu{repeat}.jsonl").write_text(json.dumps(row) + "\n")
    report = summarize(tmp_path, phases)
    assert report["requests"] == 6
    assert report["modes"]["serial"]["speedup_accepted"]
    batch = report["modes"]["batched_experimental"]
    assert not batch["speedup_accepted"] and batch["exact_requests"] == 0
    assert batch["output_tokens_per_second"] == .2  # combined phase walls, not summed GPU rates
    assert len(report["mismatches"]) == 2 and report["mismatches"][0]["first_divergence"] == 1
    assert report["baseline_cross_repeat_mismatches"] == []
    assert batch["by_category"]["natural"] == {"requests": 2, "mismatches": 2}
