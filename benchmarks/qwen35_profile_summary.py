"""Summarize captured Nsight SQLite GPU kernels without double-counting overlap.

Input must contain the measured capture only. Kernel category attribution is a
name-based heuristic; ambiguous GEMMs are not assigned to MLP or attention.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative kernel duration")
        if end is None or start > end:
            total += stop-start
        elif stop > end:
            total += stop-end
        end = max(end, stop) if end is not None else stop
    return total


def category(name):
    name = name.lower()
    if "nccl" in name or "allreduce" in name or "all_reduce" in name:
        return "communication"
    if any(part in name for part in ("gated_delta", "gdn", "delta_rule", "causal_conv", "chunk_scan")):
        return "gdn_or_conv"
    if any(part in name for part in ("flash", "paged_attention", "fmha", "attention")):
        return "attention"
    if any(part in name for part in ("gemm", "matmul", "cublas", "_linear", "xmma")):
        return "gemm_unattributed"
    if any(part in name for part in ("norm", "silu", "rotary", "rope", "elementwise")):
        return "norm_or_elementwise"
    return "other"


def summarize(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables or "StringIds" not in tables:
            raise ValueError("Nsight kernel/StringIds tables missing; export a CUDA kernel trace")
        columns = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        name_col = next((name for name in ("demangledName", "shortName", "name") if name in columns), None)
        if name_col is None or not {"start", "end", "deviceId"} <= columns:
            raise ValueError("unsupported Nsight kernel schema")
        # Column identifier is selected exclusively from the fixed whitelist above.
        query = f'SELECT k.start,k.end,k.deviceId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k."{name_col}"=s.id'
        rows = list(db.execute(query))
    if not rows:
        raise ValueError("trace contains no GPU kernels")
    devices = defaultdict(list)
    kernels = defaultdict(lambda: {"calls": 0, "sum_ns": 0, "max_ns": 0})
    for start, end, device, name in rows:
        if name is None:
            raise ValueError("kernel name did not resolve through StringIds")
        devices[device].append((start, end))
        key = (device, name)
        item = kernels[key]
        duration = end-start
        if duration < 0:
            raise ValueError("negative kernel duration")
        item["calls"] += 1
        item["sum_ns"] += duration
        item["max_ns"] = max(item["max_ns"], duration)
    result = {"kind": "diagnostic-profile", "devices": {}, "kernels": [],
              "notes": ["Input must exclude model loading, graph capture and warm-up.",
                        "Kernel duration sums may overlap across streams; they are not wall time.",
                        "Coverage uses first-to-last kernel span, excludes memcpy and CPU boundary gaps; it is not device utilization.",
                        "Categories are heuristics, not operator/architecture attribution; inspect NVTX and call stacks.",
                        "Trace device coverage must be checked against TP rank count before drawing multi-GPU conclusions."]}
    for device, intervals in sorted(devices.items()):
        span = max(end for _, end in intervals)-min(start for start, _ in intervals)
        covered = union_ns(intervals)
        categories = defaultdict(int)
        for (dev, name), item in kernels.items():
            if dev == device:
                categories[category(name)] += item["sum_ns"]
        result["devices"][str(device)] = {"kernel_calls": len(intervals), "kernel_span_ns": span,
                                           "kernel_union_ns": covered, "kernel_gap_ns": span-covered,
                                           "kernel_span_coverage": covered/span if span else None,
                                           "category_sum_ns_overlapping": dict(categories)}
    for (device, name), item in sorted(kernels.items(), key=lambda kv: kv[1]["sum_ns"], reverse=True):
        result["kernels"].append({"device": device, "name": name, "category": category(name), **item})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = summarize(args.sqlite)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
