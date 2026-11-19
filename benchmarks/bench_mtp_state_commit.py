"""Paired native-engine measurements of replay vs intermediate state commit.

One resident engine per process. C>1 compares ordinary decode and speculative
cohorts with the same prefill scheduling and single-token decode arithmetic.
No target answers are used to generate real checkpoint MTP drafts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

from zjhvllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/data/jhzhou/huggingface/Qwen3.5-27B")
    parser.add_argument("--workload", type=Path, default=Path("benchmarks/qwen35_mtp_8gpu_20261106/workload.json"))
    parser.add_argument("--ids", nargs="*", type=int, default=[2, 39, 76, 127])
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--port", type=int, default=2371)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-graph", action="store_true", help="C1 eager and graph in one resident engine")
    parser.add_argument("--graph-control", action="store_true", help="add no-draft Graph verification control (not native decode)")
    parser.add_argument("--linear-backend", choices=["decode_equivalent", "batch_invariant"], default="decode_equivalent")
    parser.add_argument("--compare-legacy", action="store_true", help="also retain the old cuBLAS normal-decode reference")
    args = parser.parse_args()
    if args.output.exists() or args.capacity < 1 or args.repeats < 1:
        parser.error("require new output path, positive capacity and repeats")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workload = json.loads(args.workload.read_text())
    items = [row for row in workload if not args.ids or row["id"] in args.ids]
    if args.ids and {row["id"] for row in items} != set(args.ids):
        parser.error("unknown request ID")
    if args.ids:
        if len(set(args.ids)) != len(args.ids):
            parser.error("request IDs must be distinct")
        by_id = {row["id"]: row for row in items}
        items = [by_id[request_id] for request_id in args.ids]
    args.output.mkdir(parents=True)
    for name in ("zjhvllm", "tests"):
        shutil.copytree(name, args.output / "source" / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(__file__, args.output / "source/benchmark.py")
    modes = ["baseline", "replay", "intermediate"] if args.capacity == 1 else ["baseline", "intermediate_batch"]
    if args.compare_graph:
        if args.capacity != 1:
            parser.error("graph comparison currently requires capacity=1")
        modes = ["baseline", "intermediate", "graph"]
        if args.graph_control:
            modes.append("graph_no_draft_control")
    elif args.graph_control:
        parser.error("--graph-control requires --compare-graph")
    if args.compare_legacy:
        modes.insert(0, "legacy_baseline")
    report = {"capacity": args.capacity, "repeats": args.repeats, "measurement_complete": False,
              "args": vars(args) | {"workload": str(args.workload), "output": str(args.output)},
              "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "baseline_contract": f"MTP weights resident; same fixed cohorts; linear backend={args.linear_backend}; legacy comparison separately retained",
              "numerical_validation_passed": False, "gpu": torch.cuda.get_device_name(),
              "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(args.output / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.output / "source").rglob("*.py")}, "phases": []}
    (args.output / "workload.json").write_text(json.dumps(items, indent=2) + "\n")
    engine = None
    try:
        engine = LLM(args.model, enable_mtp=True, max_num_seqs=args.capacity,
                     max_model_len=4096, max_num_batched_tokens=256, enforce_eager=True,
                     qwen35_linear_backend=args.linear_backend,
                     gpu_memory_utilization=.85, distributed_port=args.port)
        runner = engine.model_runner
        pages = len(engine.scheduler.block_manager.free_block_ids)
        slots = len(engine.scheduler.free_state_slots)
        report["state_history_bytes"] = runner.verification_history.nbytes
        params = [SamplingParams(greedy=True, max_tokens=min(args.tokens, row["max_tokens"]), ignore_eos=False)
                  for row in items]
        prompts = [row["prompt_token_ids"] for row in items]

        def run(mode, tokens=prompts, sampling=params):
            if mode == "legacy_baseline":
                from zjhvllm.models.qwen3_5 import Qwen3_5Linear
                modules = [m for m in runner.model.modules() if isinstance(m, Qwen3_5Linear)]
                for module in modules:
                    module.backend = "decode_equivalent"
                try:
                    return run("baseline", tokens, sampling)
                finally:
                    for module in modules:
                        module.backend = args.linear_backend
            if mode == "baseline":
                # Fixed cohorts also used by speculative generation: avoid
                # conflating refill policy with speculative execution.
                outputs = []
                for start in range(0, len(tokens), args.capacity):
                    outputs.extend(engine.generate(tokens[start:start + args.capacity],
                                                    sampling[start:start + args.capacity], use_tqdm=False))
                return outputs
            if mode == "graph_no_draft_control":
                # Diagnostic control: same scratch/checkpoint/commit machinery,
                # but one captured target input and no MTP draft computation.
                # This is deliberately not labelled native graphed decode.
                from zjhvllm.engine.verification_graph import VerificationGraph
                original = runner.run_verification
                def one_row(ids, positions, history):
                    graph = getattr(runner, "control_graph", None)
                    if graph is None or graph.history is not history:
                        max_blocks = (runner.config.max_model_len + runner.block_size - 1) // runner.block_size
                        graph = VerificationGraph(runner.model, ids, positions, history, max_blocks)
                        runner.control_graph = graph
                    return graph.replay(ids, positions)
                runner.run_verification = one_row
                outputs = []
                try:
                    for prompt, param in zip(tokens, sampling):
                        seq_id = engine.add_request(prompt, param)
                        seq = engine.seq_map[seq_id]
                        rounds = []
                        while not seq.is_finished:
                            if not seq.num_completion_tokens:
                                engine.step()
                            else:
                                _, stats = engine.speculative_step(seq_id, [], verification_mode="batched_experimental")
                                rounds.append(stats)
                        outputs.append({"token_ids": list(seq.completion_token_ids), "rounds": rounds})
                        engine.seq_map.pop(seq_id)
                    return outputs
                finally:
                    runner.run_verification = original
            if mode == "intermediate_batch":
                return engine.generate_speculative_batch(tokens, sampling)
            runner.config.verification_state_commit = mode == "intermediate"
            if mode == "graph":
                runner.config.verification_state_commit = True
            runner.config.verification_cuda_graph = mode == "graph"
            try:
                return [engine.generate_speculative(prompt, param, verification_mode="batched_experimental")
                        for prompt, param in zip(tokens, sampling)]
            finally:
                runner.config.verification_state_commit = True
                runner.config.verification_cuda_graph = False

        warm_tokens = [items[0]["prompt_token_ids"]] * args.capacity
        warm_params = [SamplingParams(greedy=True, max_tokens=6)] * args.capacity
        for mode in modes:
            run(mode, warm_tokens, warm_params)
        engine.seq_map.clear()
        for repeat in range(args.repeats):
            for mode in modes if repeat % 2 == 0 else reversed(modes):
                torch.cuda.synchronize()
                graph = getattr(runner, "verification_graph", None)
                prior_replays = graph.replays if graph is not None else 0
                control = getattr(runner, "control_graph", None)
                prior_control_replays = control.replays if control is not None else 0
                begin = perf_counter()
                outputs = run(mode)
                if len(outputs) != len(items):
                    raise RuntimeError("generation returned an incomplete request set")
                torch.cuda.synchronize()
                seconds = perf_counter() - begin
                rounds = [r for output in outputs for r in output.get("rounds", [])]
                phase = {"repeat": repeat, "mode": mode, "seconds": seconds,
                         "graph_replays": (getattr(runner.verification_graph, "replays", 0) - prior_replays)
                            if hasattr(runner, "verification_graph") else 0,
                         "control_graph_replays": (runner.control_graph.replays - prior_control_replays)
                            if hasattr(runner, "control_graph") else 0,
                         "output_tokens": sum(len(output["token_ids"]) for output in outputs),
                         "rows": [{"id": item["id"], "token_ids": output["token_ids"]}
                                  for item, output in zip(items, outputs)],
                         "proposed": sum(r["proposed"] for r in rounds),
                         "accepted": sum(r["accepted"] for r in rounds),
                         "replay_tokens": sum(r["replay_tokens"] for r in rounds),
                         "target_verification_forwards": sum(r["target_forwards"] for r in rounds),
                         "rounds": [output.get("rounds", []) for output in outputs],
                         "resources_reclaimed": (len(engine.scheduler.block_manager.free_block_ids) == pages
                                                 and len(engine.scheduler.free_state_slots) == slots)}
                phase["output_tok_s"] = phase["output_tokens"] / seconds
                report["phases"].append(phase)
                (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"C{args.capacity} repeat{repeat} {mode}: {phase['output_tok_s']:.3f} tok/s, "
                      f"replay={phase['replay_tokens']} resources={phase['resources_reclaimed']}", flush=True)
        baselines = {p["repeat"]: {r["id"]: r["token_ids"] for r in p["rows"]}
                     for p in report["phases"] if p["mode"] == "baseline"}
        report["legacy_mismatches"] = [{"repeat": p["repeat"], "id": r["id"]}
                                for p in report["phases"] if p["mode"] == "legacy_baseline" for r in p["rows"]
                                if r["token_ids"] != baselines[p["repeat"]][r["id"]]]
        report["mismatches"] = [{"repeat": p["repeat"], "mode": p["mode"], "id": r["id"]}
                                for p in report["phases"] for r in p["rows"]
                                if p["mode"] != "legacy_baseline"
                                if r["token_ids"] != baselines[p["repeat"]][r["id"]]]
        report["baseline_cross_repeat_passed"] = all(b == baselines[0] for b in baselines.values())
        report["legacy_compatibility_passed"] = (not report["legacy_mismatches"]) if args.compare_legacy else None
        report["measurement_complete"] = True
        report["numerical_validation_passed"] = (not report["mismatches"]
            and report["baseline_cross_repeat_passed"] and all(p["resources_reclaimed"] for p in report["phases"]))
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if engine is not None:
            engine.exit()
    if not report["numerical_validation_passed"]:
        raise SystemExit("strict paired verification failed")
    if report["legacy_compatibility_passed"] is False:
        raise SystemExit("new arithmetic differs from legacy cuBLAS; legacy mismatch records preserved")


if __name__ == "__main__":
    main()
