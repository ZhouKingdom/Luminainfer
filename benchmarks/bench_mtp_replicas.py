"""Eight independent MTP-1 replicas, paired outputs, C=1 per GPU.

Closed-loop native-engine latency; no HTTP, queue-backlog or TP measurement.
Experimental token mismatches invalidate speedup acceptance, but are retained.
"""
import argparse
import ast
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import time
import traceback

from benchmarks.service_metrics import quantiles


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def make_workload(model, math_path, seed):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    root = Path(__file__).resolve().parents[2]
    rng = random.Random(seed)
    qa_path = root / "AletheiaRAG-main/eval_data/qa_100.jsonl"
    qa = [json.loads(line) for line in qa_path.read_text().splitlines() if line]
    rng.shuffle(qa)
    math = [json.loads(line) for line in math_path.read_text().splitlines() if line]
    if len(math) != 30:
        raise ValueError("expected 30 distinct local AIME-2025 problems")
    items = []

    def add(category, text, source, limit=None):
        if limit:
            # Bound the content, retaining the chat template and question.
            text = tokenizer.decode(tokenizer.encode(text)[:limit])
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if not isinstance(ids, list) or not ids or not all(isinstance(token, int) for token in ids):
            raise TypeError("chat template must return a nonempty token-id list")
        index = len(items)
        items.append({"id": index, "category": category, "source": source,
                      "prompt_token_ids": ids, "input_tokens": len(ids),
                      "max_tokens": 64 if index % 2 == 0 else 128})

    for row in qa[:34]:
        add("natural", row["question"] + " Explain your answer clearly and briefly.", str(qa_path) + "#" + row["id"])
    for row in math:
        add("math", row["text"] + "\nGive a concise derivation and the final answer.", row["url"])
    code_candidates = []
    for path in sorted((root / "Lumina/zjhvllm").rglob("*.py")):
        source = path.read_text()
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno - node.lineno >= 12:
                prefix = "\n".join(lines[:node.lineno + 2])[-7000:]
                code_candidates.append((str(path) + ":" + str(node.lineno), prefix))
    rng.shuffle(code_candidates)
    if len(code_candidates) < 32:
        raise ValueError("not enough distinct source functions")
    for source, prefix in code_candidates[:32]:
        prefix = tokenizer.decode(tokenizer.encode(prefix)[-960:])
        add("code", "Complete this Python function using the surrounding code. Return only the continuation.\n```python\n" + prefix,
            source)
    # Keep the pre-cleanup corpus stable when active documentation changes.
    from zipfile import ZipFile
    with ZipFile(root / "Lumina/benchmarks/文档历史归档_20261107.zip") as archive:
        docs = "\n\n".join(archive.read(name).decode("utf-8") for name in
                              ("README.md", "TUTORIAL.md", "SPECULATIVE_DECODING.md"))
    document = tokenizer.encode(docs)
    for index in range(32):
        length = [255, 256, 257, 511, 512, 513, 1024, 2048][index % 8]
        start = (index * 193) % max(1, len(document) - 2048)
        text = tokenizer.decode(document[start:start + length])
        add("long_context", "Read this technical excerpt and explain its main mechanism, limitations and validation evidence.\n\n" + text,
            "local Lumina documentation, offset=" + str(start))
    if len(items) != 128 or len({tuple(r["prompt_token_ids"]) for r in items}) != 128:
        seen, duplicates = {}, []
        for item in items:
            key = tuple(item["prompt_token_ids"])
            if key in seen:
                duplicates.append((item["source"], seen[key]))
            seen[key] = item["source"]
        raise ValueError(f"workload must contain 128 distinct prompts: count={len(items)}, duplicates={duplicates}")
    return items


def worker(connection, gpu, port, options):
    os.setsid()  # Parent cleanup owns this group, including future children.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from zjhvllm import LLM, SamplingParams
    engine = None
    try:
        engine = LLM(options["model"], enable_mtp=True, max_num_seqs=1, max_model_len=4096,
                     max_num_batched_tokens=256, enforce_eager=True,
                     verification_cuda_graph=options.get("verification_cuda_graph", False),
                     gpu_memory_utilization=.85, distributed_port=port)
        for mode in ["baseline", "serial", "batched_experimental"]:
            params = SamplingParams(max_tokens=6, ignore_eos=False)
            if mode == "baseline":
                engine.generate(["Explain a hash table."], params, use_tqdm=False)
            else:
                engine.generate_speculative("Explain a hash table.", params, verification_mode=mode)
        engine.seq_map.clear()
        manager = engine.scheduler.block_manager
        free_pages = len(manager.free_block_ids)
        free_slots = len(engine.scheduler.free_state_slots)
        connection.send({"type": "ready", "gpu": gpu, "pid": os.getpid(), "gpu_name": torch.cuda.get_device_name(),
                         "mtp_weights": len(engine.model_runner.model.mtp.loaded_weight_names),
                         "kv_blocks": free_pages, "state_slots": free_slots})
        while True:
            job = connection.recv()
            if job is None:
                break
            mode = job["mode"]
            path = Path(options["output"]) / f"{job['name']}.gpu{gpu}.jsonl"
            while time.perf_counter() < job["start_at"]:
                time.sleep(.01)
            torch.cuda.reset_peak_memory_stats()
            with path.open("w") as output:
                for item in job["requests"]:
                    params = SamplingParams(greedy=True, max_tokens=item["max_tokens"], ignore_eos=False)
                    seq_id = engine.add_request(item["prompt_token_ids"], params)
                    seq = engine.seq_map[seq_id]
                    proposed = accepted = replay = forwards = 0
                    prefill_forwards = 0
                    delivery_gaps, round_times = [], []
                    previous_delivery = None
                    while not seq.is_finished:
                        start = time.perf_counter()
                        before = seq.num_completion_tokens
                        if not before or mode == "baseline" or seq.max_tokens - before <= 1:
                            engine.step()
                            forwards += 1
                            prefill_forwards += int(engine.last_step_info["is_prefill"])
                        else:
                            draft, _ = engine.model_runner.call("draft_mtp", seq)
                            _, stats = engine.speculative_step(seq_id, draft, verification_mode=mode)
                            proposed += stats["proposed"]
                            accepted += stats["accepted"]
                            replay += stats["replay_tokens"]
                            forwards += stats["target_forwards"]
                        now = time.perf_counter()
                        round_times.append(now - start)
                        if seq.num_completion_tokens > before:
                            if previous_delivery is not None:
                                delivery_gaps.append(now - previous_delivery)
                            previous_delivery = now
                    torch.cuda.synchronize()
                    tokens = list(seq.completion_token_ids)
                    clean = (len(manager.free_block_ids) == free_pages and len(engine.scheduler.free_state_slots) == free_slots)
                    row = {"id": item["id"], "category": item["category"], "gpu": gpu, "mode": mode,
                           "repeat": job["repeat"], "input_tokens": item["input_tokens"], "max_tokens": item["max_tokens"],
                           "output_tokens": len(tokens), "token_ids": tokens, "started_at": seq.arrival_time,
                           "finished_at": seq.finished_time, "ttft_seconds": seq.first_token_time - seq.arrival_time,
                           "e2e_seconds": seq.finished_time - seq.arrival_time,
                           "tpot_seconds": ((seq.token_times[-1] - seq.token_times[0]) / (len(tokens) - 1)) if len(tokens) > 1 else None,
                           "delivery_gaps_seconds": delivery_gaps, "round_seconds": round_times,
                           "proposed": proposed, "accepted": accepted, "replay_tokens": replay,
                           "target_forwards": forwards, "prefill_forwards": prefill_forwards, "resources_reclaimed": clean}
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    engine.seq_map.pop(seq_id)
                    connection.send({"type": "progress", "gpu": gpu, "phase": job["name"], "id": item["id"],
                                     "output_tokens": len(tokens), "resources_reclaimed": clean})
                    if not clean:
                        raise RuntimeError("KV/state slot leak")
            connection.send({"type": "done", "gpu": gpu, "phase": job["name"],
                             "peak_bytes": torch.cuda.max_memory_allocated(), "path": str(path)})
    except BaseException:
        connection.send({"type": "error", "gpu": gpu, "traceback": traceback.format_exc()})
    finally:
        if engine is not None:
            engine.exit()
        connection.close()


def free_ports(count):
    sockets = []
    for _ in range(count):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    return ports


def summarize(directory, phases):
    rows = [json.loads(line) for phase in phases for path in directory.glob(phase["name"] + ".gpu*.jsonl")
            for line in path.read_text().splitlines()]
    baseline = {(r["repeat"], r["id"]): r for r in rows if r["mode"] == "baseline"}
    report = {"replicas": 8, "concurrency_per_replica": 1, "total_concurrency": 8,
              "requests": len(rows), "phases": phases, "modes": {}, "mismatches": []}
    for mode in ["baseline", "serial", "batched_experimental"]:
        selected = [r for r in rows if r["mode"] == mode]
        if not selected:
            continue
        paired = [r for r in selected if (r["repeat"], r["id"]) in baseline]
        mismatch = [r for r in paired if r["token_ids"] != baseline[(r["repeat"], r["id"])]["token_ids"]]
        for row in mismatch:
            expected = baseline[(row["repeat"], row["id"])]["token_ids"]
            first = next((i for i, (a, b) in enumerate(zip(expected, row["token_ids"])) if a != b), min(len(expected), len(row["token_ids"])))
            report["mismatches"].append({"mode": mode, "repeat": row["repeat"], "id": row["id"],
                                         "category": row["category"], "first_divergence": first})
        wall = sum(p["wall_seconds"] for p in phases if p["mode"] == mode)
        proposed = sum(r["proposed"] for r in selected)
        summary = {"requests": len(selected), "paired_requests": len(paired), "exact_requests": len(paired) - len(mismatch),
                   "resources_reclaimed": all(r["resources_reclaimed"] for r in selected),
                   "output_tokens": sum(r["output_tokens"] for r in selected), "wall_seconds": wall,
                   "ttft_seconds": quantiles([r["ttft_seconds"] for r in selected]),
                   "tpot_seconds": quantiles([r["tpot_seconds"] for r in selected if r["tpot_seconds"] is not None]),
                   "e2e_seconds": quantiles([r["e2e_seconds"] for r in selected]),
                   "delivery_gap_seconds": quantiles([v for r in selected for v in r["delivery_gaps_seconds"]]),
                   "proposed": proposed, "accepted": sum(r["accepted"] for r in selected),
                   "replay_tokens": sum(r["replay_tokens"] for r in selected),
                   "target_forwards": sum(r["target_forwards"] for r in selected),
                   "by_category": {category: {"requests": len([r for r in paired if r["category"] == category]),
                                                "mismatches": len([r for r in mismatch if r["category"] == category])}
                                   for category in sorted({r["category"] for r in selected})}}
        summary["output_tokens_per_second"] = summary["output_tokens"] / wall
        summary["acceptance_rate"] = summary["accepted"] / proposed if proposed else None
        summary["speedup_accepted"] = len(paired) == len(selected) and not mismatch and summary["resources_reclaimed"]
        report["modes"][mode] = summary
    repeats = sorted({r["repeat"] for r in rows})
    ids = sorted({r["id"] for r in rows})
    report["baseline_cross_repeat_mismatches"] = [i for i in ids if len({tuple(baseline[(repeat, i)]["token_ids"])
        for repeat in repeats if (repeat, i) in baseline}) > 1]
    report["tail_note"] = "256 samples/mode: empirical p99, not production SLA; delivery-gap is per emission burst, not token ITL"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/data/jhzhou/huggingface/Qwen3.5-27B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--math-input", type=Path, default=Path("benchmarks/qwen35_mtp_scale_inputs/math.jsonl"))
    parser.add_argument("--gpus", type=int, nargs=8, default=list(range(8)))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--workload", type=Path, help="Replay exact archived token IDs instead of rebuilding inputs")
    parser.add_argument("--verification-cuda-graph", action="store_true", help="opt-in C1 MTP-1 graph replay")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or len(set(args.gpus)) != 8:
        parser.error("require a new output directory, positive repeats, and 8 distinct GPUs")
    args.output.mkdir(parents=True)
    if args.workload is None:
        items = make_workload(args.model, args.math_input, args.seed)
    else:
        items = json.loads(args.workload.read_text())
        if (not isinstance(items, list) or len(items) != 128
                or {r["id"] for r in items} != set(range(128))
                or len({tuple(r["prompt_token_ids"]) for r in items}) != 128):
            parser.error("archived workload must contain IDs 0..127 and 128 distinct prompts")
        for row in items:
            tokens = row["prompt_token_ids"]
            if (not tokens or any(type(t) is not int or t < 0 for t in tokens)
                    or row["input_tokens"] != len(tokens) or type(row["max_tokens"]) is not int
                    or row["max_tokens"] < 1 or len(tokens) + row["max_tokens"] > 4096):
                parser.error(f"invalid archived input: {row['id']}")
    write_json(args.output / "workload.json", items)
    shutil.copytree("zjhvllm", args.output / "source/zjhvllm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, args.output / "source/bench_mtp_replicas.py")
    write_json(args.output / "manifest.json", {"args": vars(args) | {"output": str(args.output), "math_input": str(args.math_input),
                                                                   "workload": str(args.workload) if args.workload else None},
               "workload_sha256": hashlib.sha256((args.output / "workload.json").read_bytes()).hexdigest(),
               "baseline": "normal decode, MTP weights resident; no HTTP; EOS obeyed",
               "verification_arithmetic": "tokenwise_decode_shapes",
               "verification_cuda_graph": args.verification_cuda_graph,
               "verification_state_strategy": "intermediate (bounded workspace; replay fallback for oversized windows)",
               "source_hashes": {str(p.relative_to(args.output / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.output / "source").rglob("*.py")}})
    context = mp.get_context("spawn")
    processes, pipes, phases, ready = [], [], [], []
    status = {"state": "loading", "completed_generations": 0, "planned_generations": len(items) * 3 * args.repeats}
    write_json(args.output / "status.json", status)
    try:
        for gpu, port in zip(args.gpus, free_ports(8)):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, gpu, port, {"model": args.model, "output": str(args.output),
                                                                           "verification_cuda_graph": args.verification_cuda_graph}))
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
        write_json(args.output / "processes.json", [{"pid": p.pid, "gpu": g} for p, g in zip(processes, args.gpus)])

        def receive_until(kind):
            pending = set(pipes)
            last_message = time.monotonic()
            while pending:
                for pipe in wait(pending, timeout=5):
                    message = pipe.recv()
                    last_message = time.monotonic()
                    if message["type"] == "error":
                        raise RuntimeError(str(message))
                    if message["type"] == "progress":
                        status["completed_generations"] += 1
                    elif message["type"] == kind:
                        pending.remove(pipe)
                        ready.append(message)
                    write_json(args.output / "status.json", status | {"last_message": message, "updated_at": time.time()})
                if any(not p.is_alive() for p in processes):
                    raise RuntimeError("worker exited unexpectedly")
                if time.monotonic() - last_message > 300:
                    raise TimeoutError("no worker progress for 300 seconds")

        receive_until("ready")
        write_json(args.output / "workers.json", ready)
        ready.clear()
        for repeat in range(args.repeats):
            # Cost-balanced shards, then rotate physical GPUs between repeats.
            shards, loads = [[] for _ in args.gpus], [0.] * 8
            for item in sorted(items, key=lambda r: r["max_tokens"] + .03 * r["input_tokens"], reverse=True):
                index = min(range(8), key=lambda i: loads[i])
                shards[index].append(item)
                loads[index] += item["max_tokens"] + .03 * item["input_tokens"]
            for shard in shards:
                random.Random(args.seed + repeat).shuffle(shard)
            modes = ["baseline", "serial", "batched_experimental"] if repeat % 2 == 0 else ["batched_experimental", "serial", "baseline"]
            for mode in modes:
                name = f"repeat{repeat}_{mode}"
                status.update(state="running", phase=name)
                print(name, "starting", flush=True)
                start = time.perf_counter() + 1
                for index, pipe in enumerate(pipes):
                    pipe.send({"name": name, "mode": mode, "repeat": repeat, "start_at": start,
                               "requests": shards[(index + repeat) % 8]})
                receive_until("done")
                rows = [json.loads(line) for message in ready for line in Path(message["path"]).read_text().splitlines()]
                phase = {"name": name, "mode": mode, "repeat": repeat, "requests": len(rows),
                         "wall_seconds": max(r["finished_at"] for r in rows) - start,
                         "peak_bytes_by_gpu": {m["gpu"]: m["peak_bytes"] for m in ready}}
                phases.append(phase)
                ready.clear()
                write_json(args.output / "summary.json", summarize(args.output, phases))
                print(name, "finished", phase["wall_seconds"], flush=True)
        status["state"] = "complete"
    except BaseException:
        status.update(state="failed", traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output / "status.json", status | {"updated_at": time.time()})
        for pipe in pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=5)
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            for process in processes:
                try:
                    # Even an exited leader may leave children in its group.
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
            for process in processes:
                process.join(timeout=2)
        for pipe in pipes:
            pipe.close()


if __name__ == "__main__":
    main()
