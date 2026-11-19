"""Shared offline Qwen3.5 benchmark. No framework imports until run/prepare.

Run in each engine's own environment. A burst is an offline engine test, not
HTTP serving latency, a fixed active batch, or an open-loop arrival process.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "qwen35-offline-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_identity(model):
    model = Path(model).resolve()
    names = ["config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    return {"path": str(model), "files": {name: digest(model/name) for name in names if (model/name).exists()},
            "weights": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                        for p in sorted(model.glob("*.safetensors"))],
            "weight_identity_note": "weight inventory/mtime plus index hash; not full weight-content hashes"}


def environment(engine):
    versions = {}
    for name in ("vllm", "torch", "transformers", "triton", "flash-attn"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    supported = None
    if engine == "vllm" and versions["vllm"]:
        package = metadata.distribution("vllm")
        supported = any(Path(package.locate_file(f"vllm/model_executor/models/{name}.py")).exists()
                        for name in ("qwen3_5", "qwen3_5_text"))
    def command(argv):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
            return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"error": str(error)}
    return {"python": sys.executable, "python_version": platform.python_version(),
            "packages": versions, "engine": engine, "vllm_qwen35_source_present": supported,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"]),
            "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])}


def require_idle_gpus(tp):
    """Fail before CUDA init on a shared machine; never terminate other jobs."""
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != tp or any(not item.strip() for item in visible):
        raise ValueError("set CUDA_VISIBLE_DEVICES explicitly to exactly TP physical IDs/UUIDs")
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {}
    for row in gpu_rows.splitlines():
        index, uuid = (part.strip() for part in row.split(","))
        mapping[index] = uuid
        mapping[uuid] = uuid
    selected = [mapping[item.strip()] for item in visible]
    if len(set(selected)) != tp:
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate devices")
    process_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = [row for row in process_rows.splitlines() if row.split(",")[0].strip() in selected]
    if busy:
        raise RuntimeError(f"selected GPUs have existing compute processes: {busy}")


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = random.Random(args.seed)
    texts = ([line for line in args.texts.read_text().splitlines() if line.strip()] if args.texts else [
        "解释混合注意力模型中递归状态与 KV 缓存的区别，并给出具体例子。",
        "Write a binary search function and explain its time and space complexity.",
        "分析分布式推理中计算、内存带宽和通信开销之间的关系。",
        "Explain why throughput and latency measure different properties of an inference engine.",
    ])
    if not texts:
        raise ValueError("text corpus is empty")
    rows = []
    for index in range(args.requests):
        text = f"Request {index}: " + texts[rng.randrange(len(texts))]
        ids = tokenizer.encode(text, add_special_tokens=False)
        filler = tokenizer.encode(" " + text, add_special_tokens=False)
        if not ids or not filler:
            raise ValueError("tokenizer produced empty input")
        ids = (ids + filler * (1 + args.input_tokens // len(filler)))[:args.input_tokens]
        rows.append({"id": str(index), "input_ids": ids, "max_tokens": args.output_tokens})
    value = {"schema": SCHEMA, "model": model_identity(args.model), "seed": args.seed,
             "corpus": {"kind": "repeated/truncated natural text; synthetic performance workload",
                        "source_sha256": digest(args.texts) if args.texts else None},
             "requests": rows, "arrival": "simultaneous-burst", "sampling": "greedy-ignore-eos"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, value)


def validate_workload(workload):
    if (workload.get("schema") != SCHEMA or workload.get("arrival") != "simultaneous-burst"
            or workload.get("sampling") != "greedy-ignore-eos"):
        raise ValueError("unsupported benchmark contract")
    rows = workload["requests"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("requests must have unique IDs")
    for row in rows:
        if (not row["input_ids"] or type(row["max_tokens"]) is not int or row["max_tokens"] < 1
                or any(type(t) is not int or t < 0 for t in row["input_ids"])):
            raise ValueError("invalid request tokens or output budget")


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    def q(p):
        position = (len(values)-1)*p
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi]-values[lo])*(position-lo)
    return {"count": len(values), "mean": sum(values)/len(values),
            "p50": q(.5), "p95": q(.95), "p99": q(.99)}


class Recorder:
    """Timestamp actual output observations; never fabricate per-token times."""
    def __init__(self, rows, clock=perf_counter):
        self.clock = clock
        self.rows = {r["id"]: {"input_tokens": len(r["input_ids"]), "budget": r["max_tokens"],
                               "tokens": [], "observations": [], "finished": False} for r in rows}
        self.start = clock()

    def observe(self, request_id, tokens, finished):
        row = self.rows[request_id]
        if row["finished"]:
            raise ValueError("output after request completion")
        previous = row["tokens"]
        if len(tokens) < len(previous) or tokens[:len(previous)] != previous:
            raise ValueError("engine outputs must be cumulative, stable token IDs")
        now = self.clock()-self.start
        if len(tokens) > len(previous):
            row["observations"].append({"seconds": now, "tokens": len(tokens)-len(previous)})
            row["tokens"] = list(tokens)
        if finished:
            if len(tokens) != row["budget"]:
                raise ValueError("output length differs from fixed ignore-EOS budget")
            row["finished"] = True
            row["finished_seconds"] = now

    def summary(self, wall):
        if not all(r["finished"] for r in self.rows.values()):
            raise ValueError("incomplete run")
        rows = list(self.rows.values())
        output = sum(len(r["tokens"]) for r in rows)
        single = all(o["tokens"] == 1 for r in rows for o in r["observations"])
        ttft = [r["observations"][0]["seconds"] for r in rows]
        tpot = [(r["observations"][-1]["seconds"]-r["observations"][0]["seconds"])/(len(r["tokens"])-1)
                for r in rows if len(r["tokens"]) > 1] if single else []
        itl = [b["seconds"]-a["seconds"] for r in rows
               for a, b in zip(r["observations"], r["observations"][1:])] if single else []
        return {"wall_seconds": wall, "requests": len(rows), "output_tokens": output,
                "output_tok_s": output/wall, "request_s": len(rows)/wall,
                "ttft_seconds": quantiles(ttft), "tpot_seconds": quantiles(tpot),
                "itl_seconds": quantiles(itl),
                "e2e_seconds": quantiles([r["finished_seconds"] for r in rows]),
                "single_token_observations": single,
                "tail_note": "exploratory: fewer than 1000 requests" if len(rows) < 1000 else "1000+ requests"}


class LuminaAdapter:
    def __init__(self, args):
        from zjhvllm import LLM, SamplingParams
        self.params = SamplingParams
        self.engine = LLM(args.model, tensor_parallel_size=args.tp, enable_mtp=False, quantization=None,
                          max_num_seqs=args.capacity, max_model_len=args.max_model_len,
                          max_num_batched_tokens=args.token_budget, gpu_memory_utilization=args.memory,
                          enforce_eager=args.eager, distributed_port=args.port)

    def add(self, rows):
        self.ids = {}
        self.tokens = {}
        for r in rows:
            sid = self.engine.add_request(r["input_ids"], self.params(greedy=True, ignore_eos=True, max_tokens=r["max_tokens"]))
            self.ids[sid] = r["id"]
            self.tokens[sid] = []

    def step(self):
        completed, _, stream = self.engine.step()
        changed = set()
        for sid, token in stream:
            self.tokens[sid].append(token)
            changed.add(sid)
        done = {sid for sid, _ in completed}
        return [(self.ids[sid], self.tokens[sid], sid in done) for sid in sorted(changed | done)]

    def done(self):
        return self.engine.is_finished()

    def clear(self):
        self.engine.seq_map.clear()

    def close(self):
        self.engine.exit()


class VllmAdapter:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.params = SamplingParams
        # Official unmodified vLLM. No disabling fused kernels or async scheduling.
        self.llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
                       max_num_seqs=args.capacity, max_model_len=args.max_model_len,
                       max_num_batched_tokens=args.token_budget, gpu_memory_utilization=args.memory,
                       enforce_eager=args.eager, enable_prefix_caching=False,
                       enable_chunked_prefill=True, language_model_only=True,
                       mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32",
                       generation_config="vllm", seed=0)
        self.engine = self.llm.llm_engine

    def add(self, rows):
        for r in rows:
            self.engine.add_request(r["id"], {"prompt_token_ids": r["input_ids"]},
                                    self.params(temperature=0, ignore_eos=True, max_tokens=r["max_tokens"],
                                                repetition_penalty=1., seed=0))

    def step(self):
        return [(r.request_id, list(r.outputs[0].token_ids), r.finished) for r in self.engine.step()]

    def done(self):
        return not self.engine.has_unfinished_requests()

    def clear(self):
        pass

    def close(self):
        # v0.26 exposes shutdown on the EngineCore client, not LLMEngine.
        shutdown = getattr(self.engine, "shutdown", None)
        if shutdown is None:
            shutdown = getattr(getattr(self.engine, "engine_core", None), "shutdown", None)
        if shutdown is not None:
            shutdown()
        else:
            raise RuntimeError("vLLM engine has no known shutdown API")


def run(args):
    workload = json.loads(args.workload.read_text())
    validate_workload(workload)
    if model_identity(args.model) != workload["model"]:
        raise ValueError("model/tokenizer identity differs from workload")
    if max(len(r["input_ids"])+r["max_tokens"] for r in workload["requests"]) > args.max_model_len:
        raise ValueError("workload exceeds max_model_len")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "complete": False, "kind": "profile" if args.profile else "performance",
              "args": vars(args) | {"model": str(Path(args.model).resolve()), "workload": str(args.workload), "output": str(args.output)},
              "environment": environment(args.engine), "workload_sha256": digest(args.workload),
              "model": workload["model"], "warmups": [], "phases": [],
              "contract": "offline simultaneous burst; submission+prefill+decode; host-observed engine outputs; no HTTP, no tokenization; no prefix cache, no speculation, no quantization"}
    report["args"].pop("function", None)
    def save():
        write_json(args.output/"result.json", report)
    save()
    adapter = None
    try:
        if args.engine == "vllm" and report["environment"]["vllm_qwen35_source_present"] is not True:
            raise RuntimeError("vLLM environment has no Qwen3.5 implementation; use a separate compatible official environment")
        require_idle_gpus(args.tp)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        shutil.copytree(ROOT/"zjhvllm", args.output/"source"/"zjhvllm", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(__file__, args.output/"source"/"benchmark.py")
        report["source_sha256"] = {str(p.relative_to(args.output/"source")): digest(p)
                                    for p in (args.output/"source").rglob("*.py")}
        adapter = (LuminaAdapter if args.engine == "lumina" else VllmAdapter)(args)
        report["cuda"] = {"runtime": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(args.tp)]}
        report["resolved_engine_config"] = str(adapter.engine.model_runner.config if args.engine == "lumina"
                                                else adapter.engine.vllm_config)
        if args.engine == "lumina":
            report["rank_metadata"] = adapter.engine.model_runner.call("parallel_metadata")
        save()
        for iteration in range(args.warmups + args.repeats):
            measured = iteration >= args.warmups
            torch.cuda.synchronize()
            profiling = args.profile and measured
            if profiling:
                torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range("qwen35_measured" if measured else "qwen35_warmup"):
                    recorder = Recorder(workload["requests"])
                    adapter.add(workload["requests"])
                    while not adapter.done():
                        for rid, tokens, finished in adapter.step():
                            recorder.observe(rid, tokens, finished)
                    torch.cuda.synchronize()
                    wall = perf_counter()-recorder.start
            finally:
                if profiling:
                    torch.cuda.profiler.stop()
            phase = {"iteration": iteration, "summary": recorder.summary(wall), "requests": recorder.rows}
            report["phases" if measured else "warmups"].append(phase)
            adapter.clear()
            save()
        report["repeat_outputs_equal"] = all(
            {k: r["tokens"] for k, r in p["requests"].items()} ==
            {k: r["tokens"] for k, r in report["phases"][0]["requests"].items()} for p in report["phases"])
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if adapter is not None:
            try:
                adapter.close()
                report["shutdown_called"] = True
            except BaseException as error:
                report["shutdown_error"] = repr(error)
                report["complete"] = False
                save()
                raise
        save()


def compare(args):
    left, right = (json.loads(p.read_text()) for p in (args.lumina, args.vllm))
    for report, engine in ((left, "lumina"), (right, "vllm")):
        if (not report["complete"] or report["kind"] != "performance" or not report["repeat_outputs_equal"]
                or report["environment"]["engine"] != engine):
            raise ValueError("require complete, repeat-stable, non-profile runs of the specified engines")
    if left["workload_sha256"] != right["workload_sha256"] or left["model"] != right["model"]:
        raise ValueError("workload/model mismatch")
    for key in ("tp", "capacity", "max_model_len", "token_budget", "memory", "eager", "warmups", "repeats"):
        if left["args"][key] != right["args"][key]:
            raise ValueError(f"configuration mismatch: {key}")
    if left["environment"]["cuda_visible_devices"] != right["environment"]["cuda_visible_devices"]:
        raise ValueError("GPU placement mismatch")
    def pooled(report):
        phases = report["phases"]
        return sum(p["summary"]["output_tokens"] for p in phases)/sum(p["summary"]["wall_seconds"] for p in phases)
    lrows, rrows = left["phases"][0]["requests"], right["phases"][0]["requests"]
    if lrows.keys() != rrows.keys():
        raise ValueError("request IDs differ")
    pairs = []
    for key in lrows:
        a, b = lrows[key]["tokens"], rrows[key]["tokens"]
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        pairs.append({"id": key, "exact": a == b, "common_prefix": prefix, "tokens": len(a)})
    value = {"lumina_output_tok_s": pooled(left), "vllm_output_tok_s": pooled(right),
             "lumina_over_vllm": pooled(left)/pooled(right), "output_pairs": pairs,
             "exact_sequences": sum(p["exact"] for p in pairs), "sequences": len(pairs),
             "note": "output equivalence is reported separately; this is not HF logits/task-quality validation"}
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, value)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--model", required=True)
    prep.add_argument("--requests", type=positive, default=32)
    prep.add_argument("--input-tokens", type=positive, default=128)
    prep.add_argument("--output-tokens", type=positive, default=128)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--texts", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(function=prepare)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--engine", choices=["lumina", "vllm"], required=True)
    preflight.add_argument("--output", type=Path, required=True)
    def preflight_run(args):
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, environment(args.engine))
    preflight.set_defaults(function=preflight_run)
    bench = commands.add_parser("run")
    bench.add_argument("--engine", choices=["lumina", "vllm"], required=True)
    bench.add_argument("--model", required=True)
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=2)
    bench.add_argument("--capacity", type=positive, default=32)
    bench.add_argument("--max-model-len", type=positive, default=4096)
    bench.add_argument("--token-budget", type=positive, default=2048)
    bench.add_argument("--memory", type=float, default=.85)
    bench.add_argument("--warmups", type=positive, default=1)
    bench.add_argument("--repeats", type=positive, default=3)
    bench.add_argument("--port", type=int, default=2571)
    bench.add_argument("--eager", action="store_true")
    bench.add_argument("--profile", action="store_true", help="CUDA profiler API capture; excluded from performance comparisons")
    bench.set_defaults(function=run)
    comp = commands.add_parser("compare")
    comp.add_argument("--lumina", type=Path, required=True)
    comp.add_argument("--vllm", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    comp.set_defaults(function=compare)
    args = parser.parse_args()
    if args.command == "run" and not 0 < args.memory < 1:
        parser.error("--memory must be between 0 and 1")
    if args.command == "run" and args.profile and args.repeats != 1:
        parser.error("--profile requires --repeats 1 for a single measured capture")
    args.function(args)


if __name__ == "__main__":
    main()
