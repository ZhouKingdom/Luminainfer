"""Phase + kernel-level profile of one TP2 C4 MTP step.

Answers the handoff's open question: where does a step_speculative round
actually spend time — draft, target rowwise forward, IPC/collectives, or
snapshot/commit? All instrumentation is benchmark-local (monkey-patched on
ModelRunner), matching the existing bench_mtp_net_gain convention; production
runtime files are unchanged.

Run (output dir must not exist):
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /home/data/jhzhou/conda_envs/lumina/bin/python \
    benchmarks/profile_mtp_phases.py --tp 2 --port 2522 \
    --output benchmarks/profile_mtp_phases_<tag>
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from zjhvllm import LLM, SamplingParams
from zjhvllm.engine.model_runner import ModelRunner


def _cuda_event_timing(method_name, phase):
    """Wrap a ModelRunner method to record wall + CUDA event time per phase."""
    original = getattr(ModelRunner, method_name)

    def wrapped(self, *args, **kwargs):
        start_wall = perf_counter()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        try:
            result = original(self, *args, **kwargs)
        finally:
            end_ev.record()
        torch.cuda.synchronize()
        elapsed_wall = perf_counter() - start_wall
        elapsed_cuda = start_ev.elapsed_time(end_ev) / 1000.0
        acc = getattr(self, "_prof", None)
        if acc is None:
            acc = self._prof = {}
        acc.setdefault(phase, []).append((elapsed_wall, elapsed_cuda))
        return result

    wrapped.__name__ = method_name
    return wrapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tp", type=int, default=2, choices=[1, 2, 4, 8])
    p.add_argument("--port", type=int, default=2522)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--warmup-steps", type=int, default=6)
    p.add_argument("--profiled-steps", type=int, default=12)
    p.add_argument("--backend", type=str, default="decode_equivalent",
                   choices=["decode_equivalent", "batched", "batch_invariant"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Instrument the phases of a speculative round.
    ModelRunner.draft_mtp = _cuda_event_timing("draft_mtp", "draft")
    ModelRunner.run_verification = _cuda_event_timing("run_verification", "verify")
    ModelRunner.verify_greedy_batch = _cuda_event_timing("verify_greedy_batch", "verify_batch_total")

    report = {"args": vars(args), "complete": False, "phases": []}
    engine = None
    try:
        engine = LLM(
            "/home/data/jhzhou/huggingface/Qwen3.5-27B",
            tensor_parallel_size=args.tp,
            enable_mtp=True,
            max_num_seqs=4,
            max_model_len=2048,
            max_num_batched_tokens=1024,
            verification_cuda_graph=True,
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            distributed_port=args.port,
            qwen35_linear_backend=args.backend,
        )
        runner = engine.model_runner

        items = json.loads(Path("benchmarks/qwen35_mtp_8gpu_20261106/workload.json").read_text())
        indices = [2, 6, 39, 76, 119, 127, 119, 127]
        lengths = [36, 25, 106, 128, 64, 96, 255, 256]
        prompts = [items[i]["prompt_token_ids"][:n] for i, n in zip(indices, lengths)]
        params = SamplingParams(greedy=True, max_tokens=args.tokens, ignore_eos=True)

        def run_cohort(start, profile=False, max_steps=256):
            """Prefill then speculative-decode one 4-request cohort.

            Returns the per-step phase dicts when profiling, else None.
            """
            ids = [engine.add_request(prompt, params) for prompt in prompts[start:start + 4]]
            engine.step()  # prefill (not profiled)
            profiled = []
            profiler = None
            for step in range(max_steps):
                if engine.is_finished():
                    break
                runner._prof = {}
                if profile and step == 2:
                    profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                     torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False, with_stack=False)
                    profiler.__enter__()
                engine.step_speculative()
                if profile and step == 2:
                    profiler.__exit__(None, None, None)
                if profile:
                    profiled.append(dict(runner._prof))
            return profiled, profiler

        # Warm up to completion on the first cohort (captures verification Graph,
        # JIT-compiles Triton kernels, etc.).
        run_cohort(0, profile=False)
        engine.seq_map.clear()

        # Profile a fresh cohort on the second half of the workload.
        profiled, profiler = run_cohort(4, profile=True, max_steps=args.profiled_steps)

        # Kernel-level breakdown of the one profiled round (rank 0).
        if profiler is not None:
            kernel_rows = []
            for evt in profiler.key_averages():
                if evt.device_type == torch.autograd.DeviceType.CUDA:
                    kernel_rows.append({
                        "name": evt.key,
                        "count": evt.count,
                        "cuda_total_ms": evt.self_device_time_total / 1000.0,
                    })
            kernel_rows.sort(key=lambda r: -r["cuda_total_ms"])
            report["top_cuda_kernels"] = kernel_rows[:40]
            families = {}
            for r in kernel_rows:
                n = r["name"]
                if "gated_delta" in n or "delta" in n:
                    fam = "gdn_delta"
                elif "conv" in n.lower():
                    fam = "conv1d"
                elif "flash" in n.lower() or "attention" in n.lower() or "fmha" in n.lower():
                    fam = "attention"
                elif "gemm" in n.lower() or "cublas" in n.lower() or "mm" in n or "dot" in n:
                    fam = "gemm"
                elif "nccl" in n.lower():
                    fam = "nccl"
                elif "elementwise" in n or "mul" in n or "add" in n or "copy" in n or "cat" in n:
                    fam = "elementwise"
                elif "norm" in n.lower() or "rms" in n.lower():
                    fam = "norm"
                elif "argmax" in n.lower():
                    fam = "argmax"
                else:
                    fam = "other"
                families[fam] = families.get(fam, 0.0) + r["cuda_total_ms"]
            report["cuda_family_ms"] = dict(sorted(families.items(), key=lambda kv: -kv[1]))

        def summarize(records):
            out = {}
            for phase, entries in records.items():
                walls = [e[0] for e in entries]
                cudas = [e[1] for e in entries]
                out[phase] = {
                    "calls": len(entries),
                    "wall_total_s": sum(walls),
                    "wall_mean_ms": 1000 * sum(walls) / len(entries),
                    "cuda_total_s": sum(cudas),
                    "cuda_mean_ms": 1000 * sum(cudas) / len(entries),
                }
            return out

        report["per_step"] = [summarize(step) for step in profiled]
        report["aggregate"] = summarize(
            {phase: [e for step in profiled for e in step.get(phase, [])]
             for phase in {"draft", "verify", "verify_batch_total"}}
        )
        report["complete"] = True
    finally:
        if engine is not None:
            engine.exit()

    # Flatten phase timing into the top-level report for easy reading.
    agg = report["aggregate"]
    print("\n=== TP%d C4 MTP per-round phase timing (mean over %d steps) ===" % (
        args.tp, args.profiled_steps))
    for phase in ("draft", "verify", "verify_batch_total"):
        if phase in agg:
            a = agg[phase]
            print(f"  {phase:20s} calls={a['calls']:3d}  "
                  f"wall={a['wall_mean_ms']:7.3f} ms  cuda={a['cuda_mean_ms']:7.3f} ms")
    fam = report.get("cuda_family_ms")
    if fam:
        total = sum(fam.values())
        print("\n=== CUDA kernel family breakdown (one profiled round, rank 0) ===")
        for k, v in fam.items():
            print(f"  {k:12s} {v:8.2f} ms  ({100 * v / total:5.1f}%)")
    print("\n=== Top CUDA kernels ===")
    for r in report.get("top_cuda_kernels", [])[:20]:
        print(f"  {r['cuda_total_ms']:8.2f} ms x{r['count']:5d}  {r['name']}")
    (args.output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
