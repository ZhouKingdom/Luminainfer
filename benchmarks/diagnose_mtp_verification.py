"""Replay fixed verification windows from identical committed hybrid state.

This is a numerical diagnostic, never a performance benchmark. Both paths see
the same pending token, draft, private pages and state checkpoint. The traced
generation commits SERIAL results so later comparisons never use forked history.
"""
import argparse
from contextlib import contextmanager
from copy import copy
import hashlib
import json
from pathlib import Path
import shutil

import torch

from zjhvllm import LLM, SamplingParams
from zjhvllm.engine.speculative import HybridCheckpoint
from zjhvllm.utils.context import get_context, reset_context


@contextmanager
def interventions(runner, choices):
    """Diagnostic ablations only; patches are scoped to target verification."""
    from zjhvllm.models.qwen3_5 import GatedDeltaNet, Qwen3_5RMSNorm
    from zjhvllm.layers.attention import Attention, store_kvcache
    from zjhvllm.layers.paged_attention import paged_attention_decode
    patches = []
    def patch(module, forward):
        patches.append((module, module.forward))
        module.forward = forward
    try:
        for module in runner.model.modules():
            if (("projection" in choices and isinstance(module, torch.nn.Linear))
                    or ("norm" in choices and isinstance(module, Qwen3_5RMSNorm))):
                original = module.forward
                def rowwise(x, original=original):
                    if x.ndim >= 2 and x.shape[0] > 1:
                        return torch.cat([original(row) for row in x.split(1)], dim=0)
                    return original(x)
                patch(module, rowwise)
            if "recurrence" in choices and isinstance(module, GatedDeltaNet):
                original = module.forward
                def recurrent(x, module=module, original=original):
                    context = get_context()
                    if not context.is_prefill:
                        return original(x)
                    assert len(context.state_slots) == 1
                    return torch.cat([module._forward_decode(row, context.state_slots)
                                      for row in x.split(1)], dim=0)
                patch(module, recurrent)
            if "attention" in choices and isinstance(module, Attention):
                original = module.forward
                def attention(q, k, v, module=module, original=original):
                    context = get_context()
                    if not context.is_prefill:
                        return original(q, k, v)
                    assert not module.use_fp8 and context.block_tables.shape[0] == 1
                    store_kvcache(k, v, module.k_cache, module.v_cache, context.slot_mapping)
                    start = context.max_seqlen_k - q.shape[0]
                    return torch.cat([paged_attention_decode(row.contiguous(), module.k_cache, module.v_cache,
                                          context.block_tables,
                                          torch.tensor([start + i + 1], device=q.device, dtype=torch.int32), module.scale)
                                      for i, row in enumerate(q.split(1))], dim=0)
                patch(module, attention)
        yield
    finally:
        for module, original in reversed(patches):
            module.forward = original


def difference(a, b):
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape)}
    delta = a.float() - b.float()
    return {"exact": torch.equal(a, b), "max_abs": delta.abs().max().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "different": int(torch.count_nonzero(delta)), "elements": a.numel()}


def top_rows(logits):
    values, ids = logits.float().topk(5, dim=-1)
    return [{"argmax": a, "ids": i, "values": v, "gap": v[0] - v[1]}
            for a, i, v in zip(logits.argmax(-1).tolist(), ids.tolist(), values.tolist())]


class WindowDiagnostic:
    def __init__(self, runner, limit, choices=()):
        self.runner, self.limit = runner, limit
        self.choices = choices
        self.original = runner.verify_greedy
        self.rounds = []

    @torch.inference_mode()
    def compare(self, seq, draft):
        runner = self.runner
        checkpoint = HybridCheckpoint.capture(runner.S_cache, runner.conv_cache, seq.state_slot)
        start, count = seq.num_cached_tokens, len(draft) + 1
        saved_hidden = runner.target_hidden[start:start + count].clone()
        tokens = seq.token_ids + list(draft)
        trial = copy(seq)
        trace, handles = {}, []

        def hook(name):
            def collect(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    trace.setdefault(name, []).append(output.detach().cpu().clone())
            return collect

        # Conv layouts differ between decode/prefill; observe its projection and
        # the enclosing attention output instead. Other outputs have token dim 0.
        for name, module in runner.model.model.named_modules():
            if name and not isinstance(module, (torch.nn.ModuleList, torch.nn.Conv1d)):
                handles.append(module.register_forward_hook(hook(name)))
        try:
            serial_logits = []
            for index in range(count):
                trial.token_ids = tokens[:seq.num_tokens + index]
                trial.num_tokens = len(trial.token_ids)
                trial.last_token = trial.token_ids[-1]
                trial.block_table = seq.block_table[:(trial.num_tokens + runner.block_size - 1) // runner.block_size]
                ids, positions = runner.prepare_decode([trial])
                hidden = runner.model(ids, positions)
                serial_logits.append(runner.model.compute_logits(hidden).cpu())
            serial = {name: torch.cat(rows, dim=0) for name, rows in trace.items()}
            state = runner.S_cache[seq.state_slot].cpu().clone()
            conv = runner.conv_cache[seq.state_slot].cpu().clone()
            # Compare only the logical, valid KV window, never uninitialized pages.
            indices = torch.tensor([seq.block_table[p // runner.block_size] * runner.block_size
                                    + p % runner.block_size for p in range(start + count)],
                                   device=runner.kv_cache.device)
            kv = runner.kv_cache.flatten(2, 3).index_select(2, indices).cpu()
            checkpoint.restore(runner.S_cache, runner.conv_cache)
            trace.clear()
            trial.token_ids, trial.num_tokens = tokens, len(tokens)
            trial.last_token, trial.num_scheduled_tokens = tokens[-1], count
            trial.block_table = seq.block_table
            ids, positions = runner.prepare_prefill([trial])
            from zjhvllm.engine.model_runner import ModelRunner
            ModelRunner.prepare_verification_context(start, count, positions.device)
            context = get_context()
            metadata = {"positions": positions.tolist(), "slot_mapping": context.slot_mapping.tolist(),
                        "block_table": list(seq.block_table), "state_slot": seq.state_slot,
                        "cached_tokens": start, "history": list(seq.token_ids), "draft": list(draft)}
            with interventions(runner, self.choices):
                hidden = runner.model(ids, positions)
                batched_logits = runner.model.compute_logits(hidden, all_tokens=True).cpu()
            operators = [{"name": name, **difference(serial[name], torch.cat(rows, dim=0))}
                         for name, rows in trace.items()]
            kv_batch = runner.kv_cache.flatten(2, 3).index_select(2, indices).cpu()
            serial_logits = torch.cat(serial_logits)
            result = {**metadata, "operators": operators,
                      "first_different_operator": next((r["name"] for r in operators if not r.get("exact")), None),
                      "logits": difference(serial_logits, batched_logits),
                      "serial_top5": top_rows(serial_logits), "batched_top5": top_rows(batched_logits),
                      "states": [difference(a, b) for a, b in zip(state, runner.S_cache[seq.state_slot].cpu())],
                      "conv": [difference(a, b) for a, b in zip(conv, runner.conv_cache[seq.state_slot].cpu())],
                      "kv_committed_prefix": difference(kv[:, :, :start], kv_batch[:, :, :start]),
                      "kv_window": difference(kv[:, :, start:], kv_batch[:, :, start:])}
            return result
        finally:
            for handle in handles:
                handle.remove()
            checkpoint.restore(runner.S_cache, runner.conv_cache)
            runner.target_hidden[start:start + count].copy_(saved_hidden)
            reset_context()

    def __call__(self, seq, draft, max_outputs, verification_mode="serial"):
        if len(self.rounds) < self.limit:
            result = self.compare(seq, draft)
            result["completion_tokens_before"] = seq.num_completion_tokens
            self.rounds.append(result)
            print(f"window {len(self.rounds)}: first={result['first_different_operator']}, "
                  f"logits max_abs={result['logits']['max_abs']}", flush=True)
        return self.original(seq, draft, max_outputs, verification_mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/data/jhzhou/huggingface/Qwen3.5-27B")
    parser.add_argument("--workload", type=Path, default=Path("benchmarks/qwen35_mtp_8gpu_20261106/workload.json"))
    parser.add_argument("--request-id", type=int, default=2)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--port", type=int, default=2361)
    parser.add_argument("--interventions", nargs="*", choices=["projection", "recurrence", "attention", "norm"], default=[])
    parser.add_argument("--ablation-sweep", action="store_true")
    parser.add_argument("--check-ids", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence")
    if args.tokens < 2 or args.windows < 1:
        parser.error("require tokens >= 2 and windows >= 1")
    workload = json.loads(args.workload.read_text())
    item = next(row for row in workload if row["id"] == args.request_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path("zjhvllm").rglob("*.py")) + [Path(__file__)]
    report = {"request_id": args.request_id, "interventions": args.interventions,
              "batch_path": "production_decode_equivalent",
              "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "model": args.model, "max_num_batched_tokens": 256, "max_model_len": 4096,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "outputs": {}, "windows": [],
              "measurement_complete": False}
    snapshot = args.output.with_suffix(".source")
    snapshot.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for path in sources:
        target = snapshot / path.resolve().relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    llm, diagnostic = None, None
    try:
        llm = LLM(args.model, enable_mtp=True, max_num_seqs=1, max_model_len=4096,
                  max_num_batched_tokens=256, enforce_eager=True,
                  gpu_memory_utilization=.85, distributed_port=args.port)
        runner = llm.model_runner
        params = SamplingParams(greedy=True, max_tokens=min(args.tokens, item["max_tokens"]), ignore_eos=False)
        free_pages = len(llm.scheduler.block_manager.free_block_ids)
        free_slots = len(llm.scheduler.free_state_slots)
        diagnostic = WindowDiagnostic(runner, args.windows, args.interventions)
        prompt = item["prompt_token_ids"]
        report["outputs"]["baseline"] = llm.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        for mode in ("serial", "batched_experimental"):
            result = llm.generate_speculative(prompt, params, verification_mode=mode)
            report["outputs"][mode] = result["token_ids"]
            report[mode + "_rounds"] = result["rounds"]
            print(f"{mode}: exact={result['token_ids'] == report['outputs']['baseline']}", flush=True)
        runner.verify_greedy = diagnostic
        result = llm.generate_speculative(prompt, params, verification_mode="serial")
        report["outputs"]["traced_serial"] = result["token_ids"]
        report["trace_preserved_serial"] = result["token_ids"] == report["outputs"]["serial"]
        if args.interventions:
            variants = [args.interventions]
            if args.ablation_sweep:
                variants = [["projection"], ["projection", "recurrence"],
                            ["projection", "attention"], ["projection", "recurrence", "attention"],
                            ["projection", "recurrence", "attention", "norm"]]
            report["ablations"] = []
            for choices in variants:
                def ablated(seq, draft, max_outputs, verification_mode="serial"):
                    with interventions(runner, choices):
                        return diagnostic.original(seq, draft, max_outputs, verification_mode)
                runner.verify_greedy = ablated
                result = llm.generate_speculative(prompt, params, verification_mode="batched_experimental")
                report["ablations"].append({"choices": choices, "token_ids": result["token_ids"],
                                            "exact": result["token_ids"] == report["outputs"]["baseline"]})
                print(f"ablated {choices}: exact={report['ablations'][-1]['exact']}", flush=True)
        if args.check_ids:
            report["holdout"] = []
            for request_id in args.check_ids:
                row = next(row for row in workload if row["id"] == request_id)
                params = SamplingParams(greedy=True, max_tokens=min(args.tokens, row["max_tokens"]), ignore_eos=False)
                runner.verify_greedy = diagnostic.original
                baseline = llm.generate([row["prompt_token_ids"]], params, use_tqdm=False)[0]["token_ids"]
                batch = llm.generate_speculative(row["prompt_token_ids"], params,
                                                 verification_mode="batched_experimental")["token_ids"]
                choices = args.interventions
                if args.interventions:
                    runner.verify_greedy = ablated
                actual = llm.generate_speculative(row["prompt_token_ids"], params,
                                                  verification_mode="batched_experimental")["token_ids"]
                report["holdout"].append({"request_id": request_id, "baseline": baseline,
                                           "original_batch": batch, "ablated_batch": actual,
                                           "original_exact": baseline == batch, "ablated_exact": baseline == actual})
                print(f"request {request_id}: original={baseline == batch}, ablated={baseline == actual}", flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["resources_reclaimed"] = (len(llm.scheduler.block_manager.free_block_ids) == free_pages
                                         and len(llm.scheduler.free_state_slots) == free_slots)
        if not report["trace_preserved_serial"] or not report["resources_reclaimed"]:
            raise RuntimeError("diagnostic changed serial results or leaked resources")
        report["measurement_complete"] = True
        report["batched_numerical_validation_passed"] = report["outputs"]["baseline"] == report["outputs"]["batched_experimental"]
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if diagnostic is not None:
            report["windows"] = diagnostic.rounds
            runner.verify_greedy = diagnostic.original
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
