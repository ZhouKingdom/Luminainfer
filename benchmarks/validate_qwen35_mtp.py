"""Real MTP-1 acceptance and paired generation; retains numerical failures."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from zjhvllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/data/jhzhou/huggingface/Qwen3.5-27B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.tokens < 4 or args.repeats < 1:
        parser.error("require --tokens >= 4, --repeats >= 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.model, enable_mtp=True, enforce_eager=True, max_num_seqs=1,
              max_model_len=1024, max_num_batched_tokens=128,
              gpu_memory_utilization=.85, distributed_port=2351)
    runner = llm.model_runner
    params = SamplingParams(greedy=True, max_tokens=args.tokens, ignore_eos=True)
    report = {"model": args.model, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "draft_source": "checkpoint_mtp", "speculative_tokens": 1, "repeats": args.repeats,
              "mtp_weight_names": sorted(runner.model.mtp.loaded_weight_names),
              "mtp_parameters": sum(p.numel() for p in runner.model.mtp.parameters()),
              "cases": [], "sources": {}}
    for name in ["models/qwen3_5_mtp.py", "models/qwen3_5.py", "engine/model_runner.py", "engine/llm_engine.py",
                 "engine/speculative.py", "engine/block_manager.py", "config.py", "utils/loader.py"]:
        report["sources"][name] = hashlib.sha256((Path("zjhvllm") / name).read_bytes()).hexdigest()
    free_pages, free_slots = len(llm.scheduler.block_manager.free_block_ids), len(llm.scheduler.free_state_slots)
    try:
        raw = ["Explain how a hash table handles collisions, with an example.",
               "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n",
               "The quick brown fox jumps over the lazy dog. " * 80]
        prompts = [llm.tokenizer.encode(text)[:513] for text in raw]
        # Warm target, real MTP prefill/extend, and batched verification kernels.
        llm.generate_speculative(prompts[0], SamplingParams(max_tokens=6, ignore_eos=True),
                                 verification_mode="batched_experimental")
        for label, prompt in zip(["natural", "code", "chunked_513"], prompts):
            for repeat in range(args.repeats):
                torch.cuda.synchronize()
                start = perf_counter()
                baseline = llm.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
                torch.cuda.synchronize()
                baseline_seconds = perf_counter() - start
                for backend in ["serial", "batched_experimental"]:
                    start = perf_counter()
                    result = llm.generate_speculative(prompt, params, verification_mode=backend)
                    torch.cuda.synchronize()
                    seconds = perf_counter() - start
                    proposed = sum(r["proposed"] for r in result["rounds"])
                    accepted = sum(r["accepted"] for r in result["rounds"])
                    case = {"prompt": label, "prompt_tokens": len(prompt), "repeat": repeat,
                            "backend": backend, "baseline": baseline, "actual": result["token_ids"],
                            "exact_tokens": baseline == result["token_ids"], "baseline_seconds": baseline_seconds,
                            "seconds": seconds, "proposed": proposed, "accepted": accepted,
                            "acceptance_rate": accepted / proposed if proposed else None,
                            "rounds": result["rounds"],
                            "pages_reclaimed": len(llm.scheduler.block_manager.free_block_ids) == free_pages,
                            "slots_reclaimed": len(llm.scheduler.free_state_slots) == free_slots}
                    report["cases"].append(case)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"{label}/{backend}: exact={case['exact_tokens']}, accepted={accepted}/{proposed}, "
                          f"baseline={baseline_seconds:.3f}s, mtp={seconds:.3f}s", flush=True)
        # Fail after MTP attention has written private KV, then retry the draft.
        seq_id = llm.add_request(prompts[0], params)
        seq = llm.seq_map[seq_id]
        while not seq.num_completion_tokens:
            llm.step()
        state, conv = runner.S_cache.clone(), runner.conv_cache.clone()
        original = runner.model.mtp.norm.forward

        def fail_after_attention(hidden):
            original(hidden)
            raise RuntimeError("injected MTP failure")

        runner.model.mtp.norm.forward = fail_after_attention
        caught = False
        try:
            runner.draft_mtp(seq)
        except RuntimeError as error:
            if "injected MTP" not in str(error):
                raise
            caught = True
        finally:
            runner.model.mtp.norm.forward = original
        restored = (runner.model.mtp.cache.length == 0 and torch.equal(state, runner.S_cache)
                    and torch.equal(conv, runner.conv_cache))
        draft, _ = runner.draft_mtp(seq)
        llm.speculative_step(seq_id, draft)
        while not seq.is_finished:
            llm.step()
        actual = list(seq.completion_token_ids)
        baseline = llm.generate([prompts[0]], params, use_tqdm=False)[0]["token_ids"]
        report["draft_fault"] = {"caught": caught, "restored": restored, "resumed_exact": actual == baseline}
        report["serial_passed"] = all(c["exact_tokens"] and c["pages_reclaimed"] and c["slots_reclaimed"]
                                       for c in report["cases"] if c["backend"] == "serial")
        report["batched_passed"] = all(c["exact_tokens"] and c["pages_reclaimed"] and c["slots_reclaimed"]
                                        for c in report["cases"] if c["backend"] == "batched_experimental")
        report["passed"] = report["serial_passed"] and report["batched_passed"] and all(report["draft_fault"].values())
    except BaseException as error:
        report["passed"], report["error"] = False, repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        llm.exit()
    if not report["passed"]:
        raise SystemExit("MTP validation has failures; see JSON")


if __name__ == "__main__":
    main()
