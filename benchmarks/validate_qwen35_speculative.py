"""Real-checkpoint acceptance/rejection validation; oracle drafts are TEST ONLY.

Writes failures as well as successes. Timings include snapshot/replay and are
not a representative speculative performance benchmark (oracle test drafts).
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from zjhvllm import LLM, SamplingParams
from zjhvllm.utils.context import get_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/data/jhzhou/huggingface/Qwen3.5-27B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--verification-mode", choices=["serial", "batched_experimental"], default="serial")
    args = parser.parse_args()
    if args.tokens < 6:
        parser.error("--tokens must be >= 6")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.model, enforce_eager=True, max_num_seqs=1, max_model_len=1024,
              max_num_batched_tokens=128, gpu_memory_utilization=.85, distributed_port=2349)
    params = SamplingParams(greedy=True, max_tokens=args.tokens, ignore_eos=True)
    report = {"model": args.model, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "verification_mode": args.verification_mode,
              "test_only_oracle_drafts": True, "cases": [], "fault_cases": [], "sources": {}}
    for name in ["engine/speculative.py", "engine/model_runner.py", "engine/block_manager.py",
                 "engine/llm_engine.py", "models/qwen3_5.py"]:
        report["sources"][name] = hashlib.sha256((Path("zjhvllm") / name).read_bytes()).hexdigest()
    free_pages = len(llm.scheduler.block_manager.free_block_ids)
    free_slots = len(llm.scheduler.free_state_slots)
    try:
        natural = llm.tokenizer.encode("Explain how a hash table handles collisions, with an example.")
        repeated = llm.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 100)
        prompts = [("natural", natural), ("page_boundary_255", repeated[:255]),
                   ("chunked_513", repeated[:513])]
        for label, prompt in prompts:
            start = perf_counter()
            baseline = llm.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
            torch.cuda.synchronize()
            baseline_seconds = perf_counter() - start
            for mode in ["oracle", "reject_first", "reject_middle", "prompt_lookup"]:
                rounds = []
                start = perf_counter()
                if mode == "prompt_lookup":
                    result = llm.generate_speculative(prompt, params, ngram_size=2,
                                                      verification_mode=args.verification_mode,
                                                      draft_method="prompt_lookup", max_draft_tokens=4)
                    tokens, rounds = result["token_ids"], result["rounds"]
                else:
                    seq_id = llm.add_request(prompt, params)
                    seq = llm.seq_map[seq_id]
                    while seq.num_completion_tokens == 0 and not seq.is_finished:
                        llm.step()
                    while not seq.is_finished:
                        position = seq.num_completion_tokens
                        count = min(4, len(baseline) - position - 1)
                        if count < 1:
                            llm.step()
                            continue
                        draft = list(baseline[position:position + count])
                        if mode != "oracle":
                            index = 0 if mode == "reject_first" else count // 2
                            draft[index] = (draft[index] + 1) % llm.model_runner.config.hf_config.vocab_size
                        _, stats = llm.speculative_step(seq_id, draft, verification_mode=args.verification_mode)
                        rounds.append(stats)
                    tokens = list(seq.completion_token_ids)
                torch.cuda.synchronize()
                case = {"prompt": label, "prompt_tokens": len(prompt), "mode": mode,
                        "baseline": baseline, "actual": tokens, "exact_tokens": tokens == baseline,
                        "baseline_seconds": baseline_seconds, "seconds": perf_counter() - start,
                        "rounds": rounds, "pages_reclaimed": len(llm.scheduler.block_manager.free_block_ids) == free_pages,
                        "slots_reclaimed": len(llm.scheduler.free_state_slots) == free_slots}
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{label}/{mode}: exact={case['exact_tokens']}, rounds={len(rounds)}, "
                      f"seconds={case['seconds']:.3f}", flush=True)
        # Inject after real computation has written S/conv/KV, including a
        # failure in the rejection replay itself, then resume ordinary decode.
        prompt = natural
        baseline = llm.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        runner = llm.model_runner
        for backend, fail_call in [("serial", 1), ("batched_experimental", 1), ("batched_experimental", 2)]:
            seq_id = llm.add_request(prompt, params)
            seq = llm.seq_map[seq_id]
            while not seq.num_completion_tokens:
                llm.step()
            state = runner.S_cache[seq.state_slot].clone()
            conv = runner.conv_cache[seq.state_slot].clone()
            old_pages, old_tokens = list(seq.block_table), list(seq.token_ids)
            original_forward = runner.model.forward
            calls = 0

            def failing_forward(*inputs):
                nonlocal calls
                hidden = original_forward(*inputs)
                calls += 1
                if calls == fail_call:
                    raise RuntimeError("injected real-checkpoint forward failure")
                return hidden

            runner.model.forward = failing_forward
            caught = False
            try:
                wrong = (baseline[1] + 1) % runner.config.hf_config.vocab_size
                llm.speculative_step(seq_id, [wrong] + baseline[2:5], verification_mode=backend)
            except RuntimeError as error:
                if "injected real-checkpoint" not in str(error):
                    raise
                caught = True
            finally:
                runner.model.forward = original_forward
            restored = (torch.equal(state, runner.S_cache[seq.state_slot])
                        and torch.equal(conv, runner.conv_cache[seq.state_slot])
                        and seq.token_ids == old_tokens and seq.block_table == old_pages
                        and seq.num_cached_tokens == len(old_tokens) - 1
                        and get_context().state_slots is None)
            while not seq.is_finished:
                llm.step()
            fault = {"backend": backend, "fail_call": fail_call, "caught": caught, "restored": restored,
                     "resumed_exact": seq.completion_token_ids == baseline,
                     "pages_reclaimed": len(llm.scheduler.block_manager.free_block_ids) == free_pages,
                     "slots_reclaimed": len(llm.scheduler.free_state_slots) == free_slots}
            report["fault_cases"].append(fault)
            print(f"fault {backend}/{fail_call}: {fault}", flush=True)
        report["passed"] = all(c["exact_tokens"] and c["pages_reclaimed"] and c["slots_reclaimed"]
                               for c in report["cases"])
        report["passed"] &= all(all(c[k] for k in ["caught", "restored", "resumed_exact", "pages_reclaimed",
                                                   "slots_reclaimed"]) for c in report["fault_cases"])
    except BaseException as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        llm.exit()
    if not report["passed"]:
        raise SystemExit("strict speculative validation failed; see output JSON")


if __name__ == "__main__":
    main()
