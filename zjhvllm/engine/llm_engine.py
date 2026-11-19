import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
from zjhvllm.config import Config
from zjhvllm.sampling_params import SamplingParams
from zjhvllm.engine.sequence import Sequence
from zjhvllm.engine.scheduler import Scheduler
from zjhvllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 实际并行进程数：EP 模式用 expert_parallel_size，否则 tensor_parallel_size
        world_size = config.expert_parallel_size if config.expert_parallel_size > 1 else config.tensor_parallel_size
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        if world_size > 1:
            self.model_runner.call("set_eos", config.eos)
        self.scheduler = Scheduler(config)
        self.seq_map: dict[int, Sequence] = {}   # seq_id -> Sequence（用于延迟统计）
        self._pending_exports = {}
        self._accepted_prefills = {}
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return  # 幂等：显式 exit() 后再触发 atexit 时避免二次清理
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        self.scheduler.add(seq)
        self.seq_map[seq.seq_id] = seq
        return seq.seq_id

    def export_prefill(self, seq_id):
        """Snapshot without releasing source pages. Source stepping pauses.

        The coordinator must ACK only after destination import succeeds, or
        abort to resume source execution. V1 permits one outstanding handoff.
        """
        from zjhvllm.engine.handoff import capture_prefill
        if self._pending_exports:
            raise RuntimeError("a P-D handoff is already pending")
        seq=self.seq_map[seq_id]
        if seq not in self.scheduler.running or seq.is_finished:
            raise ValueError("only an active, fully prefilled request can migrate")
        packet=capture_prefill(self.model_runner,seq)
        self._pending_exports[packet.transfer_id]=seq_id
        return packet

    def finish_export(self, transfer_id, *, accepted):
        """Release on destination ACK; abort preserves the source request."""
        seq_id=self._pending_exports.pop(transfer_id)
        if accepted:
            seq=self.seq_map.pop(seq_id)
            self.scheduler.running.remove(seq)
            self.scheduler.block_manager.deallocate(seq)
        return seq_id

    def accept_prefill(self, packet):
        """Import once per transfer ID; retries return the original request ID.

        The receipt ledger is in-memory for this engine's lifetime. Durable
        retries, expiry and bounded retention belong to the future coordinator.
        """
        from zjhvllm.engine.handoff import restore_prefill
        packet.validate(self.model_runner)
        if packet.transfer_id in self._accepted_prefills:
            return self._accepted_prefills[packet.transfer_id]
        seq=Sequence(packet.prompt_token_ids,packet.sampling)
        seq.arrival_time=packet.arrival_time
        seq.first_token_time=packet.first_token_time
        seq.token_times=[packet.first_token_time]
        seq.last_ran_seed=packet.last_ran_seed
        self.scheduler.admit_prefilled(seq,packet.first_token,
            lambda target:restore_prefill(self.model_runner,target,packet))
        self.seq_map[seq.seq_id]=seq
        self._accepted_prefills[packet.transfer_id]=seq.seq_id
        return seq.seq_id

    def latency_stats(self) -> tuple[list[float], list[float]]:
        """返回 (ttfts, itls) 两个列表。

        ttfts: 每个已完成请求的 TTFT（first_token_time - arrival_time）
        itls:  每个已完成请求的相邻 token 间隔（ITL，token_times 相邻差）
        只统计已有 first_token_time 与 finished_time 的序列（即至少生成 1 个 token 且已结束）。
        """
        ttfts = []
        itls = []
        for seq in self.seq_map.values():
            if seq.first_token_time is not None and seq.finished_time is not None:
                ttfts.append(seq.first_token_time - seq.arrival_time)
                for i in range(1, len(seq.token_times)):
                    itls.append(seq.token_times[i] - seq.token_times[i - 1])
        return ttfts, itls

    def step(self):
        if self._pending_exports:
            raise RuntimeError("P-D source is paused until handoff ACK or abort")
        with torch.cuda.nvtx.range("step"):
            with torch.cuda.nvtx.range("schedule"):
                seqs, is_prefill = self.scheduler.schedule()
            return self._run_scheduled(seqs, is_prefill)

    def _run_scheduled(self, seqs, is_prefill):
        """Execute an already reserved scheduler batch (also capacity fallback)."""
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        work_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        cached_tokens = sum(seq.num_cached_tokens for seq in seqs)
        run_start = perf_counter()
        with torch.cuda.nvtx.range("run"):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        torch.cuda.synchronize()   # 确保 GPU 完成，postprocess 记录的时间戳才准确（同步幂等）
        elapsed = perf_counter()-run_start
        if self.scheduler.scheduling_policy == "adaptive":
            self.scheduler.adaptive.observe(is_prefill,work_tokens,len(seqs),cached_tokens,elapsed)
        self.last_step_info = {"is_prefill":is_prefill,"tokens":work_tokens,"sequences":len(seqs),
                               "cached_tokens":cached_tokens,"seconds":elapsed}
        if self.scheduler.scheduling_policy == "adaptive":
            self.last_step_info["decision"] = dict(self.scheduler.adaptive.last_decision)
        with torch.cuda.nvtx.range("postprocess"):
            stream = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, stream

    def is_finished(self):
        return self.scheduler.is_finished()

    def _draft_speculative(self, seqs, max_draft_tokens=None):
        runner = self.model_runner
        configured = getattr(runner.config, "mtp_num_steps", 1)
        steps = configured if max_draft_tokens is None else max_draft_tokens
        if type(steps) is not int or not 1 <= steps <= configured:
            raise ValueError("draft steps must fit the configured mtp_num_steps")
        limits = [min(steps, seq.max_tokens-seq.num_completion_tokens-1,
                      runner.config.max_model_len-len(seq)) for seq in seqs]
        eligible = [seq for seq,limit in zip(seqs,limits) if limit > 0]
        if eligible and steps > 1:
            results = runner.call("draft_mtp_k_batch", eligible, [limit for limit in limits if limit > 0])
        elif eligible and getattr(runner.config, "mtp_batch_draft", False):
            results = runner.call("draft_mtp_batch", eligible)
        else:
            results = [runner.call("draft_mtp", seq) for seq in eligible]
        if len(results) != len(eligible):
            raise ValueError("packed MTP draft result count mismatch")
        by_id = {seq.seq_id: result for seq, result in zip(eligible, results)}
        ordered = [by_id.get(seq.seq_id, ([], {})) for seq in seqs]
        return [item[0] for item in ordered], [item[1] for item in ordered]

    def step_speculative(self, *, max_draft_tokens=None):
        """Schedule one prefill or MTP-1 round, allowing admission every round.

        Compatible with add_request while other requests are running. Returns
        the same (finished, token_count, stream) interface as ordinary step().
        Prefills and capacity fallback use the already reserved ordinary batch.
        """
        if self._pending_exports:
            raise RuntimeError("P-D source is paused")
        runner = self.model_runner
        if getattr(runner.model, "mtp", None) is None:
            raise ValueError("step_speculative requires enable_mtp=True")
        if not getattr(runner.config, "verification_state_commit", True):
            raise ValueError("continuous MTP requires intermediate state commit")
        for seq in (*self.scheduler.waiting, *self.scheduler.running):
            if (not seq.greedy or seq.repetition_penalty != 1 or seq.top_k != -1
                    or seq.top_p != 1 or seq.max_tokens < 1):
                raise ValueError("continuous MTP requires unmodified greedy sampling")
        seqs, is_prefill = self.scheduler.schedule()
        self.last_speculative_stats = {}
        if not seqs:
            return [], 0, []
        if is_prefill:
            return self._run_scheduled(seqs, is_prefill)
        cache = runner.model.mtp.cache
        starts = {seq.state_slot: cache.get_length(seq.state_slot) for seq in seqs}
        cached = sum(seq.num_cached_tokens for seq in seqs)
        begin = perf_counter()
        drafts, extras = [], []
        try:
            drafts, extras = self._draft_speculative(seqs, max_draft_tokens)
            outputs, stats = self.speculative_step_batch([seq.seq_id for seq in seqs], drafts)
        except BaseException as error:
            if runner.world_size > 1:
                runner.call("restore_mtp_lengths", starts)
            else:
                for slot, length in starts.items():
                    cache.set_length(slot, length)
            if isinstance(error, MemoryError):
                return self._run_scheduled(seqs, False)
            raise
        elapsed = perf_counter() - begin
        work = sum(1 + len(draft) for draft in drafts)
        self.last_speculative_stats = {seq.seq_id: stat | extra for seq, stat, extra in zip(seqs, stats, extras)}
        self.last_step_info = {"is_prefill": False, "tokens": work, "sequences": len(seqs),
                               "cached_tokens": cached, "seconds": elapsed, "speculative": True}
        if self.scheduler.scheduling_policy == "adaptive":
            self.scheduler.adaptive.observe(False, work, len(seqs), cached, elapsed)
        stream = [(seq.seq_id, token) for seq, output in zip(seqs, outputs) for token in output]
        finished = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return finished, -len(stream), stream

    def speculative_step(self, seq_id: int, draft_token_ids: list[int], *, verification_mode="serial"):
        """Experimental single-request greedy verification, separate from step()."""
        runner = self.model_runner
        if verification_mode not in ("serial", "batched_experimental"):
            raise ValueError("unknown verification mode")
        if (not runner.has_linear_state
                or runner.config.hf_config.model_type != "qwen3_5_text"):
            raise NotImplementedError("speculation requires Qwen3.5")
        seq = self.seq_map[seq_id]
        if (self._pending_exports or self.scheduler.waiting or list(self.scheduler.running) != [seq]
                or seq.is_finished or seq.num_cached_tokens != len(seq) - 1):
            raise ValueError("speculation requires one fully prefilled running request")
        if not seq.greedy or seq.repetition_penalty != 1 or seq.top_k != -1 or seq.top_p != 1:
            raise ValueError("speculation currently supports unmodified greedy argmax only")
        remaining = seq.max_tokens - seq.num_completion_tokens
        draft = list(draft_token_ids)
        if len(draft) > 32:
            raise ValueError("verification supports at most 32 draft tokens")
        if remaining < 1 or len(draft) >= remaining:
            raise ValueError("draft must leave room for the correction/bonus token")
        if len(seq) + len(draft) > runner.config.max_model_len:
            raise ValueError("verification window exceeds model context")
        vocab = runner.config.hf_config.vocab_size
        if any(type(token) is not int or not 0 <= token < vocab for token in draft):
            raise ValueError("draft contains an invalid token id")
        manager = self.scheduler.block_manager
        old_pages = len(seq.block_table)
        manager.reserve_private(seq, len(seq) + len(draft))
        try:
            output, stats = runner.call("verify_greedy", seq, draft, remaining, verification_mode)
        except BaseException:
            manager.truncate_private(seq, old_pages)
            raise
        # Reclaim rejected pages before publishing metadata. Old suffix bytes in
        # the final retained page are hidden by num_cached_tokens/context_lens.
        committed = seq.num_cached_tokens + len(output)
        manager.truncate_private(seq, (committed + manager.block_size - 1) // manager.block_size)
        for token in output:
            seq.num_scheduled_tokens = 1
            self.scheduler.postprocess([seq], [token], False)
        return output, stats

    def generate_speculative(self, prompt: str | list[int], sampling_params: SamplingParams,
                             *, max_draft_tokens=None, ngram_size=3, verification_mode="serial", draft_method="mtp"):
        """Checkpoint MTP-1 draft by default, or explicit prompt_lookup.

        Returns token IDs and per-round acceptance/replay counts. Use the usual
        generate() for batching and sampling. batched_experimental traverses the
        target once per window using single-token decode arithmetic inside each
        operator; this preserves the reference but does not imply a speedup.
        A failed round remains recoverable through ordinary step().
        """
        from zjhvllm.engine.speculative import prompt_lookup
        runner = self.model_runner
        if max_draft_tokens is None:
            max_draft_tokens = getattr(runner.config,"mtp_num_steps",1) if draft_method=="mtp" else 1
        if not self.is_finished() or self._pending_exports:
            raise ValueError("generate_speculative requires an idle engine")
        if max_draft_tokens < 1 or max_draft_tokens > 32 or ngram_size < 1:
            raise ValueError("require 1 <= max_draft_tokens <= 32 and ngram_size >= 1")
        if verification_mode not in ("serial", "batched_experimental"):
            raise ValueError("unknown verification mode")
        if draft_method not in ("mtp", "prompt_lookup"):
            raise ValueError("unknown draft method")
        if draft_method == "mtp" and not 1 <= max_draft_tokens <= getattr(runner.config,"mtp_num_steps",1):
            raise ValueError("draft steps must fit the configured mtp_num_steps")
        if draft_method == "mtp" and getattr(runner.model, "mtp", None) is None:
            raise ValueError("MTP requires constructing the engine with enable_mtp=True")
        if (not runner.has_linear_state
                or runner.config.hf_config.model_type != "qwen3_5_text"):
            raise NotImplementedError("speculation requires Qwen3.5")
        if (not sampling_params.greedy or sampling_params.repetition_penalty != 1
                or sampling_params.top_k != -1 or sampling_params.top_p != 1 or sampling_params.max_tokens < 1):
            raise ValueError("speculation requires positive max_tokens and unmodified greedy argmax")
        tokens = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not tokens or len(tokens) + sampling_params.max_tokens > runner.config.max_model_len:
            raise ValueError("empty prompt or prompt plus generation exceeds context")
        seq_id = self.add_request(tokens, sampling_params)
        seq = self.seq_map[seq_id]
        while not seq.is_finished and seq.num_completion_tokens == 0:
            self.step()
        rounds = []
        while not seq.is_finished:
            budget = min(max_draft_tokens, seq.max_tokens - seq.num_completion_tokens - 1)
            draft_stats = {}
            draft_cache_start = None
            if draft_method == "mtp" and budget:
                cache = runner.model.mtp.cache
                draft_cache_start = (cache.get_length(seq.state_slot) if hasattr(cache, "get_length") else cache.length)
                if max_draft_tokens > 1:
                    [(draft,draft_stats)] = runner.call("draft_mtp_k_batch", [seq], [budget])
                else:
                    draft, draft_stats = runner.call("draft_mtp", seq)
            else:
                draft = prompt_lookup(seq.token_ids, budget, ngram_size) if draft_method == "prompt_lookup" else []
            if not draft:
                self.step()
                continue
            try:
                _, stats = self.speculative_step(seq_id, draft, verification_mode=verification_mode)
            except MemoryError:
                # A generated MTP draft already extended its KV. Ordinary step
                # commits one more target input, making the next extension valid.
                self.step()
                continue
            except BaseException:
                if draft_cache_start is not None:
                    cache = runner.model.mtp.cache
                    if runner.world_size > 1:
                        runner.call("restore_mtp_lengths", {seq.state_slot: draft_cache_start})
                    elif hasattr(cache, "set_length"):
                        cache.set_length(seq.state_slot, draft_cache_start)
                    else:
                        cache.length = draft_cache_start
                raise
            stats.update(draft_stats)
            rounds.append(stats)
        return {"text": self.tokenizer.decode(seq.completion_token_ids),
                "token_ids": seq.completion_token_ids, "rounds": rounds}

    def speculative_step_batch(self, seq_ids, draft_token_ids):
        """Verify running requests together, committing each accepted prefix.

        An error before publication restores every request's recurrent state
        and newly reserved pages. Completed requests release their own slot.
        """
        runner = self.model_runner
        if self._pending_exports:
            raise RuntimeError("P-D source is paused")
        if not seq_ids or len(seq_ids) != len(draft_token_ids) or len(set(seq_ids)) != len(seq_ids):
            raise ValueError("distinct request IDs and one draft list per request required")
        seqs = [self.seq_map[seq_id] for seq_id in seq_ids]
        drafts, budgets = [], []
        for seq, raw in zip(seqs, draft_token_ids):
            if seq not in self.scheduler.running or seq.is_finished or len(seq) != seq.num_cached_tokens + 1:
                raise ValueError("each request must be fully prefilled and running")
            if not seq.greedy or seq.repetition_penalty != 1 or seq.top_k != -1 or seq.top_p != 1:
                raise ValueError("batched speculation requires unmodified greedy sampling")
            draft = list(raw)
            budget = seq.max_tokens - seq.num_completion_tokens
            if budget < 1 or len(draft) >= budget or len(draft) > 32:
                raise ValueError("draft must leave room for correction/bonus")
            if len(seq) + len(draft) > runner.config.max_model_len:
                raise ValueError("verification window exceeds context")
            if any(type(t) is not int or not 0 <= t < runner.config.hf_config.vocab_size for t in draft):
                raise ValueError("invalid draft token")
            drafts.append(draft)
            budgets.append(budget)
        manager = self.scheduler.block_manager
        old_pages = [len(seq.block_table) for seq in seqs]
        try:
            for seq, draft in zip(seqs, drafts):
                manager.reserve_private(seq, len(seq) + len(draft))
            outputs, stats = runner.call("verify_greedy_batch", seqs, drafts, budgets)
        except BaseException:
            for seq, count in zip(seqs, old_pages):
                manager.truncate_private(seq, count)
            raise
        for seq, output in zip(seqs, outputs):
            committed = seq.num_cached_tokens + len(output)
            manager.truncate_private(seq, (committed + manager.block_size - 1) // manager.block_size)
            for token in output:
                seq.num_scheduled_tokens = 1
                self.scheduler.postprocess([seq], [token], False)
        return outputs, stats

    def generate_speculative_batch(self, prompts, sampling_params, *, max_draft_tokens=None, continuous=True):
        """Generate MTP-1 with continuous admission, preserving input order.

        continuous=False retains the fixed-cohort execution for paired controls.
        Online callers can add_request() and call step_speculative() directly.
        """
        runner = self.model_runner
        if not self.is_finished() or self._pending_exports:
            raise ValueError("batched speculative generation requires an idle engine")
        if max_draft_tokens is None:max_draft_tokens=getattr(runner.config,"mtp_num_steps",1)
        if (getattr(runner.model, "mtp", None) is None or type(max_draft_tokens) is not int
                or not 1 <= max_draft_tokens <= getattr(runner.config,"mtp_num_steps",1)):
            raise ValueError("batched generation requires MTP and steps within configured mtp_num_steps")
        if not getattr(runner.config, "verification_state_commit", True):
            raise ValueError("batched MTP requires intermediate state commit")
        prompts = list(prompts)
        params = ([sampling_params] * len(prompts) if isinstance(sampling_params, SamplingParams)
                  else list(sampling_params))
        if len(params) != len(prompts):
            raise ValueError("one sampling configuration per prompt required")
        encoded = []
        for prompt, param in zip(prompts, params):
            if (not param.greedy or param.repetition_penalty != 1 or param.top_k != -1
                    or param.top_p != 1 or param.max_tokens < 1):
                raise ValueError("batched MTP requires positive output budgets and unmodified greedy sampling")
            tokens = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
            if not tokens or len(tokens) + param.max_tokens > runner.config.max_model_len:
                raise ValueError("invalid prompt length")
            encoded.append(tokens)
        if continuous:
            seqs = [self.seq_map[self.add_request(tokens, param)] for tokens, param in zip(encoded, params)]
            rounds = {seq.seq_id: [] for seq in seqs}
            while not self.is_finished():
                self.step_speculative(max_draft_tokens=max_draft_tokens)
                for seq_id, stats in self.last_speculative_stats.items():
                    rounds[seq_id].append(stats)
            return [{"text": self.tokenizer.decode(seq.completion_token_ids),
                     "token_ids": list(seq.completion_token_ids), "rounds": rounds[seq.seq_id]} for seq in seqs]
        capacity = min(runner.config.max_num_seqs, runner.config.num_state_slots)
        results = []
        for start in range(0, len(prompts), capacity):
            seqs = [self.seq_map[self.add_request(tokens, param)]
                    for tokens, param in zip(encoded[start:start + capacity], params[start:start + capacity])]
            rounds = {seq.seq_id: [] for seq in seqs}
            # Preserve ordinary scheduler/prefill behavior until every live
            # cohort request has its one-pending-token frontier initialized.
            while any(not seq.is_finished and (not seq.num_completion_tokens
                                               or seq.num_cached_tokens != len(seq) - 1) for seq in seqs):
                self.step()
            while active := [seq for seq in seqs if not seq.is_finished]:
                if any(seq not in self.scheduler.running or not seq.state_ready
                       or seq.num_cached_tokens != len(seq) - 1 for seq in active):
                    # A capacity fallback can preempt/recompute a request.
                    # Re-enter ordinary scheduling until its state is ready.
                    self.step()
                    continue
                cache = runner.model.mtp.cache
                starts = {seq.state_slot: cache.get_length(seq.state_slot) for seq in active}
                drafts, draft_stats = [], []
                try:
                    drafts, draft_stats = self._draft_speculative(active, max_draft_tokens)
                    outputs, stats = self.speculative_step_batch([seq.seq_id for seq in active], drafts)
                except BaseException as error:
                    if getattr(runner, "world_size", 1) > 1:
                        runner.call("restore_mtp_lengths", starts)
                    else:
                        for slot, length in starts.items():
                            cache.set_length(slot, length)
                    if isinstance(error, MemoryError):
                        self.step()
                        continue
                    raise
                for seq, info, extra in zip(active, stats, draft_stats):
                    rounds[seq.seq_id].append(info | extra)
            results.extend({"text": self.tokenizer.decode(seq.completion_token_ids),
                            "token_ids": list(seq.completion_token_ids), "rounds": rounds[seq.seq_id]} for seq in seqs)
        return results

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # print(f"DEBUG: use_tqdm = {use_tqdm}")
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            torch.cuda.synchronize()
            t = perf_counter()
            output, num_tokens, _ = self.step()
            torch.cuda.synchronize()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
