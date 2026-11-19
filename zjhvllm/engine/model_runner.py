import pickle
import gc
from datetime import timedelta
from copy import copy
from contextlib import contextmanager
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from random import randint

from zjhvllm.config import Config
from zjhvllm.engine.sequence import Sequence
from zjhvllm.models.registry import get_model_cls
from zjhvllm.layers.sampler import Sampler
from zjhvllm.utils.context import set_context, get_context, reset_context
from zjhvllm.utils.loader import load_model
from zjhvllm.utils.tp import set_tp


def _restore_spec_sequence(state):
    seq = Sequence.__new__(Sequence)
    seq.__dict__.update(state)
    return seq


class _SpecSequencePacket:
    """Full request snapshot for speculation; ordinary decode stays compact."""
    def __init__(self, seq):
        self.state = seq.__dict__

    def __reduce__(self):
        return _restore_spec_sequence, (self.state,)


def _spec_packet(value):
    if isinstance(value, Sequence):
        return _SpecSequencePacket(value)
    if isinstance(value, list):
        return [_spec_packet(item) for item in value]
    return value


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # 并行维度：EP>1 时 world_size=EP（非 MoE 层 replicated，tp=1）；否则纯 TP
        if config.expert_parallel_size > 1:
            self.world_size = config.expert_parallel_size
            self.tp_size = 1
        else:
            self.world_size = config.tensor_parallel_size
            self.tp_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        set_tp(self.tp_size, 0 if self.tp_size == 1 else rank)

        dist.init_process_group("nccl", f"tcp://127.0.0.1:{config.distributed_port}", world_size=self.world_size, rank=rank,
                                timeout=timedelta(minutes=3))
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = get_model_cls(hf_config.model_type)(hf_config)
        if config.quantization == "int8_weight_only":
            from zjhvllm.layers.weight_only import prepare_qwen35_int8_mlp
            self.quantized_mlp_count = prepare_qwen35_int8_mlp(self.model)
        if config.enable_mtp:
            from zjhvllm.models.qwen3_5_mtp import Qwen3_5MTP
            self.model.mtp = Qwen3_5MTP(hf_config, config.max_model_len, slots=config.max_num_seqs)
            hidden_shape = (config.max_model_len, hf_config.hidden_size)
            if config.max_num_seqs > 1:
                hidden_shape = (config.max_num_seqs, *hidden_shape)
            self.target_hidden = torch.empty(hidden_shape, dtype=hf_config.dtype)
        if config.expert_parallel_size > 1 and getattr(hf_config, "num_experts", None):
            # EP 模式：每卡只加载本地专家
            num_local = hf_config.num_experts // config.expert_parallel_size
            load_model(self.model, config.model,
                       expert_start=rank * num_local, expert_end=(rank + 1) * num_local)
        else:
            load_model(self.model, config.model)
        if config.quantization is not None:
            from zjhvllm.layers.weight_only import validate_quantized_weights
            validate_quantized_weights(self.model)
        if getattr(config, "qwen35_linear_backend", "decode_equivalent") == "batch_invariant":
            from zjhvllm.models.qwen3_5 import Qwen3_5Linear
            if hf_config.model_type != "qwen3_5_text" or hf_config.dtype != torch.bfloat16:
                raise ValueError("batch-invariant linear currently requires BF16 Qwen3.5")
            for module in self.model.modules():
                if isinstance(module, Qwen3_5Linear):
                    module.backend = "batch_invariant"
        self.sampler = Sampler()
        self.has_linear_state = self.allocate_linear_state()  # 需在 warmup 前分配，warmup 会跑 linear 层
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            if (not self.has_linear_state or
                    (hf_config.model_type == "qwen3_5_text" and not config.enable_mtp
                     and config.kv_cache_dtype != "fp8")):
                self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=f"zjhvllm_{config.distributed_port}", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=f"zjhvllm_{config.distributed_port}")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        for name in ("verification_graph", "verification_graphs", "verification_graph_pool"):
            if hasattr(self, name):
                delattr(self, name)
        # Fault tracebacks can retain captured tensors in Python cycles.
        # Release them while every rank's NCCL communicator is still alive.
        gc.collect()
        torch.cuda.synchronize()
        if self.world_size > 1:
            dist.barrier()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def parallel_metadata(self):
        local = {"rank": self.rank, "tp": self.tp_size,
                 "parameter_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
                 "quantization": getattr(self.config, "quantization", None),
                 "quantized_mlp_count": getattr(self, "quantized_mlp_count", 0),
                 "quantization_scale_bytes": sum(b.numel() * b.element_size()
                                                 for n, b in self.model.named_buffers()
                                                 if n.endswith("weight_scale")),
                 "state_shape": list(self.S_cache.shape) if self.has_linear_state else None,
                 "kv_shape": list(self.kv_cache.shape), "kv_blocks": self.config.num_kvcache_blocks,
                 "state_slots": self.config.num_state_slots,
                 "graph_buckets": getattr(self, "graph_bs", [])}
        if getattr(self.model, "mtp", None) is not None:
            local["mtp"] = {"loaded_groups": len(self.model.mtp.loaded_weight_names),
                            "kv_shape": list(self.model.mtp.cache._keys.shape),
                            "lengths": list(self.model.mtp.cache.lengths)}
        local["verification_graphs"] = [{"sizes": list(key), "replays": graph.replays}
                                        for key, graph in getattr(self, "verification_graphs", {}).items()]
        if self.world_size == 1:
            return [local]
        result = [None] * self.world_size
        dist.all_gather_object(result, local)
        return result

    def set_eos(self, eos):
        self.config.eos = eos
        if self.world_size > 1:
            dist.barrier()

    def set_decode_eager(self, enabled):
        """Synchronized benchmark control; captured buffers remain resident."""
        self.enforce_eager = bool(enabled)
        if self.world_size > 1:
            dist.barrier()

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        if method_name in ("draft_mtp", "draft_mtp_batch", "draft_mtp_k_batch", "verify_greedy", "verify_greedy_batch"):
            args = tuple(_spec_packet(arg) for arg in args)
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise ValueError("worker command exceeds shared-memory capacity")
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        if self.world_size > 1 and method_name in ("draft_mtp", "draft_mtp_batch", "draft_mtp_k_batch", "verify_greedy", "verify_greedy_batch"):
            return self._speculative_transaction(method, args,
                                                 draft_only=method_name in ("draft_mtp", "draft_mtp_batch", "draft_mtp_k_batch"))
        return method(*args)

    @contextmanager
    def _speculative_prepare(self):
        """Agree on recoverable preparation failures before entering forward."""
        error = None
        try:
            yield
        except Exception as exc:
            error = exc
        if getattr(self, "world_size", 1) > 1:
            statuses = [None] * self.world_size
            dist.all_gather_object(statuses, None if error is None else f"rank {self.rank}: {type(error).__name__}: {error}")
            failures = [status for status in statuses if status is not None]
            if failures and error is None:
                if all("MemoryError:" in failure for failure in failures):
                    raise MemoryError("; ".join(failures))
                raise RuntimeError("distributed speculation preparation failed: " + "; ".join(failures))
        if error is not None:
            raise error

    def _speculative_transaction(self, method, args, *, draft_only=False):
        """Vote before publication; recover rank-local errors after collectives.

        A failed NCCL collective/process is fatal to the distributed engine.
        Recoverable Python failures at transaction boundaries keep workers alive.
        """
        from zjhvllm.engine.speculative import HybridCheckpoint
        seqs = args[0] if isinstance(args[0], list) else [args[0]]
        try:
            with self._speculative_prepare():
                checkpoints = ([] if draft_only else
                    [HybridCheckpoint.capture(self.S_cache, self.conv_cache, seq.state_slot) for seq in seqs])
                cache = getattr(self.model, "mtp", None)
                lengths = list(cache.cache.lengths) if cache is not None else None
        except Exception:
            if self.rank == 0:
                raise
            return None
        error, result = None, None
        try:
            result = method(*args)
        except Exception as exc:
            error = exc
        statuses = [None] * self.world_size
        dist.all_gather_object(statuses, None if error is None else f"rank {self.rank}: {type(error).__name__}: {error}")
        failures = [status for status in statuses if status is not None]
        if failures:
            for checkpoint in checkpoints:
                checkpoint.restore(self.S_cache, self.conv_cache)
            if lengths is not None:
                cache.cache.lengths[:] = lengths
            reset_context()
        # Ensure worker rollback is complete before rank 0 publishes/falls back.
        dist.barrier()
        if failures and self.rank == 0:
            if error is not None:
                raise error
            if all("MemoryError:" in failure for failure in failures):
                raise MemoryError("; ".join(failures))
            raise RuntimeError("distributed speculation rolled back: " + "; ".join(failures))
        return result

    def restore_mtp_lengths(self, lengths):
        for slot, length in lengths.items():
            self.model.mtp.cache.set_length(slot, length)
        if self.world_size > 1:
            dist.barrier()

    def _root_predictions(self, predictions):
        if getattr(self, "world_size", 1) > 1:
            dist.broadcast(predictions, src=0)
        return predictions

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for i, seq in enumerate(seqs):
            seq.num_scheduled_tokens = seq_len
            if self.has_linear_state:
                seq.state_slot = i % self.config.num_state_slots
                seq.state_ready = True  # warmup 直接复用零初始化状态，跳过 prepare 清零
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_linear_state(self) -> bool:
        """为 linear attention 层（Qwen3.5 的 Gated DeltaNet）分配递归状态缓存。

        返回模型是否有 linear 层（决定是否跳过 CUDA graph）。S 用 fp32（对齐 transformers
        的 mamba_ssm_dtype=float32），conv_state 用 bf16。每个 slot 对应一个序列，
        slot 数按剩余显存保守分配（27B 每槽约147MiB/TP，各rank取共同容量）。
        """
        hf_config = self.config.hf_config
        layer_types = getattr(hf_config, "layer_types", None)
        num_linear = sum(1 for t in layer_types if t == "linear_attention") if layer_types else 0
        if num_linear == 0:
            self.config.num_state_slots = 0
            return False

        nk, nv = hf_config.linear_num_key_heads // self.tp_size, hf_config.linear_num_value_heads // self.tp_size
        hk, hv = hf_config.linear_key_head_dim, hf_config.linear_value_head_dim
        conv_dim = 2 * nk * hk + nv * hv
        kernel = hf_config.linear_conv_kernel_dim
        per_slot = num_linear * (nv * hk * hv * 4 + conv_dim * (kernel - 1) * 2)

        free, total = torch.cuda.mem_get_info()
        num_slots = max(1, int(free * 0.25 // per_slot))  # 剩余显存 1/4 保守给递归状态
        num_slots = min(num_slots, self.config.max_num_seqs)
        if self.world_size > 1:
            common = torch.tensor([num_slots], dtype=torch.int64, device="cuda")
            dist.all_reduce(common, op=dist.ReduceOp.MIN)
            num_slots = int(common.item())
        self.config.num_state_slots = num_slots

        self.S_cache = torch.zeros(num_slots, num_linear, nv, hk, hv, dtype=torch.float32)
        self.conv_cache = torch.zeros(num_slots, num_linear, conv_dim, kernel - 1, dtype=hf_config.dtype)

        lin_idx = 0
        for module in self.model.modules():
            if hasattr(module, "S_cache") and hasattr(module, "conv_cache"):
                module.S_cache = self.S_cache[:, lin_idx]
                module.conv_cache = self.conv_cache[:, lin_idx]
                lin_idx += 1
        if getattr(self.config, "enable_mtp", False):
            # Reserve MTP-1 scratch before KV sizing, not from unbudgeted
            # memory after the KV pool has consumed the configured budget.
            steps = getattr(self.config, "mtp_num_steps", 1)
            history = self.prepare_verification_history((steps + 1) * num_slots)
            if steps > 1 and history is None:
                raise ValueError("increase verification_state_budget_mb for the configured MTP window")
        return True

    def prepare_verification_history(self, count, cu_seqlens=None):
        from zjhvllm.engine.speculative import HybridStateHistory
        from zjhvllm.models.qwen3_5 import GatedDeltaNet
        if not getattr(self.config, "verification_state_commit", True):
            return None
        layers = [m for m in getattr(self.model, "modules", lambda: [])() if isinstance(m, GatedDeltaNet)]
        if not layers:
            return None
        per_step = (self.S_cache[0].numel() * self.S_cache.element_size()
                    + self.conv_cache[0].numel() * self.conv_cache.element_size())
        budget = (getattr(self.config, "verification_state_budget_mb", 512) * 1024**2
                  * getattr(self.config, "max_num_seqs", 1))
        if count * per_step > budget:
            # Generic prompt-lookup windows can be wider than MTP-1. Preserve
            # their correct restore/replay path without unbounded allocations.
            return None
        history = getattr(self, "verification_history", None)
        if history is None or len(history.states) < count:
            try:
                history = HybridStateHistory(self.S_cache, self.conv_cache, count)
            except torch.OutOfMemoryError as error:
                raise MemoryError("insufficient verification state workspace") from error
            self.verification_history = history
        return history.bind(layers, count, cu_seqlens)

    def run_verification(self, ids, positions, history):
        """Bounded exact-layout graph cache, including TP collectives.

        Never pad hybrid requests or change their recurrent update count.
        Wider lookup windows / FP8 / excess layouts retain eager verification.
        """
        context = get_context()
        context.decode_batched = (getattr(self.config, "qwen35_linear_backend", "decode_equivalent") == "batched")
        context.verification_fused_gdn = getattr(self.config, "verification_fused_gdn", False)
        cu = context.cu_seqlens_q_cpu
        sizes = [b - a for a, b in zip(cu[:-1], cu[1:])] if cu is not None else []
        use_graph = (getattr(self.config, "verification_cuda_graph", False)
                     and history is not None and sizes and max(sizes) <= 9
                     and len(sizes) <= 8
                     and getattr(self.config, "kv_cache_dtype", "auto") != "fp8"
                     and ids.is_cuda)
        self.last_verification_graph_replay = False
        if use_graph:
            from zjhvllm.engine.verification_graph import VerificationGraph
            graphs = getattr(self, "verification_graphs", None)
            if graphs is None or getattr(self, "verification_graph_history", None) is not history:
                self.verification_graphs = graphs = {}
                self.verification_graph_history = history
                self.verification_graph_pool = None
            key = ("fused_gdn", *sizes) if context.verification_fused_gdn else tuple(sizes)
            graph = graphs.get(key)
            if graph is None and len(graphs) < getattr(self.config, "verification_graph_max_layouts", 16):
                max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
                graph = VerificationGraph(self.model, ids, positions, history, max_blocks,
                                          pool=self.verification_graph_pool, prepare_guard=self._speculative_prepare)
                graphs[key] = graph
                if self.verification_graph_pool is None:
                    self.verification_graph_pool = graph.graph.pool()
            if graph is not None:
                self.verification_graph = graph
                self.last_verification_graph_replay = True
                return graph.replay(ids, positions)
        hidden = self.model(ids, positions)
        return hidden, self.model.compute_logits(hidden, all_tokens=True).argmax(-1)

    def set_verification_graph(self, enabled):
        self.config.verification_cuda_graph = bool(enabled)
        if self.world_size > 1:
            dist.barrier()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = max(1, hf_config.num_key_value_heads // self.tp_size)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        use_fp8 = config.kv_cache_dtype == "fp8"
        kv_dtype = torch.float8_e4m3fn if use_fp8 else hf_config.dtype
        # hybrid 模型（如 Qwen3.5）只有 full_attention 层用 KV cache，linear 层用递归状态
        layer_types = getattr(hf_config, "layer_types", None)
        num_kv_layers = sum(1 for t in layer_types if t == "full_attention") if layer_types else hf_config.num_hidden_layers
        # 单个 block 占用的字节数（KV 数据 + fp8 时的 per-token scale）
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * kv_dtype.itemsize
        if use_fp8:
            block_bytes += 2 * num_kv_layers * self.block_size * 4  # fp32 scale（K+V 各一）
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - (used - peak + current)) // block_bytes
        if self.world_size > 1:
            common = torch.tensor([config.num_kvcache_blocks], dtype=torch.int64, device="cuda")
            dist.all_reduce(common, op=dist.ReduceOp.MIN)
            config.num_kvcache_blocks = int(common.item())
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, head_dim, dtype=kv_dtype)
        self.kv_cache_scale = None
        if use_fp8:
            self.kv_cache_scale = torch.empty(2, num_kv_layers, config.num_kvcache_blocks,
                                              self.block_size, dtype=torch.float32)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if use_fp8:
                    module.k_scale = self.kv_cache_scale[0, layer_id]
                    module.v_scale = self.kv_cache_scale[1, layer_id]
                layer_id += 1

    def prepare_state_slots(self, seqs: list[Sequence]):
        """构建递归状态 slot 索引张量，并清零首次 prefill（state_ready=False）序列的状态。"""
        if not self.has_linear_state:
            return None
        slots = []
        for seq in seqs:
            if seq.state_slot is None:
                seq.state_slot = 0  # warmup 占位（warmup 已置 state_ready=True，不会误清零）
            if not seq.state_ready:
                self.S_cache[seq.state_slot].zero_()
                self.conv_cache[seq.state_slot].zero_()
                if getattr(self.model, "mtp", None) is not None:
                    self.model.mtp.cache.set_length(seq.state_slot, 0)
                seq.state_ready = True
            slots.append(seq.state_slot)
        return torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q_cpu = cu_seqlens_q
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_slots = self.prepare_state_slots(seqs)
        keep_host = getattr(self.config, "cache_schedule_metadata", True)
        slots_cpu = [seq.state_slot for seq in seqs] if self.has_linear_state and keep_host else None
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, state_slots,
                    cu_seqlens_q_cpu=cu_q_cpu if keep_host else None, state_slots_cpu=slots_cpu)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        state_slots = self.prepare_state_slots(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, state_slots=state_slots)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """把每个序列的采样参数打包成设备张量，供采样器批量消费。"""
        temperatures = [seq.temperature for seq in seqs]
        top_ks = [seq.top_k for seq in seqs]
        top_ps = [seq.top_p for seq in seqs]
        penalties = [seq.repetition_penalty for seq in seqs]
        greedy_flags = [int(seq.greedy) for seq in seqs]

        # seed：greedy 不需要随机数，统一给 0 避免无谓的 Philox 开销；
        # 无固定 seed 的序列使用一次性的随机种子（保证每次运行不同）。
        seeds = []
        for seq in seqs:
            if seq.greedy:
                seeds.append(0)
            elif seq.seed is None:
                seeds.append(randint(1, 2**63 - 1))
            else:
                if seq.last_ran_seed is None:
                    seq.last_ran_seed = seq.seed
                seeds.append(seq.last_ran_seed)

        # 序列 → 已生成历史（用于 repetition penalty；仅当任一序列启用惩罚时才有意义）
        max_hist = max((seq.num_completion_tokens for seq in seqs if seq.repetition_penalty != 1.0), default=0)
        if max_hist:
            hist_token_ids = torch.zeros(len(seqs), max_hist, dtype=torch.int32, pin_memory=True)
            for i, seq in enumerate(seqs):
                hist_token_ids[i, :seq.num_completion_tokens] = torch.tensor(
                    seq.completion_token_ids, dtype=torch.int32
                )
            hist_lens = torch.tensor([seq.num_completion_tokens for seq in seqs], dtype=torch.int32)
            hist_token_ids = hist_token_ids.cuda(non_blocking=True)
            hist_lens = hist_lens.cuda(non_blocking=True)
        else:
            hist_token_ids = hist_lens = None

        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        penalties = torch.tensor(penalties, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        seeds = torch.tensor(seeds, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        greedy_flags = torch.tensor(greedy_flags, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return (temperatures, top_ks, top_ps, penalties, seeds, greedy_flags, hist_token_ids, hist_lens)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if (not is_prefill and getattr(self.config, "enable_mtp", False)
                and getattr(self.config, "max_num_seqs", 1) > 1):
            # MTP-enabled multi-request baseline and verification must share
            # single-token arithmetic, even as active batch membership changes.
            get_context().decode_equivalent = True
            get_context().decode_batched = (getattr(self.config, "qwen35_linear_backend", "decode_equivalent") == "batched")
        hybrid_graph = (self.has_linear_state and hasattr(self, "graphs")
                        and input_ids.size(0) in self.graphs)
        if is_prefill or self.enforce_eager or (self.has_linear_state and not hybrid_graph) or input_ids.size(0) > 512:
            hidden = self.model(input_ids, positions)
            self.record_target_hidden(hidden, positions)
            return self.model.compute_logits(hidden)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[bs if self.has_linear_state else next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if self.has_linear_state:
                graph_vars["state_slots"][:bs].copy_(context.state_slots)
            graph.replay()
            self.decode_graph_replays = getattr(self, "decode_graph_replays", 0) + 1
            self.record_target_hidden(graph_vars["outputs"][:bs], positions)
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.no_grad()
    def record_target_hidden(self, hidden, positions):
        if hasattr(self, "target_hidden"):
            if self.target_hidden.ndim == 2:
                self.target_hidden[positions] = hidden.detach()
            else:
                context = get_context()
                if context.is_prefill:
                    cu = context.cu_seqlens_q_cpu or context.cu_seqlens_q.tolist()
                    slots = context.state_slots_cpu or context.state_slots.tolist()
                    for b, slot in enumerate(slots):
                        start, end = cu[b:b + 2]
                        self.target_hidden[slot, positions[start:end]] = hidden[start:end].detach()
                else:
                    self.target_hidden[context.state_slots.long(), positions] = hidden.detach()

    @torch.inference_mode()
    def draft_mtp(self, seq: Sequence):
        """Predict ONE future token from checkpoint MTP, using committed pairs.

        Pair at draft position t is (target post-norm h_t, token at t+1).
        Only committed target positions are extended, so rejected target suffixes
        never enter draft KV. KV length is restored if draft computation fails.
        """
        with self._speculative_prepare():
            mtp = getattr(self.model, "mtp", None)
            if mtp is None:
                raise ValueError("construct the engine with enable_mtp=True")
            if seq.num_tokens != seq.num_cached_tokens + 1 or seq.state_slot is None or not seq.state_ready:
                raise ValueError("MTP requires a prefilled request with one pending token")
        from contextlib import nullcontext
        selection = mtp.cache.for_slot(seq.state_slot) if hasattr(mtp.cache, "for_slot") else nullcontext()
        with selection:
            with self._speculative_prepare():
                start, end = mtp.cache.length, seq.num_cached_tokens
                if not 0 <= start < end:
                    raise ValueError("MTP needs new committed target hidden states")
            try:
                with self._speculative_prepare():
                    ids = torch.tensor(seq.token_ids[start + 1:end + 1], dtype=torch.int64, device=self.target_hidden.device)
                    target = self.target_hidden if self.target_hidden.ndim == 2 else self.target_hidden[seq.state_slot]
                hidden = mtp(self.model.model.embed_tokens(ids), target[start:end])
                token = self._root_predictions(self.model.lm_head(hidden[-1:]).argmax(-1)).item()
                return [token], {"draft_source": "mtp", "draft_extend_tokens": end - start,
                                 "mtp_cache_tokens": end}
            except BaseException:
                mtp.cache.length = start
                raise

    @torch.inference_mode()
    def draft_mtp_batch(self, seqs: list[Sequence]):
        """One packed draft/head pass and one prediction broadcast per batch."""
        with self._speculative_prepare():
            mtp = getattr(self.model, "mtp", None)
            if mtp is None or not seqs:
                raise ValueError("packed draft requires MTP and nonempty requests")
            slots = [seq.state_slot for seq in seqs]
            if None in slots or len(set(slots)) != len(slots):
                raise ValueError("packed draft requires distinct initialized slots")
            starts, ends = [], []
            for seq in seqs:
                if seq.num_tokens != seq.num_cached_tokens + 1 or not seq.state_ready:
                    raise ValueError("MTP requires prefilled requests with one pending token")
                start, end = mtp.cache.get_length(seq.state_slot), seq.num_cached_tokens
                if not 0 <= start < end:
                    raise ValueError("MTP needs new committed target hidden states")
                starts.append(start)
                ends.append(end)
        try:
            with self._speculative_prepare():
                sizes = [end - start for start, end in zip(starts, ends)]
                ids = torch.tensor([token for seq, start, end in zip(seqs, starts, ends)
                                    for token in seq.token_ids[start + 1:end + 1]],
                                   dtype=torch.int64, device=self.target_hidden.device)
                target = torch.cat([(self.target_hidden if self.target_hidden.ndim == 2
                                     else self.target_hidden[slot])[start:end]
                                    for slot, start, end in zip(slots, starts, ends)])
                last_rows, offset = [], 0
                for size in sizes:
                    offset += size
                    last_rows.append(offset - 1)
                last_rows = torch.tensor(last_rows, device=ids.device, dtype=torch.int64)
            hidden = mtp.forward_batch(self.model.model.embed_tokens(ids), target, slots, sizes)
            tokens = self._root_predictions(self.model.lm_head(hidden[last_rows]).argmax(-1)).tolist()
            return [([token], {"draft_source": "mtp_batched", "draft_extend_tokens": size,
                               "mtp_cache_tokens": end})
                    for token, size, end in zip(tokens, sizes, ends)]
        except BaseException:
            for slot, start in zip(slots, starts):
                mtp.cache.set_length(slot, start)
            raise

    @torch.inference_mode()
    def draft_mtp_k_batch(self, seqs: list[Sequence], limits: list[int]):
        """Recursive top-1 chain, reusing the one-layer checkpoint (SGLang).

        The first extension uses only committed target hidden/token pairs.
        Later steps condition on the preceding draft hidden and prediction.
        Their KV is speculative even if the target later accepts every token:
        always truncate it and rebuild from true target hidden next round.
        """
        with self._speculative_prepare():
            mtp = getattr(self.model, "mtp", None)
            if mtp is None or not seqs or len(seqs) != len(limits):
                raise ValueError("recursive draft requires MTP and one limit per request")
            if any(type(k) is not int or not 1 <= k <= 8 for k in limits):
                raise ValueError("recursive draft limits must be between 1 and 8")
            slots = [seq.state_slot for seq in seqs]
            if None in slots or len(set(slots)) != len(slots):
                raise ValueError("recursive draft requires distinct initialized slots")
            starts, ends = [], []
            for seq, limit in zip(seqs, limits):
                if seq.num_tokens != seq.num_cached_tokens + 1 or not seq.state_ready:
                    raise ValueError("MTP requires prefilled requests with one pending token")
                start, end = mtp.cache.get_length(seq.state_slot), seq.num_cached_tokens
                if not 0 <= start < end or end + limit - 1 > mtp.cache._keys.shape[1]:
                    raise ValueError("invalid committed MTP frontier or recursive KV capacity")
                starts.append(start); ends.append(end)
        try:
            with self._speculative_prepare():
                sizes = [end-start for start,end in zip(starts,ends)]
                ids = torch.tensor([token for seq,start,end in zip(seqs,starts,ends)
                                    for token in seq.token_ids[start+1:end+1]],
                                   dtype=torch.int64, device=self.target_hidden.device)
                target = torch.cat([(self.target_hidden if self.target_hidden.ndim==2
                                     else self.target_hidden[slot])[start:end]
                                    for slot,start,end in zip(slots,starts,ends)])
                offsets, total = [], 0
                for size in sizes:
                    total += size; offsets.append(total-1)
                last_rows = torch.tensor(offsets, device=ids.device, dtype=torch.int64)
                proposals = torch.full((len(seqs),max(limits)), -1, dtype=torch.int64, device=ids.device)
                active_steps = [[i for i,limit in enumerate(limits) if limit > step]
                                for step in range(max(limits))]
                active_indices = [torch.tensor(active, device=ids.device, dtype=torch.int64)
                                  for active in active_steps]
                keep_indices = [None] + [torch.tensor([active_steps[step-1].index(i)
                                                       for i in active_steps[step]],
                                                      device=ids.device, dtype=torch.int64)
                                         for step in range(1,max(limits))]
            hidden = mtp.forward_batch(self.model.model.embed_tokens(ids), target, slots, sizes)[last_rows]
            for step in range(max(limits)):
                if step:
                    # All request/selection metadata was prepared before the
                    # collective sequence; no per-step host vote or token read.
                    indices = keep_indices[step]
                    hidden = mtp.forward_batch(self.model.model.embed_tokens(prediction[indices]), hidden[indices],
                        [slots[i] for i in active_steps[step]], [1]*len(active_steps[step]))
                prediction = self._root_predictions(self.model.lm_head(hidden).argmax(-1))
                proposals[:,step].index_copy_(0,active_indices[step],prediction)
            tokens = proposals.tolist()  # one host transfer, no per-step .item()
            for slot,end in zip(slots,ends):
                mtp.cache.set_length(slot,end)
            return [(row[:limit], {"draft_source":"mtp_recursive", "draft_steps":limit,
                                   "draft_extend_tokens":size,"draft_recursive_pairs":limit-1,
                                   "mtp_cache_tokens":end})
                    for row,limit,size,end in zip(tokens,limits,sizes,ends)]
        except BaseException:
            for slot,start in zip(slots,starts):
                mtp.cache.set_length(slot,start)
            raise

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        try:
            with torch.cuda.nvtx.range("prepare"):
                input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
                sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
            with torch.cuda.nvtx.range("run_model"):
                logits = self.run_model(input_ids, positions, is_prefill)
            with torch.cuda.nvtx.range("sampler"):
                return self.sampler(logits, *sample_args).tolist() if self.rank == 0 else None
        finally:
            reset_context()

    @torch.inference_mode()
    def verify_greedy(self, seq: Sequence, draft: list[int], max_outputs: int, verification_mode="serial"):
        """Reference serial verification, or opt-in batched restore/replay.

        KV reservation/metadata commit belong to the engine. Only the suffix at
        positions >= num_cached_tokens is written. Exceptions restore S+conv;
        the engine then unpublishes the temporary KV pages.
        """
        from zjhvllm.engine.speculative import HybridCheckpoint, greedy_prefix
        with self._speculative_prepare():
            if (not self.has_linear_state
                    or self.config.hf_config.model_type != "qwen3_5_text"):
                raise NotImplementedError("experimental verification requires Qwen3.5")
            if seq.state_slot is None or not seq.state_ready or seq.num_tokens != seq.num_cached_tokens + 1:
                raise ValueError("verification requires initialized state and one pending token")
            if verification_mode not in ("serial", "batched_experimental"):
                raise ValueError("unknown verification mode")
            if max_outputs < 1:
                raise ValueError("positive output budget required")
            checkpoint = HybridCheckpoint.capture(self.S_cache, self.conv_cache, seq.state_slot)
            trial = copy(seq)
            trial.token_ids = seq.token_ids + list(draft)
            trial.num_tokens = len(trial.token_ids)
            trial.last_token = trial.token_ids[-1]
            trial.num_scheduled_tokens = len(draft) + 1
        try:
            if verification_mode == "serial":
                # Correctness reference: use exactly the ordinary decode shapes
                # and kernels. No acceleration claim: this is up to K+1 forwards.
                output, accepted = [], 0
                for index in range(min(max_outputs, len(draft) + 1)):
                    trial.num_tokens = seq.num_tokens + index
                    trial.token_ids = (seq.token_ids + draft)[:trial.num_tokens]
                    trial.last_token = trial.token_ids[-1]
                    trial.block_table = seq.block_table[:(trial.num_tokens + self.block_size - 1) // self.block_size]
                    with self._speculative_prepare():
                        ids, positions = self.prepare_decode([trial])
                    token = self._root_predictions(self.run_model(ids, positions, False).argmax(-1)).item()
                    output.append(token)
                    matches = index < len(draft) and draft[index] == token
                    accepted += int(matches)
                    if not matches or (not seq.ignore_eos and token == self.config.eos):
                        break
                return output, {"proposed": len(draft), "accepted": accepted, "replay_tokens": 0,
                                "snapshot_bytes": checkpoint.nbytes, "verification_mode": verification_mode,
                                "target_forwards": len(output)}
            with self._speculative_prepare():
                ids, positions = self.prepare_prefill([trial])
                self.prepare_verification_context(seq.num_cached_tokens, len(ids), positions.device)
                history = self.prepare_verification_history(len(ids))
                get_context().verification_state = history
            hidden, predictions = self.run_verification(ids, positions, history)
            target = self._root_predictions(predictions).tolist()
            output, accepted = greedy_prefix(draft, target, max_outputs,
                                               None if seq.ignore_eos else self.config.eos)
            replay_tokens = 0
            if history is not None:
                history.commit(self.S_cache, self.conv_cache, seq.state_slot, len(output))
                self.record_target_hidden(hidden[:len(output)], positions[:len(output)])
            else:
                self.record_target_hidden(hidden, positions)
            if history is None and len(output) < trial.num_scheduled_tokens:
                checkpoint.restore(self.S_cache, self.conv_cache)
                # Emitting N tokens commits N inputs: the pending token plus
                # N-1 proposals. The last emitted token is the next pending one.
                trial.num_scheduled_tokens = len(output)
                ids, positions = self.prepare_prefill([trial])
                self.prepare_verification_context(seq.num_cached_tokens, len(ids), positions.device)
                hidden = self.model(ids, positions)
                self.record_target_hidden(hidden, positions)
                replay_tokens = len(output)
            return output, {"proposed": len(draft), "accepted": accepted,
                            "replay_tokens": replay_tokens, "snapshot_bytes": checkpoint.nbytes,
                            "state_strategy": "intermediate" if history is not None else "replay",
                            "state_history_bytes": history.nbytes if history is not None else 0,
                            "state_commit_tokens": len(output),
                            "verification_graph": self.last_verification_graph_replay,
                            "verification_mode": verification_mode, "target_forwards": 1 + int(replay_tokens > 0)}
        except BaseException:
            checkpoint.restore(self.S_cache, self.conv_cache)
            raise
        finally:
            reset_context()

    @staticmethod
    def prepare_verification_context(cached_tokens, count, device):
        """Use decode arithmetic for both speculative evaluation and replay."""
        context = get_context()
        context.decode_equivalent = True
        cached = cached_tokens if isinstance(cached_tokens, list) else [cached_tokens]
        counts = count if isinstance(count, list) else [count]
        cu, lengths = [0], []
        for start, size in zip(cached, counts):
            cu.append(cu[-1] + size)
            lengths.extend(range(start + 1, start + size + 1))
        context.cu_seqlens_q_cpu = cu
        context.verification_context_lens = torch.tensor(lengths, device=device, dtype=torch.int32)

    @torch.inference_mode()
    def verify_greedy_batch(self, seqs, drafts, max_outputs):
        """One target traversal, independent greedy decisions and state commits.

        All persistent recurrent states remain unchanged until all logits and
        acceptance decisions are available. A commit fault restores every
        selected request; the engine owns KV reservation and metadata publish.
        """
        from zjhvllm.engine.speculative import HybridCheckpoint
        from zjhvllm.layers.speculative_accept import greedy_accept_batch
        with self._speculative_prepare():
            if self.config.hf_config.model_type != "qwen3_5_text":
                raise NotImplementedError("batched speculation requires Qwen3.5")
            if not seqs or len(seqs) != len(drafts) or len(seqs) != len(max_outputs):
                raise ValueError("verification request dimensions differ")
            slots = [seq.state_slot for seq in seqs]
            if None in slots or len(set(slots)) != len(slots):
                raise ValueError("verification state slots must be distinct")
            for seq, draft, budget in zip(seqs, drafts, max_outputs):
                if not seq.state_ready or len(seq) != seq.num_cached_tokens + 1:
                    raise ValueError("each verification request needs one pending token")
                if len(draft) > 32 or budget < 1:
                    raise ValueError("invalid per-request verification window")
            checkpoints = [HybridCheckpoint.capture(self.S_cache, self.conv_cache, slot) for slot in slots]
            trials, sizes, cu = [], [], [0]
            for seq, draft in zip(seqs, drafts):
                trial = copy(seq)
                trial.token_ids = seq.token_ids + list(draft)
                trial.num_tokens = len(trial.token_ids)
                trial.last_token = trial.token_ids[-1]
                trial.num_scheduled_tokens = len(draft) + 1
                sizes.append(trial.num_scheduled_tokens)
                cu.append(cu[-1] + sizes[-1])
                trials.append(trial)
        try:
            with self._speculative_prepare():
                history = self.prepare_verification_history(cu[-1], cu)
                if history is None:
                    raise MemoryError("batch exceeds intermediate verification state budget")
                ids, positions = self.prepare_prefill(trials)
                self.prepare_verification_context([s.num_cached_tokens for s in seqs], sizes, positions.device)
                get_context().verification_state = history
            hidden, predictions = self.run_verification(ids, positions, history)
            target = self._root_predictions(predictions)
            outputs, accepted = greedy_accept_batch(
                target, ids, get_context().cu_seqlens_q, max_outputs,
                [s.ignore_eos for s in seqs], self.config.eos, max(sizes))
            counts = [len(output) for output in outputs]
            history.commit_many(self.S_cache, self.conv_cache, slots, counts)
            if hasattr(self, "target_hidden"):
                for b, (slot, count) in enumerate(zip(slots, counts)):
                    dest = self.target_hidden if self.target_hidden.ndim == 2 else self.target_hidden[slot]
                    dest[positions[cu[b]:cu[b] + count]] = hidden[cu[b]:cu[b] + count]
            stats = [{"proposed": len(draft), "accepted": count, "replay_tokens": 0,
                      "snapshot_bytes": checkpoint.nbytes, "state_strategy": "intermediate",
                      "state_commit_tokens": len(output), "verification_mode": "batched_experimental",
                      "batch_requests": len(seqs), "target_forwards": int(b == 0),
                      "verification_graph": self.last_verification_graph_replay}
                     for b, (draft, count, output, checkpoint) in
                     enumerate(zip(drafts, accepted, outputs, checkpoints))]
            return outputs, stats
        except BaseException:
            for checkpoint in checkpoints:
                checkpoint.restore(self.S_cache, self.conv_cache)
            raise
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        if self.has_linear_state:
            max_bs = min(max_bs, self.config.num_state_slots, 128)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        parameter = next(self.model.parameters())
        device = parameter.device
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, dtype=parameter.dtype, device=device)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        state_slots = None
        if self.has_linear_state:
            # Never pad hybrid batches: two lanes writing the same state slot
            # would race. Uncaptured exact batch sizes fall back to eager.
            self.graph_bs = sorted({n for n in [1,2,4,8,max_bs]+list(range(16,max_bs+1,16)) if n <= max_bs})
            state_slots = torch.arange(max_bs, dtype=torch.int32, device=device)
            slot_mapping.fill_(-1)
            context_lens.fill_(1)
            self.kv_cache[:, :, 0].zero_()
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        state_slots=state_slots[:bs] if state_slots is not None else None)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            if self.has_linear_state:
                # Capture occurs before admission, so all persistent slots are
                # disposable warmup state; no large per-request backup needed.
                self.S_cache.zero_()
                self.conv_cache.zero_()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            state_slots=state_slots,
        )
