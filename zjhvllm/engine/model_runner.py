import pickle
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

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = get_model_cls(hf_config.model_type)(hf_config)
        if config.expert_parallel_size > 1 and getattr(hf_config, "num_experts", None):
            # EP 模式：每卡只加载本地专家
            num_local = hf_config.num_experts // config.expert_parallel_size
            load_model(self.model, config.model,
                       expert_start=rank * num_local, expert_end=(rank + 1) * num_local)
        else:
            load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="zjhvllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="zjhvllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        use_fp8 = config.kv_cache_dtype == "fp8"
        kv_dtype = torch.float8_e4m3fn if use_fp8 else hf_config.dtype
        # 单个 block 占用的字节数（KV 数据 + fp8 时的 per-token scale）
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * kv_dtype.itemsize
        if use_fp8:
            block_bytes += 2 * hf_config.num_hidden_layers * self.block_size * 4  # fp32 scale（K+V 各一）
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - (used - peak + current)) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, head_dim, dtype=kv_dtype)
        self.kv_cache_scale = None
        if use_fp8:
            self.kv_cache_scale = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
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
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
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
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
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
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, *sample_args).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
