"""CUDA replay of the exact request-private MTP-1 verification arithmetic.

Only scratch recurrent states are written during capture/replay. Publishing
state, hidden history and tokens remains the caller's transactional operation.
"""
from contextlib import nullcontext

import torch
import torch.distributed as dist

from zjhvllm.utils.context import get_context


class VerificationGraph:
    def __init__(self, model, ids, positions, history, max_blocks, pool=None, prepare_guard=None):
        context = get_context()
        with prepare_guard() if prepare_guard is not None else nullcontext():
            self.history = history
            self.replays = 0
            self.ids, self.positions = ids.clone(), positions.clone()
            self.slot_mapping = context.slot_mapping.clone()
            self.slots = context.state_slots.clone()
            self.cu_seqlens = context.cu_seqlens_q.clone()
            self.lengths = context.verification_context_lens.clone()
            self.block_tables = torch.zeros((len(context.cu_seqlens_q_cpu) - 1, max_blocks), device=ids.device, dtype=torch.int32)
            self.update(ids, positions)
        original = (context.slot_mapping, context.state_slots, context.cu_seqlens_q,
                    context.verification_context_lens, context.block_tables)
        context.slot_mapping, context.state_slots = self.slot_mapping, self.slots
        context.cu_seqlens_q = self.cu_seqlens
        context.verification_context_lens, context.block_tables = self.lengths, self.block_tables
        # Warm on a side stream: initialize cuBLAS/Triton before capture.
        stream = torch.cuda.Stream(device=ids.device)
        stream.wait_stream(torch.cuda.current_stream(ids.device))
        try:
            with torch.cuda.stream(stream):
                self._forward(model)
            torch.cuda.current_stream(ids.device).wait_stream(stream)
            # All ranks finish compilation/warmup before any rank captures
            # NCCL. Replay contains the same rowwise reductions as eager.
            if dist.is_initialized() and dist.get_world_size() > 1:
                torch.cuda.synchronize(ids.device)
                dist.barrier()
                torch.cuda.synchronize(ids.device)
            self.graph = torch.cuda.CUDAGraph()
            history.written.clear()
            with torch.cuda.graph(self.graph, pool=pool, stream=stream):
                self.hidden, self.target = self._forward(model)
            torch.cuda.current_stream(ids.device).wait_stream(stream)
        finally:
            (context.slot_mapping, context.state_slots, context.cu_seqlens_q,
             context.verification_context_lens, context.block_tables) = original

    def _forward(self, model):
        self.history.written.clear()
        hidden = model(self.ids, self.positions)
        self.logits = model.compute_logits(hidden, all_tokens=True)
        target = self.logits.argmax(-1)
        return hidden, target

    def update(self, ids, positions):
        context = get_context()
        self.ids.copy_(ids)
        self.positions.copy_(positions)
        self.slot_mapping.copy_(context.slot_mapping)
        self.slots.copy_(context.state_slots)
        self.cu_seqlens.copy_(context.cu_seqlens_q)
        self.lengths.copy_(context.verification_context_lens)
        width = context.block_tables.shape[1]
        self.block_tables.fill_(-1)
        self.block_tables[:, :width].copy_(context.block_tables)

    def replay(self, ids, positions):
        self.update(ids, positions)
        self.graph.replay()
        self.replays += 1
        # Python traversal is captured once, while bind clears this each round.
        self.history.written.update(self.history.layer_indices.values())
        return self.hidden, self.target
