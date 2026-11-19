import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from zjhvllm.utils.tp import get_tp_rank, get_tp_size

from zjhvllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        nn.init.normal_(self.weight)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits


class Qwen3_5VocabParallelHead(VocabParallelEmbedding):
    """Column-parallel LM head over vocab: local GEMM + all-gather.

    Unlike ParallelLMHead (which gathers to rank 0 only), every rank receives
    full-vocab logits so greedy argmax works on all ranks in the verification /
    draft path. Each rank computes only its vocab shard, then all-gathers the
    shards back together in rank order. This changes BF16 GEMM rounding vs the
    replicated single-GEMM head (accepting rounding, like qwen35_linear_backend
    = "batched"); embedding sharding above remains bit-exact.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Keep the same arithmetic as Qwen3_5Linear: rowwise (bit-exact) under
        # decode_equivalent, batched otherwise. Sharding the vocab across ranks
        # is bit-exact for the rowwise path -- each row's reduction over hidden
        # is unchanged, and all-gather only concatenates the rank-ordered shards.
        if (get_context().decode_equivalent and not get_context().decode_batched
                and hidden_states.shape[0] > 1):
            local = torch.cat([F.linear(row, self.weight) for row in hidden_states.split(1)], dim=0)
        else:
            local = F.linear(hidden_states, self.weight)
        if self.tp_size == 1:
            return local
        gathered = [torch.empty_like(local) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local)
        return torch.cat(gathered, dim=-1)
