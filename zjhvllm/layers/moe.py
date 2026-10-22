"""稀疏 Mixture-of-Experts 层，支持单卡和专家并行（EP）。

每个 token 由 router 选出 top-k 个专家，各自算 FFN 后按路由权重加权求和。
- 单卡（world_size==1）：本地 loop 遍历专家。
- EP（world_size>1，且 TP==1）：专家按 rank 分布（每卡 num_experts//world_size 个），
  token 通过 all-to-all 路由到活跃专家所在卡。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from zjhvllm.layers.activation import SiluAndMul
from zjhvllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear


class ExpertFFN(nn.Module):
    """单个专家的 SwiGLU FFN：gate_up（silu gate ⊙ up）→ down。

    复用 TP-aware 的并行线性层（MergedColumnParallelLinear / RowParallelLinear），
    这样 gate_proj/up_proj 的合并权重加载、以及 TP 分片都能复用 dense MLP 的机制。
    权重命名 gate_up_proj 与 Qwen3 的 packed_modules_mapping 对齐。
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class MoE(nn.Module):
    """稀疏 MoE 层：router（gate）→ top-k 专家选择 → 加权组合（单卡或 EP）。"""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert num_experts % self.world_size == 0, "num_experts 必须能被 world_size 整除"
        self.num_local_experts = num_experts // self.world_size
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)  # router（replicated）
        # 单卡持有全部专家；EP 只持有本地专家（每卡 num_experts//world_size 个）
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, moe_intermediate_size) for _ in range(self.num_local_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_tokens, hidden_size] → [num_tokens, hidden_size]。"""
        routing_logits = self.gate(x)                                  # [T, E]
        topk_weights, topk_ids = torch.topk(routing_logits, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights.float()
        if self.norm_topk_prob:
            topk_weights = F.softmax(topk_weights, dim=-1)             # 对 top-k 做 softmax
        topk_weights = topk_weights.to(x.dtype)

        if self.world_size == 1:
            return self._forward_local(x, topk_weights, topk_ids)
        return self._forward_ep(x, topk_weights, topk_ids)

    def _forward_local(self, x, topk_weights, topk_ids):
        out = torch.zeros_like(x)
        for k in range(self.num_experts_per_tok):
            ids = topk_ids[:, k]                                       # [T] 每个 token 的第 k 个专家
            w = topk_weights[:, k]
            for e in range(self.num_local_experts):
                mask = ids == e
                if mask.any():
                    out[mask] += w[mask].unsqueeze(-1) * self.experts[e](x[mask])
        return out

    def _forward_ep(self, x, topk_weights, topk_ids):
        """专家并行：all-to-all 路由 token 到活跃专家所在卡。"""
        T = x.shape[0]
        k = self.num_experts_per_tok
        hidden = self.hidden_size
        L = self.num_local_experts

        # 展开成 [T*k] 的 token-专家组合
        x_rep = x.repeat_interleave(k, dim=0)                          # [T*k, hidden]
        w_flat = topk_weights.reshape(-1)                              # [T*k]
        expert_flat = topk_ids.reshape(-1)                             # [T*k] 全局专家 id

        local_expert = expert_flat % L                                 # 本地专家 id
        target_rank = expert_flat // L

        # 按目标 rank 稳定排序
        order = torch.argsort(target_rank, stable=True)
        x_sorted = x_rep[order].contiguous()
        local_expert_sorted = local_expert[order].contiguous().to(torch.int32)
        target_rank_sorted = target_rank[order]

        # 各 rank 发送量 + all_gather 得到接收量
        input_split = [int((target_rank_sorted == r).sum()) for r in range(self.world_size)]
        input_split_t = torch.tensor(input_split, dtype=torch.int64, device=x.device)
        gathered_t = [torch.empty_like(input_split_t) for _ in range(self.world_size)]
        dist.all_gather(gathered_t, input_split_t)
        output_split = [int(gathered_t[s][self.rank]) for s in range(self.world_size)]

        # all-to-all dispatch
        recv_x = torch.empty(sum(output_split), hidden, device=x.device, dtype=x.dtype)
        recv_exp = torch.empty(sum(output_split), device=x.device, dtype=torch.int32)
        dist.all_to_all_single(recv_x, x_sorted, output_split_sizes=output_split, input_split_sizes=input_split)
        dist.all_to_all_single(recv_exp, local_expert_sorted, output_split_sizes=output_split, input_split_sizes=input_split)

        # 本地计算：收到的 token 用它命中的本地专家算 FFN（experts 只持有本地专家，id 已是本地 index）
        recv_out = torch.empty_like(recv_x)
        for local_id in range(L):
            mask = recv_exp == local_id
            if mask.any():
                recv_out[mask] = self.experts[local_id](recv_x[mask])

        # all-to-all combine（对称逆变换）
        back_x = torch.empty(T * k, hidden, device=x.device, dtype=x.dtype)
        dist.all_to_all_single(back_x, recv_out, output_split_sizes=input_split, input_split_sizes=output_split)

        # scatter 回原始位置，加权求和
        out_full = torch.empty_like(back_x)
        out_full[order] = back_x
        return (out_full.reshape(T, k, hidden) * w_flat.reshape(T, k, 1)).sum(dim=1)
