"""专家并行（EP）MoE 层。

与单卡 MoE（layers/moe.py）的区别：专家按 rank 分布（每个 rank 只存 num_experts//world_size
个专家），router（gate）replicated。forward 里 token 通过 all-to-all 路由到它选中的专家所在卡，
算完再 all-to-all 送回、加权求和。

数据流：
    router（每卡一份）→ top-k 专家 id
    → all-to-all dispatch（token 发到专家所在卡）
    → 本地算专家 FFN
    → all-to-all combine（结果送回）
    → 按原始位置 scatter 回来，加权求和
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from zjhvllm.layers.activation import SiluAndMul


def _randn(shape, seed):
    """按 seed 确定性生成权重（测试用，保证单卡与 EP 各 rank 的专家权重一致）。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * 0.1


class LocalExpertFFN(nn.Module):
    """EP 下单个专家的 SwiGLU FFN（普通 nn.Linear，不需要 TP 切分）。"""

    def __init__(self, hidden_size: int, intermediate_size: int, expert_id: int | None = None):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiluAndMul()
        if expert_id is not None:
            with torch.no_grad():
                self.gate_up.weight.copy_(_randn((intermediate_size * 2, hidden_size), expert_id))
                self.down.weight.copy_(_randn((hidden_size, intermediate_size), expert_id + 1_000_000))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate_up(x)))


class ExpertParallelMoE(nn.Module):
    """专家并行 MoE：router replicated + 本地专家 + all-to-all 路由。"""

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

        # router：每卡都有一份完整的（replicated）
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(_randn((num_experts, hidden_size), 999_999))

        # 本地专家：全局 id = rank * num_local_experts + local_id
        self.experts = nn.ModuleList([
            LocalExpertFFN(hidden_size, moe_intermediate_size,
                           expert_id=self.rank * self.num_local_experts + i)
            for i in range(self.num_local_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        k = self.num_experts_per_tok
        hidden = self.hidden_size
        L = self.num_local_experts

        # 1) routing（每卡完整）
        routing_logits = self.gate(x)                                   # [T, E]
        topk_weights, topk_ids = torch.topk(routing_logits, k, dim=-1)
        topk_weights = topk_weights.float()
        if self.norm_topk_prob:
            topk_weights = F.softmax(topk_weights, dim=-1)
        topk_weights = topk_weights.to(x.dtype)

        # 2) 展开成 [T*k] 的 token-专家组合
        x_rep = x.repeat_interleave(k, dim=0)                           # [T*k, hidden]
        w_flat = topk_weights.reshape(-1)                               # [T*k]
        expert_flat = topk_ids.reshape(-1)                              # [T*k] 全局专家 id

        local_expert = expert_flat % L                                  # 本地专家 id
        target_rank = expert_flat // L                                  # 目标 rank

        # 3) 按目标 rank 稳定排序
        order = torch.argsort(target_rank, stable=True)                 # [T*k]
        x_sorted = x_rep[order].contiguous()
        local_expert_sorted = local_expert[order].contiguous().to(torch.int32)
        target_rank_sorted = target_rank[order]

        # 4) 计算各 rank 的发送量，并 all_gather 交换得到接收量
        input_split = [int((target_rank_sorted == r).sum()) for r in range(self.world_size)]
        input_split_t = torch.tensor(input_split, dtype=torch.int64, device=x.device)
        gathered_t = [torch.empty_like(input_split_t) for _ in range(self.world_size)]
        dist.all_gather(gathered_t, input_split_t)
        output_split = [int(gathered_t[s][self.rank]) for s in range(self.world_size)]  # 各 rank 发给我多少

        # 5) all-to-all dispatch：把 token 发到专家所在卡
        recv_x = torch.empty(sum(output_split), hidden, device=x.device, dtype=x.dtype)
        recv_exp = torch.empty(sum(output_split), device=x.device, dtype=torch.int32)
        dist.all_to_all_single(recv_x, x_sorted, output_split_sizes=output_split, input_split_sizes=input_split)
        dist.all_to_all_single(recv_exp, local_expert_sorted, output_split_sizes=output_split, input_split_sizes=input_split)

        # 6) 本地计算：收到的每个 token 用它命中的本地专家算 FFN
        recv_out = torch.empty_like(recv_x)
        for e in range(L):
            mask = recv_exp == e
            if mask.any():
                recv_out[mask] = self.experts[e](recv_x[mask])

        # 7) all-to-all combine：结果送回（split 与 dispatch 对称互换）
        back_x = torch.empty(T * k, hidden, device=x.device, dtype=x.dtype)
        dist.all_to_all_single(back_x, recv_out,
                               output_split_sizes=input_split, input_split_sizes=output_split)

        # 8) 按原始位置 scatter 回来，加权求和
        out_full = torch.empty_like(back_x)
        out_full[order] = back_x
        return (out_full.reshape(T, k, hidden) * w_flat.reshape(T, k, 1)).sum(dim=1)
