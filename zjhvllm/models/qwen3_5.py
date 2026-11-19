"""Qwen3.5 模型结构：Hybrid Attention（Full Attention + Gated DeltaNet）。

Qwen3.5 相对 Qwen3 的核心差异是 hybrid attention：64 层 = 16 层 full attention
（标准 KV cache）+ 48 层 Gated DeltaNet（固定大小 recurrent state，不用 KV cache）。
每 4 层一个 full（`layer_types` 决定），其余 linear。

引擎集成要点：
- full 层复用 `layers/attention.py` 的 `Attention`（paged KV cache + flash_attn），
  只是多一个 output gate（`o * sigmoid(gate)`）和 Qwen3.5 特有的 q_norm/k_norm。
- linear 层的递归状态（S 和 conv_state）按序列存储，经 `get_context().state_slots`
  从模型外部的状态缓存 gather / scatter。S 是 fp32（对齐 transformers 的
  mamba_ssm_dtype=float32），conv_state 是 bf16。
- 普通推理支持 TP=1/2/4/8：按 GDN/attention 头和 MLP 维度切分，行投影 all-reduce；
  KV 头不足时复制。prefill 仍逐序列执行，长序列用 chunk + Triton 扫描。
- decode 原地更新局部递归状态，普通推理可使用原生 CUDA Graph；MTP 支持 TP=1/2/4/8，可选多卡多请求验证 Graph。

状态形状：S [num_v_heads, key_head_dim, value_head_dim]（固定大小，decode O(1)）。
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from zjhvllm.layers.attention import Attention
from zjhvllm.layers.layernorm import RMSNorm
from zjhvllm.layers.rotary_embedding import get_rope
from zjhvllm.layers.embed_head import VocabParallelEmbedding, Qwen3_5VocabParallelHead
from zjhvllm.utils.context import get_context
from zjhvllm.utils.tp import get_tp_size, get_tp_rank


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _apply_qwen35_rope(positions, query, key, rotary_emb):
    """Match HF's activation-dtype cos/sin and separate rounded products.

    The shared rotary implementation promotes products to fp32, which changes
    Qwen3.5 bf16 attention inputs and can amplify through the hybrid stack.
    """
    cos, sin = rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
    cos, sin = cos.to(query.dtype), sin.to(query.dtype)
    dim = rotary_emb.rotary_dim

    def rotate(x):
        left, right = x[..., :dim].chunk(2, dim=-1)
        rotated = torch.cat((left * cos - right * sin,
                             right * cos + left * sin), dim=-1)
        return torch.cat((rotated, x[..., dim:]), dim=-1)

    return rotate(query), rotate(key)


def _chunk_gated_delta_rule(q, k, v, g, beta, S_init, chunk_size=64, scan_backend="triton", scan_precision="ieee", scan_warps=8):
    """Gated DeltaNet 的 chunk 化实现（对齐 transformers `torch_chunk_gated_delta_rule`）。

    把 per-token 顺序递归（L 步）降到 chunk 扫描（L/chunk_size 步）：chunk 内用 UT 上三角
    系统 + triangular solve 并行，chunk 间用矩阵乘法顺序传递状态。输入均为 fp32，布局：
        q, k: [Hv, L, Dk]（已 repeat 到 value 头 + l2norm + Dk^-0.5）
        v:    [Hv, L, Dv]
        g, beta: [Hv, L]
        S_init: [Hv, Dk, Dv]
    返回 (o [L, Hv, Dv], S_final [Hv, Dk, Dv])。
    scan_backend="torch" 用相同预处理复现 torch 扫描基线；默认走 Triton。
    """
    if scan_backend not in ("torch", "triton"):
        raise ValueError(f"Unknown scan backend: {scan_backend}")
    Hv, L, Dk = q.shape
    Dv = v.shape[-1]
    decay = g

    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        decay = F.pad(decay, (0, pad))
    total = L + pad
    num_chunks = total // chunk_size

    v_beta = v * beta.unsqueeze(-1)              # β·v
    k_beta = k * beta.unsqueeze(-1)              # β·k

    q = q.reshape(Hv, num_chunks, chunk_size, Dk)
    k = k.reshape(Hv, num_chunks, chunk_size, Dk)
    k_beta = k_beta.reshape(Hv, num_chunks, chunk_size, Dk)
    v_beta = v_beta.reshape(Hv, num_chunks, chunk_size, Dv)
    decay = decay.reshape(Hv, num_chunks, chunk_size)

    # chunk 内累积 decay（log 空间 cumsum）+ 成对 decay（上三角 mask 掉，保证 causal）
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum_decay = decay.cumsum(dim=-1)                                     # [Hv, nc, chunk]
    pairwise = cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)         # [Hv, nc, chunk, chunk]
    pairwise = pairwise.masked_fill(mask, float("-inf")).exp()

    # UT 系统 + 块内 attention
    ut_system = (k_beta @ k.transpose(-1, -2)) * pairwise               # [Hv, nc, chunk, chunk]
    intra_attn = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # 三角求解（unit lower triangular）
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    S = S_init
    q = q * cum_decay.exp().unsqueeze(-1)
    k_decayed = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()                               # [Hv, nc]

    # 用 Triton 融合 chunk 间扫描（满足 tl.dot 对齐约束时）；否则回退 torch 循环
    if scan_backend == "triton" and q.is_cuda and chunk_size % 16 == 0 and Dk % 16 == 0 and Dv % 16 == 0:
        from zjhvllm.layers.gated_delta import gated_delta_scan
        o, S = gated_delta_scan(q, k_decayed, new_values, k_cumdecay, intra_attn, chunk_decay, S, input_precision=scan_precision, num_warps=scan_warps)
    else:
        o = torch.zeros(Hv, num_chunks, chunk_size, Dv, device=q.device)
        for i in range(num_chunks):
            v_new = new_values[:, i] - k_cumdecay[:, i] @ S                  # 减去旧状态已预测的部分
            inter = q[:, i] @ S                                               # 读旧状态
            o[:, i] = inter + intra_attn[:, i] @ v_new
            S = S * chunk_decay[:, i, None, None] + k_decayed[:, i].transpose(-1, -2) @ v_new

    o = o.reshape(Hv, total, Dv)[:, :L].transpose(0, 1)                  # [L, Hv, Dv]
    return o, S



class Qwen3_5Linear(nn.Linear):
    """Keep verification GEMM shapes identical to single-request decode.

    BF16 cuBLAS reductions depend on the token dimension. Computing the rows
    independently preserves the reference, including its rounded logits. This
    is a correctness path, not a fused batched GEMM optimization.
    """

    def forward(self, x):
        if hasattr(self, "weight_scale"):
            if not self.quantization_loaded:
                raise RuntimeError("INT8 weight has not been loaded")
            from zjhvllm.layers.weight_only import int8_weight_only_linear
            return int8_weight_only_linear(x, self.weight, self.weight_scale, self.bias)
        if (getattr(self, "backend", "decode_equivalent") == "batch_invariant"
                and get_context().decode_equivalent and x.shape[0] > 32):
            # Oversized verification windows keep the same arithmetic backend
            # instead of silently falling back to the old cuBLAS reductions.
            return torch.cat([self.forward(row) for row in x.split(1)])
        if (getattr(self, "backend", "decode_equivalent") == "batch_invariant"
                and x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
                and 1 <= x.shape[0] <= 32):
            if self.out_features < 256:
                return torch.cat([F.linear(row, self.weight, self.bias) for row in x.split(1)])
            from zjhvllm.layers.invariant_linear import invariant_linear
            return invariant_linear(x, self.weight, self.bias)
        if get_context().decode_equivalent and not get_context().decode_batched and x.shape[0] > 1:
            return torch.cat([F.linear(row, self.weight, self.bias) for row in x.split(1)], dim=0)
        return F.linear(x, self.weight, self.bias)


def _load_partition(param, value, parts, size, rank, dim=0):
    """Shard packed Q/K/V segments independently, retaining local Q,K,V order."""
    pieces = []
    offset = 0
    for width in parts:
        if width % size:
            raise ValueError("projection dimension must be divisible by TP partition count")
        pieces.append(value.narrow(dim, offset + rank * (width // size), width // size))
        offset += width
    param.data.copy_(torch.cat(pieces, dim=dim))


class Qwen3_5ParallelLinear(Qwen3_5Linear):
    """Column/row sharding with unchanged checkpoint parameter names."""
    def __init__(self, input_size, output_size, *, column=False, parts=None,
                 replication=1, bias=False):
        self.tp_size, self.tp_rank = get_tp_size(), get_tp_rank()
        self.column = column
        self.partitions = self.tp_size // replication
        self.partition_rank = self.tp_rank // replication
        self.parts = parts or [output_size]
        if self.tp_size % replication:
            raise ValueError("invalid KV replication factor")
        if column:
            if any(width % self.partitions for width in self.parts):
                raise ValueError("column projection does not divide TP")
            output_size //= self.partitions
        else:
            if input_size % self.tp_size:
                raise ValueError("row projection requires divisible input")
            input_size //= self.tp_size
        super().__init__(input_size, output_size, bias=bias)
        self.weight.weight_loader = self.load_weight
        if self.bias is not None:
            self.bias.weight_loader = self.load_weight

    def load_weight(self, param, value):
        if self.column:
            _load_partition(param, value, self.parts, self.partitions, self.partition_rank)
        else:
            if param.ndim == 1:
                param.data.copy_(value)
                return
            width = value.shape[1] // self.tp_size
            param.data.copy_(value.narrow(1, self.tp_rank * width, width))

    def forward(self, x):
        if (not self.column and self.tp_size > 1 and get_context().decode_equivalent
                and not get_context().decode_batched and x.shape[0] > 1):
            # NCCL may choose a different BF16 reduction protocol for a larger
            # tensor. Match both GEMM and collective shapes to C1 decode.
            return torch.cat([self.forward(row) for row in x.split(1)], dim=0)
        if not self.column and self.tp_size > 1 and self.bias is not None:
            if hasattr(self, "weight_scale"):
                from zjhvllm.layers.weight_only import int8_weight_only_linear
                if not self.quantization_loaded:
                    raise RuntimeError("INT8 weight has not been loaded")
                y = int8_weight_only_linear(x, self.weight, self.weight_scale)
            else:
                y = F.linear(x, self.weight, None)
            dist.all_reduce(y)
            return y + self.bias
        y = super().forward(x)
        if not self.column and self.tp_size > 1:
            dist.all_reduce(y)
        return y


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 的 RMSNorm：norm(x) * (1 + weight)，weight 初始化为 0（identity）。

    与 qwen3 的 `norm(x) * weight` 不同——Qwen3.5 的 norm weight 以「1+残差」形式表示。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        if get_context().decode_equivalent and not get_context().decode_batched and x.shape[0] > 1:
            # Even a one-element BF16 change from a different reduction shape
            # can flip greedy argmax after the remaining hybrid layers.
            return torch.cat([self._normalize(row) for row in x.split(1)], dim=0)
        return self._normalize(x)

    def _normalize(self, x):
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(orig_dtype)


class GatedDeltaNet(nn.Module):
    """Qwen3.5 的 linear attention 层（Gated DeltaNet，固定大小 recurrent state）。

    数学（per token，先写后读）：
        m = S·k ; δ = v - exp(g)·m ; S = exp(g)·S + β·(k⊗δ) ; o = q·S
    状态 S 形状 [num_v_heads, key_head_dim, value_head_dim]，不随序列长度增长。

    引擎形态：状态不随 hidden 传入，而是经 `get_context().state_slots` 从
    `self.S_cache` / `self.conv_cache`（模型外部按 slot 分配的显存）gather / scatter。
    """

    def __init__(self, hidden_size, num_key_heads, num_value_heads,
                 key_head_dim, value_head_dim, conv_kernel_size, rms_norm_eps):
        super().__init__()
        tp_size, tp_rank = get_tp_size(), get_tp_rank()
        if num_key_heads % tp_size or num_value_heads % tp_size:
            raise ValueError("GDN key/value heads must divide TP")
        global_key_dim = num_key_heads * key_head_dim
        global_value_dim = num_value_heads * value_head_dim
        parts = [global_key_dim, global_key_dim, global_value_dim]
        num_key_heads //= tp_size
        num_value_heads //= tp_size
        self.num_k_heads = num_key_heads
        self.num_v_heads = num_value_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps

        self.in_proj_qkv = Qwen3_5ParallelLinear(hidden_size, sum(parts), column=True, parts=parts)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=conv_kernel_size,
                                groups=self.conv_dim, bias=False, padding=conv_kernel_size - 1)
        self.conv1d.weight.weight_loader = lambda p, w: _load_partition(p, w, parts, tp_size, tp_rank)
        self.in_proj_z = Qwen3_5ParallelLinear(hidden_size, global_value_dim, column=True)
        self.in_proj_b = Qwen3_5ParallelLinear(hidden_size, num_value_heads * tp_size, column=True)
        self.in_proj_a = Qwen3_5ParallelLinear(hidden_size, num_value_heads * tp_size, column=True)
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        A = torch.empty(num_value_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias.weight_loader = lambda p, w: _load_partition(p, w, [w.numel()], tp_size, tp_rank)
        self.A_log.weight_loader = self.dt_bias.weight_loader
        self.norm = RMSNorm(value_head_dim, eps=rms_norm_eps)
        self.out_proj = Qwen3_5ParallelLinear(global_value_dim, hidden_size)

        # 由 model_runner 分配：每个 linear 层持有一份 [num_slots, ...] 的状态缓存切片
        self.S_cache = torch.tensor([])
        self.conv_cache = torch.tensor([])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        slots = context.state_slots
        if context.decode_equivalent:
            if slots is None:
                raise ValueError("decode-equivalent forward requires state slots")
            if not context.is_prefill:
                if context.decode_batched:
                    return self._forward_decode(hidden_states, slots)
                return torch.cat([self._forward_decode(hidden_states[b:b + 1], slots[b:b + 1])
                                  for b in range(len(hidden_states))], dim=0)
            history = context.verification_state
            if history is not None:
                return history.forward_layer(self, hidden_states, slots)
            cu = context.cu_seqlens_q_cpu or context.cu_seqlens_q.tolist()
            return torch.cat([self._forward_decode(hidden_states[t:t + 1], slots[b:b + 1])
                              for b in range(len(slots)) for t in range(cu[b], cu[b + 1])], dim=0)
        if context.is_prefill:
            return self._forward_prefill(hidden_states, slots)
        return self._forward_decode(hidden_states, slots)

    # ------------------------------------------------------------------ prefill
    def _forward_prefill(self, hidden_states, slots):
        """Varlen prefill; reuse host metadata to avoid per-layer scalar GPU reads."""
        context = get_context()
        cu = context.cu_seqlens_q_cpu if context.cu_seqlens_q_cpu is not None else context.cu_seqlens_q
        slot_ids = context.state_slots_cpu if context.state_slots_cpu is not None else slots
        outputs = []
        for b in range(len(cu) - 1):
            start, end = int(cu[b]), int(cu[b + 1])
            slot = int(slot_ids[b])
            S = self.S_cache[slot]
            conv = self.conv_cache[slot]
            out, S, conv = self._recurrence(hidden_states[start:end], S, conv)
            self.S_cache[slot] = S
            self.conv_cache[slot] = conv
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------ decode
    def _project_decode(self, hidden_states):
        return (self.in_proj_qkv(hidden_states), self.in_proj_z(hidden_states),
                self.in_proj_b(hidden_states).sigmoid(),
                -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias))

    def _forward_decode(self, hidden_states, slots, *, state_cache=None, conv_cache=None,
                        output_state=None, output_conv=None, output_slots=None,
                        projections=None, project_output=True):
        """decode：每个序列 1 个 token，递归更新向量化在 batch 维。"""
        state_cache = self.S_cache if state_cache is None else state_cache
        conv_cache = self.conv_cache if conv_cache is None else conv_cache
        output_conv = conv_cache if output_conv is None else output_conv
        output_slots = slots if output_slots is None else output_slots
        B = hidden_states.shape[0]
        conv = conv_cache[slots]    # [B, conv_dim, K-1] bf16

        q, k, v, g, beta, z, new_conv = self._decode_inputs(hidden_states, conv, projections)

        from zjhvllm.layers.gated_delta import gated_delta_decode
        o = gated_delta_decode(q, k, v, g, beta, state_cache, slots,
                               output_cache=output_state, output_slots=output_slots)

        o = self._z_gate(o.to(hidden_states.dtype), z)                    # [B, Hv, hv]
        o = o.reshape(B, -1).to(hidden_states.dtype)

        output_conv[output_slots] = new_conv
        return self.out_proj(o) if project_output else o

    def _decode_inputs(self, hidden_states, conv, projections=None, processed_mixed=None):
        B = hidden_states.shape[0]
        if projections is None:
            projections = self._project_decode(hidden_states)
        mixed, z, beta, g = projections
        if processed_mixed is None:
            mixed = mixed.unsqueeze(-1)    # [B, conv_dim, 1]
            concat = torch.cat([conv, mixed], dim=2)                 # [B, conv_dim, K]
            mixed = F.conv1d(concat, self.conv1d.weight, groups=self.conv_dim)  # [B, conv_dim, 1]
            new_conv = concat[:, :, -(self.conv_kernel_size - 1):]
            mixed = F.silu(mixed).transpose(1, 2)                    # [B, 1, conv_dim]
        else:
            mixed, new_conv = processed_mixed.unsqueeze(1), None

        q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, self.num_v_heads, self.head_v_dim)
        z = z.reshape(B, self.num_v_heads, self.head_v_dim)

        r = self.num_v_heads // self.num_k_heads
        q = q.repeat_interleave(r, dim=1).float()                # [B, Hv, hk]
        k = k.repeat_interleave(r, dim=1).float()
        v = v.float()
        g = g.float()
        beta = beta.float()
        q = _l2norm(q) * (self.head_k_dim ** -0.5)
        k = _l2norm(k)

        return q, k, v, g, beta, z, new_conv

    def _forward_verify_fused(self, hidden, slots, history, index):
        from zjhvllm.layers.gated_delta import gated_delta_verify, causal_conv_verify
        # GEMMs still use the chosen numerical backend; default rowwise shapes
        # and NCCL reductions are unchanged. Conv preserves its BF16 rounding
        # boundary; Q/K normalization and z gating retain single-row shapes.
        projections = self._project_decode(hidden)
        inputs, gates, convs = [], [], []
        max_steps = max(b-a for a,b in zip(history.cu_seqlens[:-1],history.cu_seqlens[1:]))
        if hidden.dtype == torch.bfloat16 and self.conv_kernel_size == 4:
            processed = F.silu(causal_conv_verify(projections[0], self.conv1d.weight, self.conv_cache,
                               slots, get_context().cu_seqlens_q, history.convs[:, index], max_steps=max_steps))
            for step in range(len(hidden)):
                values = self._decode_inputs(hidden[step:step + 1], None,
                         tuple(p[step:step + 1] for p in projections), processed_mixed=processed[step:step + 1])
                inputs.append(values[:5])
                gates.append(values[5])
        else:
            for request, (start, end) in enumerate(zip(history.cu_seqlens[:-1], history.cu_seqlens[1:])):
                conv = self.conv_cache[slots[request:request + 1]]
                for step in range(start, end):
                    values = self._decode_inputs(hidden[step:step + 1], conv,
                                                tuple(p[step:step + 1] for p in projections))
                    inputs.append(values[:5])
                    gates.append(values[5])
                    conv = values[6]
                    convs.append(conv)
        q, k, v, g, beta = [torch.cat([row[i] for row in inputs]) for i in range(5)]
        output = gated_delta_verify(q, k, v, g, beta, self.S_cache, slots,
                                    get_context().cu_seqlens_q, history.states[:, index],
                                    max_steps=max_steps)
        if convs:
            history.convs[:len(hidden), index].copy_(torch.cat(convs))
        result = torch.cat([self._z_gate(output[t:t + 1].to(hidden.dtype), gate).reshape(1, -1)
                            for t, gate in enumerate(gates)])
        return self.out_proj(result)

    # ------------------------------------------------------------------ shared
    def _recurrence(self, x, S, conv):
        """单个序列的递归：x [L, hidden]，S [Hv, hk, hv]，conv [conv_dim, K-1]。

        返回 (output [L, hidden], new_S, new_conv)。用于 prefill（L 可变）与
        分块 prefill 的续算（conv 状态跨块衔接）。
        """
        L = x.shape[0]
        mixed = self.in_proj_qkv(x).transpose(0, 1)               # [conv_dim, L]
        concat = torch.cat([conv, mixed], dim=1)                  # [conv_dim, K-1+L]
        mixed = F.conv1d(concat, self.conv1d.weight, groups=self.conv_dim)  # [conv_dim, L]
        new_conv = concat[:, -(self.conv_kernel_size - 1):]
        mixed = F.silu(mixed).transpose(0, 1)                     # [L, conv_dim]

        q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(L, self.num_k_heads, self.head_k_dim)
        k = k.reshape(L, self.num_k_heads, self.head_k_dim)
        v = v.reshape(L, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(x).reshape(L, self.num_v_heads, self.head_v_dim)
        beta = self.in_proj_b(x).sigmoid()                        # [L, Hv]
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)

        r = self.num_v_heads // self.num_k_heads
        q = q.repeat_interleave(r, dim=1).transpose(0, 1).float() # [Hv, L, hk]
        k = k.repeat_interleave(r, dim=1).transpose(0, 1).float()
        v = v.transpose(0, 1).float()                             # [Hv, L, hv]
        g = g.transpose(0, 1).float()
        beta = beta.transpose(0, 1).float()
        q = _l2norm(q) * (self.head_k_dim ** -0.5)
        k = _l2norm(k)

        if L <= 32:
            # 短序列：per-token 递归（避免 chunk 化的 pad + triangular solve 开销）
            outs = []
            for t in range(L):
                qt, kt, vt = q[:, t], k[:, t], v[:, t]
                gt, bt = g[:, t], beta[:, t]
                eg = gt.exp()
                m = torch.einsum('hd,hdv->hv', kt, S)
                delta = vt - eg[:, None] * m
                S = eg[:, None, None] * S + torch.einsum('hd,hv->hdv', bt[:, None] * kt, delta)
                outs.append(torch.einsum('hd,hdv->hv', qt, S))
            o = torch.stack(outs, dim=1).transpose(0, 1)                  # [L, Hv, hv]
        else:
            # 长序列：chunk 化（顺序步数从 L 降到 L/64）
            o, S = _chunk_gated_delta_rule(q, k, v, g, beta, S)

        o = self._z_gate(o.to(x.dtype), z)
        o = o.reshape(L, -1).to(x.dtype)
        return self.out_proj(o), S, new_conv

    def _z_gate(self, o, z):
        """z 门控 RMSNorm：o = RMSNorm(o) · norm.weight · silu(z)。"""
        # HF rounds recurrence output, normalized output, and weighted output
        # to the activation dtype before multiplying the fp32 SiLU gate.
        dtype = o.dtype
        o = o.float()
        var = o.pow(2).mean(-1, keepdim=True)
        o = o * torch.rsqrt(var + self.rms_norm_eps)
        o = o.to(dtype) * self.norm.weight
        return (o * F.silu(z.float())).to(dtype)


class Qwen3_5Attention(nn.Module):
    """Qwen3.5 的 full attention 层（标准 KV cache attention + output gate）。"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, max_position,
                 rms_norm_eps, attention_bias, rope_theta, partial_rotary_factor=1.0):
        super().__init__()
        tp_size = get_tp_size()
        if num_heads % tp_size:
            raise ValueError("attention query heads must divide TP")
        replication = max(1, tp_size // num_kv_heads)
        if max(tp_size, num_kv_heads) % min(tp_size, num_kv_heads):
            raise ValueError("KV heads must divide TP or be divisible by TP")
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # q_proj 输出 2 倍：query + gate（output gate）
        self.q_proj = Qwen3_5ParallelLinear(hidden_size, num_heads * head_dim * 2, column=True, bias=attention_bias)
        self.k_proj = Qwen3_5ParallelLinear(hidden_size, num_kv_heads * head_dim, column=True, replication=replication, bias=attention_bias)
        self.v_proj = Qwen3_5ParallelLinear(hidden_size, num_kv_heads * head_dim, column=True, replication=replication, bias=attention_bias)
        self.o_proj = Qwen3_5ParallelLinear(num_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary_emb = get_rope(head_dim, rotary_dim=rotary_dim, max_position=max_position, base=rope_theta)
        # 复用引擎的 paged KV cache + flash_attn（k_cache/v_cache 由 model_runner 注入）
        self.attn = Attention(self.num_heads, head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states):
        qg = self.q_proj(hidden_states)
        # q_proj 输出 [N, H, 2*head_dim]：每 head 的 query(head_dim) 与 gate(head_dim) 成对
        query, gate = qg.view(-1, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        gate = gate.reshape(-1, self.num_heads * self.head_dim)
        key = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_qwen35_rope(positions, query, key, self.rotary_emb)

        o = self.attn(query, key, value)                          # [N, H, D]
        o = o.reshape(-1, self.num_heads * self.head_dim) * torch.sigmoid(gate)
        return self.o_proj(o)


class Qwen3_5MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_proj = Qwen3_5ParallelLinear(hidden_size, intermediate_size, column=True)
        self.up_proj = Qwen3_5ParallelLinear(hidden_size, intermediate_size, column=True)
        self.down_proj = Qwen3_5ParallelLinear(intermediate_size, hidden_size)
        assert hidden_act == "silu"

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]

        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                config.hidden_size,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                config.rms_norm_eps,
            )
            self.self_attn = None
        else:
            self.self_attn = Qwen3_5Attention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.max_position_embeddings,
                config.rms_norm_eps,
                config.attention_bias,
                config.rope_parameters.get("rope_theta", 10000),
                config.rope_parameters.get("partial_rotary_factor", 1.0),
            )
            self.linear_attn = None

        self.mlp = Qwen3_5MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states):
        # 标准 Pre-LN 残差（对齐 transformers Qwen3.5）
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.linear_attn is not None:
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = Qwen3_5VocabParallelHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def remap_weight_name(self, name: str):
        """多模态 checkpoint 的文本权重：model.language_model.* → model.*，
        MTP 仅在显式启用后加载；跳过视觉权重。"""
        if name.startswith("model.visual."):
            return None
        if name.startswith("mtp."):
            return name if getattr(self, "mtp", None) is not None else None
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model."):]
        return name

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states, *, all_tokens=False):
        # prefill 只取每个序列最后一个 token 的 hidden（对齐 ParallelLMHead 的优化）
        context = get_context()
        if context.is_prefill and not all_tokens:
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices]
        return self.lm_head(hidden_states)
