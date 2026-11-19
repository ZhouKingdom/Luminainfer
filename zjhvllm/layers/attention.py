import torch                                    # PyTorch 主库
from torch import nn                            # 神经网络模块（nn.Module 等）
import triton                                   # Triton JIT 编译器
import triton.language as tl                    # Triton 语言前端（tl.program_id / tl.load 等）

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache   # FlashAttention 的 varlen 接口与 kvcache 接口（后者本文件未使用）
from zjhvllm.utils.context import get_context                            # 获取全局推理上下文（cu_seqlens、block_tables、slot_mapping 等）
from zjhvllm.layers.paged_attention import paged_attention_decode        # 自实现的 paged attention decode kernel


@triton.jit                                     # 用 Triton JIT 编译下面的 kernel
def store_kvcache_kernel(                       # 把当前 token 的 K/V 写入 paged KV cache 的 kernel
    key_ptr,                                    # 当前 batch 的 K 张量首地址
    key_stride,                                 # K 张量第 0 维（token 维）的 stride
    value_ptr,                                  # 当前 batch 的 V 张量首地址
    value_stride,                               # V 张量第 0 维的 stride
    k_cache_ptr,                                # K cache 首地址（展平为 [num_slots, D]）
    v_cache_ptr,                                # V cache 首地址（展平为 [num_slots, D]）
    slot_mapping_ptr,                           # slot_mapping 首地址，每个 token 对应一个 cache slot
    D: tl.constexpr,                            # 每个 token 的 K/V 展平维度 = num_heads * head_dim，编译期常量
):
    idx = tl.program_id(0)                      # 当前 program 的编号，对应第 idx 个 token（grid 大小 = N）
    slot = tl.load(slot_mapping_ptr + idx)      # 读取第 idx 个 token 要写入的 cache slot 编号
    if slot == -1: return                       # slot == -1 表示该 token 不写 cache（如 CUDA graph 的 padding），直接跳过
    key_offsets = idx * key_stride + tl.arange(0, D)     # 第 idx 个 token 在 K 张量中 D 个元素的偏移
    value_offsets = idx * value_stride + tl.arange(0, D) # 第 idx 个 token 在 V 张量中 D 个元素的偏移
    key = tl.load(key_ptr + key_offsets)                 # 从 K 张量加载该 token 的 D 个元素
    value = tl.load(value_ptr + value_offsets)           # 从 V 张量加载该 token 的 D 个元素
    cache_offsets = slot * D + tl.arange(0, D)           # 在展平 cache 中该 slot 对应 D 个元素的偏移
    tl.store(k_cache_ptr + cache_offsets, key)           # 把 K 写入 K cache 的对应 slot
    tl.store(v_cache_ptr + cache_offsets, value)         # 把 V 写入 V cache 的对应 slot


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape          # N：本轮 token 数；num_heads：KV 头数；head_dim：每头维度
    D = num_heads * head_dim                    # 每个 token 展平后的维度
    assert key.stride(-1) == 1 and value.stride(-1) == 1       # 最后一维必须连续（便于按 D 展平读写）
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # [N, H, D] 必须是紧凑布局
    assert k_cache.stride(1) == D and v_cache.stride(1) == D   # cache 第 1 维 stride 为 D（即 [num_slots, D] 视图）
    assert slot_mapping.numel() == N            # slot_mapping 长度必须与 token 数一致
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)  # 启动 kernel，grid = N


def store_kvcache_fp8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """把 K/V 量化成 fp8（e4m3），per-token scale = max(|x|) 存到 scale 张量。

    用 PyTorch 原生 fp8 cast（A100 上 Triton 的 fp8e4nv 不可用，但 torch.float8_e4m3fn 可正常存储）。
    """
    scale_k = key.float().abs().amax(dim=(1, 2))        # [N] 每个 token 的 K 绝对值最大值，作为 per-token scale
    scale_v = value.float().abs().amax(dim=(1, 2))      # [N] 每个 token 的 V 绝对值最大值
    key_fp8 = (key.float() / scale_k[:, None, None]).to(torch.float8_e4m3fn)    # 归一化到 [-1,1] 后转 fp8 e4m3
    value_fp8 = (value.float() / scale_v[:, None, None]).to(torch.float8_e4m3fn) # 同理量化 V
    # CUDA graph 路径下 slot_mapping 有 -1 padding，跳过
    valid = slot_mapping >= 0                   # 标记哪些 token 是有效的（非 padding）
    sm = slot_mapping[valid]                    # 只取有效 token 的 slot 编号
    # slot_mapping 是 flat slot 索引（block*block_size + pos），先展平成 [num_slots, H, D]
    H, D = key.shape[1], key.shape[2]           # KV 头数、每头维度
    k_cache.reshape(-1, H, D)[sm] = key_fp8[valid]   # 按 slot 索引把 fp8 K 写入 cache
    v_cache.reshape(-1, H, D)[sm] = value_fp8[valid] # 按 slot 索引把 fp8 V 写入 cache
    k_scale.reshape(-1)[sm] = scale_k[valid]         # 把 K 的 per-token scale 写入 scale 张量
    v_scale.reshape(-1)[sm] = scale_v[valid]         # 把 V 的 per-token scale 写入 scale 张量


def gather_dequant(k_cache, v_cache, k_scale, v_scale, block_table, seq_lens):
    """把 paged fp8 KV cache 反量化成连续的 bf16 张量（供 flash_attn_varlen 使用）。

    k_cache/v_cache: [num_blocks, block_size, H, D] fp8
    k_scale/v_scale: [num_blocks, block_size] fp32
    block_table:     [N, max_blocks] int32
    seq_lens:        [N] int32（每条要 gather 的 token 数）

    返回 (k_contig [total,H,D] bf16, v_contig, cu_seqlens_k [N+1])。
    """
    N = seq_lens.numel()                        # 序列条数（batch 大小）
    block_size, H, D = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]  # 从 cache 形状解析块大小、头数、每头维度
    k_list, v_list = [], []                     # 收集每条序列反量化后的连续 K/V
    cu = [0]                                    # 累积序列长度，用于构造 cu_seqlens
    for s in range(N):                          # 逐条序列处理
        L = int(seq_lens[s].item())             # 第 s 条序列要 gather 的 token 数
        if L == 0:                              # 空序列：追加空张量，cu 不变
            k_list.append(torch.zeros(0, H, D, dtype=torch.bfloat16, device=k_cache.device))  # 空 K
            v_list.append(torch.zeros(0, H, D, dtype=torch.bfloat16, device=v_cache.device))  # 空 V
            cu.append(cu[-1])                   # cu 保持不变
            continue
        n_blk = (L + block_size - 1) // block_size        # 该序列需要的 block 数（上取整）
        blks = block_table[s, :n_blk]                    # [n_blk] 该序列占用的物理 block 编号
        k_flat = k_cache[blks].reshape(-1, H, D)[:L]      # [L, H, D] fp8：按 block 取出、展平并截断到 L
        v_flat = v_cache[blks].reshape(-1, H, D)[:L]      # [L, H, D] fp8：同理取 V
        sk = k_scale[blks].reshape(-1)[:L]                # [L] fp32：对应的 K per-token scale
        sv = v_scale[blks].reshape(-1)[:L]                # [L] fp32：对应的 V per-token scale
        k_list.append((k_flat.float() * sk[:, None, None]).to(torch.bfloat16))  # 反量化 K 并转 bf16
        v_list.append((v_flat.float() * sv[:, None, None]).to(torch.bfloat16))  # 反量化 V 并转 bf16
        cu.append(cu[-1] + L)                   # 更新累积长度
    k_contig = torch.cat(k_list, dim=0)         # [total, H, D] bf16 连续 K
    v_contig = torch.cat(v_list, dim=0)         # [total, H, D] bf16 连续 V
    cu_seqlens_k = torch.tensor(cu, dtype=torch.int32, device=k_cache.device)  # [N+1] int32 累积长度
    return k_contig, v_contig, cu_seqlens_k     # 返回连续 K/V 与 cu_seqlens


class Attention(nn.Module):                     # 注意力层（prefill / decode 统一封装）

    def __init__(
        self,
        num_heads,                              # query 头数
        head_dim,                               # 每头维度
        scale,                                  # softmax 缩放因子（通常 1/sqrt(head_dim)）
        num_kv_heads,                           # KV 头数（GQA/MQA 时小于 num_heads）
    ):
        super().__init__()                      # 初始化 nn.Module 基类
        self.num_heads = num_heads              # 保存 query 头数
        self.head_dim = head_dim                # 保存每头维度
        self.scale = scale                      # 保存 softmax 缩放因子
        self.num_kv_heads = num_kv_heads        # 保存 KV 头数
        self.k_cache = self.v_cache = torch.tensor([])  # 初始为空张量，运行时由外部按需分配（bf16 或 fp8）

    @property
    def use_fp8(self):                          # 判断当前 KV cache 是否为 fp8 量化
        return self.k_cache.numel() and self.k_cache.dtype == torch.float8_e4m3fn  # cache 非空且 dtype 为 fp8 e4m3

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()                 # 获取当前推理上下文（包含阶段、cu_seqlens、block_tables、slot_mapping 等）
        k_cache, v_cache = self.k_cache, self.v_cache  # 本地引用，避免多次属性访问
        if k_cache.numel() and v_cache.numel():        # 如果 cache 已分配
            if self.use_fp8:                    # FP8 路径
                store_kvcache_fp8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)  # 量化并写入 cache
            else:                               # 普通 bf16/fp16 路径
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # 直接写入 cache

        if context.decode_equivalent:
            if not context.is_prefill:
                return torch.cat([self._decode_verification_row(
                    q[b:b + 1].contiguous(), context.block_tables[b:b + 1], context.context_lens[b:b + 1])
                    for b in range(len(q))], dim=0)
            lengths = context.verification_context_lens
            if (context.block_tables is None or lengths is None
                    or lengths.numel() != q.shape[0]):
                raise ValueError("decode-equivalent attention requires verification metadata")
            # The complete window has been written above. Each row must see
            # only its own causal prefix, using the ordinary decode kernel.
            cu = context.cu_seqlens_q_cpu or context.cu_seqlens_q.tolist()
            return torch.cat([self._decode_verification_row(
                q[i:i + 1].contiguous(), context.block_tables[b:b + 1], lengths[i:i + 1])
                for b in range(len(cu) - 1) for i in range(cu[b], cu[b + 1])], dim=0)

        if context.is_prefill:                  # ===== prefill 阶段 =====
            if context.block_tables is not None:    # 有 prefix cache / chunked prefill：需要从 cache 读历史 KV
                if self.use_fp8:                # FP8 路径
                    seq_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]  # 由 cu_seqlens 差分得到每条序列长度
                    k_contig, v_contig, cu_k = gather_dequant(                       # 反量化并 gather 出连续 KV
                        k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, seq_lens)
                    o = flash_attn_varlen_func(q, k_contig, v_contig,                # 调用 FlashAttention varlen
                                               cu_seqlens_q=context.cu_seqlens_q,     # query 侧累积长度
                                               cu_seqlens_k=cu_k,                     # key 侧累积长度（gather 后）
                                               max_seqlen_q=context.max_seqlen_q,     # query 最大序列长度
                                               max_seqlen_k=context.max_seqlen_k,     # key 最大序列长度
                                               softmax_scale=self.scale,              # softmax 缩放
                                               causal=True)                           # 因果掩码
                else:                           # 非 FP8，但 KV 在 paged cache 中
                    k, v = k_cache, v_cache     # 直接用 paged cache 作为 K/V
                    o = flash_attn_varlen_func(q, k, v,                               # FlashAttention 原生支持 paged KV
                                               max_seqlen_q=context.max_seqlen_q,     # query 最大序列长度
                                               cu_seqlens_q=context.cu_seqlens_q,     # query 累积长度
                                               max_seqlen_k=context.max_seqlen_k,     # key 最大序列长度
                                               cu_seqlens_k=context.cu_seqlens_k,     # key 累积长度
                                               softmax_scale=self.scale,              # softmax 缩放
                                               causal=True,                           # 因果掩码
                                               block_table=context.block_tables)      # 物理 block 映射表
            else:                               # 无 block_tables：首次 prefill，无历史 KV
                o = flash_attn_varlen_func(q, k, v,                                   # 直接用当前 K/V 做 varlen attention
                                           max_seqlen_q=context.max_seqlen_q,         # query 最大序列长度
                                           cu_seqlens_q=context.cu_seqlens_q,         # query 累积长度
                                           max_seqlen_k=context.max_seqlen_k,         # key 最大序列长度
                                           cu_seqlens_k=context.cu_seqlens_k,         # key 累积长度
                                           softmax_scale=self.scale,                  # softmax 缩放
                                           causal=True)                               # 因果掩码
        else:    # decode                           # ===== decode 阶段 =====
            if self.use_fp8:                    # FP8 decode 路径
                k_contig, v_contig, cu_k = gather_dequant(                            # 反量化并 gather 出连续 KV
                    k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, context.context_lens)
                cu_q = torch.arange(context.context_lens.numel() + 1, dtype=torch.int32, device=q.device)  # decode 每请求 1 个 q token，cu_seqlens_q = [0,1,2,...]
                o = flash_attn_varlen_func(q, k_contig, v_contig,                     # 用 varlen 接口做 decode
                                           cu_seqlens_q=cu_q,                         # query 累积长度
                                           cu_seqlens_k=cu_k,                         # key 累积长度
                                           max_seqlen_q=1,                            # decode 时 q 长度为 1
                                           max_seqlen_k=context.context_lens.max().item(),  # k 最大长度
                                           softmax_scale=self.scale,                  # softmax 缩放
                                           causal=True)                               # 因果掩码
            else:                               # 普通 decode 路径
                o = paged_attention_decode(q, k_cache, v_cache, context.block_tables, context.context_lens, self.scale)  # 调用 paged attention decode kernel
        return o                                # 返回注意力输出

    def _decode_verification_row(self, q, block_table, length):
        if not self.use_fp8:
            return paged_attention_decode(q, self.k_cache, self.v_cache, block_table, length, self.scale)
        # Match ordinary FP8 decode's dequantization and FA call as well.
        k, v, cu_k = gather_dequant(self.k_cache, self.v_cache, self.k_scale, self.v_scale, block_table, length)
        cu_q = torch.arange(2, device=q.device, dtype=torch.int32)
        return flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                      max_seqlen_q=1, max_seqlen_k=length.max().item(),
                                      softmax_scale=self.scale, causal=True)
