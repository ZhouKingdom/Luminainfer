import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from zjhvllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_fp8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """把 K/V 量化成 fp8（e4m3），per-token scale = max(|x|) 存到 scale 张量。

    用 PyTorch 原生 fp8 cast（A100 上 Triton 的 fp8e4nv 不可用，但 torch.float8_e4m3fn 可正常存储）。
    """
    scale_k = key.float().abs().amax(dim=(1, 2))        # [N]
    scale_v = value.float().abs().amax(dim=(1, 2))
    key_fp8 = (key.float() / scale_k[:, None, None]).to(torch.float8_e4m3fn)
    value_fp8 = (value.float() / scale_v[:, None, None]).to(torch.float8_e4m3fn)
    # CUDA graph 路径下 slot_mapping 有 -1 padding，跳过
    valid = slot_mapping >= 0
    sm = slot_mapping[valid]
    # slot_mapping 是 flat slot 索引（block*block_size + pos），先展平成 [num_slots, H, D]
    H, D = key.shape[1], key.shape[2]
    k_cache.reshape(-1, H, D)[sm] = key_fp8[valid]
    v_cache.reshape(-1, H, D)[sm] = value_fp8[valid]
    k_scale.reshape(-1)[sm] = scale_k[valid]
    v_scale.reshape(-1)[sm] = scale_v[valid]


def gather_dequant(k_cache, v_cache, k_scale, v_scale, block_table, seq_lens):
    """把 paged fp8 KV cache 反量化成连续的 bf16 张量（供 flash_attn_varlen 使用）。

    k_cache/v_cache: [num_blocks, block_size, H, D] fp8
    k_scale/v_scale: [num_blocks, block_size] fp32
    block_table:     [N, max_blocks] int32
    seq_lens:        [N] int32（每条要 gather 的 token 数）

    返回 (k_contig [total,H,D] bf16, v_contig, cu_seqlens_k [N+1])。
    """
    N = seq_lens.numel()
    block_size, H, D = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    k_list, v_list = [], []
    cu = [0]
    for s in range(N):
        L = int(seq_lens[s].item())
        if L == 0:
            k_list.append(torch.zeros(0, H, D, dtype=torch.bfloat16, device=k_cache.device))
            v_list.append(torch.zeros(0, H, D, dtype=torch.bfloat16, device=v_cache.device))
            cu.append(cu[-1])
            continue
        n_blk = (L + block_size - 1) // block_size
        blks = block_table[s, :n_blk]                    # [n_blk]
        k_flat = k_cache[blks].reshape(-1, H, D)[:L]      # [L, H, D] fp8
        v_flat = v_cache[blks].reshape(-1, H, D)[:L]
        sk = k_scale[blks].reshape(-1)[:L]                # [L] fp32
        sv = v_scale[blks].reshape(-1)[:L]
        k_list.append((k_flat.float() * sk[:, None, None]).to(torch.bfloat16))
        v_list.append((v_flat.float() * sv[:, None, None]).to(torch.bfloat16))
        cu.append(cu[-1] + L)
    k_contig = torch.cat(k_list, dim=0)
    v_contig = torch.cat(v_list, dim=0)
    cu_seqlens_k = torch.tensor(cu, dtype=torch.int32, device=k_cache.device)
    return k_contig, v_contig, cu_seqlens_k


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    @property
    def use_fp8(self):
        return self.k_cache.numel() and self.k_cache.dtype == torch.float8_e4m3fn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.use_fp8:
                store_kvcache_fp8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache / chunked prefill
                if self.use_fp8:
                    seq_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                    k_contig, v_contig, cu_k = gather_dequant(
                        k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, seq_lens)
                    o = flash_attn_varlen_func(q, k_contig, v_contig,
                                               cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=cu_k,
                                               max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                                               softmax_scale=self.scale, causal=True)
                else:
                    k, v = k_cache, v_cache
                    o = flash_attn_varlen_func(q, k, v,
                                               max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                               max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                               softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True)
        else:    # decode
            if self.use_fp8:
                k_contig, v_contig, cu_k = gather_dequant(
                    k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, context.context_lens)
                cu_q = torch.arange(context.context_lens.numel() + 1, dtype=torch.int32, device=q.device)
                o = flash_attn_varlen_func(q, k_contig, v_contig,
                                           cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                           max_seqlen_q=1, max_seqlen_k=context.context_lens.max().item(),
                                           softmax_scale=self.scale, causal=True)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o
