"""自研 paged decode attention 正确性测试：与 flash_attn_with_kvcache 逐元素对齐。

不依赖模型，直接构造随机 KV cache + block_table，验证 GQA 映射、paged 寻址、
跨块 online softmax 都与 flash-attn 一致（bf16 噪声阈值内）。

覆盖两种 GQA 配置：
  - num_heads=16 / num_kv_heads=8（num_q_per_kv=2，小 GQA）
  - num_heads=64 / num_kv_heads=8（num_q_per_kv=8，Qwen3-32B 生产配置）
"""

import pytest
import torch
from flash_attn import flash_attn_with_kvcache

from zjhvllm.layers.paged_attention import paged_attention_decode


# (num_heads, num_kv_heads, head_dim)
CONFIGS = [
    (16, 8, 64),     # num_q_per_kv = 2
    (64, 8, 128),    # num_q_per_kv = 8（Qwen3-32B）
]


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", CONFIGS)
def test_paged_attention_matches_flash_attn(num_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    num_seqs = 8
    block_size = 256
    num_blocks = 64
    max_blocks = 16
    dev = "cuda"

    q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=dev)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=dev)

    # paged 布局：每个 seq 用「交错」的物理块（非连续，测真实寻址），
    # 确保不同 seq 不共享物理块（除非巧合）
    block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device=dev)
    context_lens = torch.randint(1, 700, (num_seqs,), dtype=torch.int32, device=dev)
    for s in range(num_seqs):
        L = context_lens[s].item()
        n_blk = (L + block_size - 1) // block_size
        block_table[s, :n_blk] = torch.arange(s, s + n_blk * num_seqs, num_seqs, device=dev)

    scale = head_dim ** -0.5

    o_mine = paged_attention_decode(q, k_cache, v_cache, block_table, context_lens, scale)
    o_ref = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context_lens, block_table=block_table,
        softmax_scale=scale, causal=True,
    ).squeeze(1)

    diff = (o_mine.float() - o_ref.float()).abs()
    assert diff.max().item() < 0.05, f"max abs diff {diff.max().item():.6f} 超出 bf16 噪声阈值"
    assert torch.allclose(o_mine.float(), o_ref.float(), atol=0.05)
