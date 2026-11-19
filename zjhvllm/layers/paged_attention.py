"""自研 decode paged attention（Triton），替代 flash_attn_with_kvcache。

decode 场景：每个序列 1 个 token 的 Q，对分页的 KV cache（block_table 索引）做 attention。
与 flash_attn 库解耦的关键算子，v2 相比 v1 的两处优化：
  - GQA shared-KV：grid 从 (num_seqs, num_heads) 改成 (num_seqs, num_kv_heads)，
    一个 program 只读一遍 KV，喂给 num_heads/num_kv_heads 个共享它的 Q 头，
    KV 读取量降 8x（Qwen3-32B 是 8 个 Q 头共享 1 个 KV 头）；
  - 小内层 tile（BLOCK_N=64）+ tensor core：v1 用整页 block_size=256 做单 tile，
    把 [BLOCK, HEAD_DIM] 的中间量撑爆寄存器导致 spill；这里切成 64 一个 tile，
    Q 补零到 QBLOCK=16 后走 tl.dot（tensor core），score 和 p@V 都吃 MMA。

曾尝试 split-K（把序列按 token 切成多段并行，reduce 重归一化），实测对 memory-bound
的 decode 不划算：并行度提升被 partial 缓冲区写入 + reduce kernel 启动的开销抵消，
只在 (batch=8, ctx=4096) 一个格子有收益，却让 (batch=8, ctx=1024) 从 1.40x 退化到
2.58x。故回退，留此注释记录这个 negative result。

输入布局（与 model_runner 的 kv_cache / prepare_block_tables 对齐）：
  q:             [num_seqs, num_heads, head_dim]  bf16
  k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]  bf16
  block_table:   [num_seqs, max_blocks]  int32（-1 为 padding）
  context_lens:  [num_seqs]  int32（每个序列已缓存的 KV 长度）
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, context_lens_ptr, o_ptr,
    scale, num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_blocks,
    NUM_Q_PER_KV: tl.constexpr, QBLOCK: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)

    L = tl.load(context_lens_ptr + seq).to(tl.int32)
    dim_offs = tl.arange(0, HEAD_DIM)
    qh_offs = tl.arange(0, QBLOCK)

    # 该 kv 头对应的 num_q_per_kv 个 Q 头，补零到 QBLOCK=16 走 tensor core；
    # 补零的行 q_valid=False（qh_offs >= num_q_per_kv），不参与 softmax、不写回。
    q_head = kv_head * NUM_Q_PER_KV + qh_offs
    q_valid = qh_offs < NUM_Q_PER_KV
    q_offs = seq * num_heads * HEAD_DIM + q_head[:, None] * HEAD_DIM + dim_offs[None, :]
    q = tl.load(q_ptr + q_offs, mask=q_valid[:, None], other=0.0).to(tl.float32)

    m_i = tl.zeros([QBLOCK], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([QBLOCK], dtype=tl.float32)
    acc = tl.zeros([QBLOCK, HEAD_DIM], dtype=tl.float32)

    num_tiles = (L + BLOCK_N - 1) // BLOCK_N
    tok_offs = tl.arange(0, BLOCK_N)

    for b in range(0, num_tiles):
        tok_idx = b * BLOCK_N + tok_offs               # [BLOCK_N] 全局 token 位置
        tok_valid = tok_idx < L                        # 最后一 tile 可能不满

        # paged 寻址：token → (逻辑块, 块内偏移) → 物理块
        page_idx = tok_idx // block_size
        tok_in_page = tok_idx % block_size
        block_id = tl.load(block_table_ptr + seq * max_blocks + page_idx, mask=tok_valid, other=0)

        # K/V tile [BLOCK_N, HEAD_DIM]，只读一遍，喂给全部 Q 头
        k_offs = ((block_id * block_size + tok_in_page)[:, None] * num_kv_heads + kv_head) * HEAD_DIM \
            + dim_offs[None, :]
        k = tl.load(k_cache_ptr + k_offs, mask=tok_valid[:, None], other=0.0).to(tl.float32)

        # score [QBLOCK, BLOCK_N] = q @ K^T * scale，tensor core
        score = tl.dot(q.to(tl.bfloat16), tl.trans(k).to(tl.bfloat16)) * scale
        score = tl.where(tok_valid[None, :] & q_valid[:, None], score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_offs = ((block_id * block_size + tok_in_page)[:, None] * num_kv_heads + kv_head) * HEAD_DIM \
            + dim_offs[None, :]
        v = tl.load(v_cache_ptr + v_offs, mask=tok_valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new

    acc = acc / l_i[:, None]
    o_offs = seq * num_heads * HEAD_DIM + q_head[:, None] * HEAD_DIM + dim_offs[None, :]
    tl.store(o_ptr + o_offs, acc.to(o_ptr.dtype.element_ty), mask=q_valid[:, None])


def paged_attention_decode(q, k_cache, v_cache, block_table, context_lens, scale):
    """decode 的 paged attention 入口。q/k/v 均为 bf16。

    q:             [num_seqs, num_heads, head_dim]
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_table:   [num_seqs, max_blocks] int32
    context_lens:  [num_seqs] int32
    返回 o: [num_seqs, num_heads, head_dim]（与 q 同 dtype）
    """
    num_seqs, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks = block_table.shape[1]

    num_q_per_kv = num_heads // num_kv_heads
    assert num_q_per_kv <= 16, "QBLOCK=16 只支持 num_q_per_kv <= 16"
    assert block_size % 64 == 0, "BLOCK_N=64 需整除 block_size"

    o = torch.empty_like(q)
    grid = (num_seqs, num_kv_heads)
    _paged_attn_decode_kernel[grid](
        q, k_cache, v_cache, block_table, context_lens, o,
        scale, num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_blocks,
        NUM_Q_PER_KV=num_q_per_kv, QBLOCK=16, HEAD_DIM=head_dim, BLOCK_N=64,
    )
    return o
