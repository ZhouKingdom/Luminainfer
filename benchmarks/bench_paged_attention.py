"""对比自研 paged decode attention vs flash_attn_with_kvcache 的 decode 延迟。

用 32B 的 GQA 配置（64 Q 头 / 8 KV 头 / head_dim 128），扫 batch size 和 context_len。
decode 是 memory-bound，关键看「读 paged KV 的效率」。

用法：
    python benchmarks/bench_paged_attention.py

输出一行一个 (batch, ctx_len) 格子的 mine / flash / ratio（ratio 越小越好，
<1 表示自研 kernel 反超 flash-attn）。
"""

import time

import torch
from flash_attn import flash_attn_with_kvcache

from zjhvllm.layers.paged_attention import paged_attention_decode

num_heads = 64
num_kv_heads = 8
head_dim = 128
block_size = 256
scale = head_dim ** -0.5
dev = "cuda"
warmup, iters = 10, 50


def bench_once(fn, *args):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000  # ms


def main():
    print(f"{'batch':>6} {'ctx_len':>8} {'mine(ms)':>10} {'flash(ms)':>10} {'ratio':>7}")
    for num_seqs in [1, 8, 32, 128, 256]:
        for context_len in [256, 1024, 4096]:
            n_blk_per_seq = (context_len + block_size - 1) // block_size
            num_blocks = n_blk_per_seq * num_seqs + 8
            max_blocks = n_blk_per_seq + 1

            q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
            k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=dev)
            v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=dev)

            block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device=dev)
            for s in range(num_seqs):
                block_table[s, :n_blk_per_seq] = torch.arange(s * n_blk_per_seq, (s + 1) * n_blk_per_seq, device=dev)
            context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32, device=dev)

            t_mine = bench_once(paged_attention_decode, q, k_cache, v_cache, block_table, context_lens, scale)
            t_flash = bench_once(
                lambda: flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context_lens, block_table=block_table, softmax_scale=scale, causal=True,
                )
            )

            print(f"{num_seqs:>6} {context_len:>8} {t_mine:>10.3f} {t_flash:>10.3f} {t_mine/t_flash:>6.2f}x")


if __name__ == "__main__":
    main()
