# =============================================================================
# 文件功能：批处理采样器（Batch Sampler）
# 
# 核心职责：从模型输出的 logits（[num_seqs, vocab]）中，根据用户指定的采样参数，
# 高效、可复现地采样出下一个 token ID（[num_seqs]）。
# 
# 设计特点：
#   - 全程使用 Triton 手写内核，避免 Python 层循环和大量中间张量。
#   - 三个分离内核：温度与惩罚、top_k/top_p 过滤、采样（greedy 或 Gumbel-max）。
#   - 采用二分查找求 top_k/top_p 阈值，避免对整个词表排序，降低计算复杂度。
#   - 使用 Philox 随机数生成器，支持可复现采样（种子可控）。
#   - 所有操作在 fp32 下进行，保证数值稳定性，同时通过按行分块（tile）扫描，
#     降低显存带宽压力，与 vLLM 等工业级采样器对齐。
# =============================================================================

import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _logits_process_kernel(
    logits_ptr,          # [num_seqs, VOCAB] fp32，原地读写
    logits_stride,
    mask_ptr,            # [num_seqs, VOCAB] bool：该位置是否出现在已生成历史中
    mask_stride,
    temperatures_ptr,    # [num_seqs] fp32
    penalties_ptr,       # [num_seqs] fp32
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """
    Triton 内核 1：温度缩放 + repetition penalty（重复惩罚）。
    
    对每个序列（row）的 logits 向量进行处理：
      1. 除以 temperature（如果 temperature != 1.0）。
      2. 对历史中出现过的 token，若 logit > 0 则除以 penalty，若 logit < 0 则乘以 penalty。
         （等价于 vLLM 的 repetition penalty 实现）
    
    内核按词表分块（tile）并行，每个 tile 处理 BLOCK 个 token，而不是按历史长度循环，
    因此无论历史多长，时间复杂度均为 O(vocab)，与历史长度无关。
    """
    row = tl.program_id(0)                               # 当前处理的行（序列索引）
    base = logits_ptr + row * logits_stride              # 该行 logits 的起始指针
    temp = tl.load(temperatures_ptr + row)               # 该行的温度值
    pen = tl.load(penalties_ptr + row)                   # 该行的惩罚系数

    # 对词表分块循环，每个块大小为 BLOCK
    for i in tl.range(0, NUM_TILES):
        offs = i * BLOCK + tl.arange(0, BLOCK)           # 当前块的词汇索引偏移
        msk = offs < VOCAB                               # 有效词汇掩码（防止越界）
        x = tl.load(base + offs, mask=msk, other=0.0)    # 加载 logit

        # 温度缩放（若温度 != 1.0）
        if temp != 1.0:
            x = x / temp

        # 重复惩罚（若惩罚系数 != 1.0）
        if pen != 1.0:
            in_hist = tl.load(mask_ptr + row * mask_stride + offs, mask=msk, other=0)  # 该 token 是否在历史中
            x = tl.where(
                in_hist,
                tl.where(x > 0.0, x / pen, x * pen),     # logit>0 除 pen，logit<0 乘 pen
                x
            )

        tl.store(base + offs, x, mask=msk)               # 写回更新后的 logit


@triton.jit
def _topk_topp_kernel(
    logits_ptr,          # [num_seqs, VOCAB] fp32，原地掩码（被过滤位置置 -inf）
    logits_stride,
    top_ks_ptr,          # [num_seqs] int32；<=0 或 >=VOCAB 表示禁用
    top_ps_ptr,          # [num_seqs] fp32；>=1.0 表示禁用
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
    ITERS: tl.constexpr,
):
    """
    Triton 内核 2：top_k 和 top_p 过滤。
    
    对每个序列，计算 top_k 和 top_p 的阈值，然后将低于阈值的 logit 置为 -inf。
    采用二分查找求阈值，避免对整个词表进行排序，大幅减少计算量。
    
    步骤：
      1. 计算行内 logits 的最大值（m）和最小值（mn）（用于二分边界）。
      2. 若启用 top_k：二分查找第 k 大的 logit 值（kth）。
      3. 若启用 top_p：在 top_k 保留集上，二分查找概率累积阈值对应的 logit 值（pth）。
      4. 最终保留 logit >= max(kth, pth) 的位置，其余置 -inf。
    """
    row = tl.program_id(0)
    base = logits_ptr + row * logits_stride
    k = tl.load(top_ks_ptr + row)                        # top_k 值
    p = tl.load(top_ps_ptr + row)                        # top_p 值
    k_on = (k > 0) & (k < VOCAB)                         # top_k 是否启用
    p_on = p < 1.0                                       # top_p 是否启用
    if (k_on | p_on) == False:
        return                                           # 两者都禁用，直接返回

    # ---- pass 0：计算行内最大值 m 和最小值 mn（用于二分查找边界） ----
    m = tl.full((), -float("inf"), tl.float32)
    mn = tl.full((), float("inf"), tl.float32)
    for i in tl.range(0, NUM_TILES):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        msk = offs < VOCAB
        x = tl.load(base + offs, mask=msk, other=-float("inf"))
        m = tl.maximum(m, tl.max(x))                     # 当前块最大值，整体最大值
        # 排除无效位置（-inf），用 float("inf") 作为占位，取最小值
        mn = tl.minimum(mn, tl.min(tl.where(msk, x, float("inf"))))
    mn = tl.minimum(mn, m)                               # 保证 mn <= m

    # ---- top-k：二分求第 k 大 logit 阈值 ----
    kth = -float("inf")
    if k_on:
        lo = mn
        hi = m
        for _ in tl.range(0, ITERS):
            mid = (lo + hi) * 0.5
            cnt = 0
            for i in tl.range(0, NUM_TILES):
                offs = i * BLOCK + tl.arange(0, BLOCK)
                msk = offs < VOCAB
                x = tl.load(base + offs, mask=msk, other=-float("inf"))
                cnt += tl.sum((x >= mid) & msk).to(tl.int32)          # 统计 logit >= mid 的数量
            if cnt >= k:
                lo = mid                                 # 第 k 大值 >= mid
            else:
                hi = mid
        kth = lo                                         # 最终第 k 大阈值

    # ---- top-p：在 top-k 保留集上二分概率阈值（logit 空间） ----
    pth = -float("inf")
    if p_on:
        # 计算归一化常数 Z = sum_{x >= kth} exp(x - m)
        Z = 0.0
        for i in tl.range(0, NUM_TILES):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            msk = offs < VOCAB
            x = tl.load(base + offs, mask=msk, other=-float("inf"))
            keep = msk & (x >= kth)
            Z += tl.sum(tl.exp(tl.where(keep, x, -float("inf")) - m))
        target = p * Z                                   # 目标累积概率

        lo = mn
        hi = m
        for _ in tl.range(0, ITERS):
            mid = (lo + hi) * 0.5
            t = tl.maximum(mid, kth)                     # 保证阈值不低于 kth
            f = 0.0
            for i in tl.range(0, NUM_TILES):
                offs = i * BLOCK + tl.arange(0, BLOCK)
                msk = offs < VOCAB
                x = tl.load(base + offs, mask=msk, other=-float("inf"))
                keep = msk & (x >= t)
                f += tl.sum(tl.exp(tl.where(keep, x, -float("inf")) - m))
            if f >= target:
                lo = mid
            else:
                hi = mid
        pth = lo

    # ---- 掩码：保留 logit >= max(kth, pth) 的位置，其余置 -inf ----
    if k_on | p_on:
        th = tl.maximum(kth, pth)                        # 最终阈值
        for i in tl.range(0, NUM_TILES):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            msk = offs < VOCAB
            x = tl.load(base + offs, mask=msk, other=0.0)
            x = tl.where(msk & (x >= th), x, -float("inf"))
            tl.store(base + offs, x, mask=msk)


@triton.jit
def _sample_kernel(
    logits_ptr,          # [num_seqs, VOCAB] fp32（已含温度/惩罚/top_k/top_p 处理）
    logits_stride,
    seeds_ptr,           # [num_seqs] int64：每行采样种子（Philox）
    greedy_flags_ptr,    # [num_seqs] int32：1=greedy（argmax），0=核采样
    out_ptr,             # [num_seqs] int32：输出的 token id
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """
    Triton 内核 3：最终采样（greedy 或 Gumbel-max 核采样）。
    
    - 若 greedy 标志为 1：直接取 logit 最大的索引（argmax）。
    - 否则：使用 Gumbel-max 技巧，即 argmax_i (logit_i + Gumbel_i)，
      等价于从 softmax(logits) 分布中采样，无需显式计算 softmax。
    
    Gumbel(0,1) 可通过 u ~ Uniform(0,1) -> -log(-log(u)) 生成。
    每个 token 位置使用独立的 Philox 随机数，保证可复现性。
    """
    row = tl.program_id(0)
    base = logits_ptr + row * logits_stride
    seed = tl.load(seeds_ptr + row)                      # 该行的随机种子
    greedy = tl.load(greedy_flags_ptr + row) != 0       # 是否 greedy

    best_val = -float("inf")
    best_idx = 0
    for i in tl.range(0, NUM_TILES):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        msk = offs < VOCAB
        x = tl.load(base + offs, mask=msk, other=-float("inf"))
        if greedy:
            score = x                                     # greedy 直接用 logit
        else:
            u = tl.rand(seed, offs)                       # 生成 U(0,1)，每个位置独立
            score = x - tl.log(-tl.log(u))                # x + Gumbel(0,1)
        val, idx = tl.max(score, axis=0,return_indices=True)    # 取块内最大值及其索引
        if val > best_val:
            best_val = val
            best_idx = i * BLOCK + idx                    # 全局索引
    tl.store(out_ptr + row, best_idx)


def _build_hist_mask(
    hist_token_ids: torch.Tensor,   # [num_seqs, max_hist] int32/int64，右填充 0
    hist_lens: torch.Tensor,        # [num_seqs] int32
    num_seqs: int,
    vocab: int,
) -> torch.Tensor:
    """
    辅助函数：将“已生成历史 token”列表转换为 [num_seqs, vocab] 的 bool 掩码，
    指示每个 token 是否在对应序列的历史中出现过。
    
    padding 位置（0）会被替换为越界索引 vocab，从而避免误将真正的 token 0 标记为出现过。
    """
    max_hist = hist_token_ids.size(1)
    device = hist_token_ids.device
    cols = torch.arange(max_hist, device=device).unsqueeze(0)               # [1, max_hist]
    valid = cols < hist_lens.unsqueeze(1)                                   # [num_seqs, max_hist] 有效位置掩码
    flat_tokens = torch.where(valid, hist_token_ids.to(torch.int64), vocab) # 无效位置填充 vocab（越界）
    mask = torch.zeros(num_seqs, vocab + 1, dtype=torch.bool, device=device)  # 多一列作为越界存放区
    mask.scatter_(1, flat_tokens, True)                                     # 散列标记出现的 token
    return mask[:, :vocab].contiguous()                                    # 去掉越界列，返回 [num_seqs, vocab]


class Sampler(nn.Module):
    """批处理采样器：输入 [num_seqs, vocab] 的 logits，输出 [num_seqs] 的 token id。"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _block_for(vocab: int) -> int:
        """选择 tile 宽度：取下一个 2 的幂，但不超过 8192，平衡寄存器使用和缓存效率。"""
        return min(8192, 1 << (vocab - 1).bit_length())

    def forward(
        self,
        logits: torch.Tensor,                # [num_seqs, vocab]，bf16/fp16/fp32
        temperatures: torch.Tensor,          # [num_seqs] fp32
        top_ks: torch.Tensor,                # [num_seqs] int32
        top_ps: torch.Tensor,                # [num_seqs] fp32
        repetition_penalties: torch.Tensor,  # [num_seqs] fp32
        seeds: torch.Tensor,                 # [num_seqs] int64
        greedy_flags: torch.Tensor,          # [num_seqs] int32 (0/1)
        hist_token_ids: torch.Tensor | None = None,   # [num_seqs, max_hist] int64（有惩罚时）
        hist_lens: torch.Tensor | None = None,        # [num_seqs] int32
    ) -> torch.Tensor:
        """
        采样主流程：
          1. 将 logits 转为 fp32（内核内浮点计算需要）。
          2. 若需要重复惩罚，构建历史掩码。
          3. 调用内核 1：温度 + 惩罚。
          4. 调用内核 2：top_k + top_p 过滤。
          5. 调用内核 3：采样（greedy 或 Gumbel-max）。
          6. 返回采样得到的 token ID 张量。
        """
        num_seqs, vocab = logits.shape
        if num_seqs == 0:
            return logits.new_empty(0, dtype=torch.int32)
        assert logits.stride(1) == 1, "logits 最后一维需连续"
        logits = logits.float()              # 采样全程 fp32 保证数值精度

        BLOCK = self._block_for(vocab)                          # 每 tile 处理的词表大小
        NUM_TILES = (vocab + BLOCK - 1) // BLOCK               # 总 tile 数

        # ---- 1) 温度缩放 + 重复惩罚 ----
        if hist_token_ids is not None:
            mask = _build_hist_mask(hist_token_ids, hist_lens, num_seqs, vocab)
        else:
            mask = logits.new_zeros(1, 1, dtype=torch.bool)     # dummy，当 pen==1.0 时不读取
        _logits_process_kernel[(num_seqs,)](
            logits, logits.stride(0), mask, mask.stride(0),
            temperatures, repetition_penalties,
            VOCAB=vocab, BLOCK=BLOCK, NUM_TILES=NUM_TILES,
            num_warps=8,
        )

        # ---- 2) top_k / top_p 过滤 ----
        _topk_topp_kernel[(num_seqs,)](
            logits, logits.stride(0), top_ks, top_ps,
            VOCAB=vocab, BLOCK=BLOCK, NUM_TILES=NUM_TILES, ITERS=14,
            num_warps=8,
        )

        # ---- 3) 采样 ----
        out = torch.empty(num_seqs, dtype=torch.int32, device=logits.device)
        _sample_kernel[(num_seqs,)](
            logits, logits.stride(0), seeds, greedy_flags, out,
            VOCAB=vocab, BLOCK=BLOCK, NUM_TILES=NUM_TILES,
            num_warps=8,
        )
        return out