"""Gated DeltaNet 的 Triton 算子。

chunk 化 delta rule 的「chunk 间顺序扫描」是纯 torch 版的最大瓶颈（每个 head 顺序扫
num_chunks 个 chunk，每步 4 个批量小 matmul，存在 Python 循环与重复 launch
开销）。本文件用 Triton 把整个扫描融合成一个 kernel：递归状态 S [Dk, Dv] 驻留 SRAM，
chunk 间顺序扫描、matmul 全用 tl.dot，消除 Python 循环与 launch。

S 的列（value 维 Dv）在 delta rule 里相互独立，所以按 Dv 分块并行：grid = (Hv, Dv/DV_TILE)，
每个 program 只驻留 [Dk, DV_TILE] 的 S 子块，避免 S [128,128] 常驻 SRAM 超限。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_scan_kernel(
    qd_ptr, kd_ptr, nv_ptr, kc_ptr, intra_ptr, cd_ptr, S_ptr, o_ptr,
    nc,
    Dk: tl.constexpr, Dv: tl.constexpr, CHUNK: tl.constexpr, DV_TILE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """顺序扫描 num_chunks 个 chunk，维护递归状态 S 的一个 Dv 子块。

    输入均预计算为连续布局（head 维在最前，各 chunk 顺序排列）：
        qd    [Hv, nc, CHUNK, Dk]   = query * cum_decay.exp()
        kd    [Hv, nc, CHUNK, Dk]   = key   * (cum_decay[-1]-cum_decay).exp()
        nv    [Hv, nc, CHUNK, Dv]   = solve_triangular 解出的 new_values
        kc    [Hv, nc, CHUNK, Dk]   = solve_triangular 解出的 k_cumdecay
        intra [Hv, nc, CHUNK, CHUNK]= 块内 attention (q@k.T)*pairwise
        cd    [Hv, nc]              = cum_decay[-1].exp()（chunk 衰减标量）
        S     [Hv, Dk, Dv]          = 递归状态（读写）
        o     [Hv, nc, CHUNK, Dv]   = 输出（写）
    """
    h = tl.program_id(0)
    t = tl.program_id(1)
    dk = tl.arange(0, Dk)
    dv = t * DV_TILE + tl.arange(0, DV_TILE)
    c = tl.arange(0, CHUNK)

    # 加载初始状态子块 S [Dk, DV_TILE]
    S = tl.load(S_ptr + h * Dk * Dv + dk[:, None] * Dv + dv[None, :])

    # head 维 stride 各张量不同：qd/kd/kc 是 nc*CHUNK*Dk，nv/o 是 nc*CHUNK*Dv，intra 是 nc*CHUNK*CHUNK
    for i in range(nc):
        qd = tl.load(qd_ptr + (h * nc * CHUNK + i * CHUNK) * Dk + c[:, None] * Dk + dk[None, :])   # [CHUNK, Dk]
        kd = tl.load(kd_ptr + (h * nc * CHUNK + i * CHUNK) * Dk + c[:, None] * Dk + dk[None, :])
        kc = tl.load(kc_ptr + (h * nc * CHUNK + i * CHUNK) * Dk + c[:, None] * Dk + dk[None, :])
        nv = tl.load(nv_ptr + (h * nc * CHUNK + i * CHUNK) * Dv + c[:, None] * Dv + dv[None, :])   # [CHUNK, DV_TILE]
        intra = tl.load(intra_ptr + (h * nc * CHUNK + i * CHUNK) * CHUNK
                        + c[:, None] * CHUNK + c[None, :])                                          # [CHUNK, CHUNK]
        cd = tl.load(cd_ptr + h * nc + i)                                                           # 标量

        v_new = nv - tl.dot(kc, S, input_precision=DOT_PRECISION)                        # [CHUNK, DV_TILE]
        o_i = tl.dot(qd, S, input_precision=DOT_PRECISION) + tl.dot(intra, v_new, input_precision=DOT_PRECISION)  # [CHUNK, DV_TILE]
        S = S * cd + tl.dot(tl.trans(kd), v_new, input_precision=DOT_PRECISION)          # [Dk, DV_TILE]

        tl.store(o_ptr + (h * nc * CHUNK + i * CHUNK) * Dv + c[:, None] * Dv + dv[None, :], o_i)

    tl.store(S_ptr + h * Dk * Dv + dk[:, None] * Dv + dv[None, :], S)


def gated_delta_scan(qd, kd, nv, kc, intra, cd, S, dv_tile=16, num_warps=8, input_precision="ieee"):
    """融合的 chunk 间顺序扫描。返回 (o [Hv, nc, CHUNK, Dv], S_new [Hv, Dk, Dv])。

    S 原地更新（传入的 S 会被覆盖），返回 o 与新状态 S。
    默认保留 IEEE FP32；实验模式 input_precision="tf32x3" 用多次
    Tensor Core 乘法逼近 fp32，不等同于低精度单次 tf32。
    """
    # Projections and triangular solves may retain transposed strides.
    # The kernel uses packed head-major addressing.
    qd, kd, nv, kc, intra, cd = (
        x.contiguous() for x in (qd, kd, nv, kc, intra, cd)
    )
    if not S.is_contiguous():
        raise ValueError("gated_delta_scan requires contiguous in-place state")
    Hv, nc, CHUNK, Dk = qd.shape
    Dv = nv.shape[-1]
    o = torch.empty(Hv, nc, CHUNK, Dv, device=qd.device, dtype=torch.float32)
    _gated_delta_scan_kernel[(Hv, Dv // dv_tile)](
        qd, kd, nv, kc, intra, cd, S, o, nc,
        Dk=Dk, Dv=Dv, CHUNK=CHUNK, DV_TILE=dv_tile, DOT_PRECISION=input_precision, num_warps=num_warps, num_stages=1)
    return o, S


@triton.jit
def _delta_decode_kernel(Q, K, V, G, Beta, Cache, Slots, Out, StateOut, OutSlots,
                         H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                         STATE_STRIDE: tl.constexpr, OUT_STATE_STRIDE: tl.constexpr, TILE: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    d = tl.arange(0, triton.next_power_of_2(DK))
    v = tile * TILE + tl.arange(0, TILE)
    slot = tl.load(Slots + b).to(tl.int64)
    offset = slot * STATE_STRIDE + h * DK * DV + d[:, None] * DV + v[None, :]
    mask = (d[:, None] < DK) & (v[None, :] < DV)
    state = tl.load(Cache + offset, mask, 0)
    q = tl.load(Q + (b * H + h) * DK + d, d < DK, 0)
    k = tl.load(K + (b * H + h) * DK + d, d < DK, 0)
    value = tl.load(V + (b * H + h) * DV + v, v < DV, 0)
    decay = tl.exp(tl.load(G + b * H + h))
    beta = tl.load(Beta + b * H + h)
    prediction = tl.sum(k[:, None] * state, axis=0)
    delta = value - decay * prediction
    state = decay * state + (beta * k[:, None]) * delta[None, :]
    output = tl.sum(q[:, None] * state, axis=0)
    out_slot = tl.load(OutSlots + b).to(tl.int64)
    out_offset = out_slot * OUT_STATE_STRIDE + h * DK * DV + d[:, None] * DV + v[None, :]
    tl.store(StateOut + out_offset, state, mask)
    tl.store(Out + (b * H + h) * DV + v, output, v < DV)


def gated_delta_decode(q, k, v, g, beta, state_cache, slots, *, output_cache=None, output_slots=None):
    """One write-before-read step directly in cache; slots must be distinct.

    q/k/v are normalized fp32 [batch, heads, dim]. Cache may be a layer
    view into [slots, layers, heads, key_dim, value_dim].
    """
    if output_cache is None:
        output_cache = state_cache
    if output_slots is None:
        output_slots = slots
    batch, heads, dk = q.shape
    dv = v.shape[-1]
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    output = torch.empty_like(v)
    _delta_decode_kernel[(batch, heads, triton.cdiv(dv, 16))](
        q, k, v, g, beta, state_cache, slots, output, output_cache, output_slots,
        H=heads, DK=dk, DV=dv, STATE_STRIDE=state_cache.stride(0),
        OUT_STATE_STRIDE=output_cache.stride(0), TILE=16,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


@triton.jit
def _delta_verify_kernel(Q, K, V, G, Beta, Cache, Slots, Cu, Out, History,
                         H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                         STATE_STRIDE: tl.constexpr, HISTORY_STRIDE: tl.constexpr,
                         STEPS: tl.constexpr, TILE: tl.constexpr):
    # One program retains a request's state tile across the candidate chain.
    # Scratch records every boundary; persistent state is never published here.
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, triton.next_power_of_2(DK))
    v = tile * TILE + tl.arange(0, TILE)
    slot = tl.load(Slots + b).to(tl.int64)
    start = tl.load(Cu + b).to(tl.int64)
    end = tl.load(Cu + b + 1).to(tl.int64)
    mask = (d[:, None] < DK) & (v[None, :] < DV)
    within_head = h * DK * DV + d[:, None] * DV + v[None, :]
    state = tl.load(Cache + slot * STATE_STRIDE + within_head, mask, 0)
    for step in range(STEPS):
        row = start + step
        if row < end:
            q = tl.load(Q + (row * H + h) * DK + d, d < DK, 0)
            k = tl.load(K + (row * H + h) * DK + d, d < DK, 0)
            value = tl.load(V + (row * H + h) * DV + v, v < DV, 0)
            decay = tl.exp(tl.load(G + row * H + h))
            beta = tl.load(Beta + row * H + h)
            # Keep the existing decode operation order, tile and FP32 rounding.
            prediction = tl.sum(k[:, None] * state, axis=0)
            delta = value - decay * prediction
            state = decay * state + (beta * k[:, None]) * delta[None, :]
            output = tl.sum(q[:, None] * state, axis=0)
            tl.store(History + row * HISTORY_STRIDE + within_head, state, mask)
            tl.store(Out + (row * H + h) * DV + v, output, v < DV)


def gated_delta_verify(q, k, v, g, beta, state_cache, slots, cu_seqlens,
                       output_cache, *, max_steps=2):
    """Short varlen verification, no persistent writes, all step states saved.

    Execution follows SGLang's target_verify pattern (window loop inside a
    kernel, disable_state_update, intermediate states), with Lumina's existing
    FP32 state layout and arithmetic. This is an independent implementation.
    """
    if not 1 <= max_steps <= 9:
        raise ValueError('fused verification supports up to nine input rows per request')
    if q.shape != k.shape or q.shape[:2] != v.shape[:2] or len(output_cache) < len(q):
        raise ValueError('invalid fused verification tensor shapes')
    if len(cu_seqlens) != len(slots) + 1 or state_cache.dtype != torch.float32 or output_cache.dtype != torch.float32:
        raise ValueError('fused verification requires FP32 state and request boundaries')
    rows, heads, dk = q.shape
    dv = v.shape[-1]
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    output = torch.empty_like(v)
    _delta_verify_kernel[(len(slots), heads, triton.cdiv(dv, 16))](
        q, k, v, g, beta, state_cache, slots, cu_seqlens, output, output_cache,
        H=heads, DK=dk, DV=dv, STATE_STRIDE=state_cache.stride(0),
        HISTORY_STRIDE=output_cache.stride(0), STEPS=max_steps, TILE=16,
        num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _conv_verify_kernel(X, Weight, Cache, Slots, Cu, Out, History,
                        C: tl.constexpr, WIDTH: tl.constexpr,
                        STATE_STRIDE: tl.constexpr, HISTORY_STRIDE: tl.constexpr,
                        STEPS: tl.constexpr, BLOCK: tl.constexpr):
    request, block = tl.program_id(0), tl.program_id(1)
    channels = block * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + request).to(tl.int64)
    start = tl.load(Cu + request).to(tl.int64)
    end = tl.load(Cu + request + 1).to(tl.int64)
    for step in tl.static_range(STEPS):
        row = start + step
        if row < end:
            acc = tl.full((BLOCK,), 0, tl.float32)
            for tap in tl.static_range(WIDTH):
                if step + tap < WIDTH - 1:
                    value = tl.load(Cache + slot * STATE_STRIDE + channels * (WIDTH - 1)
                                    + step + tap, channels < C, 0)
                else:
                    value = tl.load(X + (start + step + tap - WIDTH + 1) * C + channels, channels < C, 0)
                weight = tl.load(Weight + channels * WIDTH + tap, channels < C, 0)
                acc = acc + value.to(tl.float32) * weight.to(tl.float32)
            tl.store(Out + row * C + channels, acc, channels < C)
            for tap in tl.static_range(WIDTH - 1):
                if step + tap + 1 < WIDTH - 1:
                    value = tl.load(Cache + slot * STATE_STRIDE + channels * (WIDTH - 1)
                                    + step + tap + 1, channels < C, 0)
                else:
                    value = tl.load(X + (start + step + tap - WIDTH + 2) * C + channels, channels < C, 0)
                tl.store(History + row * HISTORY_STRIDE + channels * (WIDTH - 1) + tap, value, channels < C)


def causal_conv_verify(mixed, weight, state_cache, slots, cu_seqlens, output_cache, *, max_steps=2):
    """Out-of-place short-window depthwise conv, preserving BF16 rounding.

    Follows the speculative conv-window layout used by SGLang. SiLU is applied
    by the caller after the convolution's activation-dtype rounding boundary.
    """
    if mixed.dtype != torch.bfloat16 or weight.dtype != mixed.dtype or not 1 <= max_steps <= 9:
        raise ValueError('short-window convolution requires BF16 and one to nine steps')
    channels, _, width = weight.shape
    if (width != 4 or mixed.shape[1] != channels or len(cu_seqlens) != len(slots)+1
            or len(output_cache) < len(mixed)):
        raise ValueError('invalid short-window convolution layout')
    output = torch.empty_like(mixed)
    _conv_verify_kernel[(len(slots), triton.cdiv(channels, 256))](
        mixed, weight, state_cache, slots, cu_seqlens, output, output_cache,
        C=channels, WIDTH=width, STATE_STRIDE=state_cache.stride(0),
        HISTORY_STRIDE=output_cache.stride(0), STEPS=max_steps, BLOCK=256,
        num_warps=4, enable_fp_fusion=False)
    return output
