"""Fixed-tile small-batch BF16 linear, inspired by SGLang invariant GEMM.

K partition and reduction order depend on weight shape, never on token count.
This is a new numerical backend, not an emulation of cuBLAS's M=1 heuristic.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, W, Partial, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            XS: tl.constexpr, SPLITS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mn = tl.program_id(0)
    split = tl.program_id(1)
    nt = tl.cdiv(N, BN)
    rows = mn // nt * BM + tl.arange(0, BM)
    cols = mn % nt * BN + tl.arange(0, BN)
    steps = tl.cdiv(K, BK * SPLITS)
    kk = split * steps * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(steps):
        a = tl.load(X + rows[:, None] * XS + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(W + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), other=0)
        acc = tl.dot(a, b, acc)
        kk += BK
    offset = (split * M + rows[:, None]) * N + cols[None, :]
    tl.store(Partial + offset, acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _finish(Partial, Bias, Out, M: tl.constexpr, N: tl.constexpr,
            SPLITS: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    vals = tl.load(Partial + splits[:, None] * M * N + offsets[None, :],
                   offsets[None, :] < M * N, other=0)
    value = tl.sum(vals, 0)
    if HAS_BIAS:
        value += tl.load(Bias + offsets % N, offsets < M * N, other=0).to(tl.float32)
    tl.store(Out + offsets, value, offsets < M * N)


def invariant_linear(x, weight, bias=None):
    if (not x.is_cuda or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or x.ndim != 2 or x.stride(1) != 1 or not weight.is_contiguous()):
        raise ValueError("invariant linear requires CUDA BF16 rows and contiguous weights")
    m, k = x.shape
    n, wk = weight.shape
    if wk != k or not 1 <= m <= 32 or k % 16:
        raise ValueError("invariant linear supports 1..32 rows and K divisible by 16")
    # Enough CTAs for small output dimensions; LM head already has many tiles.
    splits = 8 if n <= 17408 and k >= 1024 else 1
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _linear[(triton.cdiv(m, 16) * triton.cdiv(n, 64), splits)](
        x, weight, partial, m, n, k, x.stride(0), splits, 16, 64, 64,
        num_warps=4, num_stages=3)
    _finish[(triton.cdiv(m * n, 256),)](
        partial, bias, out, m, n, splits, bias is not None, 256, num_warps=4)
    return out
