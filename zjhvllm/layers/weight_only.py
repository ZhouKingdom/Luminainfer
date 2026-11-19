"""Experimental per-output-channel INT8 weights with BF16 activations.

CUDA dequantizes tiles inside GEMM, never materializing a full BF16 weight.
The CPU implementation is a numerical reference, not a performance backend.
"""
import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl


@torch.no_grad()
def quantize_int8(weight):
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("INT8 weight quantization requires a floating point matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("cannot quantize non-finite weights")
    # A zero channel uses scale=1, avoiding NaN and unnecessary underflow.
    maximum = weight.float().abs().amax(dim=1)
    scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    quantized = torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


@triton.jit
def _gemm(X, W, Scale, Partial, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          XS: tl.constexpr, SPLITS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile, split = tl.program_id(0), tl.program_id(1)
    nt = tl.cdiv(N, BN)
    rows = tile // nt * BM + tl.arange(0, BM)
    cols = tile % nt * BN + tl.arange(0, BN)
    steps = tl.cdiv(K, BK * SPLITS)
    kk = split * steps * BK + tl.arange(0, BK)
    scale = tl.load(Scale + cols, cols < N, other=0)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(steps):
        a = tl.load(X + rows[:, None] * XS + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        q = tl.load(W + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), other=0)
        b = (q.to(tl.float32) * scale[None, :]).to(a.dtype)
        acc = tl.dot(a, b, acc)
        kk += BK
    offsets = (split * M + rows[:, None]) * N + cols[None, :]
    tl.store(Partial + offsets, acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _finish(Partial, Bias, Y, M: tl.constexpr, N: tl.constexpr,
            SPLITS: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, SPLITS)
    values = tl.load(Partial + parts[:, None] * M * N + offsets[None, :],
                     offsets[None, :] < M * N, other=0)
    value = tl.sum(values, 0)
    if HAS_BIAS:
        value += tl.load(Bias + offsets % N, offsets < M * N, other=0).to(tl.float32)
    tl.store(Y + offsets, value, offsets < M * N)


def int8_weight_only_linear(x, weight, scale, bias=None):
    if (x.ndim < 2 or x.dtype != torch.bfloat16 or weight.ndim != 2
            or weight.dtype != torch.int8 or not weight.is_contiguous()
            or scale.shape != (weight.shape[0],) or scale.dtype != torch.float32
            or not scale.is_contiguous() or x.shape[-1] != weight.shape[1]
            or x.device != weight.device or x.device != scale.device):
        raise ValueError("W8A16 requires BF16 activations, contiguous INT8 weights and FP32 channel scales")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.device != x.device
                             or bias.dtype != x.dtype):
        raise ValueError("bias must match output channels, activation device and dtype")
    if x.device.type == "cpu":
        return F.linear(x, (weight.float() * scale[:, None]).to(x.dtype), bias)
    if not x.is_cuda:
        raise ValueError("W8A16 supports CUDA and CPU reference only")
    shape = (*x.shape[:-1], weight.shape[0])
    x = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x.shape
    n = weight.shape[0]
    if k == 0 or n == 0:
        raise ValueError("W8A16 requires nonempty weight dimensions")
    if m == 0:
        return torch.empty(shape, dtype=x.dtype, device=x.device)
    splits = 8 if m <= 32 and k >= 1024 else 1
    bm = 16 if m <= 32 else 32
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _gemm[(triton.cdiv(m, bm) * triton.cdiv(n, 64), splits)](
        x, weight, scale, partial, m, n, k, x.stride(0), splits, bm, 64, 64,
        num_warps=4, num_stages=3)
    _finish[(triton.cdiv(m * n, 256),)](
        partial, bias, out, m, n, splits, bias is not None, 256, num_warps=4)
    return out.reshape(shape)


def prepare_qwen35_int8_mlp(model):
    """Replace local TP MLP weights before loading the BF16 checkpoint.

    Reuse each projection's original loader to shard first, then quantize.
    Retain only INT8 weight + FP32 scales after each tensor is loaded.
    """
    from zjhvllm.models.qwen3_5 import Qwen3_5ParallelLinear
    count = 0
    for name, module in model.named_modules():
        if not (name.startswith("model.layers.") and ".mlp." in name
                and isinstance(module, Qwen3_5ParallelLinear)):
            continue
        if hasattr(module, "weight_scale"):
            raise ValueError("weight-only quantization has already been prepared")
        old = module.weight
        dtype = old.dtype
        original_loader = old.weight_loader
        module.weight = nn.Parameter(torch.empty_like(old, dtype=torch.int8), requires_grad=False)
        module.register_buffer("weight_scale", torch.empty(old.shape[0], device=old.device, dtype=torch.float32))
        module.quantization_loaded = False

        def loader(param, value, *, target=module, source_loader=original_loader, source_dtype=dtype):
            local = nn.Parameter(torch.empty_like(param, dtype=source_dtype), requires_grad=False)
            source_loader(local, value)
            q, scale = quantize_int8(local)
            param.data.copy_(q)
            target.weight_scale.copy_(scale)
            target.quantization_loaded = True

        module.weight.weight_loader = loader
        count += 1
    if not count:
        raise ValueError("no Qwen3.5 MLP projections found for INT8 quantization")
    return count


def validate_quantized_weights(model):
    missing = [name for name, module in model.named_modules()
               if hasattr(module, "weight_scale") and not module.quantization_loaded]
    if missing:
        raise ValueError(f"Missing INT8 source weights: {missing}")
