"""INT8 storage/loading contracts; CUDA checks require explicit opt-in."""
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file

from zjhvllm.layers.weight_only import (quantize_int8, int8_weight_only_linear,
                                       prepare_qwen35_int8_mlp, validate_quantized_weights)
from zjhvllm.models.qwen3_5 import Qwen3_5MLP, Qwen3_5ParallelLinear
from zjhvllm.utils import tp
from zjhvllm.utils.loader import load_model
from zjhvllm.utils.context import reset_context


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


@pytest.fixture(autouse=True)
def restore_tp():
    old = tp._TP_SIZE, tp._TP_RANK
    tp.set_tp(1, 0)
    yield
    tp._TP_SIZE, tp._TP_RANK = old
    reset_context()


def tiny_model():
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = Qwen3_5MLP(32, 64, "silu").to(torch.bfloat16)
    layer.other = Qwen3_5ParallelLinear(32, 32, column=True).to(torch.bfloat16)
    model.model.layers = nn.ModuleList([layer])
    return model


def test_quantization_error_zero_and_extreme_channels():
    torch.manual_seed(4)
    w = torch.randn(17, 135, dtype=torch.bfloat16)
    w[0] = 0
    w[1] *= 1000
    w[2] *= .0001
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8 and scale.dtype == torch.float32
    assert q.min() >= -127 and q.max() <= 127
    assert not q[0].any() and scale[0] == 1
    assert torch.all((w.float() - q.float()*scale[:, None]).abs() <= scale[:, None]*.5001)
    for invalid in (torch.ones(4), torch.ones(4, 4, dtype=torch.int8), torch.full((2, 2), float("nan"))):
        with pytest.raises(ValueError):
            quantize_int8(invalid)


@pytest.mark.parametrize("bias", [False, True])
def test_cpu_reference_and_noncontiguous_inputs(bias):
    x = torch.randn(2, 5, 33, dtype=torch.bfloat16).transpose(0, 1)
    q, scale = quantize_int8(torch.randn(19, 33, dtype=torch.bfloat16))
    b = torch.randn(19, dtype=x.dtype) if bias else None
    actual = int8_weight_only_linear(x, q, scale, b)
    expected = F.linear(x, (q.float()*scale[:, None]).to(x.dtype), b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        int8_weight_only_linear(x.float(), q, scale, b)


def test_checkpoint_load_storage_and_missing_weights(tmp_path):
    model = tiny_model()
    source = {name: torch.randn_like(p) for name, p in model.named_parameters()}
    before = sum(p.numel()*p.element_size() for p in model.parameters())
    assert prepare_qwen35_int8_mlp(model) == 3
    with pytest.raises(ValueError, match="Missing INT8"):
        validate_quantized_weights(model)
    with pytest.raises(RuntimeError, match="not been loaded"):
        model.model.layers[0].mlp(torch.randn(2, 32, dtype=torch.bfloat16))
    save_file(source, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))
    validate_quantized_weights(model)
    after = sum(p.numel()*p.element_size() for p in model.parameters())
    after += sum(b.numel()*b.element_size() for b in model.buffers())
    # Only the three MLP matrices are quantized; the extra projection stays BF16.
    assert after == before - 3*32*64 + (64+64+32)*4
    for name, module in model.named_modules():
        if hasattr(module, "weight_scale"):
            q, scale = quantize_int8(source[name+".weight"])
            assert torch.equal(module.weight, q) and torch.equal(module.weight_scale, scale)
            assert not module.weight.requires_grad
            assert all(p.dtype != torch.bfloat16 for n, p in module.named_parameters() if n == "weight")
    assert model.model.layers[0].other.weight.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="already"):
        prepare_qwen35_int8_mlp(model)
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    assert torch.isfinite(model.model.layers[0].mlp(x)).all()


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_quantize_after_tp_partition(world):
    torch.manual_seed(123)
    source = {name: torch.randn_like(p) for name, p in tiny_model().named_parameters()}
    for rank in range(world):
        tp.set_tp(world, rank)
        model = tiny_model()
        prepare_qwen35_int8_mlp(model)
        for name, p in model.named_parameters():
            p.weight_loader(p, source[name])
        validate_quantized_weights(model)
        for name, module in model.named_modules():
            if hasattr(module, "weight_scale"):
                shard = source[name+".weight"].chunk(world, dim=0 if module.column else 1)[rank]
                q, scale = quantize_int8(shard)
                assert torch.equal(module.weight, q)
                assert torch.equal(module.weight_scale, scale)


def test_row_parallel_bias_applied_after_reduce(monkeypatch):
    tp.set_tp(2, 0)
    model = tiny_model()
    row = model.model.layers[0].mlp.down_proj
    row.bias = nn.Parameter(torch.randn(32, dtype=torch.bfloat16))
    source = torch.randn(32, 64, dtype=torch.bfloat16)
    prepare_qwen35_int8_mlp(model)
    row.weight.weight_loader(row.weight, source)
    calls = []
    def reduce(y):
        calls.append(y.clone())
        y.mul_(2)
    monkeypatch.setattr("zjhvllm.models.qwen3_5.dist.all_reduce", reduce)
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    partial = int8_weight_only_linear(x, row.weight, row.weight_scale)
    torch.testing.assert_close(row(x), partial * 2 + row.bias, rtol=0, atol=0)
    assert len(calls) == 1 and torch.equal(calls[0], partial)


def test_quantization_config_guards(tmp_path, monkeypatch):
    from zjhvllm.config import Config
    cfg = SimpleNamespace(model_type="qwen3_5_text", dtype=torch.bfloat16, max_position_embeddings=4096)
    monkeypatch.setattr("zjhvllm.config.AutoConfig.from_pretrained", lambda *a, **kw: cfg)
    assert Config(str(tmp_path), quantization="int8_weight_only").quantization == "int8_weight_only"
    for changes in ({"quantization": "int4"}, {"enable_mtp": True}, {"qwen35_linear_backend": "batched"}):
        with pytest.raises(ValueError):
            Config(str(tmp_path), **({"quantization": "int8_weight_only"} | changes))
    cfg.dtype = torch.float32
    with pytest.raises(ValueError, match="BF16"):
        Config(str(tmp_path), quantization="int8_weight_only")
    cfg.dtype = torch.bfloat16
    cfg.quantization_config = {"quant_method": "awq"}
    with pytest.raises(ValueError, match="prequantized"):
        Config(str(tmp_path), quantization="int8_weight_only")


@pytest.mark.skipif(os.getenv("LUMINA_TEST_QUANT_CUDA") != "1", reason="explicit CUDA opt-in required")
@pytest.mark.parametrize("m,n,k", [(1, 131, 259), (4, 512, 1024), (33, 513, 1025), (128, 256, 128)])
def test_cuda_fused_dequant_and_graph(m, n, k):
    torch.manual_seed(10)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_int8(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)*.1)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)*.1
    reference = F.linear(x, (q.float()*scale[:, None]).to(x.dtype), b)
    for _ in range(3):
        y = int8_weight_only_linear(x, q, scale, b)
    # BF16 dequantized-weight reference; split reduction order may differ.
    torch.testing.assert_close(y, reference, rtol=.02, atol=.03125)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = int8_weight_only_linear(x, q, scale, b)
    x.mul_(.5)
    graph.replay()
    torch.testing.assert_close(captured, int8_weight_only_linear(x, q, scale, b), rtol=0, atol=0)
