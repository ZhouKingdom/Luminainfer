"""MoE 层正确性：router + top-k 专家选择 + 加权组合，与 token-by-token 参考实现对齐。"""

import torch
import torch.nn.functional as F

from zjhvllm.layers.moe import MoE


def _reference(x, moe):
    """逐 token 的参考实现（显式正确，用于对比 MoE 层的向量化 dispatch）。"""
    routing_logits = moe.gate(x)
    topk_weights, topk_ids = torch.topk(routing_logits, moe.num_experts_per_tok, dim=-1)
    topk_weights = topk_weights.float()
    if moe.norm_topk_prob:
        topk_weights = F.softmax(topk_weights, dim=-1)

    out = torch.zeros_like(x, dtype=torch.float32)
    for i in range(x.shape[0]):
        for k in range(moe.num_experts_per_tok):
            e = int(topk_ids[i, k].item())
            w = topk_weights[i, k]
            out[i] += w * moe.experts[e](x[i:i + 1]).squeeze(0)
    return out.to(x.dtype)


def _make_moe(hidden, inter, n_exp, n_topk, norm_topk_prob):
    torch.manual_seed(0)
    moe = MoE(hidden, inter, n_exp, n_topk, norm_topk_prob).cuda().eval()
    # 并行线性层的权重是 torch.empty（无初始化），合成测试需手动赋随机值
    for p in moe.parameters():
        if p.dim() >= 1:
            torch.nn.init.normal_(p, std=0.1)
    return moe


def test_moe_matches_reference():
    for norm in [False, True]:
        moe = _make_moe(32, 16, 4, 2, norm)
        x = torch.randn(8, 32, device="cuda")
        with torch.inference_mode():
            out = moe(x)
            ref = _reference(x, moe)
        max_diff = (out - ref).abs().max().item()
        print(f"\n[norm_topk_prob={norm}] max_abs_diff={max_diff:.2e}")
        assert torch.allclose(out, ref, atol=1e-4), f"norm_topk_prob={norm} 下 MoE 输出不一致"


def test_moe_topk_selection():
    """验证 router 确实按 top-k 选专家：直接比对 torch.topk 的专家 id。"""
    moe = _make_moe(32, 16, 8, 3, False)
    x = torch.randn(5, 32, device="cuda")
    with torch.inference_mode():
        routing_logits = moe.gate(x)
        topk_ids = torch.topk(routing_logits, 3, dim=-1).indices
    # 每个 token 选的 3 个专家互不相同（top-k 不重复）
    assert (topk_ids[:, 0] != topk_ids[:, 1]).all()
    assert (topk_ids[:, 1] != topk_ids[:, 2]).all()
    # 选中的是 logit 最大的 3 个专家
    expected = routing_logits.argsort(dim=-1, descending=True)[:, :3]
    assert (topk_ids == expected).all()


def test_moe_weight_loading(tmp_path):
    """端到端：合成 tiny Qwen3MoE 权重（HF 命名）→ safetensors → load_model → 校验权重正确合并。"""
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig

    from zjhvllm.models.qwen3_moe import Qwen3MoeForCausalLM
    from zjhvllm.utils.loader import load_model

    config = Qwen3MoeConfig(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, max_position_embeddings=128,
        rms_norm_eps=1e-6, tie_word_embeddings=False, attention_bias=True,
        torch_dtype="float32",
    )

    torch.manual_seed(0)
    weights = {
        "model.embed_tokens.weight": torch.randn(1000, 32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(16, 32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
        "model.layers.0.mlp.gate.weight": torch.randn(4, 32),  # router [num_experts, hidden]
        "model.layers.0.input_layernorm.weight": torch.randn(32),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(32),
        "model.norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(1000, 32),
    }
    for e in range(4):
        weights[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(16, 32)
        weights[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(16, 32)
        weights[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(32, 16)

    save_file(weights, str(tmp_path / "model.safetensors"))

    prev_dtype, prev_device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(config.dtype)
    torch.set_default_device("cpu")
    try:
        model = Qwen3MoeForCausalLM(config)
        load_model(model, str(tmp_path))
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)

    mlp = model.model.layers[0].mlp
    # router gate
    assert torch.allclose(mlp.gate.weight, weights["model.layers.0.mlp.gate.weight"])
    # 专家 2：gate_up_proj 应该是 [gate_proj; up_proj] 拼接
    exp2 = mlp.experts[2]
    expected = torch.cat([
        weights["model.layers.0.mlp.experts.2.gate_proj.weight"],
        weights["model.layers.0.mlp.experts.2.up_proj.weight"],
    ], dim=0)
    assert torch.allclose(exp2.gate_up_proj.weight, expected), "专家 gate/up 未正确合并进 gate_up_proj"
    assert torch.allclose(exp2.down_proj.weight, weights["model.layers.0.mlp.experts.2.down_proj.weight"])

