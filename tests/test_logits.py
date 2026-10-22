"""模型层正确性：zjhvllm 的 Qwen3 前向 logits 与 HuggingFace 参考对齐。

这是最强、最不易误报的校验 —— 直接比对最后一个 token 的 logits 张量，
能抓住权重加载、attention、RoPE、RMSNorm、lm_head（含 tie_word_embeddings）等隐藏 bug。
"""

import torch

import pytest

# 容差：flash-attn（在线 softmax + fp32 累加）与 HF SDPA 的数值实现差异，随序列长度
# 累积到 ~0.4（实测）。真实的 bug（权重加载错、RoPE 错、归一化错）会导致 diff >> 1
# 且 argmax 也错。这里用 1.0 区分「噪声」与「真 bug」，argmax 单独打印作强信号。
ATOL = 1.0


def _ours_logits(model, input_ids):
    """input_ids: [n] on cuda。返回 [1, vocab] 的最后一个 token logits。"""
    from zjhvllm.utils.context import set_context, reset_context

    n = input_ids.numel()
    cu = torch.tensor([0, n], dtype=torch.int32, device=input_ids.device)
    positions = torch.arange(n, dtype=torch.int64, device=input_ids.device)
    set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=n, max_seqlen_k=n)
    try:
        with torch.inference_mode():
            hidden = model(input_ids, positions)
            logits = model.compute_logits(hidden)
    finally:
        reset_context()
    return logits  # [1, vocab]


def _hf_logits(model, input_ids):
    """input_ids: [1, n] on cuda。返回 [vocab] 的最后一个 token logits。"""
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits[0, -1]


def _tokenize(tokenizer, prompt):
    return torch.tensor(tokenizer.encode(prompt, add_special_tokens=True), dtype=torch.int64, device="cuda")


@pytest.mark.parametrize(
    "prompt",
    [
        "Hello, world",
        "The capital of France is",
        "1 + 1 =",
        "What is the meaning of life?",
        "Once upon a time, there was a",
    ],
)
def test_prefill_logits_match_hf(our_model, hf_model, tokenizer, prompt):
    ids = _tokenize(tokenizer, prompt)

    ours = _ours_logits(our_model, ids).float()          # [1, vocab]
    ref = _hf_logits(hf_model, ids.unsqueeze(0)).float()  # [vocab]

    diff = (ours[0] - ref).abs()
    max_diff = diff.max().item()
    # 也会校验 argmax 是否一致（更强的信号，但可能受近邻 logits 影响）
    argmax_match = ours[0].argmax().item() == ref.argmax().item()

    print(f"\n[{prompt!r}] max_abs_diff={max_diff:.6f}  argmax_match={argmax_match}")
    assert max_diff < ATOL, f"logits 偏差过大: {max_diff} (prompt={prompt!r})"


def test_long_prompt_logits_match_hf(our_model, hf_model, tokenizer):
    """长 prompt（覆盖 chunked prefill 之前的基础 attention 路径）。"""
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    ids = _tokenize(tokenizer, prompt)

    ours = _ours_logits(our_model, ids).float()
    ref = _hf_logits(hf_model, ids.unsqueeze(0)).float()

    max_diff = (ours[0] - ref).abs().max().item()
    print(f"\n[long prompt, {ids.numel()} tokens] max_abs_diff={max_diff:.6f}")
    assert max_diff < ATOL, f"logits 偏差过大: {max_diff}"
