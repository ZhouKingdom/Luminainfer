"""引擎层端到端正确性：greedy 对齐、批处理隔离、前缀缓存、chunked prefill。

说明：zjhvllm 用 flash-attn、HF 用 SDPA，两者 logits 有 ~0.1~0.4 的数值噪声
（test_logits.py 已验证 logits 在该容差内一致）。当某一步 top-1/top-2 的 gap 小于
这个噪声时，greedy 的 argmax 会翻转并级联发散——这是「近邻 tie」的预期行为，
不是 bug。所以 ours-vs-HF 的 greedy 对比用「公共前缀长度」断言（真 bug 会在前几步
就分叉）；ours-vs-ours 的对比则要求精确一致（同代码路径，确定性）。
"""

import torch

from zjhvllm import SamplingParams

# ours-vs-HF 的 greedy 对比，要求至少前 N 个 token 一致（真 bug 会在前 1~2 步分叉）
MIN_COMMON_PREFIX = 8


def _greedy_ours(llm, prompt_ids, max_tokens):
    sp = SamplingParams(greedy=True, max_tokens=max_tokens, ignore_eos=False)
    return llm.generate([prompt_ids], sp, use_tqdm=False)[0]["token_ids"]


def _greedy_hf(hf_model, prompt_ids, max_tokens):
    ids = torch.tensor([prompt_ids], device="cuda")
    out = hf_model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
    return out[0].tolist()[len(prompt_ids):]


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_greedy_matches_hf(llm, hf_model, tokenizer):
    """greedy 输出与 HF 参考对齐。

    由于 zjhvllm(flash-attn) 与 HF(SDPA) 存在 ~0.1~0.4 的 logits 噪声，当某一步
    top1-top2 的 gap 小于噪声时，argmax 会翻转并级联发散（近邻 tie）。因此这里在
    分歧点验证 gap：gap 小 = 近邻 tie（预期）；gap 大 = 真 bug。
    """
    prompts = [
        "Hello, how are you",
        "The capital of France is",
        "In the year 2000, a",
    ]
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        ours = _greedy_ours(llm, ids, 32)
        ref = _greedy_hf(hf_model, ids, 32)
        d = _common_prefix_len(ours, ref)
        print(f"\n[{prompt!r}] common_prefix={d}/{len(ref)}")
        if d == len(ref):
            continue  # 完全一致
        # 分歧点 d：token ref[d] 由「prompt + 前 d 个已生成 token」的最后一个 logits 决定
        prefix = ids + ref[:d]
        with torch.inference_mode():
            logits = hf_model(torch.tensor([prefix], device="cuda")).logits[0, -1]
        top2 = torch.topk(logits, 2).values
        gap = (top2[0] - top2[1]).item()
        # gap 小 = 近邻 tie（噪声导致 argmax 翻转，预期）；gap 大 = 真 bug
        assert gap < 1.0, (
            f"分歧处 logit gap 过大 ({gap:.3f})，不是近邻 tie，疑似 bug: {prompt!r} "
            f"(ours={ours[d]} ref={ref[d]})"
        )


def test_batch_equals_individual(llm):
    prompts = ["Hello world", "The weather is nice", "Machine learning is"]
    sp = SamplingParams(greedy=True, max_tokens=16)
    batch = llm.generate(prompts, sp, use_tqdm=False)
    individual = [llm.generate([p], sp, use_tqdm=False)[0] for p in prompts]
    batch_ids = [o["token_ids"] for o in batch]
    ind_ids = [o["token_ids"] for o in individual]
    assert batch_ids == ind_ids, f"批处理结果与单条不一致:\n{batch_ids}\n{ind_ids}"


def test_prefix_cache_correctness(llm, hf_model, tokenizer):
    """前缀缓存路径的输出必须正确：第二条复用前缀 KV，结果与 HF 参考对齐。"""
    prefix = "The Eiffel Tower is located in the city of"
    sp = SamplingParams(greedy=True, max_tokens=16)

    p1 = prefix + " Paris"
    p2 = prefix + " London"
    out1 = llm.generate([p1], sp, use_tqdm=False)[0]["token_ids"]  # 填充前缀缓存
    out2 = llm.generate([p2], sp, use_tqdm=False)[0]["token_ids"]  # 应复用前缀缓存

    ref2 = _greedy_hf(hf_model, tokenizer.encode(p2, add_special_tokens=True), 16)
    common = _common_prefix_len(out2, ref2)
    print(f"\n[prefix cache] common_prefix={common}/{len(ref2)}")
    assert common >= MIN_COMMON_PREFIX, f"前缀缓存路径输出错误: ours={out2} ref={ref2}"
    assert out1 != out2  # sanity：两个不同后缀应产出不同结果


def test_greedy_is_deterministic(llm):
    sp = SamplingParams(greedy=True, max_tokens=16)
    prompt = "The answer is"
    a = llm.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
    b = llm.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
    assert a == b


def test_seed_reproducible(llm):
    sp = SamplingParams(greedy=False, temperature=1.0, top_k=50, seed=42, max_tokens=16)
    prompt = "Once upon a time"
    a = llm.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
    b = llm.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
    assert a == b, "相同 seed 应得到相同输出"


def test_chunked_prefill_matches_hf(llm, hf_model, tokenizer):
    """长 prompt 触发 chunked prefill（max_num_batched_tokens=512），输出仍与 HF 一致。"""
    prompt = "The quick brown fox jumps over the lazy dog " * 40
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    while len(ids) <= 512:  # 确保超过 512 token 以触发 chunking
        prompt += " hello"
        ids = tokenizer.encode(prompt, add_special_tokens=True)

    assert len(ids) > 512, f"prompt 不够长，未触发 chunked prefill: {len(ids)} tokens"
    ours = _greedy_ours(llm, ids, 16)
    ref = _greedy_hf(hf_model, ids, 16)
    common = _common_prefix_len(ours, ref)
    print(f"\n[chunked prefill, {len(ids)} tokens] common_prefix={common}/{len(ref)}")
    assert common >= MIN_COMMON_PREFIX, f"chunked prefill 输出与参考不一致: ours={ours} ref={ref}"


def test_batch_permutation_preserves_request_isolation(llm):
    """Keep GEMM batch shapes fixed while changing neighbors and KV slot order."""
    prompts=['Hello world','The weather is nice','Machine learning is']
    sp=SamplingParams(greedy=True,max_tokens=16,ignore_eos=True)
    reference=llm.generate(prompts,sp,use_tqdm=False)
    for order in ([2,0,1],[1,2,0]):
        outputs=llm.generate([prompts[i] for i in order],sp,use_tqdm=False)
        for index,item in zip(order,outputs):
            assert item['token_ids']==reference[index]['token_ids']
