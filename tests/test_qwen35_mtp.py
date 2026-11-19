"""MTP weight completeness, shifted conditioning and private draft cache."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from zjhvllm.engine.sequence import Sequence
from zjhvllm.models.qwen3_5_mtp import MTPAttention, Qwen3_5MTP
from zjhvllm.utils.loader import load_model


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


def config():
    return SimpleNamespace(mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
                           hidden_size=16, intermediate_size=32, hidden_act="silu", rms_norm_eps=1e-6,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                           max_position_embeddings=64, attention_bias=False, dtype=torch.float32,
                           rope_parameters={"rope_theta": 10000, "partial_rotary_factor": .5})


def initialized_mtp():
    torch.manual_seed(31)
    model = Qwen3_5MTP(config(), 64)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * .02)
    return model


@torch.inference_mode()
def test_incremental_matches_full_and_future_is_masked():
    full = initialized_mtp()
    incremental, changed = deepcopy(full), deepcopy(full)
    embedding, hidden = [torch.randn(11, 16) for _ in range(2)]
    expected = full(embedding, hidden)
    actual = torch.cat([incremental(embedding[:5], hidden[:5]), incremental(embedding[5:8], hidden[5:8]),
                        incremental(embedding[8:], hidden[8:])])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    embedding[6:] += 10
    hidden[6:] -= 10
    future = changed(embedding, hidden)
    torch.testing.assert_close(future[:6], expected[:6], atol=0, rtol=0)
    assert full.cache.length == incremental.cache.length == 11
    assert full.cache.keys.data_ptr() != incremental.cache.keys.data_ptr()


@torch.inference_mode()
def test_gqa_attention_against_explicit_reference():
    torch.manual_seed(32)
    attention = MTPAttention(16, 1, 8, torch.float32)
    q = torch.randn(7, 2, 8)
    k, v = [torch.randn(7, 1, 8) for _ in range(2)]
    outputs = torch.cat([attention(q[:4], k[:4], v[:4]), attention(q[4:], k[4:], v[4:])])
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, dim=1)) * 8**-.5
    scores.masked_fill_(torch.ones(7, 7, dtype=torch.bool).triu(1), float("-inf"))
    expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v.repeat_interleave(2, dim=1))
    torch.testing.assert_close(outputs, expected, atol=5e-7, rtol=1e-5)
    assert attention.length == 7


@torch.inference_mode()
def test_mtp_against_transformers_decoder_components():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
    )
    ours = initialized_mtp()
    cfg = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                           layer_types=["full_attention"], max_position_embeddings=64,
                           rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                            "partial_rotary_factor": .5, "mrope_section": [1, 1, 0]})
    cfg._attn_implementation = "eager"
    decoder = Qwen3_5DecoderLayer(cfg, 0).eval()
    decoder.load_state_dict(ours.layers[0].state_dict(), strict=True)
    embedding, hidden = [torch.randn(11, 16) for _ in range(2)]
    norm_e, norm_h, norm_out = [Qwen3_5RMSNorm(16, 1e-6) for _ in range(3)]
    for actual, expected in [(norm_e, ours.pre_fc_norm_embedding), (norm_h, ours.pre_fc_norm_hidden),
                             (norm_out, ours.norm)]:
        actual.load_state_dict(expected.state_dict())
    projected = torch.nn.functional.linear(torch.cat([norm_e(embedding), norm_h(hidden)], -1), ours.fc.weight)
    positions = torch.arange(11)[None]
    rope = Qwen3_5TextRotaryEmbedding(cfg)(projected[None], positions)
    mask = torch.zeros(1, 1, 11, 11).masked_fill_(torch.ones(11, 11, dtype=torch.bool).triu(1), float("-inf"))
    expected = norm_out(decoder(projected[None], rope, attention_mask=mask))[0]
    actual = ours(embedding, hidden)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("target_failure", [False, True])
def test_generation_uses_mtp_and_rolls_back_draft_on_target_failure(target_failure):
    from zjhvllm.engine.llm_engine import LLMEngine
    from zjhvllm.engine.sequence import SequenceStatus
    from zjhvllm.sampling_params import SamplingParams
    engine = LLMEngine.__new__(LLMEngine)
    seq = Sequence([1, 2], SamplingParams(max_tokens=3))
    mtp = SimpleNamespace(cache=SimpleNamespace(length=0))
    runner = SimpleNamespace(model=SimpleNamespace(mtp=mtp), world_size=1, has_linear_state=True,
                             config=SimpleNamespace(max_model_len=64, hf_config=SimpleNamespace(model_type="qwen3_5_text")))
    engine.model_runner, engine.seq_map, engine._pending_exports = runner, {seq.seq_id: seq}, {}
    engine.is_finished = lambda: True
    engine.add_request = lambda tokens, params: seq.seq_id
    engine.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))

    def prefill():
        seq.num_cached_tokens = len(seq)
        seq.append_token(5)

    def draft(name, request):
        assert name == "draft_mtp" and request is seq
        mtp.cache.length = seq.num_cached_tokens
        return [6], {"draft_source": "mtp"}

    def verify(seq_id, tokens, *, verification_mode):
        assert tokens == [6] and verification_mode == "serial"
        if target_failure:
            raise RuntimeError("target failed")
        seq.append_token(6)
        seq.append_token(7)
        seq.status = SequenceStatus.FINISHED
        return [6, 7], {"accepted": 1}

    engine.step, runner.call, engine.speculative_step = prefill, draft, verify
    if target_failure:
        with pytest.raises(RuntimeError, match="target failed"):
            engine.generate_speculative([1, 2], SamplingParams(max_tokens=3))
        assert mtp.cache.length == 0
    else:
        result = engine.generate_speculative([1, 2], SamplingParams(max_tokens=3))
        assert result["token_ids"] == [5, 6, 7] and result["rounds"][0]["draft_source"] == "mtp"


@pytest.mark.parametrize("missing", [False, True])
def test_checkpoint_mtp_weights_are_required(tmp_path, missing):
    parent = torch.nn.Module()
    parent.mtp = initialized_mtp()
    weights = {name: torch.full_like(param, (i + 1) / 100)
               for i, (name, param) in enumerate(parent.named_parameters())}
    assert len(weights) == 15
    if missing:
        weights.pop("mtp.fc.weight")
    save_file(weights, tmp_path / "model.safetensors")
    if missing:
        with pytest.raises(ValueError, match="Missing MTP checkpoint weights.*mtp.fc.weight"):
            load_model(parent, str(tmp_path))
    else:
        load_model(parent, str(tmp_path))
        assert len(parent.mtp.loaded_weight_names) == 15
        for name, param in parent.named_parameters():
            torch.testing.assert_close(param, weights[name], atol=0, rtol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_draft_uses_shifted_embeddings_and_committed_hidden(fail):
    from zjhvllm.engine.model_runner import ModelRunner
    runner = ModelRunner.__new__(ModelRunner)
    runner.target_hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    seen = []

    class Draft:
        cache = SimpleNamespace(length=1)

        def __call__(self, embedding, hidden):
            seen.append((embedding.clone(), hidden.clone()))
            self.cache.length += len(hidden)
            if fail:
                raise RuntimeError("injected draft failure")
            return hidden

    draft = Draft()
    runner.model = SimpleNamespace(mtp=draft, model=SimpleNamespace(embed_tokens=lambda ids: ids[:, None].repeat(1, 2)),
                                   lm_head=lambda hidden: torch.tensor([[0., 0., 9.]]))
    seq = Sequence([10, 11, 12, 13, 14])
    seq.num_cached_tokens, seq.state_slot, seq.state_ready = 4, 0, True
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            runner.draft_mtp(seq)
        assert draft.cache.length == 1
    else:
        token, stats = runner.draft_mtp(seq)
        assert token == [2] and stats["draft_extend_tokens"] == 3 and draft.cache.length == 4
    torch.testing.assert_close(seen[0][0], torch.tensor([[12, 12], [13, 13], [14, 14]]))
    torch.testing.assert_close(seen[0][1], runner.target_hidden[1:4])
    assert seq.token_ids == [10, 11, 12, 13, 14]
