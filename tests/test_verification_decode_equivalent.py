"""Exact hybrid verification against ordinary decode, including page crossings."""
from types import SimpleNamespace

import pytest
import torch

from zjhvllm.engine.model_runner import ModelRunner
from zjhvllm.engine.sequence import Sequence
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.sampling_params import SamplingParams
from zjhvllm.utils.context import get_context, reset_context, set_context


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cached,count", [(255, 2), (256, 5), (257, 2), (255, 33)])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("linear_backend", ["decode_equivalent", "batch_invariant"])
@torch.inference_mode()
def test_mixed_model_window_matches_decode_bitwise(cached, count, kv_dtype, use_graph, linear_backend):
    torch.manual_seed(153)
    cfg = SimpleNamespace(hidden_size=256, intermediate_size=512, hidden_act="silu",
                          model_type="qwen3_5_text",
                          num_hidden_layers=4, layer_types=["linear_attention", "full_attention"] * 2,
                          linear_num_key_heads=2, linear_num_value_heads=4,
                          linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                          rms_norm_eps=1e-6, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                          max_position_embeddings=1024, attention_bias=False,
                          rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": .25},
                          vocab_size=128, tie_word_embeddings=False)
    model = Qwen3_5ForCausalLM(cfg).cuda().bfloat16().eval()
    from zjhvllm.models.qwen3_5 import Qwen3_5Linear
    for module in model.modules():
        if isinstance(module, Qwen3_5Linear):
            module.backend = linear_backend
    states = torch.randn(3, 2, 4, 128, 128, device="cuda") * .01
    convs = torch.randn(3, 2, 1024, 3, device="cuda", dtype=torch.bfloat16) * .01
    kv = torch.randn(2, 2, 3, 256, 2, 64, device="cuda", dtype=torch.bfloat16) * .01
    kv = kv.to(kv_dtype)
    scales = torch.ones(2, 2, 3, 256, device="cuda")
    linear, full = 0, 0
    for layer in model.model.layers:
        if layer.linear_attn is not None:
            layer.linear_attn.S_cache = states[:, linear]
            layer.linear_attn.conv_cache = convs[:, linear]
            linear += 1
        else:
            layer.self_attn.attn.k_cache = kv[0, full]
            layer.self_attn.attn.v_cache = kv[1, full]
            layer.self_attn.attn.k_scale = scales[0, full]
            layer.self_attn.attn.v_scale = scales[1, full]
            full += 1
    buffers = (states, convs, kv, scales)
    initial = [x.clone() for x in buffers]
    ids = torch.randint(0, cfg.vocab_size, (count,), device="cuda")
    positions = torch.arange(cached, cached + count, device="cuda")
    pages = [2, 0, 1]
    block_table = torch.tensor([pages[:(cached + count + 255) // 256]], device="cuda", dtype=torch.int32)
    slot_mapping = torch.tensor([pages[int(p) // 256] * 256 + int(p) % 256 for p in positions.cpu()],
                                device="cuda", dtype=torch.int32)
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)
    try:
        hidden_rows, logit_rows = [], []
        for i in range(count):
            set_context(False, slot_mapping=slot_mapping[i:i + 1], state_slots=slots,
                        block_tables=block_table,
                        context_lens=torch.tensor([cached + i + 1], device="cuda", dtype=torch.int32))
            hidden = model(ids[i:i + 1], positions[i:i + 1])
            hidden_rows.append(hidden)
            logit_rows.append(model.compute_logits(hidden))
        expected_hidden, expected_logits = torch.cat(hidden_rows), torch.cat(logit_rows)
        final = [x.clone() for x in buffers]

        def window(tokens):
            for target, source in zip(buffers, initial):
                target.copy_(source)
            set_context(True, slot_mapping=slot_mapping, state_slots=slots, block_tables=block_table,
                        cu_seqlens_q=torch.tensor([0, count], device="cuda", dtype=torch.int32))
            ModelRunner.prepare_verification_context(cached, count, ids.device)
            hidden = model(tokens, positions)
            return hidden, model.compute_logits(hidden, all_tokens=True)

        actual_hidden, actual_logits = window(ids)
        assert torch.equal(actual_hidden, expected_hidden)
        assert torch.equal(actual_logits, expected_logits)
        for actual, expected in zip(buffers, final):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        # Proposals are all stored in KV up front, but future rows remain hidden.
        future = ids.clone()
        future[1:] = (future[1:] + 1) % cfg.vocab_size
        changed_hidden, changed_logits = window(future)
        assert torch.equal(changed_hidden[:1], expected_hidden[:1])
        assert torch.equal(changed_logits[:1], expected_logits[:1])
        # Exercise the actual transaction, not just model forwards: all-accept,
        # rejection/replay and truncation to the remaining output budget.
        runner = ModelRunner.__new__(ModelRunner)
        runner.model, runner.S_cache, runner.conv_cache = model, states, convs
        runner.has_linear_state, runner.world_size, runner.enforce_eager = True, 1, True
        runner.block_size = 256
        runner.config = SimpleNamespace(hf_config=cfg, eos=-1, cache_schedule_metadata=True,
                                        max_model_len=1024, verification_cuda_graph=use_graph,
                                        kv_cache_dtype="fp8" if kv_dtype == torch.float8_e4m3fn else "auto")
        runner.target_hidden = torch.zeros(1024, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        seq = Sequence([0] * cached + [int(ids[0])], SamplingParams(greedy=True, ignore_eos=True))
        seq.num_cached_tokens, seq.state_slot, seq.state_ready = cached, 2, True
        seq.block_table = pages[:(cached + 2 + 255) // 256]
        correct = int(expected_logits[0].argmax())
        for proposal, budget in [(correct, 2), ((correct + 1) % cfg.vocab_size, 2), (correct, 1)]:
            def verify(mode):
                for target, source in zip(buffers, initial):
                    target.copy_(source)
                runner.target_hidden.zero_()
                result = runner.verify_greedy(seq, [proposal], budget, mode)
                snapshots = [x.clone() for x in buffers] + [runner.target_hidden.clone()]
                assert not get_context().decode_equivalent
                return result, snapshots
            (expected, _), serial_buffers = verify("serial")
            (actual, stats), batch_buffers = verify("batched_experimental")
            assert actual == expected
            assert stats["state_strategy"] == "intermediate"
            assert stats["replay_tokens"] == 0
            assert stats["target_forwards"] == 1
            for i in (0, 1):  # Every linear state, including untouched slots.
                assert torch.equal(serial_buffers[i], batch_buffers[i])
            valid = cached + len(actual)
            indices = torch.tensor([pages[p // 256] * 256 + p % 256 for p in range(valid)], device="cuda")
            for i in (2, 3):  # Logical valid KV/scales; speculative suffix is hidden.
                a = serial_buffers[i].flatten(2, 3).index_select(2, indices)
                b = batch_buffers[i].flatten(2, 3).index_select(2, indices)
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            assert torch.equal(serial_buffers[4][:valid], batch_buffers[4][:valid])
            if hasattr(runner, "verification_graph"):
                graph = runner.verification_graph
                logits = torch.cat([model.lm_head(row) for row in graph.hidden.split(1)])
                assert torch.equal(graph.logits, logits)
        assert hasattr(runner, "verification_graph") == (use_graph and kv_dtype == torch.bfloat16)
        # A failed forward must never publish partially advanced S/conv. A
        # partially failed commit must restore the original checkpoint too.
        original_norm = model.model.norm.forward
        def failed_forward(x):
            original_norm(x)
            assert torch.equal(states, initial[0])
            assert torch.equal(convs, initial[1])
            raise RuntimeError("injected after recurrent verification")
        original_commit = runner.verification_history.commit
        def failed_commit(states, convs, slot, count):
            states[slot].copy_(runner.verification_history.states[count - 1])
            raise RuntimeError("injected partial state commit")
        for kind in ("forward", "commit"):
            for target, source in zip(buffers, initial):
                target.copy_(source)
            if kind == "forward":
                # Model monkey-patching applies to eager execution; a captured
                # graph deliberately does not re-enter Python module forwards.
                runner.config.verification_cuda_graph = False
                model.model.norm.forward = failed_forward
            else:
                runner.config.verification_cuda_graph = use_graph
                runner.verification_history.commit = failed_commit
            try:
                with pytest.raises(RuntimeError, match="injected"):
                    runner.verify_greedy(seq, [correct], 2, "batched_experimental")
                assert torch.equal(states, initial[0])
                assert torch.equal(convs, initial[1])
                assert get_context().verification_state is None
            finally:
                model.model.norm.forward = original_norm
                runner.verification_history.commit = original_commit
    finally:
        reset_context()
    assert not get_context().decode_equivalent
