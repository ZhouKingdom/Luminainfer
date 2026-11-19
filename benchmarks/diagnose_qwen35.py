"""Compare layer outputs and linear attention in isolation on HF hidden states."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zjhvllm import LLM, SamplingParams


@torch.inference_mode()
def main():
    path = '/home/data/jhzhou/huggingface/Qwen3.5-27B'
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={'': 'cuda:1'}, attn_implementation='flash_attention_2').eval()
    llm = LLM(path, enforce_eager=True, max_num_seqs=1, max_model_len=1024, max_num_batched_tokens=1024, gpu_memory_utilization=.9)
    ids = AutoTokenizer.from_pretrained(path).encode('The quick brown fox jumps over the lazy dog. ' * 10)[:65]
    ref = hf(torch.tensor([ids], device='cuda:1'), output_hidden_states=True, use_cache=False, logits_to_keep=1)
    rows = []
    handles = []
    def hook(index):
        def capture(module, inputs, output):
            expected = ref.hidden_states[index+1][0].to('cuda:0')
            if index == 63:
                return  # HF applies final norm to its last hidden state.
            diff = output.float() - expected.float()
            row = {'layer': index, 'max_abs': diff.abs().max().item(), 'rmse': diff.square().mean().sqrt().item()}
            rows.append(row)
            print(row, flush=True)
        return capture
    for i, layer in enumerate(llm.model_runner.model.model.layers):
        handles.append(layer.register_forward_hook(hook(i)))
    llm.generate([ids], SamplingParams(max_tokens=1), use_tqdm=False)
    for handle in handles:
        handle.remove()
    isolated = []
    for i in [0, 1, 2, 4, 16, 32, 60]:
        ours = llm.model_runner.model.model.layers[i]
        reference = hf.model.layers[i]
        x = reference.input_layernorm(ref.hidden_states[i])
        expected = reference.linear_attn(x)
        x = x[0].to('cuda:0')
        module = ours.linear_attn
        actual, _, _ = module._recurrence(x, torch.zeros_like(module.S_cache[0]), torch.zeros_like(module.conv_cache[0]))
        diff = actual.float() - expected[0].to('cuda:0').float()
        row = {'layer': i, 'max_abs': diff.abs().max().item(), 'rmse': diff.square().mean().sqrt().item()}
        isolated.append(row)
        print('isolated', row, flush=True)
    # Establish the reference model's own bf16 sensitivity to GEMM batch shape.
    doubled = hf(torch.tensor([ids, ids], device='cuda:1'), use_cache=False, logits_to_keep=1)
    intrinsic = (doubled.logits[0, -1].float() - ref.logits[0, -1].float()).abs()
    print('HF single vs duplicated batch', {'max_abs': intrinsic.max().item(), 'rmse': intrinsic.square().mean().sqrt().item()}, flush=True)
    from zjhvllm.utils.context import set_context, reset_context
    pos = torch.arange(len(ids), device='cuda:1')[None, None].expand(3, 1, -1)
    embeddings = hf.model.rotary_emb(ref.hidden_states[0], pos)
    full = []
    for i in [3, 7, 31, 63]:
        ours = llm.model_runner.model.model.layers[i].self_attn
        reference = hf.model.layers[i]
        x = reference.input_layernorm(ref.hidden_states[i])
        expected = reference.self_attn(x, position_embeddings=embeddings, attention_mask=None)[0]
        old_k, old_v = ours.attn.k_cache, ours.attn.v_cache
        ours.attn.k_cache = ours.attn.v_cache = torch.tensor([], device='cuda:0')
        cu = torch.tensor([0, len(ids)], dtype=torch.int32, device='cuda:0')
        set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=len(ids), max_seqlen_k=len(ids))
        actual = ours(torch.arange(len(ids), device='cuda:0'), x[0].to('cuda:0'))
        reset_context()
        ours.attn.k_cache, ours.attn.v_cache = old_k, old_v
        diff = actual.float() - expected[0].to('cuda:0').float()
        row = {'layer': i, 'max_abs': diff.abs().max().item(), 'rmse': diff.square().mean().sqrt().item()}
        full.append(row)
        print('isolated full', row, flush=True)
    with open('benchmarks/qwen35_layer_diagnostics.json', 'w') as f:
        json.dump({'layers': rows, 'isolated_linear': isolated, 'isolated_full': full}, f, indent=2)
    llm.exit()


if __name__ == '__main__':
    main()
