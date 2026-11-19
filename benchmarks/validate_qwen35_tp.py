"""Distributed small hybrid model vs unsharded FP32 decode reference."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from zjhvllm.models.qwen3_5 import Qwen3_5ForCausalLM
from zjhvllm.utils.tp import set_tp
from zjhvllm.utils.context import set_context, reset_context


def worker(rank, world, port, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,
                            world_size=world,timeout=timedelta(minutes=3))
    try:
        torch.manual_seed(918)
        torch.backends.cuda.matmul.allow_tf32=False
        cfg=SimpleNamespace(hidden_size=256,intermediate_size=512,hidden_act='silu',
            num_hidden_layers=2,layer_types=['linear_attention','full_attention'],
            linear_num_key_heads=8,linear_num_value_heads=24,linear_key_head_dim=128,
            linear_value_head_dim=128,linear_conv_kernel_dim=4,rms_norm_eps=1e-6,
            num_attention_heads=8,num_key_value_heads=2,head_dim=64,max_position_embeddings=512,
            attention_bias=True,rope_parameters={'rope_theta':10000000,'partial_rotary_factor':.25},
            vocab_size=128,tie_word_embeddings=False)
        set_tp(1,0)
        full=Qwen3_5ForCausalLM(cfg).cuda().eval()
        set_tp(world,rank)
        local=Qwen3_5ForCausalLM(cfg).cuda().eval()
        source=dict(full.named_parameters())
        with torch.no_grad():
            for name,p in local.named_parameters():
                getattr(p,'weight_loader',lambda p,w:p.copy_(w))(p,source[name])
        def buffers(model):
            linear=model.model.layers[0].linear_attn
            s=torch.zeros(2,linear.num_v_heads,128,128,device='cuda')
            c=torch.zeros(2,linear.conv_dim,3,device='cuda')
            linear.S_cache,linear.conv_cache=s,c
            attention=model.model.layers[1].self_attn
            kv=torch.zeros(2,2,256,attention.num_kv_heads,64,device='cuda')
            attention.attn.k_cache,attention.attn.v_cache=kv[0],kv[1]
            return s,c,kv
        full_buffers=buffers(full);local_buffers=buffers(local)
        errors=[]
        with torch.inference_mode():
            for step in range(4):
                ids=torch.tensor([7+step,23-step],device='cuda')
                positions=torch.tensor([step,step],device='cuda')
                results=[]
                for model in (full,local):
                    set_context(False,state_slots=torch.tensor([1,0],device='cuda',dtype=torch.int32),
                        slot_mapping=torch.tensor([256+step,step],device='cuda',dtype=torch.int32),
                        block_tables=torch.tensor([[1],[0]],device='cuda',dtype=torch.int32),
                        context_lens=torch.tensor([step+1,step+1],device='cuda',dtype=torch.int32))
                    hidden=model(ids,positions)
                    results.append(model.compute_logits(hidden))
                    reset_context()
                delta=results[0]-results[1]
                errors.append({'max_abs':delta.abs().max().item(),'rmse':delta.square().mean().sqrt().item()})
                torch.testing.assert_close(results[0],results[1],rtol=1e-3,atol=1e-3)
            expected=full_buffers[0].chunk(world,dim=1)[rank]
            torch.testing.assert_close(local_buffers[0],expected,rtol=1e-3,atol=1e-3)
        result={'rank':rank,'errors':errors,'state_max_abs':(local_buffers[0]-expected).abs().max().item(),
            'local_parameters':sum(p.numel() for p in local.parameters()),'passed':True}
        gathered=[None]*world
        dist.all_gather_object(gathered,result)
        if rank==0:Path(output).write_text(json.dumps({'tp':world,'ranks':gathered,'passed':True},indent=2)+'\n')
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--port',type=int,default=2400)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    mp.spawn(worker,args=(args.tp,args.port,args.output),nprocs=args.tp,join=True)
