"""Version-1 P→D handoff: dense, unsharded, native-dtype KV, host staging.

This is a correctness transport, not a high-performance GPU connector. The
source retains ownership until the receiver acknowledges successful import.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import torch

from zjhvllm.sampling_params import SamplingParams


def check_supported(runner):
    if runner.has_linear_state:
        raise NotImplementedError('P-D v1 does not transfer recurrent/conv states')
    if runner.world_size != 1 or runner.tp_size != 1:
        raise NotImplementedError('P-D v1 requires unsharded single-GPU workers')
    if runner.config.kv_cache_dtype != 'auto':
        raise NotImplementedError('P-D v1 requires native KV dtype; FP8 scales are not transferred')


def model_signature(runner):
    check_supported(runner)
    # Same canonical checkpoint on one host. Different pool capacities and
    # physical block IDs are allowed; weights/layout/dtype must match.
    content={'model':str(Path(runner.config.model).resolve()),
             'config':runner.config.hf_config.to_dict(),
             'kv_shape':[runner.kv_cache.shape[i] for i in (0,1,3,4,5)],
             'kv_dtype':str(runner.kv_cache.dtype)}
    return hashlib.sha256(json.dumps(content,sort_keys=True,default=str).encode()).hexdigest()


@dataclass
class PrefillHandoff:
    transfer_id: str
    signature: str
    prompt_token_ids: list[int]
    first_token: int
    sampling: SamplingParams
    last_ran_seed: int | None
    arrival_time: float
    first_token_time: float
    kv: torch.Tensor  # [2, layers, valid prompt tokens, heads, head_dim], CPU
    version: int = 1

    @property
    def nbytes(self):
        return self.kv.numel()*self.kv.element_size()

    def validate(self, runner):
        check_supported(runner)
        if self.version!=1 or self.signature!=model_signature(runner):
            raise ValueError('incompatible handoff version/checkpoint/layout')
        n=len(self.prompt_token_ids)
        expected=(2,runner.kv_cache.shape[1],n,*runner.kv_cache.shape[4:])
        if not n or self.sampling.max_tokens<2 or n+self.sampling.max_tokens>runner.config.max_model_len:
            raise ValueError('invalid handoff token counts or destination context limit')
        if tuple(self.kv.shape)!=expected or self.kv.dtype!=runner.kv_cache.dtype:
            raise ValueError('invalid handoff KV shape/dtype')
        if self.kv.device.type!='cpu' or not self.kv.is_contiguous():
            raise ValueError('P-D v1 requires contiguous CPU payload')


def token_slots(block_table, count, block_size, device):
    if len(block_table)*block_size<count:
        raise ValueError('block table does not cover cached tokens')
    slots=[block_table[i//block_size]*block_size+i%block_size for i in range(count)]
    return torch.tensor(slots,dtype=torch.long,device=device)


@torch.inference_mode()
def capture_prefill(runner, seq):
    check_supported(runner)
    if seq.num_completion_tokens!=1 or seq.num_cached_tokens!=seq.num_prompt_tokens:
        raise ValueError('export requires completed prefill and exactly one pending output token')
    if seq.max_tokens<2 or seq.first_token_time is None:
        raise ValueError('export requires an unfinished generation')
    fields=SamplingParams.__dataclass_fields__
    sampling=SamplingParams(**{name:getattr(seq,name) for name in fields})
    slots=token_slots(seq.block_table,seq.num_cached_tokens,runner.block_size,runner.kv_cache.device)
    # Export valid tokens only, never the uninitialized tail of the last page.
    packed=runner.kv_cache.flatten(2,3).index_select(2,slots).contiguous().cpu()
    return PrefillHandoff(uuid4().hex,model_signature(runner),list(seq.prompt_token_ids),seq.last_token,
        sampling,seq.last_ran_seed,seq.arrival_time,seq.first_token_time,packed)


@torch.inference_mode()
def restore_prefill(runner, seq, packet):
    packet.validate(runner)
    slots=token_slots(seq.block_table,len(packet.prompt_token_ids),runner.block_size,runner.kv_cache.device)
    runner.kv_cache.flatten(2,3).index_copy_(2,slots,packet.kv.to(runner.kv_cache.device))
    if runner.kv_cache.is_cuda:
        torch.cuda.synchronize(runner.kv_cache.device)
