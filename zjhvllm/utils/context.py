from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    state_slots: torch.Tensor | None = None
    cu_seqlens_q_cpu: list[int] | None = None
    state_slots_cpu: list[int] | None = None
    # Verification-only: retain ordinary single-request decode arithmetic while
    # traversing the target once for a pending token plus proposal window.
    decode_equivalent: bool = False
    # When True, the decode-equivalent path batches rows instead of splitting
    # each token into its own GEMM/norm. Accepts BF16 rounding differences from
    # single-token decode in exchange for far fewer, larger kernels (the
    # qwen35_linear_backend="batched" opt-in). Recurrent state routing is unchanged.
    decode_batched: bool = False
    verification_context_lens: torch.Tensor | None = None
    verification_state: object | None = None
    verification_fused_gdn: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, state_slots=None, cu_seqlens_q_cpu=None, state_slots_cpu=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, state_slots, cu_seqlens_q_cpu, state_slots_cpu)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
