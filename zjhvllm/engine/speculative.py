"""Experimental greedy prompt lookup and transactional hybrid-state verification.

This helper module contains no MTP weights or stochastic rejection sampling;
the engine supplies checkpoint MTP drafts separately. Target window verification
uses ordinary decode arithmetic within operators to preserve greedy decisions.
"""
from dataclasses import dataclass

import torch

from zjhvllm.utils.context import get_context


def prompt_lookup(tokens: list[int], max_tokens: int, ngram_size: int = 3) -> list[int]:
    if max_tokens < 0 or ngram_size < 1:
        raise ValueError("invalid prompt lookup bounds")
    if not max_tokens or len(tokens) <= ngram_size:
        return []
    suffix = tokens[-ngram_size:]
    for start in range(len(tokens) - ngram_size - 1, -1, -1):
        if tokens[start:start + ngram_size] == suffix:
            return tokens[start + ngram_size:start + ngram_size + max_tokens]
    return []


def greedy_prefix(draft: list[int], target: list[int], max_outputs: int, eos: int | None):
    """Return emitted tokens and accepted proposals; target includes a bonus row."""
    if len(target) != len(draft) + 1 or max_outputs < 1:
        raise ValueError("verification requires K+1 predictions and a positive output budget")
    output, accepted = [], 0
    for index, token in enumerate(target):
        output.append(token)
        matches = index < len(draft) and draft[index] == token
        accepted += int(matches)
        if not matches or token == eos or len(output) == max_outputs:
            break
    return output, accepted


@dataclass
class HybridCheckpoint:
    """Snapshot one request only, including EVERY linear layer and conv history."""
    slot: int
    state: torch.Tensor
    conv: torch.Tensor

    @classmethod
    def capture(cls, states, convs, slot):
        if not 0 <= slot < states.shape[0] or states.shape[0] != convs.shape[0]:
            raise ValueError("invalid hybrid state slot")
        return cls(slot, states[slot].clone(), convs[slot].clone())

    def restore(self, states, convs):
        states[self.slot].copy_(self.state)
        convs[self.slot].copy_(self.conv)

    @property
    def nbytes(self):
        return self.state.numel() * self.state.element_size() + self.conv.numel() * self.conv.element_size()


class HybridStateHistory:
    """Bounded scratch containing each verification input's post-update state.

    Kernels read the persistent state (first input) or previous scratch step,
    and write directly to the next scratch step. Persistent S/conv are not
    advanced until commit. The caller retains its checkpoint for commit faults.
    """

    def __init__(self, states, convs, capacity):
        self.states = torch.empty((capacity, *states.shape[1:]), device=states.device, dtype=states.dtype)
        self.convs = torch.empty((capacity, *convs.shape[1:]), device=convs.device, dtype=convs.dtype)
        self.indices = torch.arange(capacity, device=states.device, dtype=torch.int32)
        self.layer_indices = {}
        self.written = set()

    def bind(self, layers, count, cu_seqlens=None):
        if not 1 <= count <= len(self.states):
            raise ValueError("verification history capacity exceeded")
        if len(layers) != self.states.shape[1]:
            raise ValueError("verification history layer mismatch")
        self.layer_indices = {id(layer): i for i, layer in enumerate(layers)}
        self.count = count
        self.cu_seqlens = [0, count] if cu_seqlens is None else cu_seqlens
        if self.cu_seqlens[0] != 0 or self.cu_seqlens[-1] != count:
            raise ValueError("invalid verification history sequence boundaries")
        self.written.clear()
        return self

    def forward_layer(self, layer, hidden, slots):
        index = self.layer_indices[id(layer)]
        if len(hidden) != self.count or index in self.written:
            raise ValueError("invalid verification history traversal")
        if (get_context().verification_fused_gdn and hidden.is_cuda
                and max(b-a for a,b in zip(self.cu_seqlens[:-1],self.cu_seqlens[1:])) <= 9):
            output = layer._forward_verify_fused(hidden, slots, self, index)
            self.written.add(index)
            return output
        state_out, conv_out = self.states[:, index], self.convs[:, index]
        packed = (getattr(layer.in_proj_qkv, "backend", "decode_equivalent") == "batch_invariant"
                  or get_context().decode_batched)
        projections = layer._project_decode(hidden) if packed else None
        output = []
        for request, (start, end) in enumerate(zip(self.cu_seqlens[:-1], self.cu_seqlens[1:])):
            for step in range(start, end):
                first = step == start
                output.append(layer._forward_decode(
                    hidden[step:step + 1], slots[request:request + 1] if first else self.indices[step - 1:step],
                    state_cache=layer.S_cache if first else state_out,
                    conv_cache=layer.conv_cache if first else conv_out,
                    output_state=state_out, output_conv=conv_out,
                    output_slots=self.indices[step:step + 1],
                    projections=tuple(x[step:step + 1] for x in projections) if packed else None,
                    project_output=not packed))
        self.written.add(index)
        result = torch.cat(output, dim=0)
        return layer.out_proj(result) if packed else result

    def commit(self, states, convs, slot, count):
        self.commit_many(states, convs, [slot], [count])

    def commit_many(self, states, convs, slots, counts):
        if len(slots) != len(counts) or len(counts) != len(self.cu_seqlens) - 1:
            raise ValueError("verification commit request mismatch")
        if len(set(slots)) != len(slots) or len(self.written) != self.states.shape[1]:
            raise ValueError("incomplete verification history cannot commit")
        for i, count in enumerate(counts):
            if not 1 <= count <= self.cu_seqlens[i + 1] - self.cu_seqlens[i]:
                raise ValueError("invalid per-request state commit length")
        for i, (slot, count) in enumerate(zip(slots, counts)):
            index = self.cu_seqlens[i] + count - 1
            states[slot].copy_(self.states[index])
            convs[slot].copy_(self.convs[index])

    @property
    def nbytes(self):
        return self.states.numel() * self.states.element_size() + self.convs.numel() * self.convs.element_size()
