"""Qwen3.5 checkpoint MTP-1, sharing target embeddings and LM head.

Architecture checked against SGLang's qwen3_5_mtp.py: embedding-first
concatenation of normalized shifted token embedding and post-norm target
hidden state, projection, one full-attention decoder layer, final norm.
This independent implementation uses PyTorch SDPA for the private draft KV.
"""
from copy import deepcopy
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from zjhvllm.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm


class MTPAttention(nn.Module):
    """Request-private append-only KV; length is the publication boundary."""

    def __init__(self, capacity, heads, dim, dtype, slots=1):
        super().__init__()
        self.register_buffer("_keys", torch.empty(slots, capacity, heads, dim, dtype=dtype), persistent=False)
        self.register_buffer("_values", torch.empty_like(self._keys), persistent=False)
        self.lengths = [0] * slots
        self.active_slot = 0
        self.batch_layout = None

    @property
    def keys(self):
        return self._keys[self.active_slot]

    @property
    def values(self):
        return self._values[self.active_slot]

    @property
    def length(self):
        return self.lengths[self.active_slot]

    @length.setter
    def length(self, value):
        self.lengths[self.active_slot] = value

    def get_length(self, slot):
        return self.lengths[slot]

    def set_length(self, slot, value):
        self.lengths[slot] = value

    @contextmanager
    def for_slot(self, slot):
        if not 0 <= slot < len(self.lengths):
            raise ValueError("invalid MTP request slot")
        previous = self.active_slot
        self.active_slot = slot
        try:
            yield
        finally:
            self.active_slot = previous

    def forward(self, query, key, value):
        if self.batch_layout is not None:
            output, offset = [], 0
            for slot, size in self.batch_layout:
                with self.for_slot(slot):
                    output.append(self._forward_single(query[offset:offset + size],
                                                       key[offset:offset + size], value[offset:offset + size]))
                offset += size
            if offset != len(query):
                raise ValueError("MTP packed attention row mismatch")
            return torch.cat(output, dim=0)
        return self._forward_single(query, key, value)

    @contextmanager
    def packed(self, slots, sizes):
        if (not slots or len(slots) != len(sizes) or len(set(slots)) != len(slots)
                or self.batch_layout is not None):
            raise ValueError("invalid MTP packed request layout")
        for slot, size in zip(slots, sizes):
            if not 0 <= slot < len(self.lengths) or size < 1:
                raise ValueError("invalid MTP packed slot or row count")
            if self.lengths[slot] + size > self._keys.shape[1]:
                raise ValueError("MTP KV capacity exceeded")
        self.batch_layout = list(zip(slots, sizes))
        try:
            yield
        finally:
            self.batch_layout = None

    def _forward_single(self, query, key, value):
        start, end = self.length, self.length + len(query)
        if end > len(self.keys):
            raise ValueError("MTP KV capacity exceeded")
        self.keys[start:end].copy_(key)
        self.values[start:end].copy_(value)
        q = query.transpose(0, 1).unsqueeze(0)
        k = self.keys[:end].transpose(0, 1).unsqueeze(0)
        v = self.values[:end].transpose(0, 1).unsqueeze(0)
        if start:
            mask = torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        self.length = end
        return output.squeeze(0).transpose(0, 1).contiguous()


class Qwen3_5MTP(nn.Module):
    def __init__(self, config, capacity, slots=1):
        super().__init__()
        if getattr(config, "mtp_num_hidden_layers", 0) != 1:
            raise NotImplementedError("only one-layer Qwen3.5 MTP is supported")
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            raise NotImplementedError("dedicated MTP embeddings are unsupported")
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        layer_config = deepcopy(config)
        layer_config.layer_types = ["full_attention"]
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(layer_config, 0)])
        self.layers[0].self_attn.attn = MTPAttention(
            capacity, self.layers[0].self_attn.num_kv_heads, config.head_dim, config.dtype, slots=slots)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    @property
    def cache(self):
        return self.layers[0].self_attn.attn

    def forward(self, token_embeddings, target_hidden):
        if token_embeddings.shape != target_hidden.shape or not len(target_hidden):
            raise ValueError("MTP requires nonempty aligned token/hidden pairs")
        start = self.cache.length
        positions = torch.arange(start, start + len(target_hidden), device=target_hidden.device)
        return self._forward_pairs(token_embeddings, target_hidden, positions)

    def forward_batch(self, token_embeddings, target_hidden, slots, sizes):
        """Pack projections across requests; attention retains private KV/masks.

        Draft arithmetic may round differently from individual GEMMs. The target
        verifier still makes every acceptance decision with its own backend.
        """
        if (token_embeddings.shape != target_hidden.shape
                or len(target_hidden) != sum(sizes)):
            raise ValueError("MTP packed token/hidden rows are not aligned")
        with self.cache.packed(slots, sizes):
            starts = {slot: self.cache.get_length(slot) for slot in slots}
            positions = torch.cat([torch.arange(starts[slot], starts[slot] + size,
                                               device=target_hidden.device)
                                   for slot, size in zip(slots, sizes)])
            try:
                return self._forward_pairs(token_embeddings, target_hidden, positions)
            except BaseException:
                for slot, start in starts.items():
                    self.cache.set_length(slot, start)
                raise

    def _forward_pairs(self, token_embeddings, target_hidden, positions):
        hidden = self.fc(torch.cat([self.pre_fc_norm_embedding(token_embeddings),
                                    self.pre_fc_norm_hidden(target_hidden)], dim=-1))
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return self.norm(hidden)
