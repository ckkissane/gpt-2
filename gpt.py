from dataclasses import dataclass
import math
import torch as t
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torch import einsum
from torchtyping import TensorType
from typing import Optional


class UnidirectionalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.n_heads = num_heads

    def forward(
        self,
        x: t.Tensor,
        past_key_values: Optional[t.Tensor] = None,
        return_key_values=False,
    ):
        batch, seq_len = x.shape[:2]
        q, k, v = t.split(self.qkv_proj(x), self.hidden_size, dim=-1)
        q = rearrange(q, "b n (h l) -> b h n l", l=self.head_size)
        k = rearrange(k, "b n (h l) -> b h n l", l=self.head_size)
        v = rearrange(v, "b n (h l) -> b h n l", l=self.head_size)
        new_k, new_v = k, v

        if past_key_values is not None:
            assert x.shape == (1, 1, self.hidden_size)
            past_k, past_v = t.split(
                past_key_values.unsqueeze(0), self.head_size, dim=-1
            )
            k = t.cat([past_k, k], dim=2)
            v = t.cat([past_v, v], dim=2)
            attn_scores = einsum("bhql, bhkl -> bhqk", q, k) / math.sqrt(self.head_size)
        else:
            neg_inf = t.tensor(-1e4).to(x.device)
            q_ind = t.arange(seq_len).unsqueeze(1)
            k_ind = t.arange(seq_len).unsqueeze(0)
            mask = (q_ind < k_ind).to(x.device)
            attn_scores = einsum("bhql, bhkl -> bhqk", q, k) / math.sqrt(self.head_size)
            attn_scores = t.where(mask, neg_inf, attn_scores)

        softmaxed_attn = attn_scores.softmax(dim=-1)
        combined_v = einsum("bhqk, bhkl -> bhql", softmaxed_attn, v)
        combined_v = rearrange(combined_v, "b h q l -> b q (h l)")
        out = self.output_proj(combined_v)
        if return_key_values:
            return out, t.cat([new_k, new_v], dim=-1)
        return out


class GPT2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_norm_epsilon: float,
        dropout: float,
        num_heads: int,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UnidirectionalAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor, past_key_values=None, return_key_values=False):
        if return_key_values:
            attn_output, new_key_values = self.attn(
                self.ln1(x),
                past_key_values=past_key_values,
                return_key_values=return_key_values,
            )
            x = x + attn_output
            x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.ln2(x)))))
            return x, new_key_values
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.ln2(x)))))
            return x


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]


class GPT2(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        dropout,
        layer_norm_epsilon,
        use_cache=False,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                GPT2Block(hidden_size, layer_norm_epsilon, dropout, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.use_cache = use_cache
        head_size = hidden_size // num_heads
        self.cache_size = (num_layers, num_heads, 0, 2 * head_size)
        self.clear_cache()

    def clear_cache(self):
        self._cache_kv = t.zeros(self.cache_size).to(self.ln.weight.device)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        pos = t.arange(seq_len).to(input_ids.device)

        if not self.use_cache:
            enc = self.dropout(
                self.token_embedding(input_ids) + self.pos_embedding(pos)
            )
            enc = self.blocks(enc)
        elif self._cache_kv.shape[2] == 0:  # cache is empty
            assert input_ids.shape[0] == 1
            enc = self.dropout(
                self.token_embedding(input_ids) + self.pos_embedding(pos)
            )
            new_key_values = []
            for i, block in enumerate(self.blocks):
                enc, new_kv = block(enc, return_key_values=True)
                new_key_values.append(new_kv)
            self._cache_kv = t.cat(new_key_values, dim=0)
        else:  # utlize past keys / values in cache
            assert input_ids.shape[0] == 1
            enc = self.dropout(
                self.token_embedding(input_ids[:, -1:]) + self.pos_embedding(pos[-1:])
            )
            new_key_values = []
            for i, block in enumerate(self.blocks):
                enc, new_kv = block(
                    enc, past_key_values=self._cache_kv[i], return_key_values=True
                )
                new_key_values.append(new_kv)
            last_token_cache = t.cat(new_key_values, dim=0)
            self._cache_kv = t.cat([self._cache_kv, last_token_cache], dim=-2)

        enc = self.ln(enc)
        logits = einsum("bnl, vl -> bnv", enc, self.token_embedding.weight)
        return GPT2Output(logits=logits[:, -1, :], final_encoding=enc[:, -1, :])
