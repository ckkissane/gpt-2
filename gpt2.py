from dataclasses import dataclass
import math
import torch as t
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torch import einsum
from torchtyping import TensorType
import transformers


class UnidirectionalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.n_heads = num_heads

    def forward(self, x: t.Tensor):
        batch, seq_len = x.shape[:2]
        q, k, v = t.split(self.qkv_proj(x), self.hidden_size, dim=-1)
        q = rearrange(q, "b n (h l) -> b h n l", l=self.head_size)
        k = rearrange(k, "b n (h l) -> b h n l", l=self.head_size)
        v = rearrange(v, "b n (h l) -> b h n l", l=self.head_size)

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

    def forward(self, x: t.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.ln2(x)))))
        return x


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "seq_length", "vocab_size"]
    final_encoding: TensorType["batch_size", "seq_length", "hidden_size"]


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

        self.block_size = max_position_embeddings

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        pos = t.arange(seq_len).to(input_ids.device)
        enc = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))
        enc = self.blocks(enc)
        enc = self.ln(enc)
        logits = einsum("bnl, vl -> bnv", enc, self.token_embedding.weight)
        return GPT2Output(logits=logits, final_encoding=enc)


def _copy_weight_bias(mine, theirs, transpose=False):
    if transpose:
        mine.weight.copy_(theirs.weight.T)
    else:
        mine.weight.copy_(theirs.weight)
    if mine.bias is not None:
        mine.bias.copy_(theirs.bias)


def get_pretrained_gpt():
    pretrained_gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    config = dict(
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        hidden_size=768,
        max_position_embeddings=1024,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
    )
    my_gpt = GPT2(**config)
    for p in my_gpt.parameters():
        p.requires_grad = False

    my_gpt.token_embedding.weight.copy_(pretrained_gpt.transformer.wte.weight)
    my_gpt.pos_embedding.weight.copy_(pretrained_gpt.transformer.wpe.weight)
    _copy_weight_bias(my_gpt.ln, pretrained_gpt.transformer.ln_f)

    for my_block, hf_block in zip(my_gpt.blocks, pretrained_gpt.transformer.h):
        _copy_weight_bias(my_block.ln1, hf_block.ln_1)
        _copy_weight_bias(my_block.attn.qkv_proj, hf_block.attn.c_attn, transpose=True)
        _copy_weight_bias(
            my_block.attn.output_proj, hf_block.attn.c_proj, transpose=True
        )
        _copy_weight_bias(my_block.ln2, hf_block.ln_2)
        _copy_weight_bias(my_block.linear1, hf_block.mlp.c_fc, transpose=True)
        _copy_weight_bias(my_block.linear2, hf_block.mlp.c_proj, transpose=True)
    return my_gpt


def load_weights(GPT2Class):
    pretrained_gpt = get_pretrained_gpt()
    my_gpt = GPT2Class(
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        hidden_size=768,
        max_position_embeddings=1024,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
    )

    state_dict = {
        mykey: v
        for (k, v), mykey in zip(
            pretrained_gpt.state_dict().items(), my_gpt.state_dict().keys()
        )
    }
    my_gpt.load_state_dict(state_dict)
    return my_gpt
