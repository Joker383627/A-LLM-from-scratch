
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.dim_head = emb_dim // num_heads
        self.num_heads = num_heads
        self.QKV_param = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, is_causal):
        B, T, D = x.shape
        QKV = self.QKV_param(x).reshape(B, T, 3, self.num_heads, self.dim_head)
        Q, K, V = QKV.unbind(dim=2)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, is_causal):
        x = x + self.dropout1(self.attention(self.layernorm1(x), is_causal))
        x = x + self.dropout2(self.ffn(self.layernorm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, num_layers,
                 is_causal=True, context_length=768, vocab_size=50257):
        super().__init__()
        self.is_causal = is_causal
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.transformer_stack = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_layernorm = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        position = torch.arange(T, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(position)

        for block in self.transformer_stack:
            x = block(x, self.is_causal)

        logits = self.lm_head(self.final_layernorm(x))

        if targets is not None:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits