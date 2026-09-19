
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VOCAB_SIZE,CONTEXT_LENGTH

class Transformer(nn.Module):

    def __init__(self,emb_dim,num_heads,dropout,num_layers,causal_mask = True):
        super().__init__()

        mask = torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH, dtype=torch.bool)
        if causal_mask:
            self.register_buffer("mask",torch.tril(mask))
        else:
            self.register_buffer("mask", mask)

        self.token_embedding = nn.Embedding(VOCAB_SIZE,emb_dim,max_norm=1.0)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH,emb_dim,max_norm = 1.0)

        self.transformer_stack = nn.ModuleList([
            TransformerBlock(emb_dim,num_heads,dropout) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim,2*emb_dim),
            nn.GELU(),
            nn.Linear(2*emb_dim,VOCAB_SIZE)
        )

        self.final_layernorm = nn.LayerNorm(emb_dim)


    def forward(self,x):
        B,T = x.shape
        mask = self.mask[:T,:T]
        position = torch.arange(0,T,device=x.device)

        x = self.token_embedding(x) + self.position_embedding(position)  #shape ---> (B,T,D)

        for block in self.transformer_stack:
            x,last_attn_score = block(x,mask)

        x = self.final_layernorm(x)
        logits = self.mlp_head(x)

        return logits,last_attn_score 


class MultiHeadAttention(nn.Module):

    def __init__(self,emb_dim,num_heads):
        super().__init__()
        assert emb_dim%num_heads == 0

        self.dim_head = emb_dim//num_heads
        self.num_heads = num_heads
        self.QKV_param = nn.Linear(emb_dim,3*emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self,x,mask):
        B,T,D = x.shape

        QKV = self.QKV_param(x)
        QKV = QKV.reshape(B, T, 3, self.num_heads, self.dim_head)
        Q, K, V = QKV.unbind(dim=2)  # Each: (B, T, num_heads, dim_head) 

        out, attn_score = self.scaled_attention(Q,K,V,mask)

        out = self.out_proj(out.permute(0, 2, 1, 3).reshape(B,T,D))

        return out,attn_score

    def scaled_attention(self,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask = None):
        dim_head = Q.size(-1)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attn_score_mat = torch.matmul(Q,K.transpose(-2,-1))/(dim_head**0.5)
        if mask is not None:
            A = attn_score_mat.masked_fill(~mask,float("-inf"))
        else:
            A = attn_score_mat

        A = F.softmax(A,dim=-1)

        return A@V,A

class TransformerBlock(nn.Module):
    def __init__(self,emb_dim,num_heads,dropout):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadAttention(emb_dim,num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim,4*emb_dim),
            nn.GELU(),
            nn.Linear(4*emb_dim,2*emb_dim),
            nn.GELU(),
            nn.Linear(2*emb_dim,emb_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,mask):
        residual = x

        x,attn_score = self.attention(self.layernorm1(x),mask)
        x = self.dropout1(x) + residual

        x = self.dropout2(self.ffn(self.layernorm2(x))) + x

        return x,attn_score