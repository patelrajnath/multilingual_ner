import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    This is basic transformer model
    """

    def __init__(self, emb_dim, heads, mask_future_steps=False, multihead_shared_emb=False):
        super().__init__()
        self.emb_dim, self.heads, self.mask_future_steps = emb_dim, heads, mask_future_steps
        if multihead_shared_emb:
            self.dim_per_head = self.emb_dim // self.heads
        else:
            self.dim_per_head = self.emb_dim

        self.toqueries = nn.Linear(self.emb_dim, self.dim_per_head * heads)
        self.tovalue = nn.Linear(self.emb_dim, self.dim_per_head * heads)
        self.tokey = nn.Linear(self.emb_dim, self.dim_per_head * heads)
        self.unifyheads = nn.Linear(self.dim_per_head * heads, self.emb_dim)

    def forward(self, tensor, mask=None, kv=None):
        bs, qlen, dim = tensor.size()
        if kv is not None:
            kv = kv
        else:
            kv = tensor

        heads = self.heads
        kv_bs, kv_qlen, kv_dim = kv.size()

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        query = self.toqueries(tensor).view(bs, qlen, heads, self.dim_per_head).transpose(1, 2)
        key = self.tokey(kv).view(kv_bs, kv_qlen, heads, self.dim_per_head).transpose(1, 2)
        value = self.tovalue(kv).view(kv_bs, kv_qlen, heads, self.dim_per_head).transpose(1, 2)

        query = query / (self.dim_per_head ** (1 / 4))
        key = key / (self.dim_per_head ** (1 / 4))

        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        dot = F.softmax(scores, dim=-1)
        out = torch.matmul(dot, value)
        out = out.transpose(1, 2).contiguous().view(bs, qlen, heads * self.dim_per_head)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, ff=4, dropout=0.1, multihead_shared_emb=False):
        super().__init__()

        self.attention = SelfAttention(emb_dim, heads=heads, multihead_shared_emb=multihead_shared_emb)

        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff * emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff * emb_dim, emb_dim))

        self.do = nn.Dropout(dropout)

    def forward(self, tensor, mask=None):
        # Add and layer normalize: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.attention(self.norm1(tensor), mask))

        # Add and layer normalize: Normalize + Layer + Dropout
        tensor = tensor + self.do(self.ff(self.norm2(tensor)))

        return tensor
