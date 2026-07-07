import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange
from typing import Optional

from model.config import PLMConfig


def rope(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    b, n, l, h = x.shape
    assert h % 2 == 0, "rope requires an even head size"
    device = x.device
    freq = 1.0 / (theta ** (torch.arange(0, h, 2, device=x.device).float() / h)) # (d / 2)
    pos = torch.arange(l, device=device, dtype=freq.dtype) # l
    angle = torch.outer(pos, freq) # (l, d / 2)
    rot = torch.polar(torch.ones_like(angle), angle)
    xc = torch.view_as_complex(x.float().reshape(b, n, l, h // 2, 2))
    x_final = torch.view_as_real(xc * rot).flatten(-2).type_as(x)
    return x_final


class Attention(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size # d
        self.head_size = config.head_size # h
        self.num_heads = self.hidden_size // self.head_size # n
        self.scale = 1.0 / math.sqrt(self.head_size)

        self.layernorm_Wqkv = nn.Sequential(
            nn.LayerNorm(self.hidden_size, bias=False),
            nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        )
        self.q_ln = nn.LayerNorm(self.hidden_size, bias=False)
        self.k_ln = nn.LayerNorm(self.hidden_size, bias=False)
        self.add_heads = partial(rearrange, pattern="b l (h d) -> b h l d", h=self.num_heads)
        self.remove_heads = partial(rearrange, pattern="b h l d -> b l (h d)")
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        QKV = self.layernorm_Wqkv(x) # (b, l, 3d)
        Q, K, V = torch.chunk(QKV, 3, dim=-1) # (b, l, d)
        Q, K = self.q_ln(Q).to(Q.dtype), self.k_ln(K).to(K.dtype)
        Q, K, V = map(self.add_heads, (Q, K, V)) # (b, n, l, h)
        Q, K = rope(Q), rope(K)
        A = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask, scale=self.scale
        ) # (b, n, l, h)
        A = self.remove_heads(A) # (b, l, d)
        O = self.Wo(A)
        return O


if __name__ == "__main__":
    ### Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, d = 8, 64, 512
    X = torch.rand((b, l, d)).to(device)

    config = PLMConfig(hidden_size=d)

    attention_layer = Attention(config=config).to(device)

    Y = attention_layer(X)
    print(Y.shape)
