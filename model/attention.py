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

    x_float = x.float()
    freq = 1.0 / (theta ** (torch.arange(0, h, 2, device=x.device).float() / h))
    pos = torch.arange(l, device=x.device, dtype=freq.dtype)
    angle = torch.outer(pos, freq)
    cos = angle.cos().view(1, 1, l, h // 2)
    sin = angle.sin().view(1, 1, l, h // 2)

    x_pair = x_float.reshape(b, n, l, h // 2, 2)
    x_even, x_odd = x_pair.unbind(dim=-1)
    x_rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return x_rotated.flatten(-2).type_as(x)


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
    from pathlib import Path

    import matplotlib.pyplot as plt

    ### Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, d = 8, 64, 512
    X = torch.rand((b, l, d)).to(device)

    config = PLMConfig(hidden_size=d)

    attention_layer = Attention(config=config).to(device)

    Y = attention_layer(X)
    print(Y.shape)

    rope_length, rope_head_size = l, attention_layer.head_size
    rope_input = torch.zeros((1, 1, rope_length, rope_head_size), device=device)
    rope_input[..., 0::2] = 1.0
    rope_output = rope(rope_input)

    assert torch.allclose(
        rope_input.norm(dim=-1),
        rope_output.norm(dim=-1),
        atol=1e-5,
    ), "rope should preserve each token vector norm"
    assert torch.allclose(
        rope_input[:, :, 0, :],
        rope_output[:, :, 0, :],
        atol=1e-5,
    ), "position zero should not rotate"

    positions = torch.arange(rope_length).cpu()
    rope_values = rope_output[0, 0].detach().cpu()
    relative_similarity = (rope_values @ rope_values[0]) / rope_values[0].dot(rope_values[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    scatter = axes[0].scatter(
        rope_values[:, 0],
        rope_values[:, 1],
        c=positions,
        s=14,
        cmap="viridis",
    )
    axes[0].plot(rope_values[:, 0], rope_values[:, 1], color="black", alpha=0.25, linewidth=1)
    axes[0].set_title("First RoPE Pair")
    axes[0].set_xlabel("dim 0")
    axes[0].set_ylabel("dim 1")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=axes[0], label="position")

    heatmap = axes[1].imshow(
        rope_values.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        origin="lower",
    )
    axes[1].set_title("Rotated Head Features")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("head dim")
    fig.colorbar(heatmap, ax=axes[1], label="value")

    axes[2].plot(positions, relative_similarity)
    axes[2].set_title("Similarity to Position 0")
    axes[2].set_xlabel("position offset")
    axes[2].set_ylabel("cosine similarity")
    axes[2].set_ylim(-1.05, 1.05)

    fig.tight_layout()
    plot_path = Path("rope_effect.png").resolve()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"saved {plot_path}")
